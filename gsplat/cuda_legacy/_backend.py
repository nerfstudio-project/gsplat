import glob
import json
import os
import shutil
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import (
    _get_build_directory,
    _import_module_from_library,
    load,
)

PATH = os.path.dirname(os.path.abspath(__file__))
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
):
    """Load a JIT compiled extension."""
    # If the JIT build happens concurrently in multiple processes,
    # race conditions can occur when removing the lock file at:
    # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
    # But it's ok so we catch this exception and ignore it.
    try:
        return load(
            name,
            sources,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=build_directory,
        )
    except OSError:
        # The module should be already compiled
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    """Check if the nvcc is avaiable on the machine."""
    try:
        call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
        return True
    except FileNotFoundError:
        return False


def cuda_toolkit_version():
    """Get the cuda toolkit version."""
    cuda_home = os.path.join(os.path.dirname(shutil.which("nvcc")), "..")
    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the cuda version.")
    return cuda_version


_C = None

try:
    # try to import the compiled module (via setup.py)
    from gsplat import csrc_legacy as _C
except ImportError:
    # if failed, try with JIT compilation
    if cuda_toolkit_available():
        name = "gsplat_cuda_legacy"
        build_dir = _get_build_directory(name, verbose=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        glm_path = os.path.join(current_dir, "csrc", "third_party", "glm")

        extra_include_paths = [os.path.join(PATH, "csrc/"), glm_path]
        extra_cflags = ["-O3"]
        extra_cuda_cflags = ["-O3"]
        sources = list(glob.glob(os.path.join(PATH, "csrc/*.cu"))) + list(
            glob.glob(os.path.join(PATH, "csrc/*.cpp"))
        )

        # If JIT is interrupted it might leave a lock in the build directory.
        # We dont want it to exist in any case.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(
            os.path.join(build_dir, "gsplat_cuda_legacy.so")
        ) or os.path.exists(os.path.join(build_dir, "gsplat_cuda_legacy.lib")):
            # If the build exists, we assume the extension has been built
            # and we can load it.

            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            with Console().status(
                "[bold yellow]gsplat (legacy): Setting up CUDA (This may take a few minutes the first time)",
                spinner="bouncingBall",
            ):
                _C = load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                )
    else:
        Console().print(
            "[yellow]gsplat (legacy): No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")


__all__ = ["_C"]
