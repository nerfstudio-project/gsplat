""" 
Trigger compiling (for debugging):

VERBOSE=1 FAST_COMPILE=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"
"""

import glob
import json
import os
import shutil
import time
from subprocess import DEVNULL, call

import torch
from packaging import version
from rich.console import Console
from torch.utils.cpp_extension import _find_cuda_home  # <--- For robust CUDA detection
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    _get_build_directory,
    _import_module_from_library,
    _jit_compile,
)

PATH = os.path.dirname(os.path.abspath(__file__))
NO_FAST_MATH = os.getenv("NO_FAST_MATH", "0") == "1"
FAST_COMPILE = os.getenv("FAST_COMPILE", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
USE_PRECOMPILED_HEADERS = os.getenv("USE_PRECOMPILED_HEADERS", "0") == "1"
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"

# torch has bugs on precompiled headers before 2.2, see:
# https://github.com/nerfstudio-project/gsplat/pull/583#issuecomment-2732597080
if version.parse(torch.__version__) < version.parse("2.2") and USE_PRECOMPILED_HEADERS:
    Console().print(
        "[yellow]gsplat: Precompiled headers are enabled but torch version is lower than 2.2. Disabling it.[/yellow]"
    )
    USE_PRECOMPILED_HEADERS = False


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
):
    """Load a JIT compiled extension."""
    # Make sure the build directory exists.
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    # If the JIT build happens concurrently in multiple processes,
    # race conditions can occur when removing the lock file at:
    # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
    # But it's ok so we catch this exception and ignore it.
    try:
        if USE_PRECOMPILED_HEADERS:
            from torch.utils.cpp_extension import (
                _check_and_build_extension_h_precompiler_headers,
            )

            # Using PreCompiled Header('torch/extension.h') to reduce compile time.
            # remove: remove_extension_h_precompiler_headers()
            _check_and_build_extension_h_precompiler_headers(
                extra_cflags, extra_include_paths
            )
            head_file = os.path.join(_TORCH_PATH, "include", "torch", "extension.h")
            extra_cflags += ["-include", head_file, "-Winvalid-pch"]

        return _jit_compile(
            name,
            sources,
            extra_cflags,
            extra_cuda_cflags,
            extra_ldflags,
            extra_include_paths,
            build_directory,
            verbose,
            with_cuda=None,
            is_python_module=True,
            is_standalone=False,
            keep_intermediates=True,
        )
    except OSError:
        # The module should already be compiled if we get OSError
        return _import_module_from_library(name, build_directory, True)


def cuda_toolkit_available():
    """
    Check more robustly if the CUDA toolkit is available.
    1. Attempt to locate `CUDA_HOME` using PyTorch’s internal method.
    2. Check if nvcc is present in that location.
    """
    cuda_home = _find_cuda_home()  # This tries various heuristics
    if not cuda_home:
        return False

    # If we have a cuda_home, check if nvcc exists there:
    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc_path):
        # Maybe still on PATH, try calling "nvcc" directly:
        try:
            call(["nvcc"], stdout=DEVNULL, stderr=DEVNULL)
            return True
        except FileNotFoundError:
            return False
    return True


def cuda_toolkit_version():
    """Get the CUDA toolkit version if we found CUDA home."""
    cuda_home = _find_cuda_home()
    if not cuda_home:
        return None

    if os.path.exists(os.path.join(cuda_home, "version.txt")):
        with open(os.path.join(cuda_home, "version.txt")) as f:
            cuda_version = f.read().strip().split()[-1]
    elif os.path.exists(os.path.join(cuda_home, "version.json")):
        with open(os.path.join(cuda_home, "version.json")) as f:
            cuda_version = json.load(f)["cuda"]["version"]
    else:
        raise RuntimeError("Cannot find the CUDA version file in CUDA_HOME.")
    return cuda_version


_C = None

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from gsplat import csrc as _C
except ImportError:
    # if that fails, try with JIT compilation
    if cuda_toolkit_available():
        name = "gsplat_cuda"
        build_dir = _get_build_directory(name, verbose=False)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        glm_path = os.path.join(current_dir, "csrc", "third_party", "glm")

        extra_include_paths = [os.path.join(PATH, "include/"), glm_path]
        opt_level = "-O0" if FAST_COMPILE else "-O3"
        extra_cflags = [opt_level, "-Wno-attributes"]
        extra_cuda_cflags = [opt_level]
        if not NO_FAST_MATH:
            extra_cuda_cflags += ["-use_fast_math"]
        sources = (
            list(glob.glob(os.path.join(PATH, "csrc/*.cu")))
            + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
            + [os.path.join(PATH, "ext.cpp")]
        )

        # If JIT is interrupted it might leave a lock in the build directory.
        # We dont want it to exist in any case.
        try:
            os.remove(os.path.join(build_dir, "lock"))
        except OSError:
            pass

        if os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(
            os.path.join(build_dir, f"{name}.lib")
        ):
            # If the build exists, we assume the extension has been built
            # and we can load it.
            _C = load_extension(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
        else:
            # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
            # if the build directory exists with a lock file in it.
            shutil.rmtree(build_dir)
            tic = time.time()
            with Console().status(
                f"[bold yellow]gsplat: Setting up CUDA with MAX_JOBS={os.environ['MAX_JOBS']} (This may take a few minutes the first time)",
                spinner="bouncingBall",
            ):
                _C = load_extension(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_cuda_cflags=extra_cuda_cflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_dir,
                    verbose=VERBOSE,
                )
            toc = time.time()
            Console().print(
                f"[green]gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
            )

    else:
        Console().print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )

if need_to_unset_max_jobs:
    os.environ.pop("MAX_JOBS")


__all__ = ["_C"]
