"""
Trigger compiling (for debugging):

VERBOSE=1 DEBUG=1 TORCH_CUDA_ARCH_LIST="8.9" python -c "from gsplat.cuda._backend import _C"
"""

import glob
import json
import os
import shutil
import time
from subprocess import DEVNULL, call
from contextlib import nullcontext, contextmanager

import torch
from packaging import version
from rich.console import Console

import torch.utils.cpp_extension as jit

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
VERBOSE = os.getenv("VERBOSE", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
BUILD_CAMERA_WRAPPERS = os.getenv("BUILD_CAMERA_WRAPPERS", "1" if DEBUG else "0") == "1"

def build_gsplat():
    name = "gsplat_cuda"
    build_dir = jit._get_build_directory(name, verbose=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Make sure the build directory exists.
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)

    # Include paths -----------------------------------
    extra_include_paths = [
        os.path.join(PATH, "include/"),
        os.path.join(current_dir, "csrc", "third_party", "glm")
    ]

    # Source files ------------------------------------
    sources = (
        list(glob.glob(os.path.join(PATH, "csrc/*.cu")))
        + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
        + [os.path.join(PATH, "ext.cpp")]
    )

    # Compiler flags ----------------------------------
    extra_cflags = ["-std=c++20"]
    extra_cuda_cflags = ["--forward-unknown-opts"]

    # Debug/Release mode
    extra_cflags += ["-g","-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []

    # Silencing of warnings
    extra_cflags += ["-Wno-attributes"]

    # Whether to build the camera wrappers or not (for tests)
    if BUILD_CAMERA_WRAPPERS:
        extra_cflags += ["-DBUILD_CAMERA_WRAPPERS=1"]
    else:
        # Remove 'csrc/CameraWrappers.cu' from the sources list if it exists
        sources = [s for s in sources if not s.endswith("csrc/CameraWrappers.cu")]

    extra_cuda_cflags += extra_cflags

    # If JIT is interrupted it might leave a lock in the build directory.
    # We dont want it to exist in any case.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    @contextmanager
    def status_context():
        # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
        # if the build directory exists with a lock file in it.
        shutil.rmtree(build_dir)
        tic = time.time()
        with Console().status(
            f"[bold yellow]gsplat: Setting up CUDA with MAX_JOBS={os.environ['MAX_JOBS']} (This may take a few minutes the first time)",
            spinner="bouncingBall",
        ):
            yield

        toc = time.time()
        Console().print(
            f"[green]gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
        )

    # If the build exists, we assume the extension has been built
    # and we can load it.
    module_exists = os.path.exists(os.path.join(build_dir, f"{name}.so")) or os.path.exists(os.path.join(build_dir, f"{name}.lib"))

    with status_context() if not module_exists else nullcontext():
        # If the JIT build happens concurrently in multiple processes,
        # race conditions can occur when removing the lock file at:
        # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
        # But it's ok so we catch this exception and ignore it.
        envvars_to_remove = []
        try:
            if not MAX_JOBS:
                envvars_to_remove.append("MAX_JOBS")
                os.environ["MAX_JOBS"] = "10"

            if not NINJA_STATUS:
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "

            gsplat_module = jit.load(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_include_paths=extra_include_paths,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
            return gsplat_module
        except OSError:
            # The module should already be compiled if we get OSError
            return jit._import_module_from_library(name, build_dir, True)
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)

def cuda_toolkit_available():
    """
    Check more robustly if the CUDA toolkit is available.
    1. Attempt to locate `CUDA_HOME` using PyTorch’s internal method.
    2. Check if nvcc is present in that location.
    """
    cuda_home = jit._find_cuda_home()  # This tries various heuristics
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

_C = None

try:
    # Try to import the compiled module (via setup.py or pre-built .so)
    from gsplat import csrc as _C
except ImportError:
    # if that fails, try with JIT compilation
    if cuda_toolkit_available():
        _C = build_gsplat()
    else:
        Console().print(
            "[yellow]gsplat: No CUDA toolkit found. gsplat will be disabled.[/yellow]"
        )

__all__ = ["_C"]
