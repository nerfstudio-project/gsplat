""" 
Trigger compiling:

FAST_COMPILE=1 TORCH_CUDA_ARCH_LIST="7.5" python -c "from gsplat.cuda2._backend import _C"
"""

import glob
import json
import os
import shutil
import time
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import (
    _TORCH_PATH,
    _check_and_build_extension_h_precompiler_headers,
    _get_build_directory,
    _import_module_from_library,
    _jit_compile,
    remove_extension_h_precompiler_headers,
)


def load_extension(
    name,
    sources,
    extra_cflags=None,
    extra_cuda_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    use_pch=True,
):
    """Load a JIT compiled extension."""
    # Make sure the build directory exists.
    if build_directory:
        os.makedirs(build_directory, exist_ok=True)

    # Using PreCompile Header('torch/extension.h') to reduce compile time.
    _check_and_build_extension_h_precompiler_headers(
        extra_cflags, extra_include_paths
    )
    head_file = os.path.join(_TORCH_PATH, 'include', 'torch', 'extension.h')
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


name = "_gsplat_cuda"
build_dir = _get_build_directory(name, verbose=False)
current_dir = os.path.dirname(os.path.abspath(__file__))
extra_include_paths = [os.path.join(current_dir, "include")]
extra_cflags = []
extra_cuda_cflags = []
# extra_cuda_cflags = ["-O0", "--use_fast_math"]
sources = list(glob.glob(os.path.join(current_dir, "csrc/*.cu"))) + list(
    glob.glob(os.path.join(current_dir, "csrc/*.cpp"))
)

shutil.rmtree(build_dir) # force rebuild to reveal compile time.
tic = time.time()
with Console().status(
    f"[bold yellow]gsplat: Setting up CUDA (This may take a few minutes the first time)",
    spinner="bouncingBall",
):
    _C = load_extension(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_dir,
        verbose=True,
    )
toc = time.time()
Console().print(
    f"[green]gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
)


__all__ = ["_C"]
