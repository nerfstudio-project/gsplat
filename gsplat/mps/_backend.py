import glob
import json
import os
import shutil
from subprocess import DEVNULL, call

from rich.console import Console
from torch.utils.cpp_extension import _get_build_directory, load

PATH = os.path.dirname(os.path.abspath(__file__))


name = "gsplat_mps"
build_dir = _get_build_directory(name, verbose=False)
extra_include_paths = []  # [os.path.join(PATH, "csrc/third_party/glm")]
extra_cflags = ["-O3"]
extra_mps_cflags = ["-O3"]

_C = None
sources = list(glob.glob(os.path.join(PATH, "csrc/*.mm"))) + list(
    glob.glob(os.path.join(PATH, "csrc/*.cpp"))
)
# sources = [
#     os.path.join(PATH, "csrc/ext.cpp"),
#     os.path.join(PATH, "csrc/rasterize.cu"),
#     os.path.join(PATH, "csrc/bindings.cu"),
#     os.path.join(PATH, "csrc/forward.cu"),
#     os.path.join(PATH, "csrc/backward.cu"),
# ]

try:
    # try to import the compiled module (via setup.py)
    from gsplat import csrc as _C
except ImportError:
    # if failed, try with JIT compilation
    # If JIT is interrupted it might leave a lock in the build directory.
    # We dont want it to exist in any case.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    if os.path.exists(os.path.join(build_dir, "gsplat_mps.so")) or os.path.exists(
        os.path.join(build_dir, "gsplat_mps.lib")
    ):
        # If the build exists, we assume the extension has been built
        # and we can load it.

        _C = load(
            name=name,
            sources=sources,
            extra_cflags=extra_cflags,
            extra_include_paths=extra_include_paths,
        )
    else:
        # Build from scratch. Remove the build directory just to be safe: pytorch jit might stuck
        # if the build directory exists with a lock file in it.
        shutil.rmtree(build_dir)
        with Console().status(
            "[bold yellow]gsplat: Setting up mps (This may take a few minutes the first time)",
            spinner="bouncingBall",
        ):
            _C = load(
                name=name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_include_paths=extra_include_paths,
            )


__all__ = ["_C"]
