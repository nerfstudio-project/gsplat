from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

try:
    import torch
    import torch.utils.cpp_extension as jit
except ImportError as e:
    if "pkg_resources" in str(e):
        raise ImportError("torch.utils.cpp_extension failed to import; install setuptools<82 or upgrade PyTorch") from e
    raise

try:
    from rich.console import Console
    _console = Console()
except ImportError:
    _console = None

PATH = os.path.dirname(os.path.abspath(__file__))
_GSPLAT_ROOT = os.path.normpath(os.path.join(PATH, "..", "..", "..", ".."))
_GSPLAT_INCLUDE = os.path.join(_GSPLAT_ROOT, "hip", "include")
_GSPLAT_CSRC = os.path.join(_GSPLAT_ROOT, "hip", "csrc")
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "1" if DEBUG else "0") == "1"
HIPCC_FLAGS = os.getenv("HIPCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def _common_flags():
    cflags = ["-std=c++20"]
    hipflags = []
    if sys.platform == "darwin" and platform.machine() == "arm64":
        cflags += ["-arch", "arm64"]
    cflags += ["-g", "-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    if FAST_MATH:
        hipflags += ["-ffast-math"]
    if WITH_SYMBOLS:
        hipflags += ["-gline-tables-only"]
    if os.name != "nt":
        cflags += ["-Wno-attributes", "-Wno-unknown-pragmas"]
    cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
    parinfo = torch.__config__.parallel_info()
    if "backend: OpenMP" in parinfo and "OpenMP not found" not in parinfo and sys.platform != "darwin":
        cflags += ["-DAT_PARALLEL_OPENMP", "-fopenmp"]
    hipflags += cflags
    if HIPCC_FLAGS:
        hipflags += HIPCC_FLAGS.split()
    return cflags, hipflags


def _rocm_include_paths():
    paths = []
    roots = [os.getenv("ROCM_HOME"), os.getenv("ROCM_PATH"), os.getenv("HIP_PATH")]
    hipcc = shutil.which("hipcc")
    if hipcc:
        roots.append(os.path.dirname(os.path.dirname(os.path.realpath(hipcc))))
    for root in roots:
        if not root:
            continue
        include_dir = os.path.join(root, "include")
        if os.path.isdir(include_dir) and include_dir not in paths:
            paths.append(include_dir)
    return paths
def get_build_parameters():
    current_dir = PATH
    inference_dir = os.path.join(current_dir, "csrc", "gaussian_inference")
    sources = [os.path.join(current_dir, "ext.cpp")]
    sources += [os.path.join(inference_dir, f) for f in [
        "GaussianRenderInferenceScene.hip",
        "IntersectCommon.hip",
        "IntersectMTFused.hip",
        "MacroTileIntersect.hip",
        "MacroTileRasterize.hip",
        "Projection.hip",
        "SegmentedSort.hip",
        "SHCompression.hip",
        "SphericalHarmonics.hip",
    ]]
    extra_cflags, extra_hip_cflags = _common_flags()
    return SimpleNamespace(
        name="experimental_gaussian_render_inference_scene_hip",
        extra_include_paths=[_GSPLAT_INCLUDE, _GSPLAT_CSRC, inference_dir, os.path.join(_GSPLAT_CSRC, "third_party", "glm")] + _rocm_include_paths(),
        sources=sources,
        extra_cflags=extra_cflags,
        extra_hip_cflags=extra_hip_cflags,
        extra_ldflags=[] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"],
    )


def build_and_load_experimental_gaussian_render_inference_scene():
    build_params = get_build_parameters()
    build_dir = jit._get_build_directory(build_params.name, verbose=False)
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass
    saved = os.path.join(build_dir, "build_params.json")
    changed = False
    try:
        if os.path.exists(saved):
            with open(saved, "r") as f:
                changed = SimpleNamespace(**json.load(f)) != build_params
    except Exception:
        changed = True
    if changed:
        shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)
    with open(saved, "w") as f:
        json.dump(build_params.__dict__, f)

    @contextmanager
    def status_context():
        tic = time.time()
        msg = f"experimental: Setting up Inference HIP extension with MAX_JOBS={MAX_JOBS if MAX_JOBS else 'max'}"
        ctx = _console.status(f"[bold yellow]{msg}", spinner="bouncingBall") if _console is not None else nullcontext()
        if _console is None:
            print(msg)
        with ctx:
            yield
        print(f"experimental: Inference HIP extension set up in {time.time() - tic:.2f} seconds.")

    module_exists = any(os.path.exists(os.path.join(build_dir, f"{build_params.name}{ext}")) for ext in (".so", ".pyd"))
    with (status_context() if not module_exists or changed else nullcontext()):
        envvars_to_remove = []
        try:
            if not NINJA_STATUS:
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "
            return jit.load(
                name=build_params.name,
                sources=build_params.sources,
                extra_cflags=build_params.extra_cflags,
                extra_cuda_cflags=build_params.extra_hip_cflags,
                extra_include_paths=build_params.extra_include_paths,
                extra_ldflags=build_params.extra_ldflags,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar, None)


__all__ = ["build_and_load_experimental_gaussian_render_inference_scene", "get_build_parameters"]
