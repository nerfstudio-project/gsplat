from __future__ import annotations

import os
import pickle
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
import torch.utils.cpp_extension as jit

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
HIPCC_FLAGS = os.getenv("HIPCC_FLAGS", "")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def _common_flags():
    cflags = ["-std=c++20"]
    hipflags = []
    cflags += ["-g", "-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    if FAST_MATH:
        hipflags += ["-ffast-math"]
    if DEBUG:
        hipflags += ["-gline-tables-only"]
    if os.name != "nt":
        cflags += ["-Wno-attributes", "-Wno-sign-compare"]
    cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
    hipflags += cflags
    if HIPCC_FLAGS:
        hipflags += HIPCC_FLAGS.split()
    return cflags, hipflags


def _rocm_include_paths() -> list[str]:
    paths: list[str] = []
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
def get_build_parameters() -> SimpleNamespace:
    extra_cflags, extra_hip_cflags = _common_flags()
    return SimpleNamespace(
        name="gsplat_geometry_hip",
        extra_include_paths=_rocm_include_paths(),
        sources=[
            os.path.join(PATH, "ext.cpp"),
            os.path.join(PATH, "csrc", "quaternion.hip"),
            os.path.join(PATH, "csrc", "pose.hip"),
        ],
        extra_cflags=extra_cflags,
        extra_hip_cflags=extra_hip_cflags,
        extra_ldflags=[],
    )


@contextmanager
def _status_context(msg: str):
    tic = time.time()
    print(msg, flush=True)
    try:
        yield
    finally:
        print(f"gsplat.geometry: HIP extension ready in {time.time() - tic:.2f}s", flush=True)


def build_and_load_geometry_hip():
    build_params = get_build_parameters()
    build_dir = jit._get_build_directory(build_params.name, verbose=False)
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass
    saved = os.path.join(build_dir, "build_params.pkl")
    changed = False
    try:
        if os.path.exists(saved):
            with open(saved, "rb") as f:
                changed = pickle.load(f) != build_params
    except Exception:
        changed = True
    if changed:
        shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)
    with open(saved, "wb") as f:
        pickle.dump(build_params, f)
    module_exists = any(os.path.exists(os.path.join(build_dir, f"{build_params.name}{ext}")) for ext in (".so", ".pyd"))
    ctx = _status_context("gsplat.geometry: compiling HIP extension...") if (not module_exists or changed) else nullcontext()
    with ctx:
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


__all__ = ["build_and_load_geometry_hip", "get_build_parameters"]
