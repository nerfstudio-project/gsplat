# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT-compiles the sensorlib camera CUDA extension via torch.utils.cpp_extension.load.

Environment variables honored at import time:
    DEBUG:         Set to "1" to compile with -g -O0 instead of -O3 -DNDEBUG.
    FAST_MATH:     Set to "0" to disable -use_fast_math (default "1").
    NVCC_FLAGS:    Space-separated extra flags forwarded to nvcc.
    NINJA_STATUS:  Override the ninja progress format; defaults to "[%f/%t %r %es] ".
    VERBOSE:       Set to "1" to pass verbose=True to jit.load().

Environment variable honored at load time:
    GSPLAT_BUILD_LOCK_AGE_S: Maximum age in seconds of a stale ninja lock file
                             before it is removed (default 1800).  The build
                             directory follows torch's per-user
                             ~/.cache/torch/extensions/ (TORCH_EXTENSIONS_DIR).
"""

from __future__ import annotations

import json
import os
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
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def get_build_parameters() -> SimpleNamespace:
    """Enumerate source files and compiler flags for the CUDA extension.

    Returns:
        SimpleNamespace with fields: name, sources, extra_include_paths,
        extra_cflags, extra_cuda_cflags, extra_ldflags.
    """
    name = "gsplat_sensors_cuda"
    sources = [
        os.path.join(PATH, "ext.cpp"),
        os.path.join(PATH, "csrc", "camera_torch.cpp"),
        os.path.join(PATH, "csrc", "external_distortion_torch.cpp"),
        os.path.join(PATH, "csrc", "camera_kernel.cu"),
        os.path.join(PATH, "csrc", "camera_kernel_backward.cu"),
        os.path.join(PATH, "csrc", "ftheta_kernel.cu"),
        os.path.join(PATH, "csrc", "ftheta_kernel_backward.cu"),
    ]
    geometry_csrc = os.path.normpath(
        os.path.join(PATH, "..", "..", "..", "geometry", "kernels", "cuda", "csrc")
    )
    extra_include_paths: list[str] = [geometry_csrc]
    extra_cflags: list[str] = []
    extra_cuda_cflags: list[str] = []
    extra_ldflags: list[str] = []

    if sys.platform == "win32":
        extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
        extra_cuda_cflags += [
            "-std=c++20",
            "-allow-unsupported-compiler",
            "-Xcompiler",
            "/Zc:preprocessor",
            "-DWIN32_LEAN_AND_MEAN",
        ]
    else:
        extra_cflags = ["-std=c++20"]

    extra_cflags += ["-g", "-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []
    extra_cuda_cflags += ["-lineinfo"] if DEBUG else []
    extra_cflags += ["-Wno-attributes"]
    if os.name != "nt":
        extra_cflags += ["-Wno-sign-compare"]

    if torch.version.hip:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
    else:
        extra_cuda_cflags += ["--forward-unknown-opts", "--expt-relaxed-constexpr"]

    if sys.platform != "win32":
        extra_cuda_cflags += extra_cflags
    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
    )


@contextmanager
def _status_context(msg: str):
    """Print ``msg`` on entry and elapsed time on exit.

    Args:
        msg: Message to print before the managed block executes.
    """
    tic = time.time()
    print(msg, flush=True)
    try:
        yield
    finally:
        print(
            f"gsplat_sensors: extension ready in {time.time() - tic:.2f}s", flush=True
        )


def build_and_load_sensors_cuda():
    """JIT-compile and load the gsplat_sensors_cuda extension.

    Manages a JSON build-parameter snapshot alongside the build directory so
    that a compiler-flag change triggers a clean rebuild.  A stale ninja lock
    file (older than GSPLAT_BUILD_LOCK_AGE_S seconds) is removed before the
    build to avoid blocking on a crashed previous attempt.

    If jit.load() fails but a pre-built .so/.pyd already exists, falls back to
    importing the cached artefact directly.

    Returns:
        The loaded gsplat_sensors_cuda extension module.

    Raises:
        OSError: If both jit.load() and the cached-import fallback fail.
    """
    build_params = get_build_parameters()
    build_dir = jit._get_build_directory(build_params.name, verbose=False)

    lock_path = os.path.join(build_dir, "lock")
    lock_max_age_s = int(os.environ.get("GSPLAT_BUILD_LOCK_AGE_S", "1800"))
    try:
        age = time.time() - os.path.getmtime(lock_path)
        if age > lock_max_age_s:
            os.remove(lock_path)
    except OSError:
        pass

    saved_build_params_fname = os.path.join(build_dir, "build_params.json")
    saved_build_params = None
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, encoding="utf-8") as f:
                saved_build_params = json.load(f)
            build_params_changed = saved_build_params != vars(build_params)
    except Exception as e:
        print(
            f"gsplat_sensors: rebuilding (could not load saved build params: {e})",
            flush=True,
        )

    if build_params_changed and saved_build_params is not None:
        shutil.rmtree(build_dir, ignore_errors=True)
    os.makedirs(build_dir, exist_ok=True)

    with open(saved_build_params_fname, "w", encoding="utf-8") as f:
        json.dump(vars(build_params), f, sort_keys=True)

    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.pyd"))
    ctx = (
        _status_context("gsplat_sensors: compiling registration extension...")
        if (not module_exists or build_params_changed)
        else nullcontext()
    )

    envvars_to_remove: list[str] = []
    try:
        with ctx:
            if not NINJA_STATUS:
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "
            return jit.load(
                name=build_params.name,
                sources=build_params.sources,
                extra_cflags=build_params.extra_cflags,
                extra_cuda_cflags=build_params.extra_cuda_cflags,
                extra_include_paths=build_params.extra_include_paths,
                extra_ldflags=build_params.extra_ldflags,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
    except OSError as jit_error:
        if module_exists and not build_params_changed:
            try:
                return jit._import_module_from_library(
                    build_params.name, build_dir, True
                )
            except OSError as import_error:
                raise OSError(
                    "Failed to load cached gsplat_sensors_cuda after jit.load() failed.\n"
                    f"jit.load error: {jit_error!r}\n"
                    f"cached import error: {import_error!r}"
                ) from import_error
        raise
    finally:
        for envvar in envvars_to_remove:
            os.environ.pop(envvar, None)


__all__ = ["build_and_load_sensors_cuda", "get_build_parameters"]
