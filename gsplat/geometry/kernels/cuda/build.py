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

"""JIT build for the gsplat_geometry_cuda extension (quaternion ops, etc.)."""

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
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def _rocm_lib_dir() -> str | None:
    """Locate the pip ROCm SDK lib dir holding libamdhip64.

    torch resolves ROCM_HOME from the hipcc on PATH (here ``.venv/bin/hipcc``),
    so it emits ``-L<venv>/lib -lamdhip64`` — but the runtime lib actually lives
    in the ``_rocm_sdk_devel`` site-package. Return that dir so the JIT link can
    find libamdhip64; None when no pip SDK is present (e.g. system /opt/rocm).
    """
    import site

    candidates = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)
    for sp in candidates:
        for name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
            lib = os.path.join(sp, name, "lib")
            if os.path.exists(os.path.join(lib, "libamdhip64.so")):
                return lib
    return None


def _rocm_sdk_include_dir() -> str | None:
    """Locate the pip ROCm SDK include dir that ships hipsparse/hipsparse.h.

    torch resolves ROCM_HOME from the hipcc on PATH and emits its ROCm includes
    from ``_rocm_sdk_core``, but torch's own ``ATen/hip/HIPContextLight.h`` pulls
    in ``hipsparse/hipsparse.h`` which only ships in ``_rocm_sdk_devel``. Return
    that include dir so it can be added via ``-isystem``; None when no pip SDK is
    present (e.g. system /opt/rocm, where the headers are already unified).
    """
    import site

    candidates = list(site.getsitepackages())
    user_site = site.getusersitepackages()
    if user_site:
        candidates.append(user_site)
    for sp in candidates:
        for name in ("_rocm_sdk_devel", "_rocm_sdk_core"):
            inc = os.path.join(sp, name, "include")
            if os.path.isfile(os.path.join(inc, "hipsparse", "hipsparse.h")):
                return inc
    return None


def get_build_parameters() -> SimpleNamespace:
    name = "gsplat_geometry_cuda"
    # The csrc dir holds the .cuh headers included by the .cu sources. On ROCm,
    # torch's JIT hipify only hipifies headers it can find via header_include_dirs
    # (fed from extra_include_paths); without this the .cuh files are never
    # hipified/copied into the generated hip/csrc tree and the build fails with
    # "'pose.cuh' file not found".
    extra_include_paths: list[str] = [os.path.join(PATH, "csrc")]
    sources = [
        os.path.join(PATH, "ext.cpp"),
        os.path.join(PATH, "csrc", "quaternion.cu"),
        os.path.join(PATH, "csrc", "pose.cu"),
    ]

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

    if not torch.version.hip:
        extra_cuda_cflags += ["--forward-unknown-opts"]
    extra_cflags += ["-g", "-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []
    extra_cuda_cflags += ["-lineinfo"] if DEBUG else []
    extra_cflags += ["-Wno-attributes"]
    if os.name != "nt":
        extra_cflags += ["-Wno-sign-compare"]

    if torch.version.hip:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
        rocm_lib = _rocm_lib_dir()
        if rocm_lib is not None:
            extra_ldflags += [f"-L{rocm_lib}", f"-Wl,-rpath,{rocm_lib}"]
        # hipsparse.h (pulled in by torch's ATen/hip headers) ships only in the
        # _rocm_sdk_devel include tree, which torch's JIT does not add. Feed it
        # via -isystem; extra_cflags is merged into extra_cuda_cflags below, so
        # both host and device compiles pick it up.
        rocm_inc = _rocm_sdk_include_dir()
        if rocm_inc is not None:
            extra_cflags += [f"-isystem{rocm_inc}"]
        # ROCm's clang defaults to the newest installed gcc libstdc++. gcc-14's
        # <format> pulls in bits/unicode.h, which has a libstdc++ bug (`++this;`
        # instead of `++*this;`) that clang rejects but g++ tolerates. Pin clang
        # to a gcc toolchain whose headers compile — prefer 13, fall back to 12.
        # Clang-only flag: goes in extra_cuda_cflags, not extra_cflags.
        for _gcc_ver in ("13", "12"):
            _gcc_dir = f"/usr/lib/gcc/x86_64-linux-gnu/{_gcc_ver}"
            if os.path.isdir(_gcc_dir):
                extra_cuda_cflags.append(f"--gcc-install-dir={_gcc_dir}")
                break
    else:
        extra_cuda_cflags += ["--expt-relaxed-constexpr"]

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
    tic = time.time()
    print(msg, flush=True)
    try:
        yield
    finally:
        print(
            f"gsplat.geometry: CUDA extension ready in {time.time() - tic:.2f}s",
            flush=True,
        )


def build_and_load_geometry_cuda():
    build_params = get_build_parameters()
    build_dir = jit._get_build_directory(build_params.name, verbose=False)

    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    saved_build_params_fname = os.path.join(build_dir, "build_params.pkl")
    saved_build_params = None
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, "rb") as f:
                saved_build_params = pickle.load(f)
            build_params_changed = saved_build_params != build_params
    except Exception as e:
        print(
            f"gsplat.geometry: rebuilding (could not load saved build params: {e})",
            flush=True,
        )

    if build_params_changed and saved_build_params is not None:
        shutil.rmtree(build_dir, ignore_errors=True)

    if build_dir:
        os.makedirs(build_dir, exist_ok=True)

    with open(saved_build_params_fname, "wb") as f:
        pickle.dump(build_params, f)

    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.pyd"))

    ctx = (
        _status_context(
            "gsplat.geometry: compiling CUDA extension (first run may take a few minutes)..."
        )
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
                    "Failed to load cached gsplat.geometry CUDA extension after "
                    f"jit.load() failed for build directory {build_dir!r}.\n"
                    f"jit.load error: {jit_error!r}\n"
                    f"cached import error: {import_error!r}"
                ) from import_error
        raise
    finally:
        for envvar in envvars_to_remove:
            os.environ.pop(envvar, None)


__all__ = ["build_and_load_geometry_cuda", "get_build_parameters"]
