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

"""Build parameters for the experimental Inference render CUDA extension."""

import os
import sys
import platform
import json
import shutil
import time
from contextlib import nullcontext, contextmanager
from types import SimpleNamespace

from torch.utils.cpp_extension import CUDA_HOME

try:
    import torch.utils.cpp_extension as jit
except ImportError as e:
    if "pkg_resources" in str(e):
        raise ImportError(
            "torch.utils.cpp_extension failed to import because 'pkg_resources' "
            "is no longer available in setuptools >= 82. "
            "Fix: pip install 'setuptools<82'"
        ) from e
    raise

try:
    from rich.console import Console

    _console = Console()
except ImportError:
    _console = None

# Directory of this build.py file (gsplat/experimental/render/kernels/cuda/)
PATH = os.path.dirname(os.path.abspath(__file__))

# Compute the path to gsplat/cuda/include/ relative to this file's location.
# Layout: <gsplat_pkg>/experimental/render/kernels/cuda/build.py
#         <gsplat_pkg>/cuda/include/
# Walk up four levels to the gsplat package root, then descend into cuda/.
_GSPLAT_ROOT = os.path.normpath(os.path.join(PATH, "..", "..", "..", ".."))
_GSPLAT_INCLUDE = os.path.join(_GSPLAT_ROOT, "cuda", "include")
_GSPLAT_CSRC = os.path.join(_GSPLAT_ROOT, "cuda", "csrc")

DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "1" if DEBUG else "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def _rocm_sdk_include_dir():
    """Locate the pip ROCm SDK include dir that ships hipsparse/hipsparse.h.

    torch emits its ROCm includes from ``_rocm_sdk_core``, but torch's own
    ``ATen/hip`` headers pull in ``hipsparse/hipsparse.h`` which only ships in
    ``_rocm_sdk_devel``. Return that include dir so it can be added via
    ``-isystem``; None when no pip SDK is present (e.g. system /opt/rocm).
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


def _rocm_lib_dir():
    """Locate the pip ROCm SDK lib dir holding the libamdhip64.so link symlink.

    torch links against ``_rocm_sdk_core/lib``, which only ships the versioned
    ``libamdhip64.so.7`` — the unversioned ``libamdhip64.so`` that ``-lamdhip64``
    needs lives in ``_rocm_sdk_devel/lib``. Return that dir; None when no pip SDK
    is present (e.g. system /opt/rocm, where the link name already resolves).
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


def get_build_parameters():
    import torch

    name = "experimental_gaussian_render_inference_scene_cuda"
    current_dir = PATH

    gaussian_render_inference_dir = os.path.join(
        current_dir, "csrc", "gaussian_inference"
    )
    _glm_dir = os.path.join(_GSPLAT_CSRC, "third_party", "glm")
    # Include paths -----------------------------------
    extra_include_paths = [
        _GSPLAT_CSRC,  # for Config.h (included as #include "Config.h")
        gaussian_render_inference_dir,
    ]
    # GLM is a header-only template library that must NOT be hipified. torch's
    # JIT hipify treats every extra_include_paths entry as a tree to rewrite,
    # and it mangles GLM's relative includes while dropping its .inl files
    # (breaking the build). Under HIP, pass GLM as a plain -I compile flag so
    # hipify leaves it untouched; on CUDA it can stay in extra_include_paths.
    _glm_include_flag = f"-I{_glm_dir}"
    # cuda/include holds the shared headers (Common.h, Utils.cuh, RocmCeilDiv.cuh).
    # If left in extra_include_paths under HIP, hipify mirrors it into a divergent
    # hip/include copy and rewrites the .hip sources to include that copy, while
    # Utils.cuh keeps pulling cuda/include — two paths defeat #pragma once and
    # every shared symbol is redefined. Pass it as a raw -I under HIP (same as GLM)
    # so hipify never copies it and all includes resolve to the single cuda tree.
    _gsplat_include_flag = f"-I{_GSPLAT_INCLUDE}"
    if not torch.version.hip:
        extra_include_paths.insert(0, _GSPLAT_INCLUDE)
        extra_include_paths.append(_glm_dir)

    # Fix for CUDA 12+ in conda environment
    if CUDA_HOME and os.path.isdir(os.path.join(CUDA_HOME, "targets")):
        for arch in os.listdir(os.path.join(CUDA_HOME, "targets")):
            if os.path.isdir(p := os.path.join(CUDA_HOME, "targets", arch, "include")):
                extra_include_paths.append(p)
                if os.path.isdir(
                    p := os.path.join(CUDA_HOME, "targets", arch, "include", "cccl")
                ):
                    extra_include_paths.append(p)

    # Source files ------------------------------------
    sources = [os.path.join(current_dir, "ext.cpp")]
    if os.path.isdir(gaussian_render_inference_dir):
        sources.append(
            os.path.join(
                gaussian_render_inference_dir, "GaussianRenderInferenceScene.cu"
            )
        )
    if os.path.isdir(gaussian_render_inference_dir):
        sources += [
            os.path.join(gaussian_render_inference_dir, f)
            for f in [
                "IntersectCommon.cu",
                "IntersectMTFused.cu",
                "MacroTileIntersect.cu",
                "MacroTileRasterize.cu",
                "Projection.cu",
                "SegmentedSort.cu",
                "SHCompression.cu",
                "SphericalHarmonics.cu",
            ]
        ]

    # Compiler flags ----------------------------------
    extra_cflags = []
    extra_cuda_cflags = []
    extra_ldflags = []

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

    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    if not torch.version.hip:
        extra_cuda_cflags += ["--forward-unknown-opts"]

    if DEBUG:
        extra_cflags += ["-g", "-O0"]
        if sys.platform != "win32":
            extra_cflags += ["-Wall"]
            extra_cuda_cflags += [
                "-Xcompiler=-Werror",
                "--Werror",
                "all-warnings",
            ]
        else:
            extra_cflags += ["/Zi", "/Od"]
            extra_cuda_cflags += ["-Od"]
    else:
        if sys.platform != "win32":
            extra_cflags += ["-O3", "-DNDEBUG"]
        else:
            extra_cflags += ["/O2", "-DNDEBUG"]
            extra_cuda_cflags += ["-O2", "-DNDEBUG"]

    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []

    if not torch.version.hip:
        extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes"]
        extra_cflags += ["-Wno-unknown-pragmas"]

    extra_ldflags += [] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    if WITH_SYMBOLS:
        extra_cuda_cflags += ["-lineinfo"] if not torch.version.hip else []

    if torch.version.hip:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
        # GLM and cuda/include passed as raw -I so hipify never rewrites them.
        # (extra_cflags is later folded into extra_cuda_cflags, covering nvcc.)
        extra_cflags.append(_glm_include_flag)
        extra_cflags.append(_gsplat_include_flag)
        # hipsparse.h (pulled in by torch's ATen/hip headers) ships only in the
        # _rocm_sdk_devel include tree, which torch's JIT does not add.
        _rocm_inc = _rocm_sdk_include_dir()
        if _rocm_inc is not None:
            extra_cflags.append(f"-isystem{_rocm_inc}")
            # ROCm 10.x ships libcu++ under include/hipccl/, not include/.
            # Utils.cuh includes <cuda/std/limits>; a raw -I keeps hipify from
            # rewriting the CCCL tree (same reason GLM is not in extra_include_paths).
            _hipccl = os.path.join(_rocm_inc, "hipccl")
            if os.path.isdir(os.path.join(_hipccl, "cuda", "std")):
                extra_cflags.append(f"-I{_hipccl}")
        # torch links against _rocm_sdk_core/lib, which only has the versioned
        # libamdhip64.so.7; the unversioned link name -lamdhip64 needs lives in
        # _rocm_sdk_devel/lib. Add it so the final .so link resolves.
        _rocm_lib = _rocm_lib_dir()
        if _rocm_lib is not None:
            extra_ldflags += [f"-L{_rocm_lib}", f"-Wl,-rpath,{_rocm_lib}"]
        # ROCm's clang defaults to the newest installed gcc libstdc++. gcc-14's
        # <format> pulls in bits/unicode.h, which has a libstdc++ bug (`++this;`
        # instead of `++*this;`) that clang rejects but g++ tolerates. Pin clang
        # to a gcc toolchain whose headers compile — prefer 13, fall back to 12.
        # This is a clang-only flag, so it must go in the CUDA (clang) flags and
        # NOT extra_cflags (which also drives the host g++ compile of ext.cpp).
        for _gcc_ver in ("13", "12"):
            _gcc_dir = f"/usr/lib/gcc/x86_64-linux-gnu/{_gcc_ver}"
            if os.path.isdir(_gcc_dir):
                extra_cuda_cflags.append(f"--gcc-install-dir={_gcc_dir}")
                break
        if not DEBUG:
            # Suppress hipcc -Wpass-failed=transform-warning from #pragma unroll.
            extra_cuda_cflags += ["-Wno-pass-failed"]
    else:
        extra_cuda_cflags += ["--expt-relaxed-constexpr"]

    parinfo = torch.__config__.parallel_info()
    if (
        "backend: OpenMP" in parinfo
        and "OpenMP not found" not in parinfo
        and sys.platform != "darwin"
    ):
        extra_cflags += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_cflags += ["/openmp"]
            extra_cuda_cflags += ["-Xcompiler", "/openmp"]
        else:
            extra_cflags += ["-fopenmp"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DAT_PARALLEL_OPENMP"]
    else:
        print("Compiling without OpenMP...")

    if sys.platform != "win32":
        extra_cuda_cflags += extra_cflags

    if DEBUG and sys.platform != "win32":
        extra_cflags += ["-Werror"]

    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
    )


def build_and_load_experimental_gaussian_render_inference_scene():
    build_params = get_build_parameters()

    if not hasattr(jit, "_get_build_directory"):
        raise RuntimeError(
            "torch.utils.cpp_extension.jit._get_build_directory is missing — "
            "PyTorch upgrade broke a private API. Update "
            "gsplat/experimental/render/kernels/cuda/build.py."
        )

    build_dir = jit._get_build_directory(build_params.name, verbose=False)

    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    saved_build_params_fname = os.path.join(build_dir, "build_params.json")
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, "r") as f:
                saved = SimpleNamespace(**json.load(f))
            build_params_changed = saved != build_params
    except Exception:
        build_params_changed = True

    if build_params_changed:
        shutil.rmtree(build_dir, ignore_errors=True)

    os.makedirs(build_dir, exist_ok=True)

    with open(saved_build_params_fname, "w") as f:
        json.dump(build_params.__dict__, f)

    @contextmanager
    def status_context():
        tic = time.time()
        msg = (
            f"experimental: Setting up Inference CUDA extension with "
            f"MAX_JOBS={MAX_JOBS if MAX_JOBS else 'max'} "
            f"(This may take a few minutes the first time)"
        )
        if _console is not None:
            ctx = _console.status(f"[bold yellow]{msg}", spinner="bouncingBall")
        else:
            print(msg)
            ctx = nullcontext()
        with ctx:
            yield
        toc = time.time()
        if _console is not None:
            _console.print(
                f"[green]experimental: Inference CUDA extension set up in "
                f"{toc - tic:.2f} seconds.[/green]"
            )
        else:
            print(
                f"experimental: Inference CUDA extension set up in "
                f"{toc - tic:.2f} seconds."
            )

    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.pyd"))

    with (
        status_context() if not module_exists or build_params_changed else nullcontext()
    ):
        envvars_to_remove = []
        try:
            if not NINJA_STATUS:
                envvars_to_remove.append("NINJA_STATUS")
                os.environ["NINJA_STATUS"] = "[%f/%t %r %es] "

            module = jit.load(
                name=build_params.name,
                sources=build_params.sources,
                extra_cflags=build_params.extra_cflags,
                extra_cuda_cflags=build_params.extra_cuda_cflags,
                extra_include_paths=build_params.extra_include_paths,
                extra_ldflags=build_params.extra_ldflags,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
            return module
        except OSError:
            return jit._import_module_from_library(build_params.name, build_dir, True)
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)


__all__ = [
    "build_and_load_experimental_gaussian_render_inference_scene",
    "get_build_parameters",
]
