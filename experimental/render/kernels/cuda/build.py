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

# Directory of this build.py file (experimental/render/kernels/cuda/)
PATH = os.path.dirname(os.path.abspath(__file__))

# Compute the path to gsplat/cuda/include/ relative to this file's location.
# Layout: <repo_root>/experimental/render/kernels/cuda/build.py
#         <repo_root>/gsplat/cuda/include/
_REPO_ROOT = os.path.normpath(os.path.join(PATH, "..", "..", "..", ".."))
_GSPLAT_INCLUDE = os.path.join(_REPO_ROOT, "gsplat", "cuda", "include")
_GSPLAT_CSRC = os.path.join(_REPO_ROOT, "gsplat", "cuda", "csrc")

DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "1" if DEBUG else "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"


def get_build_parameters():
    name = "experimental_gaussian_render_inference_scene_cuda"
    current_dir = PATH

    gaussian_render_inference_dir = os.path.join(
        current_dir, "csrc", "gaussian_inference"
    )
    # Include paths -----------------------------------
    extra_include_paths = [
        _GSPLAT_INCLUDE,
        _GSPLAT_CSRC,  # for Config.h (included as #include "Config.h")
        gaussian_render_inference_dir,
        os.path.join(_GSPLAT_CSRC, "third_party", "glm"),
    ]

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

    extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes"]
        extra_cflags += ["-Wno-unknown-pragmas"]

    extra_ldflags += [] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    if WITH_SYMBOLS:
        extra_cuda_cflags += ["-lineinfo"]

    import torch

    if torch.version.hip:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
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
            "experimental/render/kernels/cuda/build.py."
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
