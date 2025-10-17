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

import os
import shutil
import time
import glob
import sys
import torch
import platform
from types import SimpleNamespace
import torch.utils.cpp_extension as jit
from contextlib import nullcontext, contextmanager
from rich.console import Console

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
FAST_MATH = os.getenv("FAST_MATH", "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
USE_PRECOMPILED_HEADERS = os.getenv("USE_PRECOMPILED_HEADERS", "0") == "1"
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

def get_build_parameters():
    name = "gsplat_cuda"
    current_dir = os.path.dirname(os.path.abspath(__file__))

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
    extra_cflags = []
    extra_cuda_cflags = []
    extra_ldflags = []

    if sys.platform == "win32":
        extra_cflags += ["-DWIN32_LEAN_AND_MEAN"]
        extra_cuda_cflags += ["-allow-unsupported-compiler"]

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    extra_cuda_cflags += ["--forward-unknown-opts"]

    # Debug/Release mode
    extra_cflags += ["-g","-O0"] if DEBUG else ["-O3", "-DNDEBUG"]
    extra_cuda_cflags += ["-use_fast_math"] if FAST_MATH else []

    # Silencing of warnings
    extra_cflags += ["-Wno-attributes"]
    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
    extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-sign-compare"]

    extra_ldflags += [] if WITH_SYMBOLS else ["-s"]

    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
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
        extra_cflags += ["/openmp"] if sys.platform == "win32" else ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    extra_cuda_cflags += extra_cflags
    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags
    )

def build_and_load_gsplat():
    build_params = get_build_parameters()

    build_dir = jit._get_build_directory(build_params.name, verbose=False)
    # Make sure the build directory exists.
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)


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
    module_exists = os.path.exists(os.path.join(build_dir, f"{build_params.name}.so")) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.lib"))

    with status_context() if not module_exists else nullcontext():
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
                name=build_params.name,
                sources=build_params.sources,
                extra_cflags=build_params.extra_cflags,
                extra_cuda_cflags=build_params.extra_cuda_cflags,
                extra_include_paths=build_params.extra_include_paths,
                extra_ldflags=build_params.extra_ldflags,
                build_directory=build_dir,
                verbose=VERBOSE,
            )
            return gsplat_module
        except OSError:
            # The module should already be compiled if we get OSError
            return jit._import_module_from_library(build_params.name, build_dir, True)
        finally:
            for envvar in envvars_to_remove:
                os.environ.pop(envvar)

__all__ = [
    "get_build_parameters"
]
