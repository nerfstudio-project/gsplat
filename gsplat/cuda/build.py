# SPDX-FileCopyrightText: Copyright 2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
import json
from types import SimpleNamespace

try:
    import torch.utils.cpp_extension as jit
except ImportError as e:
    if "pkg_resources" in str(e):
        raise ImportError(
            "torch.utils.cpp_extension failed to import because 'pkg_resources' "
            "is no longer available in setuptools >= 82. "
            "Fix: pip install 'setuptools<82'\n"
            "This is a known issue with PyTorch < 2.9. "
            "Alternatively, upgrade to PyTorch >= 2.9."
        ) from e
    raise
from contextlib import nullcontext, contextmanager

try:
    from rich.console import Console

    _console = Console()
except ImportError:
    _console = None

PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "0") == "1"
# ROCm's -ffast-math is more aggressive than CUDA's -use_fast_math: it perturbs a
# few ill-conditioned projection-covariance gradients enough to exceed gsplat's
# own test tolerances (a handful of single-element outliers). Default fast-math
# OFF on ROCm so the test suite passes out of the box; users can opt back in with
# FAST_MATH=1. CUDA keeps the historical default of on.
FAST_MATH = os.getenv("FAST_MATH", "0" if torch.version.hip else "1") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
NVCC_FLAGS = os.getenv("NVCC_FLAGS", "")
MAX_JOBS = os.getenv("MAX_JOBS")
NINJA_STATUS = os.getenv("NINJA_STATUS")
VERBOSE = os.getenv("VERBOSE", "0") == "1"

# On ROCm, 3DGUT needs cuda::std::optional. The ROCm stack ships no libcu++, but
# the header-only ROCm/libhipcxx port provides the cuda::std namespace. Point
# LIBHIPCXX_INCLUDE at its include/ dir (a clone of github.com/ROCm/libhipcxx) to
# build 3DGUT on ROCm; if unset, 3DGUT stays off so a plain pip install does not
# hard-error on the missing header. Opt in with BUILD_3DGUT=1. CUDA is unchanged.
LIBHIPCXX_INCLUDE = os.getenv("LIBHIPCXX_INCLUDE")
_rocm_3dgut_default = "1" if (torch.version.hip and LIBHIPCXX_INCLUDE) else "0"
BUILD_3DGUT = os.getenv("BUILD_3DGUT", _rocm_3dgut_default if torch.version.hip else None)
BUILD_3DGS = os.getenv("BUILD_3DGS")
BUILD_2DGS = os.getenv("BUILD_2DGS")
BUILD_ADAM = os.getenv("BUILD_ADAM")
BUILD_RELOC = os.getenv("BUILD_RELOC")
BUILD_LOSSES = os.getenv("BUILD_LOSSES")
BUILD_CAMERA_WRAPPERS = os.getenv("BUILD_CAMERA_WRAPPERS", "1" if DEBUG else "0") == "1"

NUM_CHANNELS = os.getenv("NUM_CHANNELS")


def get_build_parameters():
    name = "gsplat_cuda"
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Include paths -----------------------------------
    extra_include_paths = [
        os.path.join(PATH, "include/"),
        os.path.join(current_dir, "csrc", "third_party", "glm"),
    ]

    # Source files ------------------------------------
    # *_winhip.cu are generated .cu shims of the host .cpp wrappers (see the
    # win32+HIP block near the end of this function); exclude them from the glob
    # so a rebuild does not pick up stale copies as standalone device sources.
    sources = (
        [s for s in glob.glob(os.path.join(PATH, "csrc/*.cu"))
         if not s.endswith("_winhip.cu")]
        + list(glob.glob(os.path.join(PATH, "csrc/*.cpp")))
        + [os.path.join(PATH, "ext.cpp")]
    )

    # Compiler flags ----------------------------------
    extra_cflags = []
    extra_cuda_cflags = []
    extra_ldflags = []

    if sys.platform == "win32":
        extra_cflags += ["/std:c++20", "/Zc:preprocessor", "-DWIN32_LEAN_AND_MEAN"]
        if torch.version.hip:
            # The Windows device compiler is amdclang (gcc-style driver), not
            # nvcc/cl: -allow-unsupported-compiler is nvcc-only and the MSVC
            # /Zc:preprocessor conformance flag is meaningless to clang (whose
            # preprocessor is already conforming). Only -std and the portable
            # -D survive to the hipcc device pass; the MSVC host (.cpp via cl)
            # still gets /Zc:preprocessor above.
            extra_cuda_cflags += ["-std=c++20", "-DWIN32_LEAN_AND_MEAN"]
        else:
            extra_cuda_cflags += [
                "-std=c++20",
                "-allow-unsupported-compiler",
                "-Xcompiler",
                "/Zc:preprocessor",
                "-DWIN32_LEAN_AND_MEAN",
            ]
    else:
        extra_cflags = ["-std=c++20"]

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_cflags += ["-arch", "arm64"]
        extra_ldflags += ["-arch", "arm64"]

    # --forward-unknown-opts is an nvcc-only flag; hipcc/clang++ rejects it.
    if not torch.version.hip:
        extra_cuda_cflags += ["--forward-unknown-opts"]

    # Debug/Release mode
    # MSVC (cl) does not support -O3/-O0; use -O2/-Od (torch converts - to /)
    if sys.platform == "win32":
        if DEBUG:
            extra_cflags += ["/Zi", "/Od"]
            extra_cuda_cflags += ["-Od"]
        else:
            extra_cflags += ["/O2", "-DNDEBUG"]
            extra_cuda_cflags += ["-O2", "-DNDEBUG"]
    else:
        if DEBUG:
            extra_cflags += ["-g", "-O0", "-Wall"]
            extra_cuda_cflags += [
                "-lineinfo",
                # nvcc intercepts bare -Werror as its own --Werror flag, so
                # pass it via -Xcompiler instead of --forward-unknown-opts.
                "-Xcompiler=-Werror",
                "--Werror",
                "all-warnings",
            ]
        else:
            extra_cflags += ["-O3", "-DNDEBUG"]
    if FAST_MATH:
        # nvcc spells it -use_fast_math; hipcc/clang++ uses -ffast-math.
        extra_cuda_cflags += ["-ffast-math"] if torch.version.hip else ["-use_fast_math"]

    # Silencing of warnings
    # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses.
    # -diag-suppress is nvcc-only; hipcc/clang++ does not accept it.
    if not torch.version.hip:
        extra_cuda_cflags += ["-diag-suppress", "20012,186"]
    if not os.name == "nt":
        extra_cflags += ["-Wno-attributes"]
        # #pragma unroll is standard CUDA idiom but unknown to gcc
        extra_cflags += ["-Wno-unknown-pragmas"]

    if BUILD_2DGS is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_2DGS={BUILD_2DGS}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_2DGS={BUILD_2DGS}"]
    if BUILD_3DGS is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_3DGS={BUILD_3DGS}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_3DGS={BUILD_3DGS}"]
    if BUILD_3DGUT is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_3DGUT={BUILD_3DGUT}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_3DGUT={BUILD_3DGUT}"]
    if BUILD_ADAM is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_ADAM={BUILD_ADAM}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_ADAM={BUILD_ADAM}"]
    if BUILD_RELOC is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_RELOC={BUILD_RELOC}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_RELOC={BUILD_RELOC}"]
    if BUILD_LOSSES is not None:
        extra_cflags += [f"-DGSPLAT_BUILD_LOSSES={BUILD_LOSSES}"]
        if sys.platform == "win32":
            extra_cuda_cflags += [f"-DGSPLAT_BUILD_LOSSES={BUILD_LOSSES}"]
    if BUILD_CAMERA_WRAPPERS:
        extra_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DGSPLAT_BUILD_CAMERA_WRAPPERS=1"]
    else:
        # Remove 'csrc/CameraWrappers.cu' from the sources list if it exists
        sources = [s for s in sources if not s.endswith("csrc/CameraWrappers.cu")]

    extra_ldflags += [] if WITH_SYMBOLS or sys.platform == "win32" else ["-s"]

    if torch.version.hip:
        # USE_ROCM was added to later versions of PyTorch.
        # Define here to support older PyTorch versions as well:
        extra_cflags += ["-DUSE_ROCM", "-U__HIP_NO_HALF_CONVERSIONS__"]
        # 3DGUT's cuda::std::optional comes from ROCm/libhipcxx (header-only). Add
        # its include/ to the device-compile flags so <cuda/std/*> resolves under
        # hipcc. NVIDIA gets cuda::std from the CUDA toolkit, so this is ROCm-only.
        if LIBHIPCXX_INCLUDE:
            extra_cuda_cflags += [f"-I{LIBHIPCXX_INCLUDE}"]
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
            if torch.version.hip:
                # amdclang on Windows is a gcc-style driver: it wants -fopenmp,
                # not MSVC's /openmp (which it would treat as an input file).
                extra_cuda_cflags += ["-fopenmp"]
            else:
                extra_cuda_cflags += ["-Xcompiler", "/openmp"]
        else:
            extra_cflags += ["-fopenmp"]
        if sys.platform == "win32":
            extra_cuda_cflags += ["-DAT_PARALLEL_OPENMP"]
    else:
        print("Compiling without OpenMP...")

    if sys.platform != "win32":
        extra_cuda_cflags += extra_cflags

    # Add -Werror after the copy so it reaches gcc but not nvcc (see DEBUG block above).
    if DEBUG and sys.platform != "win32":
        extra_cflags += ["-Werror"]

    if NUM_CHANNELS is not None:
        # nvcc has a bug where you need to escape the commas in macro values defined with -D.
        extra_cuda_cflags += [
            '-DGSPLAT_NUM_CHANNELS="' + NUM_CHANNELS.replace(",", "\\,") + '"'
        ]
        # gcc would not grok the backslash, so here we just pass NUM_CHANNELS as is.
        extra_cflags += [f"-DGSPLAT_NUM_CHANNELS={NUM_CHANNELS}"]

    extra_cuda_cflags += [] if NVCC_FLAGS == "" else NVCC_FLAGS.split(" ")

    if sys.platform == "win32" and torch.version.hip:
        # The host .cpp op-wrappers include c10/cuda/CUDAGuard.h, which on ROCm
        # pulls in the HIP runtime headers (amd_hip_vector_types.h etc.). Those
        # use GCC __attribute__ syntax that MSVC cl.exe -- the host compiler torch
        # picks for .cpp -- cannot parse. amdclang (hipcc) handles them, so route
        # each host wrapper through the device toolchain by handing it to torch
        # with a .cu extension. The shims are byte copies placed next to their
        # source so every relative #include resolves identically; they carry no
        # device code, so the extra device pass compiles nothing. Linux and CUDA
        # keep the .cpp host TUs untouched (this whole block is win32+HIP only).
        shim_sources = []
        for s in sources:
            if s.endswith(".cpp"):
                shim = s[:-4] + "_winhip.cu"
                shutil.copyfile(s, shim)
                shim_sources.append(shim)
            else:
                shim_sources.append(s)
        sources = shim_sources

    return SimpleNamespace(
        name=name,
        extra_include_paths=extra_include_paths,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
    )


def build_and_load_gsplat():
    build_params = get_build_parameters()

    build_dir = jit._get_build_directory(build_params.name, verbose=False)

    # If JIT is interrupted it might leave a lock in the build directory.
    # We dont want it to exist in any case.
    try:
        os.remove(os.path.join(build_dir, "lock"))
    except OSError:
        pass

    # Check if the build parameters have changed since last build (if any
    saved_build_params_fname = os.path.join(build_dir, "build_params.json")
    saved_build_params = None
    build_params_changed = False
    try:
        if os.path.exists(saved_build_params_fname):
            with open(saved_build_params_fname, "r") as f:
                saved_build_params = SimpleNamespace(**json.load(f))
            build_params_changed = saved_build_params != build_params
    except Exception as e:
        if _console is not None:
            _console.print(
                f"[bold yellow]gsplat: rebuilding due to error loading saved build parameters: {e}"
            )
        else:
            print(
                f"gsplat: rebuilding due to error loading saved build parameters: {e}"
            )

    # If parameters have changed,
    if build_params_changed:
        # Build gsplat from scratch
        shutil.rmtree(build_dir)
        # Print out what triggered the rebuild (for debugging...)
        if saved_build_params is not None:
            if _console is not None:
                _console.print(
                    f"[bold yellow]gsplat: rebuilding due to build parameter change"
                )
            else:
                print("gsplat: rebuilding due to build parameter change")
            saved_dict = saved_build_params.__dict__
            current_dict = build_params.__dict__
            for k in sorted(set(saved_dict) | set(current_dict)):
                saved_val = saved_dict.get(k, "<missing>")
                current_val = current_dict.get(k, "<missing>")
                if saved_val != current_val:
                    if _console is not None:
                        _console.print(f"[white] old {k}: {saved_val}")
                        _console.print(f"[white] new {k}: {current_val}")
                    else:
                        print(f"  old {k}: {saved_val}")
                        print(f"  new {k}: {current_val}")

    # Make sure the build directory exists.
    if build_dir:
        os.makedirs(build_dir, exist_ok=True)

    # Save our current build parameters
    with open(saved_build_params_fname, "w") as f:
        json.dump(build_params.__dict__, f)

    @contextmanager
    def status_context():
        tic = time.time()
        msg = f"gsplat: Setting up CUDA with MAX_JOBS={MAX_JOBS if MAX_JOBS else 'max'} (This may take a few minutes the first time)"
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
                f"[green]gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds.[/green]"
            )
        else:
            print(
                f"gsplat: CUDA extension has been set up successfully in {toc - tic:.2f} seconds."
            )

    # If the build exists, we assume the extension has been built
    # and we can load it.
    module_exists = os.path.exists(
        os.path.join(build_dir, f"{build_params.name}.so")
    ) or os.path.exists(os.path.join(build_dir, f"{build_params.name}.lib"))

    with (
        status_context() if not module_exists or build_params_changed else nullcontext()
    ):
        # If the JIT build happens concurrently in multiple processes,
        # race conditions can occur when removing the lock file at:
        # https://github.com/pytorch/pytorch/blob/e3513fb2af7951ddf725d8c5b6f6d962a053c9da/torch/utils/cpp_extension.py#L1736
        # But it's ok so we catch this exception and ignore it.
        envvars_to_remove = []
        try:
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


__all__ = ["get_build_parameters"]
