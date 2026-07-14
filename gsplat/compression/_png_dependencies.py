# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate CuPy against the CUDA toolkit used to build gsplat."""

import importlib


def _cuda_major_used_to_build_gsplat() -> int:
    """Return the CUDA ABI major against which gsplat was built."""

    # Read the toolkit version embedded in the loaded gsplat binary. Runtime
    # Torch can be replaced after gsplat is built, so torch.version.cuda is not
    # authoritative for selecting a binary-compatible CuPy package.
    from gsplat.cuda._wrapper import _build_config

    cuda_version = _build_config().get("cuda_version")
    if (
        isinstance(cuda_version, bool)
        or not isinstance(cuda_version, int)
        or cuda_version < 1000
    ):
        raise ImportError(
            "PNG compression could not determine the CUDA version used to build "
            "gsplat; reinstall gsplat with its CUDA extension"
        )
    return cuda_version // 1000


def _install_hint(cuda_major: int) -> str:
    """Return guidance for selecting a build-compatible CuPy distribution."""

    return (
        "install `gsplat[png]`; its build metadata selects CuPy for CUDA "
        f"{cuda_major} used to build gsplat"
    )


def validate_cupy() -> None:
    """Require a CuPy build matching the CUDA major used to build gsplat."""

    gsplat_cuda_major = _cuda_major_used_to_build_gsplat()
    try:
        cupy = importlib.import_module("cupy")
    except ModuleNotFoundError as exc:
        # Preserve failures from CuPy's transitive imports. Only a genuinely
        # absent CuPy package should be replaced with gsplat-specific guidance.
        if exc.name != "cupy":
            raise
        raise ImportError(
            "PNG compression requires a CUDA-matched CuPy package; "
            f"{_install_hint(gsplat_cuda_major)}"
        ) from exc

    cupy_cuda_major = int(cupy.cuda.runtime.runtimeGetVersion()) // 1000
    if cupy_cuda_major != gsplat_cuda_major:
        raise ImportError(
            f"gsplat was built for CUDA {gsplat_cuda_major}, but the installed "
            f"CuPy targets CUDA {cupy_cuda_major}; "
            f"{_install_hint(gsplat_cuda_major)}"
        )
