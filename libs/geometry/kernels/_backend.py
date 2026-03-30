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

"""Load native geometry CUDA ops, preferring prebuilt then JIT build."""

from __future__ import annotations

_GEOMETRY_CUDA = None

try:
    import gsplat_geometry_cuda as _GEOMETRY_CUDA  # pyright: ignore[reportMissingImports]
except ImportError as prebuilt_error:
    from .cuda.build import build_and_load_geometry_cuda

    try:
        _GEOMETRY_CUDA = build_and_load_geometry_cuda()
    except Exception as jit_error:
        raise ImportError(
            "Failed to load gsplat_geometry_cuda via both prebuilt import and "
            "JIT build/load.\n"
            f"Prebuilt import error: {prebuilt_error!r}\n"
            f"JIT build/load error: {jit_error!r}"
        ) from jit_error


__all__ = [
    "_GEOMETRY_CUDA",
]
