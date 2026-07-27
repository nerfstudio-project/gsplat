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

from gsplat._lazy_backend import make_lazy_backend


def _build():
    # Deferred so importing this module does not import .cuda.build.
    from .cuda.build import build_and_load_geometry_cuda

    return build_and_load_geometry_cuda()


_get_backend, __getattr__ = make_lazy_backend(
    module_name=__name__,
    public_name="_GEOMETRY_CUDA",
    prebuilt_module="gsplat_geometry_cuda",
    jit_loader=_build,
)

__all__ = ["_GEOMETRY_CUDA"]
