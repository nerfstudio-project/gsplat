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
"""Deformable / 4D Gaussian Splatting (experimental).

Ported from G-SHARP v0.2's ``training/scene`` package. See the proposal at
``docs/source/proposals/gsharp_v0_2_port.rst`` for motivation and the tracked
branch ``vnath_gsharp`` for progress.
"""

from .deformation import DeformationTable, DeformNetwork  # noqa: F401  (back-compat re-export)
from .hexplane import HexPlaneField
from .regulation import (
    hexplane_regularization,
    plane_smoothness,
    time_l1,
    time_smoothness,
)
from .strategy import DynamicStrategy

# Public API: prefer `hexplane_regularization` over calling the three
# lower-level regularizers with hand-partitioned plane lists; the
# spatial / temporal partition lives on :class:`HexPlaneField` now
# (MR-030). `DeformationTable` is intentionally NOT in `__all__`: it's
# internal bookkeeping that the trainer wires through
# `state["dynamic_mask"]` (MR-022). Back-compat callers can still
# `from gsplat.contrib.dynamic.deformation import DeformationTable`.
__all__ = [
    "DeformNetwork",
    "DynamicStrategy",
    "HexPlaneField",
    "hexplane_regularization",
    # Lower-level regularizers kept for advanced callers.
    "plane_smoothness",
    "time_l1",
    "time_smoothness",
]
