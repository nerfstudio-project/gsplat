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

"""LiDAR models sub-package - Layer 2 stateful LiDAR models.

This module provides:
- LidarModel: Stateful spinning-LiDAR model (nn.Module) wrapping a Layer 0 projection
- LidarFrame / LidarFrameSet: LiDAR frame and keyed collection types
- RowOffsetStructuredSpinningLidarProjection / SpinningDirection: public LiDAR projection type and spin enum
"""

from ...kernels.lidars import (
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
)
from .lidar_frame import LidarFrame, LidarFrameSet
from .lidar_model import LidarModel

__all__ = [
    "LidarFrame",
    "LidarFrameSet",
    "LidarModel",
    "RowOffsetStructuredSpinningLidarProjection",
    "SpinningDirection",
]
