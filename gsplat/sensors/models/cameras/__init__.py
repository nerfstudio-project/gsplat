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

"""Camera models sub-package - Layer 2 stateful camera models.

This module provides:
- CameraModel: Stateful camera model (nn.Module) wrapping a Layer 0 projection
- ImageFrame / ImageFrameGroup: Camera frame and keyed collection types
- Re-exported Layer 0 types: BivariateWindshieldDistortion, ReferencePolynomial, from_components
- Return types: ImagePointsReturn, PixelsReturn, WorldPointsToImagePointsReturn,
  WorldPointsToPixelsReturn, WorldRaysReturn
"""

from ...functional.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldRaysReturn,
)
from ...kernels.cameras import (
    BivariateWindshieldDistortion,
    ReferencePolynomial,
    from_components,
)
from .camera_model import CameraModel
from .image_frame import ImageFrame, ImageFrameGroup

__all__ = [
    "BivariateWindshieldDistortion",
    "CameraModel",
    "ImageFrame",
    "ImageFrameGroup",
    "ImagePointsReturn",
    "PixelsReturn",
    "ReferencePolynomial",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldRaysReturn",
    "from_components",
]
