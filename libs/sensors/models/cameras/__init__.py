# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
