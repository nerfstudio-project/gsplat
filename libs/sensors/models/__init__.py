# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensors models package - Layer 2 stateful sensor models.

This module provides:
- CameraModel: Base class for stateful camera models (nn.Module)
- OpenCVPinholeProjection: Pinhole projection with OpenCV distortion coefficients
- ImageFrame / ImageFrameGroup: Camera frame and keyed collection types
- Pose / DynamicPose / Trajectory: Sensor pose representations
- Return types: typed dataclasses for all projection and ray-casting outputs
- ShutterType / BivariateWindshieldDistortion / ReferencePolynomial: kernel-layer enums and types
- from_components: factory helper for building BivariateWindshieldDistortion from raw polynomial tensors
"""

from ..functional.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from ..kernels.cameras import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    NoExternalDistortion,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
    from_components,
)
from ..kernels.common.pose import DynamicPose, Pose, Trajectory
from .cameras import CameraModel, ImageFrame, ImageFrameGroup
from .common import Frame, FrameId

__all__ = [
    "BivariateWindshieldDistortion",
    "CameraModel",
    "CameraProjection",
    "DynamicPose",
    "ExternalDistortion",
    "Frame",
    "FrameId",
    "ImageFrame",
    "ImageFrameGroup",
    "ImagePointsReturn",
    "NoExternalDistortion",
    "OpenCVPinholeProjection",
    "PixelsReturn",
    "Pose",
    "ReferencePolynomial",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "ShutterType",
    "Trajectory",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldPointsToSensorAnglesReturn",
    "WorldRaysReturn",
    "from_components",
]
