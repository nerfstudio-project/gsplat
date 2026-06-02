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
    FThetaProjection,
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
    "FThetaProjection",
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
