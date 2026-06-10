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

"""Camera kernels package — Layer 0 GPU operations for camera projection.

Public API surface for the gsplat-sensorlib camera model. Exports:

- Parameter dataclasses: :class:`OpenCVPinholeProjection`,
  :class:`FThetaProjection`, :class:`BivariateWindshieldDistortion`,
  :class:`NoExternalDistortion`.
- Type aliases: :data:`CameraProjection`, :data:`ExternalDistortion`,
  :data:`REGISTERED_CAMERA_PROJECTIONS`, :data:`REGISTERED_DISTORTIONS`.
- Enums: :class:`ShutterType`, :class:`ReferencePolynomial`.
- Factory helper: :func:`from_components` (windshield).
- Validators: :func:`validate_camera_projection`.
- Constants: :data:`FTHETA_MAX_POLYNOMIAL_TERMS`.
- Kernel functions: projection, ray-casting, pose interpolation.
"""

from .ops import (
    camera_rays_to_image_points,
    generate_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    interpolate_dynamic_pose,
    pixel_grid_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
    relative_frame_times,
    unpack_dynamic_pose_components,
)
from ._projection_validate import validate_camera_projection
from .types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FISHEYE_MAX_FORWARD_POLY_TERMS,
    FTHETA_MAX_POLYNOMIAL_TERMS,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    REGISTERED_CAMERA_PROJECTIONS,
    REGISTERED_DISTORTIONS,
    ShutterType,
)
from .windshield import (
    MAX_H_POLYNOMIAL_TERMS,
    MAX_V_POLYNOMIAL_TERMS,
    from_components,
)

__all__ = [
    "BivariateWindshieldDistortion",
    "CameraProjection",
    "ExternalDistortion",
    "FISHEYE_MAX_FORWARD_POLY_TERMS",
    "FTHETA_MAX_POLYNOMIAL_TERMS",
    "FThetaProjection",
    "MAX_H_POLYNOMIAL_TERMS",
    "MAX_V_POLYNOMIAL_TERMS",
    "NoExternalDistortion",
    "OpenCVFisheyeProjection",
    "OpenCVPinholeProjection",
    "ReferencePolynomial",
    "REGISTERED_CAMERA_PROJECTIONS",
    "REGISTERED_DISTORTIONS",
    "ShutterType",
    "camera_rays_to_image_points",
    "from_components",
    "generate_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "interpolate_dynamic_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "relative_frame_times",
    "unpack_dynamic_pose_components",
    "validate_camera_projection",
]
