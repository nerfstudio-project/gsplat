# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera kernels package — Layer 0 GPU operations for camera projection.

Public API surface for the gsplat-sensorlib camera model. Exports:

- Parameter dataclasses: :class:`OpenCVPinholeProjection`,
  :class:`BivariateWindshieldDistortion`, :class:`NoExternalDistortion`.
- Type aliases: :data:`CameraProjection`, :data:`ExternalDistortion`,
  :data:`REGISTERED_CAMERA_PROJECTIONS`, :data:`REGISTERED_DISTORTIONS`.
- Enums: :class:`ShutterType`, :class:`ReferencePolynomial`.
- Factory helper: :func:`from_components` (windshield).
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
from .types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    NoExternalDistortion,
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
    "MAX_H_POLYNOMIAL_TERMS",
    "MAX_V_POLYNOMIAL_TERMS",
    "NoExternalDistortion",
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
]
