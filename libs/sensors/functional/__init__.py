# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Functional sensor projection API (Layer 1).

This package re-exports stateless projection functions for cameras and LiDARs,
wrapping the Layer 0 GPU kernels with input validation and structured return
types. Camera ops cover projection, back-projection, and ray generation;
LiDAR ops (spinning lidar geometry) are stubbed in this OpenCV-pinhole slice.
"""

from .cameras import (
    camera_rays_to_image_points,
    generate_image_points,
    image_points_to_camera_rays,
    image_points_to_world_rays_static_pose,
    image_points_to_world_rays_shutter_pose,
    pixel_grid_to_world_rays_shutter_pose,
    project_world_points_mean_pose,
    project_world_points_shutter_pose,
)
from .lidars import (
    elements_to_sensor_angles,
    generate_spinning_lidar_rays,
    inverse_project_spinning_lidar,
    sensor_angles_to_sensor_rays,
    sensor_rays_to_sensor_angles,
)
from .return_types import (
    ImagePointsReturn,
    PixelsReturn,
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)

__all__ = [
    "ImagePointsReturn",
    "PixelsReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldPointsToSensorAnglesReturn",
    "WorldRaysReturn",
    "camera_rays_to_image_points",
    "elements_to_sensor_angles",
    "generate_image_points",
    "generate_spinning_lidar_rays",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "inverse_project_spinning_lidar",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
