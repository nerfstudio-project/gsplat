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
