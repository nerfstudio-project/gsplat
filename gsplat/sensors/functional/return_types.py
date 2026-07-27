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

"""Structured return types for Layer 1 functional sensor operations.

These dataclasses provide named, optionally-populated return values for camera
and LiDAR projection functions, enabling callers to request auxiliary outputs
(poses, timestamps, validity information) without breaking function signatures.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class ImagePointsReturn:
    """Return type for ``camera_rays_to_image_points``.

    Attributes:
        image_points: (N, 2) float - projected image coordinates in pixels.
        valid_flag: (N,) bool - True for rays that project inside the image bounds.
        jacobians: (N, 2, 3) float - Jacobians of the projection with respect to
            the input camera rays, or None when not requested.
    """

    image_points: Tensor
    valid_flag: Tensor
    jacobians: Tensor | None = None


@dataclass
class PixelsReturn:
    """Return type for pixel-rounding projection functions.

    Attributes:
        pixels: (N, 2) int - discrete pixel indices corresponding to projected points.
        valid_flag: (N,) bool - True for points that project to a valid pixel.
    """

    pixels: Tensor
    valid_flag: Tensor


@dataclass
class WorldPointsToImagePointsReturn:
    """Return type for ``project_world_points_*`` functions.

    Attributes:
        image_points: (N, 2) float - projected image coordinates. Layer 1
            functional ops return one row per input point; Layer 2 model methods
            filter to valid projections unless ``return_all_projections=True``.
        T_sensor_world: (N, 4, 4) float - per-point transformation matrices,
            or None when not requested.
        valid_flag: (N,) bool - validity mask over all input points, or None.
        valid_indices: (M,) int64 - indices into the input point cloud for projections
            that passed all validity checks, or None.
        timestamps_us: (N,) int64 - per-point shutter timestamps in microseconds,
            or None when not requested.
    """

    image_points: Tensor
    T_sensor_world: Tensor | None = None
    valid_flag: Tensor | None = None
    valid_indices: Tensor | None = None
    timestamps_us: Tensor | None = None


@dataclass
class WorldPointsToPixelsReturn:
    """Return type for ``project_world_points_*`` functions that return pixel indices.

    Attributes:
        pixels: (N, 2) int - discrete pixel indices. Layer 1 functional ops return
            one row per input point; Layer 2 model methods filter to valid
            projections unless ``return_all_projections=True``.
        T_sensor_world: (N, 4, 4) float - per-point transformation matrices,
            or None when not requested.
        valid_flag: (N,) bool - validity mask over all input points, or None.
        valid_indices: (M,) int64 - indices into the input point cloud for projections
            that passed all validity checks, or None.
        timestamps_us: (N,) int64 - per-point shutter timestamps in microseconds,
            or None when not requested.
    """

    pixels: Tensor
    T_sensor_world: Tensor | None = None
    valid_flag: Tensor | None = None
    valid_indices: Tensor | None = None
    timestamps_us: Tensor | None = None


@dataclass
class WorldRaysReturn:
    """Return type for back-projection functions (``image_points_to_world_rays_*``).

    Attributes:
        world_rays: (N, 6) float - rays packed as [origin.x, origin.y, origin.z,
            direction.x, direction.y, direction.z] in the world frame.
        T_sensor_world: (N, 4, 4) float - per-ray transformation matrices,
            or None when not requested.
        timestamps_us: (N,) int64 - per-ray shutter timestamps in microseconds,
            or None when not requested.
    """

    world_rays: Tensor
    T_sensor_world: Tensor | None = None
    timestamps_us: Tensor | None = None


@dataclass
class SensorAnglesReturn:
    """Return type for ray-to-angle conversion functions.

    Attributes:
        sensor_angles: (N, 2) float - [elevation_rad, azimuth_rad] pairs in the
            sensor frame.
        valid_flag: (N,) bool - validity mask, or None when all outputs are valid.
    """

    sensor_angles: Tensor
    valid_flag: Tensor | None = None


@dataclass
class SensorRayReturn:
    """Return type for angle-to-ray conversion functions.

    Attributes:
        sensor_rays: (N, 3) float - unit direction vectors in the sensor frame.
        valid_flag: (N,) bool - validity mask, or None when all outputs are valid.
    """

    sensor_rays: Tensor
    valid_flag: Tensor | None = None


@dataclass
class WorldPointsToSensorAnglesReturn:
    """Return type for world-points-to-sensor-angles projection (LiDAR).

    Attributes:
        sensor_angles: (N, 2) float - [elevation_rad, azimuth_rad] pairs.
        T_sensor_world: (N, 4, 4) float - per-point transformation matrices,
            or None when not requested.
        valid_flag: (N,) bool - validity mask over all input points, or None.
        valid_indices: (M,) int64 - indices into the input point cloud for projections
            that passed all validity checks, or None.
        timestamps_us: (N,) int64 - per-point timestamps in microseconds, or None.
    """

    sensor_angles: Tensor
    T_sensor_world: Tensor | None = None
    valid_flag: Tensor | None = None
    valid_indices: Tensor | None = None
    timestamps_us: Tensor | None = None


__all__ = [
    "ImagePointsReturn",
    "PixelsReturn",
    "SensorAnglesReturn",
    "SensorRayReturn",
    "WorldPointsToImagePointsReturn",
    "WorldPointsToPixelsReturn",
    "WorldPointsToSensorAnglesReturn",
    "WorldRaysReturn",
]
