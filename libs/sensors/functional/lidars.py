# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiDAR functional API stubs for the OpenCV pinhole slice.

The spinning-lidar projection ops (angle conversion, ray generation, inverse
projection) are not included in this build slice. Each name raises
``NotImplementedError`` at call time so imports do not fail but misuse is
surfaced immediately at runtime.
"""

from __future__ import annotations


def _deferred(*args, **kwargs):
    raise NotImplementedError(
        "LiDAR functional APIs are deferred outside the OpenCV pinhole slice"
    )


elements_to_sensor_angles = _deferred
sensor_rays_to_sensor_angles = _deferred
sensor_angles_to_sensor_rays = _deferred
generate_spinning_lidar_rays = _deferred
inverse_project_spinning_lidar = _deferred

__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
