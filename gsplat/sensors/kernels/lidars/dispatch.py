# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-key spinning-LiDAR projection dispatch.

The camera ``projective_sensor_ops._DISPATCH_TABLES`` are keyed on a Cartesian
product of ``(projection, external_distortion)``. LiDAR has no distortion pair;
it is keyed on a single projection type. This module is the LiDAR-side sibling:
a parallel set of dispatch tables keyed on a lone projection class, leaving the
camera tables untouched.
"""

from __future__ import annotations

from collections.abc import Callable

from . import ops as lidar_ops
from .types import (
    REGISTERED_LIDAR_PROJECTION_NAMES,
    REGISTERED_LIDAR_PROJECTIONS,
    RowOffsetStructuredSpinningLidarProjection,
    script_class_name,
)

DispatchKey = RowOffsetStructuredSpinningLidarProjection

_SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.sensor_rays_to_sensor_angles,
}
_SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.sensor_angles_to_sensor_rays,
}
_ELEMENTS_TO_SENSOR_ANGLES_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.elements_to_sensor_angles,
}
_GENERATE_SPINNING_LIDAR_RAYS_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.generate_spinning_lidar_rays,
}
_INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS: dict[DispatchKey, Callable] = {
    RowOffsetStructuredSpinningLidarProjection: lidar_ops.inverse_project_spinning_lidar,
}

_DISPATCH_TABLES = {
    "sensor_rays_to_sensor_angles": _SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS,
    "sensor_angles_to_sensor_rays": _SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS,
    "elements_to_sensor_angles": _ELEMENTS_TO_SENSOR_ANGLES_BACKENDS,
    "generate_spinning_lidar_rays": _GENERATE_SPINNING_LIDAR_RAYS_BACKENDS,
    "inverse_project_spinning_lidar": _INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS,
}

# Registered name keyed by projection class, zipped to keep positional
# parallelism with REGISTERED_LIDAR_PROJECTIONS as the single source of truth.
_REGISTERED_CLASS_NAMES = dict(
    zip(REGISTERED_LIDAR_PROJECTIONS, REGISTERED_LIDAR_PROJECTION_NAMES, strict=True)
)


def _registered_name(cls: object) -> str:
    """Return the registered string name for a LiDAR projection class."""
    return _REGISTERED_CLASS_NAMES[cls]


def _lookup(table: dict[DispatchKey, Callable], projection: object) -> Callable:
    """Return the backend callable for a single LiDAR projection.

    Args:
        table: One of the module-level ``_*_BACKENDS`` dispatch dicts.
        projection: Spinning-LiDAR projection instance.

    Returns:
        Callable that implements the operation for the matched projection.

    Raises:
        TypeError: If the projection type is not registered.
    """
    projection_name = script_class_name(projection)
    for projection_cls, backend in table.items():
        if projection_name == _registered_name(projection_cls):
            return backend
    raise TypeError(f"Unsupported LiDAR projection: {projection_name}")


def sensor_rays_to_sensor_angles(sensor_rays, projection, **kwargs):
    """Convert sensor-frame ray directions to (elevation, azimuth) angles.

    Args:
        sensor_rays: ``(N, 3)`` ray directions in the sensor frame.
        projection: Spinning-LiDAR projection used only to resolve the backend.
        **kwargs: Forwarded verbatim to the resolved backend.

    Returns:
        ``(N, 2)`` ``[elevation, azimuth]`` tensor as returned by the backend.
    """
    return _lookup(_SENSOR_RAYS_TO_SENSOR_ANGLES_BACKENDS, projection)(
        sensor_rays, **kwargs
    )


def sensor_angles_to_sensor_rays(sensor_angles, projection, **kwargs):
    """Convert (elevation, azimuth) angles to unit sensor-frame ray directions.

    Args:
        sensor_angles: ``(N, 2)`` ``[elevation, azimuth]`` tensor.
        projection: Spinning-LiDAR projection used only to resolve the backend.
        **kwargs: Forwarded verbatim to the resolved backend.

    Returns:
        ``(N, 3)`` unit-direction tensor as returned by the backend.
    """
    return _lookup(_SENSOR_ANGLES_TO_SENSOR_RAYS_BACKENDS, projection)(
        sensor_angles, **kwargs
    )


def elements_to_sensor_angles(elements, projection, **kwargs):
    """Look up per-element (elevation, azimuth) angles from the projection tables.

    Args:
        elements: ``(N, 2)`` int32 ``[row, col]`` element indices.
        projection: Spinning-LiDAR projection parameters.
        **kwargs: Forwarded verbatim to the resolved backend.

    Returns:
        Tuple ``(sensor_angles, valid_flags)`` as returned by the backend.
    """
    return _lookup(_ELEMENTS_TO_SENSOR_ANGLES_BACKENDS, projection)(
        elements, projection, **kwargs
    )


def generate_spinning_lidar_rays(projection, *args, **kwargs):
    """Generate world-frame rays for a spinning LiDAR with rolling shutter.

    Args:
        projection: Spinning-LiDAR projection parameters.
        *args: Additional positional args forwarded to the backend
            (e.g. ``elements``, ``dynamic_pose``).
        **kwargs: Forwarded verbatim to the resolved backend.

    Returns:
        Tuple ``(world_rays, timestamps_us, poses_translation, poses_rotation)``
        as returned by the backend.
    """
    return _lookup(_GENERATE_SPINNING_LIDAR_RAYS_BACKENDS, projection)(
        projection, *args, **kwargs
    )


def inverse_project_spinning_lidar(projection, *args, **kwargs):
    """Inverse-project world points to spinning-LiDAR sensor angles.

    Args:
        projection: Spinning-LiDAR projection parameters.
        *args: Additional positional args forwarded to the backend
            (e.g. ``world_points``, ``dynamic_pose``).
        **kwargs: Forwarded verbatim to the resolved backend.

    Returns:
        Tuple ``(sensor_angles, valid_flags, timestamps_us, poses_translation,
        poses_rotation)`` as returned by the backend.
    """
    return _lookup(_INVERSE_PROJECT_SPINNING_LIDAR_BACKENDS, projection)(
        projection, *args, **kwargs
    )


__all__ = [
    "REGISTERED_LIDAR_PROJECTIONS",
    "_DISPATCH_TABLES",
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
