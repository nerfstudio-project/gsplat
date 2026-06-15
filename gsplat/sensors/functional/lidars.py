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

"""Stateless functional wrappers for spinning-LiDAR operations (Layer 1).

This module bridges the Layer 0 CUDA kernel bindings and the Layer 2 LiDAR model
classes. Every function here is a thin, stateless wrapper: it forwards arguments
directly to the corresponding kernel op and packages the raw tensor outputs into
the structured return types defined in :mod:`return_types`. It adds no math; the
dtype contract and device handling live in the kernel ops.
"""

from __future__ import annotations

from torch import Tensor

from ..kernels.common.pose import DynamicPose
from ..kernels.common.utils import poses_to_matrix, valid_flags_to_indices
from ..kernels.lidars import ops as _kernel_ops
from ..kernels.lidars.types import RowOffsetStructuredSpinningLidarProjection
from .return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)


def sensor_rays_to_sensor_angles(
    sensor_rays: Tensor,
    *,
    allow_device_transfer: bool = False,
) -> SensorAnglesReturn:
    """Convert sensor-frame ray directions to (elevation, azimuth) angles.

    Args:
        sensor_rays: (N, 3) ray directions in the sensor frame.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            the input requires an implicit device or dtype transfer.

    Returns:
        SensorAnglesReturn with:
            sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
            valid_flag: ``None`` (every ray maps to a valid angle).
    """
    sensor_angles = _kernel_ops.sensor_rays_to_sensor_angles(
        sensor_rays, allow_device_transfer=allow_device_transfer
    )
    return SensorAnglesReturn(sensor_angles=sensor_angles)


def sensor_angles_to_sensor_rays(
    sensor_angles: Tensor,
    *,
    allow_device_transfer: bool = False,
) -> SensorRayReturn:
    """Convert (elevation, azimuth) angles to unit sensor-frame ray directions.

    Args:
        sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            the input requires an implicit device or dtype transfer.

    Returns:
        SensorRayReturn with:
            sensor_rays: (N, 3) unit direction vectors in the sensor frame.
            valid_flag: ``None`` (every angle maps to a valid ray).
    """
    sensor_rays = _kernel_ops.sensor_angles_to_sensor_rays(
        sensor_angles, allow_device_transfer=allow_device_transfer
    )
    return SensorRayReturn(sensor_rays=sensor_rays)


def elements_to_sensor_angles(
    elements: Tensor,
    projection: RowOffsetStructuredSpinningLidarProjection,
    *,
    return_valid_flag: bool = False,
    allow_device_transfer: bool = False,
) -> SensorAnglesReturn:
    """Look up per-element (elevation, azimuth) angles from the projection tables.

    Args:
        elements: (N, 2) int32 ``[row, col]`` element indices.
        projection: Structured spinning-LiDAR projection parameters.
        return_valid_flag: If ``True``, populate ``valid_flag`` with an (N,)
            boolean mask of in-bounds elements.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input requires an implicit device or dtype transfer.

    Returns:
        SensorAnglesReturn with:
            sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
            valid_flag: (N,) bool mask, or ``None`` when not requested.
    """
    sensor_angles, valid_flag = _kernel_ops.elements_to_sensor_angles(
        elements,
        projection,
        allow_device_transfer=allow_device_transfer,
    )
    return SensorAnglesReturn(
        sensor_angles=sensor_angles,
        valid_flag=valid_flag if return_valid_flag else None,
    )


def generate_spinning_lidar_rays(
    projection: RowOffsetStructuredSpinningLidarProjection,
    elements: Tensor | None,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Generate world-frame rays for a spinning LiDAR with rolling shutter.

    Args:
        projection: Structured spinning-LiDAR projection parameters.
        elements: (N, 2) int32 ``[row, col]`` element indices, or ``None`` to
            generate the full row-major ``(n_rows x n_columns)`` grid.
        dynamic_pose: Time-varying dynamic pose (start at t=0, end at t=1).
        start_timestamp_us: Start timestamp in microseconds. Must be paired with
            ``end_timestamp_us`` (both provided or both ``None``).
        end_timestamp_us: End timestamp in microseconds. Same both-or-neither
            contract as ``start_timestamp_us``.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` with
            per-ray (N, 4, 4) sensor-to-world transformation matrices.
        return_timestamps: If ``True``, populate ``timestamps_us`` with per-ray
            absolute timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input requires an implicit device or dtype transfer.

    Returns:
        WorldRaysReturn with:
            world_rays: (N, 6) ``[origin_xyz, direction_xyz]`` in the world frame.
            T_sensor_world: (N, 4, 4) transformation matrices, or ``None``.
            timestamps_us: (N,) int64 timestamps, or ``None``.
    """
    (
        world_rays,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.generate_spinning_lidar_rays(
        projection,
        elements,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_T_sensor_world,
        allow_device_transfer=allow_device_transfer,
    )
    return WorldRaysReturn(
        world_rays=world_rays,
        T_sensor_world=poses_to_matrix(poses_t, poses_r)
        if return_T_sensor_world
        else None,
        timestamps_us=timestamps_us if return_timestamps else None,
    )


def inverse_project_spinning_lidar(
    projection: RowOffsetStructuredSpinningLidarProjection,
    world_points: Tensor,
    dynamic_pose: DynamicPose,
    *,
    max_iterations: int = 10,
    stop_mean_relative_time_error: float = 1e-4,
    stop_delta_mean_relative_time_error: float = 1e-6,
    initial_relative_time: float = 0.5,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToSensorAnglesReturn:
    """Inverse-project world points to spinning-LiDAR sensor angles.

    Args:
        projection: Structured spinning-LiDAR projection parameters.
        world_points: (N, 3) world-frame points.
        dynamic_pose: Time-varying dynamic pose (start at t=0, end at t=1).
        max_iterations: Maximum rolling-shutter solver iterations; must be in
            ``[1, 32]`` or the op raises.
        stop_mean_relative_time_error: Convergence threshold on relative-time
            change.
        stop_delta_mean_relative_time_error: Convergence threshold on the change in
            relative-time error between iterations.
        initial_relative_time: Initial relative-time seed in [0, 1].
        start_timestamp_us: Start timestamp in microseconds. Both-or-neither with
            ``end_timestamp_us``.
        end_timestamp_us: End timestamp in microseconds.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` with
            per-point (N, 4, 4) sensor-to-world transformation matrices.
        return_valid_flag: If ``True``, populate ``valid_flag`` with an (N,)
            boolean mask of in-FOV points.
        return_valid_indices: If ``True``, populate ``valid_indices`` with the
            integer indices of valid projections (derived from the validity mask).
        return_timestamps: If ``True``, populate ``timestamps_us`` with per-point
            absolute timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input requires an implicit device or dtype transfer.

    Returns:
        WorldPointsToSensorAnglesReturn with:
            sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
            T_sensor_world: (N, 4, 4) transformation matrices, or ``None``.
            valid_flag: (N,) bool mask, or ``None``.
            valid_indices: (M,) int64 indices of valid points, or ``None``.
            timestamps_us: (N,) int64 timestamps, or ``None``.
    """
    (
        sensor_angles,
        valid_flag,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.inverse_project_spinning_lidar(
        projection,
        world_points,
        dynamic_pose,
        max_iterations=max_iterations,
        stop_mean_relative_time_error=stop_mean_relative_time_error,
        stop_delta_mean_relative_time_error=stop_delta_mean_relative_time_error,
        initial_relative_time=initial_relative_time,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_timestamps=return_timestamps,
        return_poses=return_T_sensor_world,
        allow_device_transfer=allow_device_transfer,
    )
    return WorldPointsToSensorAnglesReturn(
        sensor_angles=sensor_angles,
        T_sensor_world=poses_to_matrix(poses_t, poses_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps_us if return_timestamps else None,
    )


__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
