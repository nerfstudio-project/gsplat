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

"""Layer 0 Python bindings for spinning-LiDAR kernel operations.

Provides differentiable PyTorch entry points around the hand-written CUDA LiDAR
kernels; each public function is backed by a ``torch.autograd.Function`` so
forward and backward CUDA kernels are invoked transparently.

This module covers all five LiDAR ops:
    sensor_rays_to_sensor_angles   (N, 3) -> (N, 2)
    sensor_angles_to_sensor_rays   (N, 2) -> (N, 3)
    elements_to_sensor_angles      (N, 2) int + tables -> (N, 2), valid_flags
    generate_spinning_lidar_rays   tables + poses -> (N, 6) world rays
    inverse_project_spinning_lidar (N, 3) world points -> (N, 2), valid_flags
"""

from __future__ import annotations

import torch
from torch import Tensor

from .. import _backend

_backend._SENSORS_CUDA  # noqa: B018  # ensures torch.ops gsplat_sensors registrations exist
from ..common import (
    DynamicPose,
    raise_or_target_device,
    timestamp_bounds,
    to_dev,
    unpack_dynamic_pose_components,
    zero_like,
)
from .types import (
    REGISTERED_LIDAR_PROJECTION_NAMES,
    REGISTERED_LIDAR_PROJECTIONS,
    RowOffsetStructuredSpinningLidarProjection,
    script_class_name,
)

# Registered name keyed by projection class, parallel to the two tuples.
_REGISTERED_CLASS_NAMES = dict(
    zip(REGISTERED_LIDAR_PROJECTIONS, REGISTERED_LIDAR_PROJECTION_NAMES, strict=True)
)
_SUPPORTED_PROJECTIONS = frozenset(_REGISTERED_CLASS_NAMES.values())


def _check_projection(projection: object) -> None:
    """Validate the projection is a supported LiDAR projection type."""
    if script_class_name(projection) not in _SUPPORTED_PROJECTIONS:
        raise TypeError(
            "Unsupported LiDAR projection: "
            f"{script_class_name(projection)} "
            f"(expected one of {sorted(_SUPPORTED_PROJECTIONS)})"
        )


def _projection_on_device(
    projection: RowOffsetStructuredSpinningLidarProjection,
    device: torch.device,
    dtype: torch.dtype,
    allow_device_transfer: bool,
) -> RowOffsetStructuredSpinningLidarProjection:
    """Build a device-local projection from the moved angle tables.

    Returns a fresh ``RowOffsetStructuredSpinningLidarProjection`` whose three
    angle tables are on ``device`` at ``dtype``, leaving the caller's projection
    untouched.
    """
    return RowOffsetStructuredSpinningLidarProjection(
        row_elevations_rad=to_dev(
            projection.row_elevations_rad, device, dtype, allow_device_transfer
        ),
        column_azimuths_rad=to_dev(
            projection.column_azimuths_rad, device, dtype, allow_device_transfer
        ),
        row_azimuth_offsets_rad=to_dev(
            projection.row_azimuth_offsets_rad, device, dtype, allow_device_transfer
        ),
        fov_vert_start_rad=projection.fov_vert_start_rad,
        fov_vert_span_rad=projection.fov_vert_span_rad,
        fov_horiz_start_rad=projection.fov_horiz_start_rad,
        fov_horiz_span_rad=projection.fov_horiz_span_rad,
        spinning_direction=projection.spinning_direction,
        has_row_offsets=projection.has_row_offsets,
    )


class _SensorRaysToSensorAngles(torch.autograd.Function):
    """Autograd Function wrapping the ``sensor_rays_to_sensor_angles`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(ctx, sensor_rays: Tensor) -> Tensor:
        sensor_angles = torch.ops.gsplat_sensors.sensor_rays_to_sensor_angles(
            sensor_rays
        )
        ctx.save_for_backward(sensor_rays)
        return sensor_angles

    @staticmethod
    def backward(ctx, grad_sensor_angles: Tensor | None):
        (sensor_rays,) = ctx.saved_tensors
        if grad_sensor_angles is None:
            grad_sensor_angles = zero_like((sensor_rays.shape[0], 2), sensor_rays)
        grad_sensor_rays = (
            torch.ops.gsplat_sensors.sensor_rays_to_sensor_angles_backward(
                sensor_rays, grad_sensor_angles.contiguous()
            )
        )
        return grad_sensor_rays if ctx.needs_input_grad[0] else None


class _SensorAnglesToSensorRays(torch.autograd.Function):
    """Autograd Function wrapping the ``sensor_angles_to_sensor_rays`` CUDA forward + backward kernels."""

    @staticmethod
    def forward(ctx, sensor_angles: Tensor) -> Tensor:
        sensor_rays = torch.ops.gsplat_sensors.sensor_angles_to_sensor_rays(
            sensor_angles
        )
        ctx.save_for_backward(sensor_angles)
        return sensor_rays

    @staticmethod
    def backward(ctx, grad_sensor_rays: Tensor | None):
        (sensor_angles,) = ctx.saved_tensors
        if grad_sensor_rays is None:
            grad_sensor_rays = zero_like((sensor_angles.shape[0], 3), sensor_angles)
        grad_sensor_angles = (
            torch.ops.gsplat_sensors.sensor_angles_to_sensor_rays_backward(
                sensor_angles, grad_sensor_rays.contiguous()
            )
        )
        return grad_sensor_angles if ctx.needs_input_grad[0] else None


class _ElementsToSensorAngles(torch.autograd.Function):
    """Autograd Function wrapping the ``elements_to_sensor_angles`` CUDA forward + backward kernels.

    The three angle tables are threaded as positional ``apply`` arguments (so
    ``needs_input_grad`` routes gradients to them) AND fed directly to the C++
    op, which keeps the op dtype-templated: the public layer forces float32, but
    fp64 gradcheck can pass float64 tables straight through. The projection is
    used only for the ``has_row_offsets`` flag. ``elements`` is
    non-differentiable (integer indices).
    """

    @staticmethod
    def forward(
        ctx,
        elements: Tensor,
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        projection: RowOffsetStructuredSpinningLidarProjection,
    ) -> tuple[Tensor, Tensor]:
        (
            sensor_angles,
            valid_flags,
        ) = torch.ops.gsplat_sensors.elements_to_sensor_angles(
            projection,
            elements,
            row_elevations,
            column_azimuths,
            row_offsets,
        )
        ctx.projection = projection
        ctx.save_for_backward(elements, row_elevations, column_azimuths, row_offsets)
        ctx.mark_non_differentiable(valid_flags)
        return sensor_angles, valid_flags

    @staticmethod
    def backward(
        ctx, grad_sensor_angles: Tensor | None, grad_valid_flags: Tensor | None
    ):
        del grad_valid_flags
        elements, row_elevations, column_azimuths, row_offsets = ctx.saved_tensors
        projection = ctx.projection
        if grad_sensor_angles is None:
            grad_sensor_angles = zero_like((elements.shape[0], 2), row_elevations)
        need_row_elevations_grad = ctx.needs_input_grad[1]
        need_column_azimuths_grad = ctx.needs_input_grad[2]
        need_row_offsets_grad = ctx.needs_input_grad[3]
        (
            grad_row_elevations,
            grad_column_azimuths,
            grad_row_offsets,
        ) = torch.ops.gsplat_sensors.elements_to_sensor_angles_backward(
            projection,
            elements,
            row_elevations,
            column_azimuths,
            row_offsets,
            grad_sensor_angles.contiguous(),
            need_row_elevations_grad,
            need_column_azimuths_grad,
            need_row_offsets_grad,
        )
        return (
            None,
            grad_row_elevations if need_row_elevations_grad else None,
            grad_column_azimuths if need_column_azimuths_grad else None,
            grad_row_offsets if need_row_offsets_grad else None,
            None,
        )


class _GenerateSpinningLidarRays(torch.autograd.Function):
    """Autograd Function wrapping the ``generate_spinning_lidar_rays`` CUDA forward + backward kernels.

    The three angle tables and the two control poses (translation + wxyz
    rotation) are threaded as positional ``apply`` arguments so
    ``needs_input_grad`` routes gradients to them AND fed directly to the C++ op,
    which keeps it dtype-templated: the public layer forces float32, but fp64
    gradcheck can pass float64 tables / poses straight through. ``elements`` is
    non-differentiable (integer indices), or None in generate-mode. ``timestamps_us``,
    ``poses_translation`` and ``poses_rotation`` are non-differentiable outputs.
    """

    @staticmethod
    def forward(
        ctx,
        elements: Tensor | None,
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        control_translations: Tensor,
        control_rotations: Tensor,
        projection: RowOffsetStructuredSpinningLidarProjection,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        does_generate_elements = elements is None
        elements_arg = (
            elements
            if elements is not None
            else torch.empty((0, 2), device=row_elevations.device, dtype=torch.int32)
        )
        (
            world_rays,
            timestamps_us,
            poses_translation,
            poses_rotation,
        ) = torch.ops.gsplat_sensors.generate_spinning_lidar_rays(
            projection,
            elements_arg,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            does_generate_elements,
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.projection = projection
        ctx.does_generate_elements = does_generate_elements
        ctx.save_for_backward(
            elements_arg,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
        )
        ctx.mark_non_differentiable(timestamps_us, poses_translation, poses_rotation)
        return world_rays, timestamps_us, poses_translation, poses_rotation

    @staticmethod
    def backward(
        ctx,
        grad_world_rays: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_poses_translation: Tensor | None,
        grad_poses_rotation: Tensor | None,
    ):
        del grad_timestamps_us, grad_poses_translation, grad_poses_rotation
        (
            elements,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
        ) = ctx.saved_tensors
        if grad_world_rays is None:
            count = (
                row_elevations.shape[0] * column_azimuths.shape[0]
                if ctx.does_generate_elements
                else elements.shape[0]
            )
            grad_world_rays = zero_like((count, 6), row_elevations)
        need_row_elevations_grad = ctx.needs_input_grad[1]
        need_column_azimuths_grad = ctx.needs_input_grad[2]
        need_row_offsets_grad = ctx.needs_input_grad[3]
        need_control_translations_grad = ctx.needs_input_grad[4]
        need_control_rotations_grad = ctx.needs_input_grad[5]
        (
            grad_row_elevations,
            grad_column_azimuths,
            grad_row_offsets,
            grad_control_translations,
            grad_control_rotations,
        ) = torch.ops.gsplat_sensors.generate_spinning_lidar_rays_backward(
            ctx.projection,
            elements,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            ctx.does_generate_elements,
            grad_world_rays.contiguous(),
            need_row_elevations_grad,
            need_column_azimuths_grad,
            need_row_offsets_grad,
            need_control_translations_grad,
            need_control_rotations_grad,
        )
        return (
            None,
            grad_row_elevations if need_row_elevations_grad else None,
            grad_column_azimuths if need_column_azimuths_grad else None,
            grad_row_offsets if need_row_offsets_grad else None,
            grad_control_translations if need_control_translations_grad else None,
            grad_control_rotations if need_control_rotations_grad else None,
            None,
            None,
            None,
        )


class _InverseProjectSpinningLidar(torch.autograd.Function):
    """Autograd Function wrapping the ``inverse_project_spinning_lidar`` CUDA forward + backward kernels.

    The rolling-shutter fixed point converges non-differentiably in the forward;
    the converged interpolation alpha is saved to scratch. The backward re-loads
    that alpha and takes ONE differentiable step (the LINEAR relative-time map is
    piecewise-constant, so the implicit-function correction term vanishes and the
    frozen-alpha VJP is exact). ``world_points`` and the two control poses are
    threaded as positional ``apply`` arguments so ``needs_input_grad`` routes
    gradients to them. The three angle tables are accepted positionally for
    dtype-templating but carry no gradient (no differentiable path through the
    piecewise-constant column lookup). ``valid_flags``, ``timestamps_us`` and the
    per-point poses are non-differentiable.
    """

    @staticmethod
    def forward(
        ctx,
        world_points: Tensor,
        row_elevations: Tensor,
        column_azimuths: Tensor,
        row_offsets: Tensor,
        control_translations: Tensor,
        control_rotations: Tensor,
        projection: RowOffsetStructuredSpinningLidarProjection,
        max_iterations: int,
        stop_mean_relative_time_error: float,
        stop_delta_mean_relative_time_error: float,
        initial_relative_time: float,
        start_timestamp_us: int,
        end_timestamp_us: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        (
            sensor_angles,
            valid_flags,
            timestamps_us,
            poses_translation,
            poses_rotation,
            scratch,
        ) = torch.ops.gsplat_sensors.inverse_project_spinning_lidar(
            projection,
            world_points,
            row_elevations,
            column_azimuths,
            row_offsets,
            control_translations,
            control_rotations,
            int(max_iterations),
            float(stop_mean_relative_time_error),
            float(stop_delta_mean_relative_time_error),
            float(initial_relative_time),
            int(start_timestamp_us),
            int(end_timestamp_us),
        )
        ctx.save_for_backward(
            world_points,
            control_translations,
            control_rotations,
            valid_flags,
            scratch,
        )
        ctx.mark_non_differentiable(
            valid_flags, timestamps_us, poses_translation, poses_rotation
        )
        return (
            sensor_angles,
            valid_flags,
            timestamps_us,
            poses_translation,
            poses_rotation,
        )

    @staticmethod
    def backward(
        ctx,
        grad_sensor_angles: Tensor | None,
        grad_valid_flags: Tensor | None,
        grad_timestamps_us: Tensor | None,
        grad_poses_translation: Tensor | None,
        grad_poses_rotation: Tensor | None,
    ):
        del (
            grad_valid_flags,
            grad_timestamps_us,
            grad_poses_translation,
            grad_poses_rotation,
        )
        (
            world_points,
            control_translations,
            control_rotations,
            valid_flags,
            scratch,
        ) = ctx.saved_tensors
        if grad_sensor_angles is None:
            grad_sensor_angles = zero_like((world_points.shape[0], 2), world_points)
        need_world_points_grad = ctx.needs_input_grad[0]
        need_control_translations_grad = ctx.needs_input_grad[4]
        need_control_rotations_grad = ctx.needs_input_grad[5]
        (
            grad_world_points,
            grad_control_translations,
            grad_control_rotations,
        ) = torch.ops.gsplat_sensors.inverse_project_spinning_lidar_backward(
            world_points,
            control_translations,
            control_rotations,
            valid_flags,
            scratch,
            grad_sensor_angles.contiguous(),
            need_world_points_grad,
            need_control_translations_grad,
            need_control_rotations_grad,
        )
        return (
            grad_world_points if need_world_points_grad else None,
            None,
            None,
            None,
            grad_control_translations if need_control_translations_grad else None,
            grad_control_rotations if need_control_rotations_grad else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sensor_rays_to_sensor_angles(
    sensor_rays: Tensor,
    *,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Convert sensor-frame ray directions to (elevation, azimuth) angles.

    elevation = atan2(z, hypot(x, y)); azimuth = atan2(y, x). Sensor frame is
    +X forward, +Y left, +Z up. The output dtype matches the input dtype.

    Args:
        sensor_rays: (N, 3) ray directions in sensor frame.
        allow_device_transfer: If False (default), raises if the input needs a
            device transfer before the kernel launches.

    Returns:
        sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
    """
    device = raise_or_target_device(sensor_rays, allow_device_transfer)
    rays = to_dev(sensor_rays, device, sensor_rays.dtype, allow_device_transfer)
    return _SensorRaysToSensorAngles.apply(rays)


def sensor_angles_to_sensor_rays(
    sensor_angles: Tensor,
    *,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Convert (elevation, azimuth) angles to unit sensor-frame ray directions.

    ray = (cos_e*cos(az), cos_e*sin(az), sin(elev)); unit by construction. The
    output dtype matches the input dtype.

    Args:
        sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
        allow_device_transfer: If False (default), raises if the input needs a
            device transfer before the kernel launches.

    Returns:
        sensor_rays: (N, 3) unit ray directions in sensor frame.
    """
    device = raise_or_target_device(sensor_angles, allow_device_transfer)
    angles = to_dev(sensor_angles, device, sensor_angles.dtype, allow_device_transfer)
    return _SensorAnglesToSensorRays.apply(angles)


def elements_to_sensor_angles(
    elements: Tensor,
    projection: RowOffsetStructuredSpinningLidarProjection,
    *,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor]:
    """Look up per-element (elevation, azimuth) angles from the projection tables.

    For each ``[row, col]`` element, returns ``row_elevations[row]`` and
    ``column_azimuths[col]`` (plus ``row_offsets[row]`` when the projection has
    row offsets, normalized to ``[-pi, pi)``). Out-of-bounds row/col yields
    ``valid=False`` and ``(0, 0)`` angles. Differentiable w.r.t. the three angle
    tables; ``elements`` is non-differentiable.

    This public entry point forces the angle tables to float32. The underlying
    autograd Function is dtype-templated and accepts float64 tables directly
    (used for fp64 gradcheck).

    Args:
        elements: (N, 2) int32 ``[row, col]`` element indices.
        projection: Structured spinning-LiDAR projection parameters.
        allow_device_transfer: If False (default), raises if any tensor needs a
            device transfer before the kernel launches.

    Returns:
        sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
        valid_flags: (N,) bool mask marking in-bounds elements.
    """
    _check_projection(projection)
    device = raise_or_target_device(
        projection.row_elevations_rad, allow_device_transfer
    )
    elements_dev = to_dev(elements, device, torch.int32, allow_device_transfer)
    projection_dev = _projection_on_device(
        projection, device, torch.float32, allow_device_transfer
    )
    return _ElementsToSensorAngles.apply(
        elements_dev,
        projection_dev.row_elevations_rad,
        projection_dev.column_azimuths_rad,
        projection_dev.row_azimuth_offsets_rad,
        projection_dev,
    )


def generate_spinning_lidar_rays(
    projection: RowOffsetStructuredSpinningLidarProjection,
    elements: Tensor | None,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Generate world-frame rays for a spinning LiDAR with rolling shutter.

    For each ``[row, col]`` element (or the full ``(n_rows x n_columns)`` grid
    when ``elements`` is None), the sensor pose is interpolated at a relative
    time derived from the COLUMN index (column 0 captured first, column
    ``n_columns - 1`` last): translation by LERP, rotation by SLERP of the
    two control poses. The world ray is ``[origin = interpolated translation,
    direction = R(q_interp) * sensor_ray]``. Differentiable w.r.t. the three
    angle tables and the two control poses (translation + rotation);
    ``timestamps_us`` and the per-ray poses are non-differentiable.

    This public entry point forces the angle tables, control poses, and
    ``world_rays`` to float32. The underlying autograd Function is
    dtype-templated and accepts float64 inputs directly (used for fp64
    gradcheck).

    Args:
        projection: Structured spinning-LiDAR projection parameters.
        elements: (N, 2) int32 ``[row, col]`` element indices, or None to
            generate the full row-major grid.
        dynamic_pose: Time-varying dynamic pose (start at t=0, end at t=1).
        start_timestamp_us: Start timestamp in microseconds. Must be supplied
            together with ``end_timestamp_us`` or both omitted.
        end_timestamp_us: End timestamp in microseconds. Same both-or-neither
            contract as ``start_timestamp_us``.
        return_timestamps: If True, return per-ray absolute timestamps.
        return_poses: If True, return per-ray interpolated poses.
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        world_rays: (N, 6) ``[origin.xyz, direction.xyz]`` in world frame (float32).
        timestamps_us: (N,) int64 microsecond timestamps, or None.
        poses_translation: (N, 3) per-ray translations, or None.
        poses_rotation: (N, 4) per-ray rotations (wxyz), or None.
    """
    _check_projection(projection)
    device = raise_or_target_device(
        dynamic_pose.start_pose.translation, allow_device_transfer
    )
    start, end = timestamp_bounds(start_timestamp_us, end_timestamp_us)

    projection_dev = _projection_on_device(
        projection, device, torch.float32, allow_device_transfer
    )

    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device, torch.float32, allow_device_transfer
    )
    control_translations = torch.stack(
        [start_t.reshape(3), end_t.reshape(3)]
    ).contiguous()
    control_rotations = torch.stack([start_r.reshape(4), end_r.reshape(4)]).contiguous()

    elements_dev = (
        to_dev(elements, device, torch.int32, allow_device_transfer)
        if elements is not None
        else None
    )

    (
        world_rays,
        timestamps_us,
        poses_translation,
        poses_rotation,
    ) = _GenerateSpinningLidarRays.apply(
        elements_dev,
        projection_dev.row_elevations_rad,
        projection_dev.column_azimuths_rad,
        projection_dev.row_azimuth_offsets_rad,
        control_translations,
        control_rotations,
        projection_dev,
        start,
        end,
    )
    return (
        world_rays,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
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
    return_timestamps: bool = False,
    return_poses: bool = False,
    allow_device_transfer: bool = False,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Inverse-project world points to spinning-LiDAR sensor angles.

    For each world point, solves the azimuth->relative-time rolling-shutter fixed
    point: at each iteration the 2-pose trajectory is interpolated at the current
    relative time (translation LERP, rotation SLERP), the point is mapped to the
    sensor frame via ``R(q)^T (p - t)``, normalized, converted to
    ``(elevation, azimuth)``, FOV-checked, and a fresh relative time is read from
    the LINEAR nearest-row/nearest-column table lookup. Iterates up to
    ``max_iterations`` (rejected if outside ``[1, 32]``) with the two convergence
    breaks. Differentiable
    w.r.t. ``world_points`` and the two control poses; the angle tables carry no
    gradient. ``valid_flags``, ``timestamps_us`` and the per-point poses are
    non-differentiable.

    This op is dtype-polymorphic: the output ``sensor_angles`` follow
    ``world_points.dtype``. The registered projection POD is kept at float32
    while explicit angle-table and pose tensors are aligned to the compute dtype,
    so a float64 ``world_points`` input yields float64 outputs and exercises the
    fp64 gradcheck path end to end.

    Args:
        projection: Structured spinning-LiDAR projection parameters.
        world_points: (N, 3) world-frame points.
        dynamic_pose: Time-varying dynamic pose (start at t=0, end at t=1).
        max_iterations: Maximum rolling-shutter solver iterations; must be in
            ``[1, 32]`` or the op raises.
        stop_mean_relative_time_error: Convergence threshold on relative-time
            change.
        stop_delta_mean_relative_time_error: Convergence threshold on the change
            in relative-time error between iterations.
        initial_relative_time: Initial relative-time seed in [0, 1].
        start_timestamp_us: Start timestamp in microseconds. Both-or-neither with
            ``end_timestamp_us``.
        end_timestamp_us: End timestamp in microseconds.
        return_timestamps: If True, return per-point absolute timestamps.
        return_poses: If True, return per-point interpolated poses.
        allow_device_transfer: If False (default), raises if any tensor needs a
            device or dtype transfer before the kernel launches.

    Returns:
        sensor_angles: (N, 2) ``[elevation, azimuth]`` in radians.
        valid_flags: (N,) bool mask marking in-FOV points.
        timestamps_us: (N,) int64 microsecond timestamps, or None.
        poses_translation: (N, 3) per-point translations, or None.
        poses_rotation: (N, 4) per-point rotations (wxyz), or None.
    """
    _check_projection(projection)
    device = raise_or_target_device(world_points, allow_device_transfer)
    start, end = timestamp_bounds(start_timestamp_us, end_timestamp_us)

    compute_dtype = world_points.dtype
    points = to_dev(world_points, device, compute_dtype, allow_device_transfer)
    projection_dev = _projection_on_device(
        projection, device, torch.float32, allow_device_transfer
    )
    row_elevations = to_dev(
        projection.row_elevations_rad, device, compute_dtype, allow_device_transfer
    )
    column_azimuths = to_dev(
        projection.column_azimuths_rad, device, compute_dtype, allow_device_transfer
    )
    row_offsets = to_dev(
        projection.row_azimuth_offsets_rad,
        device,
        compute_dtype,
        allow_device_transfer,
    )

    start_t, start_r, end_t, end_r = unpack_dynamic_pose_components(
        dynamic_pose, device, compute_dtype, allow_device_transfer
    )
    control_translations = torch.stack(
        [start_t.reshape(3), end_t.reshape(3)]
    ).contiguous()
    control_rotations = torch.stack([start_r.reshape(4), end_r.reshape(4)]).contiguous()

    (
        sensor_angles,
        valid_flags,
        timestamps_us,
        poses_translation,
        poses_rotation,
    ) = _InverseProjectSpinningLidar.apply(
        points,
        row_elevations,
        column_azimuths,
        row_offsets,
        control_translations,
        control_rotations,
        projection_dev,
        max_iterations,
        stop_mean_relative_time_error,
        stop_delta_mean_relative_time_error,
        initial_relative_time,
        start,
        end,
    )
    return (
        sensor_angles,
        valid_flags,
        timestamps_us if return_timestamps else None,
        poses_translation if return_poses else None,
        poses_rotation if return_poses else None,
    )


__all__ = [
    "elements_to_sensor_angles",
    "generate_spinning_lidar_rays",
    "inverse_project_spinning_lidar",
    "sensor_angles_to_sensor_rays",
    "sensor_rays_to_sensor_angles",
]
