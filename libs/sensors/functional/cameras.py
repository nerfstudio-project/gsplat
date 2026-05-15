# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless functional wrappers for OpenCV pinhole camera operations (Layer 1).

This module bridges the Layer 0 CUDA kernel bindings and the Layer 2 camera model
classes.  Every function here is a thin, stateless wrapper: it forwards arguments
directly to the corresponding kernel op and packages the raw tensor outputs into the
structured return types defined in :mod:`return_types`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..kernels.cameras import ops as _kernel_ops
from ..kernels.cameras.types import (
    ExternalDistortion,
    OpenCVPinholeProjection,
    ShutterType,
)
from ..kernels.common.utils import poses_to_matrix, valid_flags_to_indices
from ..kernels.common.pose import DynamicPose, Pose
from .return_types import (
    ImagePointsReturn,
    WorldPointsToImagePointsReturn,
    WorldRaysReturn,
)


def camera_rays_to_image_points(
    camera_rays: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> ImagePointsReturn:
    """Project camera-space rays to image points using an OpenCV pinhole model.

    Args:
        camera_rays: (N, 3) unit direction vectors in camera space.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model (e.g. windshield) applied
            after standard lens distortion, or ``NoExternalDistortion``.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.  Set
            ``True`` to permit the transfer at the cost of a synchronization.

    Returns:
        ImagePointsReturn with:
            image_points: (N, 2) projected image coordinates.
            valid_flag: (N,) boolean mask indicating which projections fall inside
                the image plane and within valid angular bounds.
    """
    image_points, valid_flag = _kernel_ops.camera_rays_to_image_points(
        camera_rays,
        projection,
        external_distortion,
        allow_device_transfer=allow_device_transfer,
    )
    return ImagePointsReturn(image_points=image_points, valid_flag=valid_flag)


def image_points_to_camera_rays(
    image_points: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    *,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Back-project image points to unit direction vectors in camera space.

    Inverts the projection performed by :func:`camera_rays_to_image_points`.  The
    output rays are not yet transformed into world space; rotate them by the sensor
    pose to obtain world-space directions.

    Args:
        image_points: (N, 2) image coordinates.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model applied during projection,
            whose inverse is applied here, or ``NoExternalDistortion``.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        (N, 3) unit direction vectors in camera space.
    """
    return _kernel_ops.image_points_to_camera_rays(
        image_points,
        projection,
        external_distortion,
        allow_device_transfer=allow_device_transfer,
    )


def project_world_points_mean_pose(
    world_points: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToImagePointsReturn:
    """Project world points using the temporally-averaged (mean) sensor pose.

    Interpolates the dynamic pose to a single representative pose at the midpoint
    of the exposure window rather than compensating per-point for sensor motion.
    Suitable for global-shutter cameras or as a fast approximate path when rolling-
    shutter distortion is negligible.

    Args:
        world_points: (N, 3) world coordinates.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model, or ``NoExternalDistortion``.
        resolution: (width, height) in pixels used for validity checking.
        dynamic_pose: Time-varying dynamic pose interpolated over the exposure window.
        start_timestamp_us: Start timestamp in microseconds for timestamp
            computation.  Defaults to 0 when ``None``; must be paired with
            ``end_timestamp_us`` (both provided or both ``None``, otherwise the
            kernel op raises ``ValueError``).
        end_timestamp_us: End timestamp in microseconds for timestamp
            computation.  Same both-or-neither contract as ``start_timestamp_us``.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` in the
            return value with per-point (N, 4, 4) transformation matrices.
        return_valid_flag: If ``True``, populate ``valid_flag`` in the return value
            with an (N,) boolean mask of projections that landed inside the image.
        return_valid_indices: If ``True``, populate ``valid_indices`` in the return
            value with the integer indices of valid projections (derived from
            ``valid_flag``; implies the validity mask is computed internally).
        return_timestamps: If ``True``, populate ``timestamps_us`` in the return
            value with per-point exposure timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        WorldPointsToImagePointsReturn with projected image points and optional
        auxiliary data.
    """
    (
        image_points,
        valid_flag,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.project_world_points_mean_pose(
        world_points,
        projection,
        external_distortion,
        dynamic_pose,
        resolution,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        return_valid_flags=return_valid_flag or return_valid_indices,
        return_timestamps=return_timestamps,
        return_poses=return_T_sensor_world,
        allow_device_transfer=allow_device_transfer,
    )
    return WorldPointsToImagePointsReturn(
        image_points=image_points,
        T_sensor_world=poses_to_matrix(poses_t, poses_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps_us if return_timestamps else None,
    )


def project_world_points_shutter_pose(
    world_points: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    max_iterations: int = 10,
    stop_mean_error_px: float = 0.001,
    stop_delta_mean_error_px: float = 0.00001,
    initial_relative_time: float = 0.5,
    return_T_sensor_world: bool = False,
    return_valid_flag: bool = False,
    return_valid_indices: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldPointsToImagePointsReturn:
    """Project world points using rolling-shutter compensation.

    Each point is projected at the pose interpolated to its specific scanline
    timestamp, determined by an iterative solver.  Use this in preference to
    :func:`project_world_points_mean_pose` when the sensor has a rolling shutter or
    when motion blur correction is required.

    Args:
        world_points: (N, 3) world coordinates.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model, or ``NoExternalDistortion``.
        resolution: (width, height) in pixels, used for both validity checking and
            scanline-time computation.
        shutter_type: Shutter mode (e.g. ``ShutterType.ROLLING_TOP_TO_BOTTOM``).
        dynamic_pose: Time-varying dynamic pose interpolated over the exposure window.
        start_timestamp_us: Start timestamp in microseconds for timestamp
            computation.  Defaults to 0 when ``None``; must be paired with
            ``end_timestamp_us`` (both provided or both ``None``, otherwise the
            kernel op raises ``ValueError``).
        end_timestamp_us: End timestamp in microseconds for timestamp
            computation.  Same both-or-neither contract as ``start_timestamp_us``.
        max_iterations: Maximum number of solver iterations for per-point pose
            refinement.
        stop_mean_error_px: Convergence threshold (applied per point): stop when
            the approximate reprojection error is below this value in pixels.
        stop_delta_mean_error_px: Convergence threshold (applied per point): stop
            when the change in projected image point between iterations is below
            this value in pixels.
        initial_relative_time: Initial relative time in [0, 1] used to seed the
            iterative solver for each point.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` in the
            return value with per-point (N, 4, 4) transformation matrices.
        return_valid_flag: If ``True``, populate ``valid_flag`` in the return value
            with an (N,) boolean mask of projections that landed inside the image.
        return_valid_indices: If ``True``, populate ``valid_indices`` in the return
            value with the integer indices of valid projections (derived from
            ``valid_flag``; implies the validity mask is computed internally).
        return_timestamps: If ``True``, populate ``timestamps_us`` in the return
            value with per-point exposure timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        WorldPointsToImagePointsReturn with projected image points and optional
        auxiliary data.
    """
    (
        image_points,
        valid_flag,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.project_world_points_shutter_pose(
        world_points,
        projection,
        external_distortion,
        resolution,
        shutter_type,
        dynamic_pose,
        start_timestamp_us=start_timestamp_us,
        end_timestamp_us=end_timestamp_us,
        max_iterations=max_iterations,
        stop_mean_error_px=stop_mean_error_px,
        stop_delta_mean_error_px=stop_delta_mean_error_px,
        initial_relative_time=initial_relative_time,
        return_valid_flags=return_valid_flag or return_valid_indices,
        return_timestamps=return_timestamps,
        return_poses=return_T_sensor_world,
        allow_device_transfer=allow_device_transfer,
    )
    return WorldPointsToImagePointsReturn(
        image_points=image_points,
        T_sensor_world=poses_to_matrix(poses_t, poses_r)
        if return_T_sensor_world
        else None,
        valid_flag=valid_flag if return_valid_flag else None,
        valid_indices=valid_flags_to_indices(valid_flag)
        if return_valid_indices
        else None,
        timestamps_us=timestamps_us if return_timestamps else None,
    )


def image_points_to_world_rays_static_pose(
    image_points: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    pose: Pose,
    *,
    timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Back-project image points to world-space rays using a fixed sensor pose.

    Each image point is unprojected through the lens model into a camera-space ray,
    then rotated into world space using the provided static pose.

    Args:
        image_points: (N, 2) image coordinates.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model, or ``NoExternalDistortion``.
        pose: Static sensor-to-world pose used to rotate camera-space rays
            into world space.
        timestamp_us: Timestamp in microseconds associated with the static pose.
            If ``None``, the kernel uses its own default.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` in the
            return value with per-ray (N, 4, 4) transformation matrices.
        return_timestamps: If ``True``, populate ``timestamps_us`` in the return
            value with per-ray timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        WorldRaysReturn with:
            world_rays: (N, 6) rays packed as (origin_xyz, direction_xyz).
            T_sensor_world: (N, 4, 4) transformation matrices, or ``None``.
            timestamps_us: (N,) timestamps in microseconds, or ``None``.
    """
    (
        world_rays,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.image_points_to_world_rays_static_pose(
        image_points,
        projection,
        external_distortion,
        pose,
        timestamp_us=timestamp_us,
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


def image_points_to_world_rays_shutter_pose(
    image_points: Tensor,
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Back-project image points to world-space rays with rolling-shutter pose interpolation.

    Unlike :func:`image_points_to_world_rays_static_pose`, the pose used for each
    ray is interpolated from the dynamic pose according to the scanline timestamp
    implied by the image point's row coordinate (or column coordinate for the
    ``ROLLING_LEFT_TO_RIGHT`` / ``ROLLING_RIGHT_TO_LEFT`` shutter modes) and the
    given shutter type.

    Args:
        image_points: (N, 2) image coordinates.
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model, or ``NoExternalDistortion``.
        resolution: (width, height) in pixels, used to derive per-scanline times
            (per row for vertical shutters, per column for horizontal shutters).
        shutter_type: Shutter mode (e.g. ``ShutterType.ROLLING_TOP_TO_BOTTOM``).
        dynamic_pose: Time-varying dynamic pose interpolated over the exposure window.
        start_timestamp_us: Start timestamp in microseconds for timestamp
            computation.  Defaults to 0 when ``None``; must be paired with
            ``end_timestamp_us`` (both provided or both ``None``, otherwise the
            kernel op raises ``ValueError``).
        end_timestamp_us: End timestamp in microseconds for timestamp
            computation.  Same both-or-neither contract as ``start_timestamp_us``.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` in the
            return value with per-ray (N, 4, 4) transformation matrices.
        return_timestamps: If ``True``, populate ``timestamps_us`` in the return
            value with per-ray exposure timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        WorldRaysReturn with:
            world_rays: (N, 6) rays packed as (origin_xyz, direction_xyz).
            T_sensor_world: (N, 4, 4) transformation matrices, or ``None``.
            timestamps_us: (N,) per-ray exposure timestamps in microseconds, or ``None``.
    """
    (
        world_rays,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.image_points_to_world_rays_shutter_pose(
        image_points,
        projection,
        external_distortion,
        resolution,
        shutter_type,
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


def pixel_grid_to_world_rays_shutter_pose(
    projection: OpenCVPinholeProjection,
    external_distortion: ExternalDistortion,
    resolution: tuple[int, int],
    shutter_type: ShutterType,
    dynamic_pose: DynamicPose,
    *,
    start_timestamp_us: int | None = None,
    end_timestamp_us: int | None = None,
    return_T_sensor_world: bool = False,
    return_timestamps: bool = False,
    allow_device_transfer: bool = False,
) -> WorldRaysReturn:
    """Back-project every pixel in a full image grid to world-space rays.

    Convenience wrapper that generates the dense (H * W, 2) pixel-center grid
    internally and then dispatches to the rolling-shutter ray kernel.  Equivalent
    to calling :func:`generate_image_points` followed by
    :func:`image_points_to_world_rays_shutter_pose`, but spares the caller the
    explicit ``generate_image_points`` call.

    Args:
        projection: OpenCV pinhole intrinsic parameters.
        external_distortion: External distortion model, or ``NoExternalDistortion``.
        resolution: (width, height) in pixels defining the grid dimensions.
        shutter_type: Shutter mode (e.g. ``ShutterType.ROLLING_TOP_TO_BOTTOM``).
        dynamic_pose: Time-varying dynamic pose interpolated over the exposure window.
        start_timestamp_us: Start timestamp in microseconds for timestamp
            computation.  Defaults to 0 when ``None``; must be paired with
            ``end_timestamp_us`` (both provided or both ``None``, otherwise the
            kernel op raises ``ValueError``).
        end_timestamp_us: End timestamp in microseconds for timestamp
            computation.  Same both-or-neither contract as ``start_timestamp_us``.
        return_T_sensor_world: If ``True``, populate ``T_sensor_world`` in the
            return value with per-ray (N, 4, 4) transformation matrices, where
            N = width * height.
        return_timestamps: If ``True``, populate ``timestamps_us`` in the return
            value with per-ray exposure timestamps in microseconds.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            any input tensor requires an implicit device or dtype transfer.

    Returns:
        WorldRaysReturn with:
            world_rays: (H * W, 6) rays in row-major order, packed as
                (origin_xyz, direction_xyz).
            T_sensor_world: (H * W, 4, 4) transformation matrices, or ``None``.
            timestamps_us: (H * W,) per-ray exposure timestamps in microseconds,
                or ``None``.
    """
    (
        world_rays,
        timestamps_us,
        poses_t,
        poses_r,
    ) = _kernel_ops.pixel_grid_to_world_rays_shutter_pose(
        projection,
        external_distortion,
        resolution,
        shutter_type,
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


def generate_image_points(
    resolution: tuple[int, int],
    device: torch.device | str = "cuda",
    *,
    allow_device_transfer: bool = False,
) -> Tensor:
    """Generate pixel-center image coordinates for a full image grid.

    Produces a dense ``(H, W, 2)`` tensor of ``(x, y)`` image-plane coordinates
    at the center of every pixel.  The first two axes are indexed ``[y, x]``
    (row-major), with ``x`` in channel 0 and ``y`` in channel 1.  Reshape to
    ``(H * W, 2)`` to feed any function that accepts an ``image_points``
    argument.

    Args:
        resolution: (width, height) in pixels.
        device: Target device for the output tensor.  Defaults to ``"cuda"``.
        allow_device_transfer: If ``False`` (default), raises ``RuntimeError`` when
            ``device`` is not a CUDA device.

    Returns:
        ``(height, width, 2)`` float32 tensor of pixel-center coordinates;
        ``out[y, x] == (x + 0.5, y + 0.5)``.
    """
    return _kernel_ops.generate_image_points(
        resolution, device=device, allow_device_transfer=allow_device_transfer
    )


__all__ = [
    "camera_rays_to_image_points",
    "generate_image_points",
    "image_points_to_camera_rays",
    "image_points_to_world_rays_static_pose",
    "image_points_to_world_rays_shutter_pose",
    "pixel_grid_to_world_rays_shutter_pose",
    "project_world_points_mean_pose",
    "project_world_points_shutter_pose",
]
