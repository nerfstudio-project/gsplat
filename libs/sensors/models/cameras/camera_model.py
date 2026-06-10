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

"""
Stateful camera model wrapping Layer 0 projection and external-distortion parameters.

Layer 2 stateful camera model wrapping a Layer 0 CameraProjection and ExternalDistortion
as nn.Module state, along with sensor resolution and ShutterType. Dispatches to Layer 1
functional ops and Layer 0 CUDA kernels for projection and unprojection. Callers work
entirely through this class; the underlying kernel types remain an implementation detail.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ... import functional as F
from ...functional.return_types import (
    ImagePointsReturn,
    PixelsReturn,
    WorldPointsToImagePointsReturn,
    WorldPointsToPixelsReturn,
    WorldRaysReturn,
)
from ...kernels.cameras import ops as _camera_ops
from ...kernels.cameras.types import (
    BivariateWindshieldDistortion,
    CameraProjection,
    ExternalDistortion,
    FThetaProjection,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ShutterType,
    script_class_name,
)
from ...kernels.common.pose import DynamicPose, Pose
from ..common.utils import (
    compute_scaled_resolution,
    filter_by_validity,
    valid_flags_to_indices,
)


def _move_projection(projection: CameraProjection, fn) -> CameraProjection:
    """Apply ``fn`` to the tensor fields of a TorchScript projection struct.

    Dispatches per concrete projection type so device/dtype moves rebuild the
    struct with the moved tensors while preserving its static configuration.
    """
    class_name = script_class_name(projection)
    if class_name == "OpenCVPinholeProjection":
        return OpenCVPinholeProjection(
            focal_length=fn(projection.focal_length),
            principal_point=fn(projection.principal_point),
            radial_coeffs=fn(projection.radial_coeffs),
            tangential_coeffs=fn(projection.tangential_coeffs),
            thin_prism_coeffs=fn(projection.thin_prism_coeffs),
            resolution=projection.resolution,
        )
    if class_name == "FThetaProjection":
        return FThetaProjection(
            principal_point=fn(projection.principal_point),
            fw_poly=fn(projection.fw_poly),
            bw_poly=fn(projection.bw_poly),
            A=fn(projection.A),
            resolution=projection.resolution,
            reference_polynomial=int(projection.reference_polynomial),
            fw_poly_degree=int(projection.fw_poly_degree),
            bw_poly_degree=int(projection.bw_poly_degree),
            newton_iterations=int(projection.newton_iterations),
            max_angle=float(projection.max_angle),
            min_2d_norm=float(projection.min_2d_norm),
        )
    if class_name == "OpenCVFisheyeProjection":
        return OpenCVFisheyeProjection(
            principal_point=fn(projection.principal_point),
            focal_length=fn(projection.focal_length),
            forward_poly=fn(projection.forward_poly),
            approx_backward_factor=fn(projection.approx_backward_factor),
            resolution=projection.resolution,
            newton_iterations=int(projection.newton_iterations),
            max_angle=float(projection.max_angle),
            min_2d_norm=float(projection.min_2d_norm),
        )
    raise TypeError(f"Unknown camera projection class: {class_name}")


def _move_external_distortion(
    external_distortion: ExternalDistortion, fn
) -> ExternalDistortion:
    """Apply ``fn`` to the tensor fields of a TorchScript external-distortion struct.

    Returns the original instance unchanged when it is ``NoExternalDistortion`` (no
    tensor fields), or when ``fn`` returns the identical tensor object (``is``-check)
    for the held coefficients; otherwise rebuilds the struct with the new tensor.
    """
    class_name = script_class_name(external_distortion)
    if class_name == "NoExternalDistortion":
        return external_distortion
    if class_name == "BivariateWindshieldDistortion":
        new_coeffs = fn(external_distortion.distortion_coeffs)
        if new_coeffs is external_distortion.distortion_coeffs:
            return external_distortion
        return BivariateWindshieldDistortion(
            new_coeffs,
            int(external_distortion.reference_polynomial),
            int(external_distortion.h_poly_degree),
            int(external_distortion.v_poly_degree),
        )
    raise TypeError(f"Unknown external distortion class: {class_name}")


class CameraModel(nn.Module):
    """Stateful camera model combining projection, distortion, resolution, and shutter type.

    Wraps a Layer 0 CameraProjection and ExternalDistortion as nn.Module parameters,
    making them available for gradient-based optimization. Provides a unified projection
    and unprojection interface that dispatches to Layer 1 functional ops for static,
    mean-pose, and rolling-shutter variants.

    The class is not abstract: a single ``CameraModel`` covers all supported projection
    types (e.g. OpenCV pinhole) by holding the concrete projection object.

    Note on Timestamps:
        - Static pose functions accept an optional ``timestamp_us`` integer (microseconds)
          that tags the resulting per-point timestamps.
        - Dynamic pose-based functions (``mean_pose``, ``shutter_pose``) accept
          ``start_timestamp_us`` / ``end_timestamp_us`` microsecond boundaries.
          ``mean_pose`` uses a fixed mid-frame time of ``0.5`` for all points;
          ``shutter_pose`` derives a per-point relative frame time in ``[0, 1]``
          from image-row position and shutter type.

    Example usage::

        from libs.sensors.kernels.cameras.types import (
            NoExternalDistortion,
            OpenCVPinholeProjection,
            ShutterType,
        )
        from libs.sensors.kernels.common.pose import Pose
        from libs.sensors.models.cameras.camera_model import CameraModel

        projection = OpenCVPinholeProjection(
            focal_length=...,
            principal_point=...,
            radial_coeffs=...,
            tangential_coeffs=...,
            thin_prism_coeffs=...,
            resolution=(1920, 1080),
        )
        camera = CameraModel(
            projection=projection,
            external_distortion=NoExternalDistortion(),
            resolution=(1920, 1080),
            shutter_type=ShutterType.GLOBAL,
        )

        # Project world points using a fixed sensor pose
        result = camera.world_points_to_image_points_static_pose(
            world_points, pose, return_valid_flag=True
        )

    Attributes:
        external_distortion: Layer 0 external distortion parameters
            (``NoExternalDistortion`` or ``BivariateWindshieldDistortion``).
        resolution: ``(width, height)`` in pixels.
        shutter_type: Rolling or global shutter behavior controlling per-row timing.
    """

    def __init__(
        self,
        projection: CameraProjection,
        external_distortion: ExternalDistortion,
        resolution: tuple[int, int],
        shutter_type: ShutterType,
    ):
        """Initialize the camera model.

        Args:
            projection: Layer 0 camera projection parameters (e.g.
                ``OpenCVPinholeProjection``). Must not be ``None``; pass a concrete
                instance explicitly.
            external_distortion: Layer 0 external distortion parameters. Must not be
                ``None``; pass ``NoExternalDistortion()`` when no distortion is needed.
            resolution: ``(width, height)`` sensor resolution in pixels.
            shutter_type: Rolling or global shutter behavior.
        """
        super().__init__()
        if projection is None:
            raise TypeError(
                "projection=None is not supported; pass a concrete projection "
                "(e.g. OpenCVPinholeProjection(...)) explicitly"
            )
        if external_distortion is None:
            raise TypeError(
                "external_distortion=None is not supported; pass NoExternalDistortion() explicitly"
            )
        self._projection = projection
        self.external_distortion = external_distortion
        self.resolution = resolution
        self.shutter_type = ShutterType(shutter_type)

    @property
    def projection(self) -> CameraProjection:
        """Layer 0 camera projection parameters."""
        return self._projection

    def _apply(self, fn):
        """Override ``nn.Module._apply`` to move TorchScript projection structs."""
        super()._apply(fn)
        self._projection = _move_projection(self._projection, fn)
        self.external_distortion = _move_external_distortion(
            self.external_distortion, fn
        )
        return self

    def _build_image_points_return(
        self,
        result: WorldPointsToImagePointsReturn,
        return_T_sensor_world: bool,
        return_valid_flag: bool,
        return_valid_indices: bool,
        return_timestamps: bool,
        return_all_projections: bool,
    ) -> WorldPointsToImagePointsReturn:
        """Build ``WorldPointsToImagePointsReturn`` with common post-processing.

        Filters image points, poses, and timestamps to the valid subset (unless
        ``return_all_projections`` is set) and promotes the validity mask and indices
        to the return struct only when explicitly requested.
        """
        valid = result.valid_flag
        image_points = filter_by_validity(
            result.image_points, valid, return_all_projections
        )
        t_sensor_world = filter_by_validity(
            result.T_sensor_world, valid, return_all_projections
        )
        timestamps = filter_by_validity(
            result.timestamps_us, valid, return_all_projections
        )
        return WorldPointsToImagePointsReturn(
            image_points=image_points,
            T_sensor_world=t_sensor_world if return_T_sensor_world else None,
            valid_flag=valid if return_valid_flag else None,
            valid_indices=valid_flags_to_indices(valid)
            if return_valid_indices
            else None,
            timestamps_us=timestamps if return_timestamps else None,
        )

    def world_points_to_image_points_static_pose(
        self,
        world_points: Tensor,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using fixed sensor pose.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            pose: Static sensor → world pose (``Pose`` object with translation and
                rotation).
            timestamp_us: Optional timestamp in microseconds assigned to every
                projected point; used only when ``return_timestamps=True``.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToImagePointsReturn`` with projected image points and
            optional auxiliary data.
        """
        dynamic_pose = DynamicPose.from_static_pose(pose)
        result = F.project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            dynamic_pose,
            start_timestamp_us=timestamp_us,
            end_timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world,
            return_valid_flag,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

    def world_points_to_image_points_mean_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using mean pose (not compensating for sensor motion).

        The mean pose is the SLERP-interpolated pose at normalized time ``0.5``,
        applied uniformly to all points regardless of their image-row position.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToImagePointsReturn`` with projected image points.
        """
        result = F.project_world_points_mean_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world,
            return_valid_flag,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

    def world_points_to_image_points_shutter_pose(
        self,
        world_points: Tensor,
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
        return_all_projections: bool = False,
    ) -> WorldPointsToImagePointsReturn:
        """Project world points using rolling-shutter compensation.

        Each point is projected with the sensor pose interpolated to the normalized
        time corresponding to its image row (or column), accounting for sensor motion
        across the frame. Convergence is determined iteratively.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            max_iterations: Maximum number of iterations for the rolling-shutter solver.
            stop_mean_error_px: Convergence threshold on mean reprojection error in
                pixels.
            stop_delta_mean_error_px: Convergence threshold on the change in mean
                reprojection error between iterations in pixels.
            initial_relative_time: Initial normalized time in ``[0, 1]`` used as the
                starting estimate for the per-point shutter time.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToImagePointsReturn`` with projected image points.
        """
        result = F.project_world_points_shutter_pose(
            world_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            self.shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            initial_relative_time=initial_relative_time,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        return self._build_image_points_return(
            result,
            return_T_sensor_world,
            return_valid_flag,
            return_valid_indices,
            return_timestamps,
            return_all_projections,
        )

    def _points_to_pixels_return(
        self,
        result: WorldPointsToImagePointsReturn,
    ) -> WorldPointsToPixelsReturn:
        """Convert a ``WorldPointsToImagePointsReturn`` to ``WorldPointsToPixelsReturn``."""
        return WorldPointsToPixelsReturn(
            pixels=self.image_points_to_pixels(result.image_points),
            T_sensor_world=result.T_sensor_world,
            valid_flag=result.valid_flag,
            valid_indices=result.valid_indices,
            timestamps_us=result.timestamps_us,
        )

    def world_points_to_pixels_static_pose(
        self,
        world_points: Tensor,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using fixed sensor pose.

        Convenience wrapper around
        :meth:`world_points_to_image_points_static_pose` that converts sub-pixel
        image coordinates to integer pixel indices via ``floor``.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            pose: Static sensor → world pose (``Pose`` object with translation and
                rotation).
            timestamp_us: Optional timestamp in microseconds assigned to every
                projected point.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToPixelsReturn`` with integer pixel indices.
        """
        result = self.world_points_to_image_points_static_pose(
            world_points,
            pose,
            timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )
        return self._points_to_pixels_return(result)

    def world_points_to_pixels_mean_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using mean pose (not compensating for sensor motion).

        Convenience wrapper around
        :meth:`world_points_to_image_points_mean_pose` that converts sub-pixel
        image coordinates to integer pixel indices via ``floor``.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToPixelsReturn`` with integer pixel indices.
        """
        result = self.world_points_to_image_points_mean_pose(
            world_points,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )
        return self._points_to_pixels_return(result)

    def world_points_to_pixels_shutter_pose(
        self,
        world_points: Tensor,
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
        return_all_projections: bool = False,
    ) -> WorldPointsToPixelsReturn:
        """Project world points to pixel indices using rolling-shutter compensation.

        Convenience wrapper around
        :meth:`world_points_to_image_points_shutter_pose` that converts sub-pixel
        image coordinates to integer pixel indices via ``floor``.

        Args:
            world_points: ``(N, 3)`` world coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            max_iterations: Maximum number of iterations for the rolling-shutter solver.
            stop_mean_error_px: Convergence threshold on mean reprojection error in
                pixels.
            stop_delta_mean_error_px: Convergence threshold on the change in mean
                reprojection error between iterations in pixels.
            initial_relative_time: Initial normalized time in ``[0, 1]`` used as the
                starting estimate for the per-point shutter time.
            return_T_sensor_world: If ``True``, return per-point poses as
                ``(N, 4, 4)`` matrices.
            return_valid_flag: If ``True``, include the per-point validity mask in
                the return struct.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps in
                microseconds.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToPixelsReturn`` with integer pixel indices.
        """
        result = self.world_points_to_image_points_shutter_pose(
            world_points,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            max_iterations=max_iterations,
            stop_mean_error_px=stop_mean_error_px,
            stop_delta_mean_error_px=stop_delta_mean_error_px,
            initial_relative_time=initial_relative_time,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=return_valid_flag,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            return_all_projections=return_all_projections,
        )
        return self._points_to_pixels_return(result)

    def image_points_to_world_rays_static_pose(
        self,
        image_points: Tensor,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project image points to world rays using fixed sensor pose.

        Args:
            image_points: ``(N, 2)`` image coordinates.
            pose: Static sensor → world pose (``Pose`` object with translation and
                rotation).
            timestamp_us: Optional timestamp in microseconds assigned to every ray.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        return F.image_points_to_world_rays_static_pose(
            image_points,
            self.projection,
            self.external_distortion,
            pose,
            timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    def image_points_to_world_rays_mean_pose(
        self,
        image_points: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project using mean pose (not compensating for sensor motion).

        Interpolates the dynamic pose at normalized time ``0.5`` (mid-frame) and
        delegates to :meth:`image_points_to_world_rays_static_pose`. The mid-frame
        timestamp is the arithmetic mean of ``start_timestamp_us`` and
        ``end_timestamp_us`` when both are provided.

        Args:
            image_points: ``(N, 2)`` image coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        pose = _camera_ops.mean_pose_to_static_pose(
            dynamic_pose, image_points.device, torch.float32
        )
        timestamp = None
        if start_timestamp_us is not None and end_timestamp_us is not None:
            # Integer math; avoids float round-trip imprecision above 2^53 us.
            timestamp = (start_timestamp_us + end_timestamp_us) // 2
        return self.image_points_to_world_rays_static_pose(
            image_points,
            pose,
            timestamp_us=timestamp,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def image_points_to_world_rays_shutter_pose(
        self,
        image_points: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project using rolling-shutter compensation.

        Each image point is back-projected with the sensor pose interpolated to the
        normalized time derived from its image-row (or column) position, as determined
        by ``self.shutter_type``.

        Args:
            image_points: ``(N, 2)`` image coordinates.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        return F.image_points_to_world_rays_shutter_pose(
            image_points,
            self.projection,
            self.external_distortion,
            self.resolution,
            self.shutter_type,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    def pixels_to_world_rays_static_pose(
        self,
        pixels: Tensor,
        pose: Pose,
        *,
        timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using fixed sensor pose.

        Converts pixels to image points via :meth:`pixels_to_image_points`, then
        delegates to :meth:`image_points_to_world_rays_static_pose`.

        Args:
            pixels: ``(N, 2)`` integer pixel indices.
            pose: Static sensor → world pose (``Pose`` object with translation and
                rotation).
            timestamp_us: Optional timestamp in microseconds assigned to every ray.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        return self.image_points_to_world_rays_static_pose(
            self.pixels_to_image_points(pixels),
            pose,
            timestamp_us=timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_mean_pose(
        self,
        pixels: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using mean pose (not compensating for sensor motion).

        Converts pixels to image points via :meth:`pixels_to_image_points`, then
        delegates to :meth:`image_points_to_world_rays_mean_pose`.

        Args:
            pixels: ``(N, 2)`` integer pixel indices.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        return self.image_points_to_world_rays_mean_pose(
            self.pixels_to_image_points(pixels),
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def pixels_to_world_rays_shutter_pose(
        self,
        pixels: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Back-project pixel indices to world rays using rolling-shutter compensation.

        Converts pixels to image points via :meth:`pixels_to_image_points`, then
        delegates to :meth:`image_points_to_world_rays_shutter_pose`.

        Args:
            pixels: ``(N, 2)`` integer pixel indices.
            dynamic_pose: Time-varying dynamic pose spanning the frame interval.
            start_timestamp_us: Start timestamp in microseconds for the frame interval.
            end_timestamp_us: End timestamp in microseconds for the frame interval.
            return_T_sensor_world: If ``True``, return per-ray poses as
                ``(N, 4, 4)`` matrices.
            return_timestamps: If ``True``, return per-ray timestamps in microseconds.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in world frame.
        """
        return self.image_points_to_world_rays_shutter_pose(
            self.pixels_to_image_points(pixels),
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
        )

    def camera_rays_to_image_points(
        self, camera_rays: Tensor, *, return_jacobians: bool = False
    ) -> ImagePointsReturn:
        """Convert camera rays to image points.

        Projects normalized camera-frame rays through the projection and external
        distortion models to obtain image coordinates. When ``return_jacobians=True``,
        the ``(N, 2, 3)`` Jacobian of image-point coordinates with respect to camera
        ray components is computed via autograd on an isolated gradient leaf so the
        graph does not propagate back to the caller's ``camera_rays`` tensor.

        Args:
            camera_rays: ``(N, 3)`` normalized ray directions in the camera frame.
            return_jacobians: If ``True``, compute and return per-point Jacobians
                ``(N, 2, 3)`` of image coordinates with respect to camera rays.
                The returned ``image_points`` will be detached from the autograd
                graph; pass ``return_jacobians=False`` if a differentiable value
                path is required.

        Returns:
            ``ImagePointsReturn`` with image points, per-point validity mask, and
            optional Jacobians.
        """
        if not return_jacobians:
            return F.camera_rays_to_image_points(
                camera_rays,
                self.projection,
                self.external_distortion,
                allow_device_transfer=True,
            )
        # Compute the value path against an isolated grad-tracked leaf so the
        # autograd graph for the Jacobian doesn't leak back to the caller's
        # camera_rays. The returned image_points is detached for the same
        # reason; pass return_jacobians=False if you need a differentiable
        # value path.
        rays_grad = camera_rays.detach().clone().requires_grad_(True)
        result = F.camera_rays_to_image_points(
            rays_grad,
            self.projection,
            self.external_distortion,
            allow_device_transfer=True,
        )
        grads = [
            torch.autograd.grad(
                result.image_points[:, coord].sum(),
                rays_grad,
                retain_graph=True,
            )[0]
            for coord in range(2)
        ]
        return ImagePointsReturn(
            image_points=result.image_points.detach(),
            valid_flag=result.valid_flag,
            jacobians=torch.stack(grads, dim=1),
        )

    def camera_rays_to_pixels(self, camera_rays: Tensor) -> PixelsReturn:
        """Convert camera rays to pixel indices.

        Args:
            camera_rays: ``(N, 3)`` normalized ray directions in the camera frame.

        Returns:
            ``PixelsReturn`` with integer pixel indices and a per-point validity mask.
        """
        result = self.camera_rays_to_image_points(camera_rays)
        return PixelsReturn(
            pixels=self.image_points_to_pixels(result.image_points),
            valid_flag=result.valid_flag,
        )

    def image_points_to_camera_rays(self, image_points: Tensor) -> Tensor:
        """Convert image points to camera rays.

        Args:
            image_points: ``(N, 2)`` image coordinates.

        Returns:
            ``(N, 3)`` normalized ray directions in the camera frame.
        """
        return F.image_points_to_camera_rays(
            image_points,
            self.projection,
            self.external_distortion,
            allow_device_transfer=True,
        )

    def pixels_to_camera_rays(self, pixels: Tensor) -> Tensor:
        """Convert pixel indices to camera rays.

        Args:
            pixels: ``(N, 2)`` integer pixel indices.

        Returns:
            ``(N, 3)`` normalized ray directions in the camera frame.
        """
        return self.image_points_to_camera_rays(self.pixels_to_image_points(pixels))

    def pixels_to_image_points(self, pixels: Tensor) -> Tensor:
        """Convert pixel indices to continuous image coordinates (pixel centers).

        Args:
            pixels: ``(N, 2)`` integer pixel indices.

        Returns:
            ``(N, 2)`` float image coordinates at pixel centers (``pixel + 0.5``).
        """
        return pixels.to(dtype=torch.float32) + 0.5

    def image_points_to_pixels(self, image_points: Tensor) -> Tensor:
        """Convert continuous image coordinates to integer pixel indices.

        Args:
            image_points: ``(N, 2)`` float image coordinates.

        Returns:
            ``(N, 2)`` integer pixel indices (``floor`` of image coordinates).
        """
        return torch.floor(image_points).to(torch.int32)

    def image_points_relative_frame_times(self, image_points: Tensor) -> Tensor:
        """Get relative frame-times [0, 1] based on image coordinates and rolling shutter.

        For global shutter, returns zero for all points. For rolling-shutter variants,
        returns the normalized scanline time derived from the image-row (or column)
        position and ``self.shutter_type``.

        Args:
            image_points: ``(N, 2)`` image coordinates.

        Returns:
            ``(N,)`` float tensor of relative frame times in ``[0, 1]``.
        """
        return _camera_ops.relative_frame_times(
            image_points, self.resolution, self.shutter_type
        )

    def transform(
        self,
        image_domain_scale: float | tuple[float, float],
        image_domain_offset: tuple[float, float] = (0.0, 0.0),
        new_resolution: tuple[int, int] | None = None,
    ) -> "CameraModel":
        """Apply image domain transformation to camera parameters.

        Creates a new ``CameraModel`` with intrinsic parameters scaled and shifted to
        reflect a crop or resize of the image domain. The external distortion and
        shutter type are preserved unchanged.

        Args:
            image_domain_scale: Isotropic scale factor (``float``) or anisotropic
                per-axis scale ``(sx, sy)`` applied to the image domain.
            image_domain_offset: ``(ox, oy)`` offset in the scaled image domain,
                used to account for cropping. Defaults to ``(0.0, 0.0)``.
            new_resolution: Explicit ``(width, height)`` for the output model. When
                ``None``, the resolution is derived from ``image_domain_scale`` and the
                current resolution.

        Returns:
            New ``CameraModel`` with transformed projection parameters and updated
            resolution.
        """
        if isinstance(image_domain_scale, tuple):
            sx, sy = image_domain_scale
        else:
            sx = sy = image_domain_scale
        new_model_resolution = compute_scaled_resolution(
            self.resolution, image_domain_scale, new_resolution
        )
        new_projection = self._projection.transform(
            (sx, sy), image_domain_offset, new_model_resolution
        )
        return CameraModel(
            projection=new_projection,
            external_distortion=self.external_distortion,
            resolution=new_model_resolution,
            shutter_type=self.shutter_type,
        )


__all__ = ["CameraModel"]
