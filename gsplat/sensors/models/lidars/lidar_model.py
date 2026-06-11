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

"""Stateful spinning-LiDAR model wrapping Layer 0 projection parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ... import functional as F
from ...functional.return_types import (
    SensorAnglesReturn,
    SensorRayReturn,
    WorldPointsToSensorAnglesReturn,
    WorldRaysReturn,
)
from ...kernels.common.pose import DynamicPose
from ...kernels.lidars.types import (
    RowOffsetStructuredSpinningLidarProjection,
    SpinningDirection,
    script_class_name,
)
from ..common.utils import filter_by_validity, valid_flags_to_indices

_TWO_PI = 2.0 * torch.pi


def _move_lidar_projection(
    projection: RowOffsetStructuredSpinningLidarProjection, fn
) -> RowOffsetStructuredSpinningLidarProjection:
    class_name = script_class_name(projection)
    if class_name == "RowOffsetStructuredSpinningLidarProjection":
        return RowOffsetStructuredSpinningLidarProjection(
            row_elevations_rad=fn(projection.row_elevations_rad),
            column_azimuths_rad=fn(projection.column_azimuths_rad),
            row_azimuth_offsets_rad=fn(projection.row_azimuth_offsets_rad),
            fov_vert_start_rad=projection.fov_vert_start_rad,
            fov_vert_span_rad=projection.fov_vert_span_rad,
            fov_horiz_start_rad=projection.fov_horiz_start_rad,
            fov_horiz_span_rad=projection.fov_horiz_span_rad,
            spinning_direction=projection.spinning_direction,
            has_row_offsets=projection.has_row_offsets,
        )
    raise TypeError(f"Unknown LiDAR projection class: {class_name}")


class LidarModel(nn.Module):
    """Stateful row-offset structured spinning-LiDAR model.

    Wraps a Layer 0 ``RowOffsetStructuredSpinningLidarProjection`` as nn.Module
    state, making its angle tables available for gradient-based optimization.
    Provides ray generation, angle conversion, and inverse-projection methods that
    dispatch to Layer 1 functional ops.

    The projection is held as a plain attribute (not an ``nn`` buffer); its tensors
    are moved across devices/dtypes by overriding ``_apply``.

    Note on timestamps:
        Dynamic pose-based functions accept ``start_timestamp_us`` /
        ``end_timestamp_us`` microsecond boundaries. Per-element relative frame
        times in ``[0, 1]`` derive from column index (forward) or refined
        rolling-shutter time (inverse).

    Example usage::

        from gsplat.sensors.models import (
            RowOffsetStructuredSpinningLidarProjection,
        )
        from gsplat.sensors.models.lidars import LidarModel

        projection = RowOffsetStructuredSpinningLidarProjection(...)
        lidar = LidarModel(projection=projection)

        # Generate world rays with rolling shutter using a dynamic pose
        result = lidar.elements_to_world_rays_shutter_pose(elements, dynamic_pose)

    Attributes:
        projection: Layer 0 row-offset structured spinning projection parameters.
    """

    _fov_eps_rad: float

    def __init__(
        self,
        projection: RowOffsetStructuredSpinningLidarProjection,
        fov_eps_factor: float = 4.0,
    ):
        """Initialize the spinning-LiDAR model.

        Args:
            projection: Layer 0 row-offset structured spinning projection
                parameters. Must not be ``None``; pass a concrete instance
                explicitly.
            fov_eps_factor: Factor for the FOV epsilon used in validity checks to
                account for accumulated numerical error. The validity check adds
                ``2 * fov_eps_factor * float32_eps`` of slack at the FOV boundary.
                Default ``4.0``.
        """
        super().__init__()
        if projection is None:
            raise TypeError(
                "projection=None is not supported; pass a concrete "
                "RowOffsetStructuredSpinningLidarProjection(...) explicitly"
            )
        self._projection = projection
        self._fov_eps_rad = fov_eps_factor * torch.finfo(torch.float32).eps

    @property
    def projection(self) -> RowOffsetStructuredSpinningLidarProjection:
        """Layer 0 row-offset structured spinning projection parameters."""
        return self._projection

    def _apply(self, fn):
        """Override ``nn.Module._apply`` to move the projection's angle tables."""
        super()._apply(fn)
        self._projection = _move_lidar_projection(self._projection, fn)
        return self

    # ------------------------------------------------------------------
    # Sensor ray / angle conversions
    # ------------------------------------------------------------------

    def sensor_rays_to_sensor_angles(
        self,
        sensor_rays: Tensor,
        *,
        normalized: bool = True,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        """Convert sensor-frame rays to ``[elevation, azimuth]`` angles.

        Args:
            sensor_rays: ``(N, 3)`` direction vectors in the sensor frame.
            normalized: If ``True`` (default), assumes rays are already
                unit-length. If ``False``, normalizes before conversion.
            return_valid_flag: If ``True``, include a per-ray FOV validity mask.

        Returns:
            ``SensorAnglesReturn`` with ``[elevation, azimuth]`` angles and an
            optional validity mask.
        """
        if not normalized:
            sensor_rays = torch.nn.functional.normalize(sensor_rays, dim=-1)
        result = F.sensor_rays_to_sensor_angles(sensor_rays, allow_device_transfer=True)
        valid_flag = (
            self._valid_sensor_angles(result.sensor_angles)
            if return_valid_flag
            else None
        )
        return SensorAnglesReturn(
            sensor_angles=result.sensor_angles, valid_flag=valid_flag
        )

    def sensor_angles_to_sensor_rays(
        self,
        sensor_angles: Tensor,
        *,
        return_valid_flag: bool = False,
    ) -> SensorRayReturn:
        """Convert ``[elevation, azimuth]`` angles to unit sensor-frame rays.

        Args:
            sensor_angles: ``(N, 2)`` ``[elevation, azimuth]`` in radians.
            return_valid_flag: If ``True``, include a per-angle FOV validity mask.

        Returns:
            ``SensorRayReturn`` with unit ray directions and an optional validity
            mask.
        """
        result = F.sensor_angles_to_sensor_rays(
            sensor_angles, allow_device_transfer=True
        )
        valid_flag = (
            self._valid_sensor_angles(sensor_angles) if return_valid_flag else None
        )
        return SensorRayReturn(sensor_rays=result.sensor_rays, valid_flag=valid_flag)

    def elements_to_sensor_angles(
        self,
        elements: Tensor,
        *,
        return_valid_flag: bool = False,
    ) -> SensorAnglesReturn:
        """Look up ``[elevation, azimuth]`` angles for ``[row, col]`` elements.

        Args:
            elements: ``(N, 2)`` int ``[row, col]`` element indices.
            return_valid_flag: If ``True``, include a per-element in-bounds mask.

        Returns:
            ``SensorAnglesReturn`` with angles and an optional in-bounds mask.
        """
        return F.elements_to_sensor_angles(
            elements,
            self.projection,
            return_valid_flag=return_valid_flag,
            allow_device_transfer=True,
        )

    def elements_to_sensor_rays(self, elements: Tensor) -> Tensor:
        """Convert ``[row, col]`` elements to unit sensor-frame rays.

        Args:
            elements: ``(N, 2)`` int ``[row, col]`` element indices.

        Returns:
            ``(N, 3)`` unit direction vectors in the sensor frame.
        """
        angles = self.elements_to_sensor_angles(elements).sensor_angles
        return self.sensor_angles_to_sensor_rays(angles).sensor_rays

    def elements_to_sensor_points(
        self, elements: Tensor, element_distances: Tensor
    ) -> Tensor:
        """Convert ``[row, col]`` elements and distances to 3D sensor points.

        Args:
            elements: ``(N, 2)`` int ``[row, col]`` element indices.
            element_distances: ``(N,)`` distances in meters.

        Returns:
            ``(N, 3)`` points in the sensor frame.
        """
        sensor_rays = self.elements_to_sensor_rays(elements)
        return sensor_rays * element_distances.unsqueeze(-1)

    # ------------------------------------------------------------------
    # World ray generation with rolling shutter
    # ------------------------------------------------------------------

    def elements_to_world_rays_shutter_pose(
        self,
        elements: Tensor | None,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        return_T_sensor_world: bool = False,
        return_timestamps: bool = False,
    ) -> WorldRaysReturn:
        """Generate world-frame rays using rolling-shutter compensation.

        Args:
            elements: ``(N, 2)`` int ``[row, col]`` element indices, or ``None`` to
                generate the full row-major ``(n_rows x n_columns)`` grid.
            dynamic_pose: Time-varying dynamic pose (start at ``t=0``, end at
                ``t=1``).
            start_timestamp_us: Start timestamp in microseconds.
            end_timestamp_us: End timestamp in microseconds.
            return_T_sensor_world: If ``True``, return per-ray ``(N, 4, 4)`` poses.
            return_timestamps: If ``True``, return per-ray timestamps.

        Returns:
            ``WorldRaysReturn`` with ``world_rays: (N, 6)`` packed
            ``[origin_xyz, direction_xyz]`` in the world frame.
        """
        return F.generate_spinning_lidar_rays(
            self.projection,
            elements,
            dynamic_pose,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )

    # ------------------------------------------------------------------
    # World points to sensor angles with rolling shutter
    # ------------------------------------------------------------------

    def world_points_to_sensor_angles_shutter_pose(
        self,
        world_points: Tensor,
        dynamic_pose: DynamicPose,
        *,
        start_timestamp_us: int | None = None,
        end_timestamp_us: int | None = None,
        max_iterations: int = 10,
        stop_mean_relative_time_error: float = 1e-4,
        stop_delta_mean_relative_time_error: float = 1e-6,
        initial_relative_time: float = 0.5,
        return_T_sensor_world: bool = False,
        return_valid_flag: bool = False,
        return_valid_indices: bool = False,
        return_timestamps: bool = False,
        return_all_projections: bool = False,
    ) -> WorldPointsToSensorAnglesReturn:
        """Inverse-project world points to sensor angles with rolling shutter.

        Args:
            world_points: ``(N, 3)`` world-frame points.
            dynamic_pose: Time-varying dynamic pose (start at ``t=0``, end at
                ``t=1``).
            start_timestamp_us: Start timestamp in microseconds.
            end_timestamp_us: End timestamp in microseconds.
            max_iterations: Maximum rolling-shutter solver iterations; must be in
                ``[1, 32]`` or the op raises.
            stop_mean_relative_time_error: Convergence threshold on relative-time
                change.
            stop_delta_mean_relative_time_error: Convergence threshold on the change
                in relative-time error between iterations.
            initial_relative_time: Initial relative-time seed in ``[0, 1]``.
            return_T_sensor_world: If ``True``, return per-point ``(N, 4, 4)`` poses.
            return_valid_flag: If ``True``, include the in-FOV validity mask.
            return_valid_indices: If ``True``, return indices of valid projections.
            return_timestamps: If ``True``, return per-point timestamps.
            return_all_projections: If ``True``, return all projections including
                those flagged invalid; otherwise only valid points are returned.

        Returns:
            ``WorldPointsToSensorAnglesReturn`` with sensor angles and optional
            auxiliary data.
        """
        result = F.inverse_project_spinning_lidar(
            self.projection,
            world_points,
            dynamic_pose,
            max_iterations=max_iterations,
            stop_mean_relative_time_error=stop_mean_relative_time_error,
            stop_delta_mean_relative_time_error=stop_delta_mean_relative_time_error,
            initial_relative_time=initial_relative_time,
            start_timestamp_us=start_timestamp_us,
            end_timestamp_us=end_timestamp_us,
            return_T_sensor_world=return_T_sensor_world,
            return_valid_flag=True,
            return_valid_indices=return_valid_indices,
            return_timestamps=return_timestamps,
            allow_device_transfer=True,
        )
        valid = result.valid_flag
        return WorldPointsToSensorAnglesReturn(
            sensor_angles=filter_by_validity(
                result.sensor_angles, valid, return_all_projections
            ),
            T_sensor_world=filter_by_validity(
                result.T_sensor_world, valid, return_all_projections
            )
            if return_T_sensor_world
            else None,
            valid_flag=valid if return_valid_flag else None,
            valid_indices=valid_flags_to_indices(valid)
            if return_valid_indices
            else None,
            timestamps_us=filter_by_validity(
                result.timestamps_us, valid, return_all_projections
            )
            if return_timestamps
            else None,
        )

    # ------------------------------------------------------------------
    # Relative frame times
    # ------------------------------------------------------------------

    def sensor_angles_relative_frame_times(self, sensor_angles: Tensor) -> Tensor:
        """Get relative frame-times ``[0, 1]`` for sensor-angle coordinates.

        Uses the linear nearest-column fallback: the nearest row determines the
        per-row azimuth offset (subtracted), then the nearest column (by wrapped
        angular distance) gives the column index, normalized to ``[0, 1]``. The
        spinning direction is baked into the host-built ``column_azimuths_rad``
        ordering, so no runtime direction flip is applied.

        Args:
            sensor_angles: ``(N, 2)`` ``[elevation, azimuth]`` in radians.

        Returns:
            ``(N,)`` float relative frame times in ``[0, 1]``.
        """
        proj = self.projection
        elevations = sensor_angles[:, 0]
        azimuths = sensor_angles[:, 1]

        # Nearest row by elevation (no wrapping; elevation never wraps).
        row_idx = torch.argmin(
            torch.abs(elevations.unsqueeze(-1) - proj.row_elevations_rad.unsqueeze(0)),
            dim=-1,
        )

        adjusted_azimuths = azimuths
        if proj.has_row_offsets:
            adjusted_azimuths = azimuths - proj.row_azimuth_offsets_rad[row_idx]

        # Nearest column by wrapped angular distance (+-pi seam handling).
        az_diff = adjusted_azimuths.unsqueeze(-1) - proj.column_azimuths_rad.unsqueeze(
            0
        )
        col_idx = torch.argmin(torch.abs(self._normalize_angle(az_diff)), dim=-1)

        n_columns = self.n_columns
        if n_columns <= 1:
            return torch.zeros_like(elevations)
        return col_idx.to(sensor_angles.dtype) / (n_columns - 1)

    @staticmethod
    def _normalize_angle(angle: Tensor) -> Tensor:
        """Wrap angles to ``[-pi, pi)`` via the fmod-based formula."""
        wrapped = torch.fmod(angle + torch.pi, _TWO_PI)
        wrapped = torch.where(wrapped < 0, wrapped + _TWO_PI, wrapped)
        return wrapped - torch.pi

    # ------------------------------------------------------------------
    # FOV validity
    # ------------------------------------------------------------------

    def _relative_sensor_angles(self, sensor_angles: Tensor) -> Tensor:
        """Compute relative ``[elevation, azimuth]`` offsets from the FOV start.

        Vertical is always measured clockwise from the top (``fov_vert_start``).
        Horizontal direction follows ``spinning_direction``.
        """
        proj = self.projection
        elevations = sensor_angles[:, 0]
        azimuths = sensor_angles[:, 1]

        rel_elevation = proj.fov_vert_start_rad - elevations
        if int(proj.spinning_direction) == int(SpinningDirection.COUNTERCLOCKWISE):
            rel_azimuth = azimuths - proj.fov_horiz_start_rad
        else:
            rel_azimuth = proj.fov_horiz_start_rad - azimuths

        rel_azimuth = rel_azimuth % _TWO_PI
        return torch.stack([rel_elevation, rel_azimuth], dim=-1)

    def _valid_sensor_angles(self, sensor_angles: Tensor) -> Tensor:
        """Return ``(N,)`` bool: True for angles within the FOV (with eps slack)."""
        proj = self.projection
        rel = self._relative_sensor_angles(sensor_angles)
        # 2 * eps slack: 1x eps is inherited from the FOV start in the relative
        # angle, so this effectively checks 1x eps on the FOV end boundary.
        return torch.logical_and(
            torch.logical_and(
                rel[:, 0] >= -2 * self._fov_eps_rad,
                rel[:, 0] <= proj.fov_vert_span_rad + 2 * self._fov_eps_rad,
            ),
            rel[:, 1] <= proj.fov_horiz_span_rad + 2 * self._fov_eps_rad,
        )

    def valid_sensor_angles(self, sensor_angles: Tensor) -> Tensor:
        """Check whether sensor angles fall within the sensor FOV.

        Args:
            sensor_angles: ``(N, 2)`` ``[elevation, azimuth]`` in radians.

        Returns:
            ``(N,)`` bool mask, True for angles within the FOV.
        """
        return self._valid_sensor_angles(sensor_angles)

    def forward(self, *args, **kwargs):
        """Forward pass is not implemented for LiDAR models.

        LiDAR models are used via their projection methods (e.g.
        ``elements_to_world_rays_shutter_pose``) rather than as part of a forward
        pass.
        """
        raise NotImplementedError(
            "LidarModel.forward() is not implemented. Use projection methods like "
            "elements_to_world_rays_shutter_pose directly."
        )

    @property
    def n_rows(self) -> int:
        """Number of rows (vertical channels)."""
        return int(self.projection.row_elevations_rad.shape[0])

    @property
    def n_columns(self) -> int:
        """Number of columns (horizontal samples)."""
        return int(self.projection.column_azimuths_rad.shape[0])

    @property
    def n_elements(self) -> int:
        """Total number of elements (``n_rows * n_columns``)."""
        return self.n_rows * self.n_columns

    @property
    def fov_vert(self) -> tuple[float, float]:
        """Vertical FOV as ``(start_rad, span_rad)``."""
        return (
            self.projection.fov_vert_start_rad,
            self.projection.fov_vert_span_rad,
        )

    @property
    def fov_horiz(self) -> tuple[float, float]:
        """Horizontal FOV as ``(start_rad, span_rad)``."""
        return (
            self.projection.fov_horiz_start_rad,
            self.projection.fov_horiz_span_rad,
        )

    @property
    def spinning_direction(self) -> SpinningDirection:
        """Azimuth sweep direction of the spinning LiDAR."""
        return SpinningDirection(int(self.projection.spinning_direction))


__all__ = ["LidarModel"]
