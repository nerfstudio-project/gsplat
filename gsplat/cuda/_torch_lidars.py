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

"""Reference Lidar model implementation for GSplat.

This module contains reference lidar model implementations.
"""

import math
import torch
import copy
from typing import Optional, Tuple
from dataclasses import dataclass
from torch import Tensor
from gsplat._helper import assert_shape
from ._lidar import (
    SpinningDirection,
    FOV,
    LidarModelParameters,
    StructuredLidarModelParameters,
    SpinningLidarModelParameters,
    StructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
)

from ._wrapper import RowOffsetStructuredSpinningLidarModelParametersExt

from ._math import (
    _safe_normalize,
)

from ._torch_cameras import _BaseCameraModel

# TODO: The hierarchy should be rooted at new class SensorModel.
class _LidarModel:
    def __init__(self, params: LidarModelParameters) -> None:
        self.params = params

    def __getattr__(self, name):
        params = object.__getattribute__(self, "params")
        if hasattr(params, name):
            return getattr(params, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# TODO: Since we don't have SensorModel yet, we make StructuredLidarModel inherit from _BaseCameraModel,
# as it is close to the structured model (With width/height).
class _StructuredLidarModel(_LidarModel, _BaseCameraModel):
    def __init__(self, params: StructuredLidarModelParameters) -> None:
        _LidarModel.__init__(self, params)

        # TODO: passing shutter type because the base expects it, but Lidar doesn't use it.
        # This is a clear sign of design smell, the camera/sensor class hierarchy needs a revamp
        # to properly support lidar sensors.
        _BaseCameraModel.__init__(self, width=params.n_columns, height=params.n_rows)


class _SpinningLidarModel(_LidarModel):
    def __init__(self, params: SpinningLidarModelParameters) -> None:
        super().__init__(params)

    def relative_angle(
        self,
        angle_ref,
        angle,
        spinning_direction: SpinningDirection,
        *,
        scale: float = 1,
    ) -> float:
        rel_angle = self.relative_clock_rotation(angle_ref, angle, spinning_direction)
        period = scale * 2 * math.pi
        return rel_angle % period

    def angle_range_wrap_around(
        self, start_angle, end_angle, *, scale: float = 1
    ) -> bool:
        return torch.abs(end_angle - start_angle) >= scale * 2 * math.pi

    def relative_sensor_angles(self, angles: Tensor, *, scale: float = 1) -> Tensor:
        angles_az = angles[..., 0]
        angles_el = angles[..., 1]
        rel_azimuth = self.relative_angle(
            self.fov_horiz_rad.start * scale,
            angles_az,
            self.spinning_direction,
            scale=scale,
        )
        rel_elevation = self.relative_clock_rotation(
            self.fov_vert_rad.start * scale,
            angles_el,
            SpinningDirection.CLOCKWISE,
        )

        return torch.stack([rel_azimuth, rel_elevation], dim=-1)

    def valid_sensor_angles(self, angles: Tensor, *, scale: float = 1) -> Tensor:
        angles_az = angles[..., 0]
        angles_el = angles[..., 1]
        # Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
        fov_vert_start_adj = self.fov_vert_rad.start + self.fov_eps_rad
        if self.spinning_direction == SpinningDirection.CLOCKWISE:
            fov_horiz_start_adj = self.fov_horiz_rad.start + self.fov_eps_rad
        else:
            fov_horiz_start_adj = self.fov_horiz_rad.start - self.fov_eps_rad

        rel_elevation = self.relative_clock_rotation(
            fov_vert_start_adj * scale, angles_el, SpinningDirection.CLOCKWISE
        )
        rel_azimuth = self.relative_angle(
            fov_horiz_start_adj * scale,
            angles_az,
            self.spinning_direction,
            scale=scale,
        )

        # Also account for accumulated numerical errors via some epsilon in FOV check
        # (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
        #  so effectively this checks 1x eps on the end of the FOV)
        return (
            rel_elevation <= scale * (self.fov_vert_rad.span + self.fov_eps_rad * 2)
        ) & (rel_azimuth <= scale * (self.fov_horiz_rad.span + self.fov_eps_rad * 2))


class _StructuredSpinningLidarModel(_StructuredLidarModel, _SpinningLidarModel):
    def __init__(self, params: StructuredLidarModelParameters) -> None:
        super().__init__(params)


class _RowOffsetStructuredSpinningLidarModel(_StructuredSpinningLidarModel):
    ANGLE_TO_PIXEL_SCALING_FACTOR = 1024.0

    def __init__(self, params: RowOffsetStructuredSpinningLidarModelParametersExt):
        params = copy.copy(params)

        # Force using our reference fov
        # NOTE: there must be no other params attributes that depend on fov,
        # or else they need to be updated as well
        object.__setattr__(params, "fov_horiz_rad", self._compute_fov_horiz_rad(params))
        object.__setattr__(params, "fov_vert_rad", self._compute_fov_vert_rad(params))

        super().__init__(params)

    @property
    def focal_lengths(self) -> Tensor:
        # Just a dummy focal length.
        # TODO: fix the class hierarchy to properly accomodate lidar sensors
        return torch.ones(2, dtype=torch.float32, device=self.row_elevations_rad.device)

    @property
    def principal_points(self) -> Tensor:
        # Just return a dummy tensor, as Lidar doesn't have principal point.
        # TODO: fix the class hierarchy to properly accomodate lidar sensors
        return torch.tensor(
            [self.width / 2, self.height / 2], dtype=torch.float32, device=self.device
        )

    def relative_clock_rotation(
        self, angle_ref, angle, spinning_direction: SpinningDirection
    ):
        if spinning_direction == SpinningDirection.CLOCKWISE:
            # Clockwise: going from ref to angle in CW direction
            return angle_ref - angle
        else:
            # Counter-clockwise: going from ref to angle in CCW direction
            return angle - angle_ref

    @staticmethod
    def _compute_fov_vert_rad(
        params: RowOffsetStructuredSpinningLidarModelParameters,
    ) -> FOV:
        start = params.row_elevations_rad[0].item()
        span = start - params.row_elevations_rad[-1].item()
        return FOV(start=start, span=span, direction=SpinningDirection.CLOCKWISE)

    @staticmethod
    def _compute_fov_horiz_rad(
        params: RowOffsetStructuredSpinningLidarModelParameters,
    ) -> FOV:
        # Reconstruct first and last (wrapped) element azimuths per elevation once to obtain FoV bounds
        azimuth_extremes = (
            params.column_azimuths_rad[None, [0, params.n_columns - 1]]
            + params.row_azimuth_offsets_rad[:, None]
        )

        # Determine extremum in first element
        if params.spinning_direction == SpinningDirection.COUNTER_CLOCKWISE:
            # azimuths are in increasing order
            start = azimuth_extremes[:, 0].min().item()
            span = azimuth_extremes[:, -1].max().item() - start
        else:
            # azimuths are in decreasing order
            start = azimuth_extremes[:, 0].max().item()
            span = start - azimuth_extremes[:, -1].min().item()
        return FOV(
            start=start,
            span=min(span, 2 * math.pi),
            direction=params.spinning_direction,
        )

    def camera_ray_to_image_point(
        self, camera_ray: Tensor, margin_factor: float = 0.0
    ) -> tuple[Tensor, Tensor]:
        """Forward projection: 3D camera ray → 2D image point.

        Args:
            camera_ray: Camera rays [..., 3] (x, y, z direction vectors)
            margin_factor: Margin factor for FOV bounds checking

        Returns:
            image_points: Image points [..., 2] with [column, row] = [azimuth, elevation]
            valid_flags: Boolean flags [...] indicating if projection is valid
        """
        # Normalize rays
        ray_normalized = _safe_normalize(camera_ray)

        # Convert to spherical coordinates (same as CUDA implementation)
        ray_x = ray_normalized[..., 0]
        ray_y = ray_normalized[..., 1]
        ray_z = ray_normalized[..., 2]
        azimuth = torch.atan2(ray_y, ray_x)
        elevation = torch.asin(ray_z)

        # Scale angles to pixel space
        column = azimuth * self.ANGLE_TO_PIXEL_SCALING_FACTOR
        row = elevation * self.ANGLE_TO_PIXEL_SCALING_FACTOR

        # image_point = [x, y] = [column, row] = [azimuth, elevation]
        image_point = torch.stack([column, row], dim=-1)

        # Validation: compute relative angles for FOV checking

        # Compute relative angles from FOV start
        angles = torch.stack([azimuth, elevation], dim=-1)
        rel_angles = self.relative_sensor_angles(angles)
        rel_az = rel_angles[..., 0]
        rel_el = rel_angles[..., 1]

        # Apply margin for FOV checking
        margin_elevation = margin_factor * self.fov_vert_rad.span
        margin_azimuth = margin_factor * self.fov_horiz_rad.span

        # Check if within FOV+margin
        valid = (
            (rel_el <= self.fov_vert_rad.span + margin_elevation)
            & (rel_az <= self.fov_horiz_rad.span + margin_azimuth)
            & (rel_el >= -margin_elevation)
            & (rel_az >= -margin_azimuth)
        )

        return image_point, valid

    def element_to_image_point(self, row: Tensor, col: Tensor) -> Tensor:
        """Convert element indices to image points in scaled angle space.

        Mirrors the CUDA ``element_to_image_point`` in Lidars.cuh: looks up
        raw sensor angles and scales them, without modulo-based normalization
        on elevation.

        Args:
            row: Elevation (row) indices [...] in [0, n_rows).
            col: Azimuth (column) indices [...] in [0, n_columns).

        Returns:
            image_points: [..., 2] as (azimuth * SCALE, elevation * SCALE).
        """
        params = self.params
        el = params.row_elevations_rad[row]
        az = params.column_azimuths_rad[col] + params.row_azimuth_offsets_rad[row]
        # Normalize azimuth to (-pi, pi] via conditional subtract/add
        az = torch.where(az > torch.pi, az - 2 * torch.pi, az)
        az = torch.where(az <= -torch.pi, az + 2 * torch.pi, az)
        azimuth = az * self.ANGLE_TO_PIXEL_SCALING_FACTOR
        elevation = el * self.ANGLE_TO_PIXEL_SCALING_FACTOR
        return torch.stack([azimuth, elevation], dim=-1)

    def image_point_to_camera_ray(self, image_point: Tensor) -> tuple[Tensor, Tensor]:
        """Inverse projection: 2D image point → 3D camera ray.

        Args:
            image_point: Image points [..., 2] with [column, row] = [azimuth, elevation] in scaled angle space

        Returns:
            camera_rays: Camera rays [..., 3] (x, y, z direction vectors)
            valid_flags: Boolean flags [...] indicating if point is valid
        """
        column = image_point[..., 0]
        row = image_point[..., 1]

        # Convert from scaled angle space to angles
        # image_point = [x, y] = [column, row] = [azimuth, elevation]
        kToAngle = 1.0 / self.ANGLE_TO_PIXEL_SCALING_FACTOR
        angles_az = column * kToAngle
        angles_el = row * kToAngle

        # Convert to Cartesian coordinates
        cos_elevation = torch.cos(angles_el)
        camera_ray = torch.stack(
            [
                torch.cos(angles_az) * cos_elevation,
                torch.sin(angles_az) * cos_elevation,
                torch.sin(angles_el),
            ],
            dim=-1,
        )

        camera_ray = _safe_normalize(camera_ray)

        angles = torch.stack([angles_az, angles_el], dim=-1)
        return camera_ray, self.valid_sensor_angles(angles)

    def shutter_relative_frame_time(self, image_point: Tensor) -> Tensor:
        """Compute relative frame time for rolling shutter.

        Args:
            image_point: Image points [..., 2] with [column, row] = [azimuth, elevation] in scaled angle space

        Returns:
            relative_times: Relative frame times [...] in range [0, 1]
        """
        column = image_point[..., 0]
        row = image_point[..., 1]

        # Convert from scaled angle space back to angles
        kToAngle = 1.0 / self.ANGLE_TO_PIXEL_SCALING_FACTOR
        angles = torch.stack([column * kToAngle, row * kToAngle], dim=-1)

        assert torch.all(self.valid_sensor_angles(angles))

        # 0. Compute the map resolution in radians
        map_resolution_horiz_rad: float = self.fov_horiz_rad.span / (
            self.angles_to_columns_map.shape[1] - 1
        )
        map_resolution_vert_rad: float = self.fov_vert_rad.span / (
            self.angles_to_columns_map.shape[0] - 1
        )

        # 1.compute relative angles wrt. FOV start
        rel_angles = self.relative_sensor_angles(angles)
        rel_az = rel_angles[..., 0]
        rel_el = rel_angles[..., 1]

        # 2. Compute the indices into the lookup map
        ivert = torch.clamp(
            rel_el / map_resolution_vert_rad + 0.5,
            0,
            self.angles_to_columns_map.shape[0] - 1,
        ).to(torch.int)
        ihoriz = torch.clamp(
            rel_az / map_resolution_horiz_rad + 0.5,
            0,
            self.angles_to_columns_map.shape[1] - 1,
        ).to(torch.int)

        # 3. Lookup column index from the angle-to-column map
        col_idx = self.angles_to_columns_map[ivert, ihoriz]

        # Compute relative frame time
        # We're assuming that the lidar captures the whole row instantly,
        # so the relative frame time is basically the column index.
        return col_idx.to(image_point.dtype) / (self.n_columns - 1)
