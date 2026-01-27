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
from torch import Tensor
import torch
import math
from scipy import spatial as scipy_spatial
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


class SpinningDirection(Enum):
    CLOCKWISE = 0
    COUNTER_CLOCKWISE = 1


def relative_clock_rotation(
    angle_ref: Tensor, angle: Tensor, direction: SpinningDirection
) -> Tensor:
    if direction == SpinningDirection.CLOCKWISE:
        # Clockwise: going from ref to angle in CW direction
        rel_angle = angle_ref - angle
    else:
        # Counter-clockwise: going from ref to angle in CCW direction
        rel_angle = angle - angle_ref
    return rel_angle


def relative_angle(
    angle_ref: Tensor,
    angle: Tensor,
    direction: SpinningDirection,
    period: float = 2 * math.pi,
) -> float:
    rel_angle = relative_clock_rotation(angle_ref, angle, direction)

    # Normalize it between [0, period)
    return rel_angle % period


def angle_range_wrap_around(
    start_angle, end_angle, period: float = 2 * math.pi
) -> bool:
    return torch.abs(end_angle - start_angle) >= period


@dataclass(kw_only=True)
class SphericalUnitCoord:
    elevation: Tensor
    azimuth: Tensor

    def __post_init__(self):
        assert self.elevation.dtype == self.azimuth.dtype
        assert self.elevation.device == self.azimuth.device
        assert self.elevation.shape == self.azimuth.shape

    def to(self, *, dtype: torch.dtype, device: torch.device) -> "SphericalUnitCoord":
        return SphericalUnitCoord(
            elevation=self.elevation.to(dtype=dtype, device=device),
            azimuth=self.azimuth.to(dtype=dtype, device=device),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.elevation.dtype

    @property
    def device(self) -> torch.device:
        return self.elevation.device

    @property
    def shape(self) -> tuple:
        return self.elevation.shape


@dataclass
class FOV:
    """Represents a field-of-view with start and span in radians"""

    def __post_init__(self):
        assert self.span >= 0

    # Start angle of the field-of-view in radians
    start: float
    # Span of the valid field-of-view region in radians in [0, 2π]
    span: float
    # Direction of the valid field-of-view region, either clockwise or counter-clockwise
    direction: SpinningDirection

    @property
    def end(self) -> float:
        if self.direction == SpinningDirection.COUNTER_CLOCKWISE:
            return self.start + self.span
        else:
            return self.start - self.span


@dataclass(frozen=True, kw_only=True)
class LidarCameraParameters:
    """Raw lidar camera parameters"""

    row_elevations_rad: Tensor
    column_azimuths_rad: Tensor
    row_azimuth_offsets_rad: Tensor

    spinning_frequency_hz: float
    spinning_direction: SpinningDirection

    fov_eps_factor: int = 4

    @property
    def device(self):
        return self.row_elevations_rad.device

    @property
    def dtype(self):
        return self.row_elevations_rad.dtype

    @property
    def n_columns(self):
        return self.column_azimuths_rad.shape[0]

    @property
    def n_rows(self):
        return self.row_elevations_rad.shape[0]

    @property
    def fov_eps_rad(self) -> float:
        return self.fov_eps_factor * torch.finfo(self.row_elevations_rad.dtype).eps

    def __hash__(self) -> int:
        return hash(
            (
                int(torch.hash_tensor(self.row_elevations_rad).item()),
                int(torch.hash_tensor(self.column_azimuths_rad).item()),
                int(torch.hash_tensor(self.row_azimuth_offsets_rad).item()),
                self.spinning_frequency_hz,
                self.spinning_direction,
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, LidarCameraParameters):
            return NonImplemented
        else:
            return (
                self.spinning_direction == other.spinning_direction
                and self.spinning_frequency_hz == other.spinning_frequency_hz
                and torch.equal(self.row_elevations_rad, other.row_elevations_rad)
                and torch.equal(self.column_azimuths_rad, other.column_azimuths_rad)
                and torch.equal(
                    self.row_azimuth_offsets_rad, other.row_azimuth_offsets_rad
                )
            )

    def __post_init__(self):
        assert self.spinning_frequency_hz > 0.0

        assert self.row_elevations_rad.dtype == torch.float32
        assert self.row_elevations_rad.shape == (self.n_rows,)
        assert self.row_azimuth_offsets_rad.dtype == torch.float32
        assert self.row_azimuth_offsets_rad.shape == (self.n_rows,)
        assert self.column_azimuths_rad.dtype == torch.float32
        assert self.column_azimuths_rad.shape == (self.n_columns,)

        # Check elevation angles are sorted consistently
        relative_row_elevations_rad = relative_angle(
            self.row_elevations_rad[0],
            self.row_elevations_rad,
            SpinningDirection.CLOCKWISE,
        )
        assert torch.all(
            torch.diff(relative_row_elevations_rad) > 0
        ), "Row elevation angles must be sorted in descending order (cw)"
        assert torch.all(
            ~angle_range_wrap_around(
                self.row_elevations_rad[0], self.row_elevations_rad
            )
        ), "Row elevation angles must not wrap around the start element"

        # Check order of column azimuth angles is consistent with spinning direction
        relative_column_azimuths_rad = relative_angle(
            self.column_azimuths_rad[0],
            self.column_azimuths_rad,
            self.spinning_direction,
        )

        assert torch.all(
            torch.diff(relative_column_azimuths_rad) > 0
        ), "Column azimuth angles must be sorted in the spinning direction so the diff between relative angles of consecutive columns should always be positive"

        assert torch.all(
            ~angle_range_wrap_around(
                self.column_azimuths_rad[0], self.column_azimuths_rad
            )
        ), "Column azimuth angles (without offsets) must not wrap around the start element"

    _cache_fov_vert: FOV | None = field(init=False, repr=False, default=None)

    @property
    def fov_vert_rad(self) -> FOV:
        """Returns the vertical field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        if self._cache_fov_vert is None:
            start_rad = self.row_elevations_rad[0].item()
            span_rad = relative_angle(
                start_rad, self.row_elevations_rad[-1], SpinningDirection.CLOCKWISE
            ).item()
            assert span_rad >= 0

            # Bypass frozen setattr so that we can store this cached value
            object.__setattr__(
                self,
                "_cache_fov_vert",
                FOV(
                    start=start_rad,
                    span=span_rad,
                    direction=SpinningDirection.CLOCKWISE,
                ),
            )

        return self._cache_fov_vert

    _cache_fov_horiz: FOV | None = field(init=False, repr=False, default=None)

    @property
    def fov_horiz_rad(self) -> FOV:
        """Returns the horizontal field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        if self._cache_fov_horiz is None:
            # Reconstruct first and last (wrapped) element azimuths per elevation once to obtain FoV bounds
            azimuth_extremes_rad = (
                self.column_azimuths_rad[None, [0, self.n_columns - 1]]
                + self.row_azimuth_offsets_rad[:, None]
            )

            # Determine extremum in first element
            if self.spinning_direction == SpinningDirection.COUNTER_CLOCKWISE:
                start_rad = azimuth_extremes_rad[:, 0].min().item()
            else:
                start_rad = azimuth_extremes_rad[:, 0].max().item()

            end_rad = azimuth_extremes_rad[:, -1]

            # Check if the azimuth angles of last element wrap around over the start element
            if torch.any(angle_range_wrap_around(start_rad, end_rad)):
                span_rad = 2 * torch.pi
            else:
                span_rad = (
                    relative_angle(start_rad, end_rad, self.spinning_direction)
                    .max()
                    .item()
                )

            object.__setattr__(
                self,
                "_cache_fov_horiz",
                FOV(start=start_rad, span=span_rad, direction=self.spinning_direction),
            )

        return self._cache_fov_horiz


def relative_sensor_angles(
    lidar: LidarCameraParameters, coord: SphericalUnitCoord
) -> SphericalUnitCoord:
    # Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
    fov_vert_start_adj = lidar.fov_vert_rad.start + lidar.fov_eps_rad
    if lidar.spinning_direction == SpinningDirection.CLOCKWISE:
        # angles increase from right->left  bottom->top
        fov_horiz_start_adj = lidar.fov_horiz_rad.start + lidar.fov_eps_rad
    else:
        # angles increase from left->right or top->bottom.
        fov_horiz_start_adj = lidar.fov_horiz_rad.start - lidar.fov_eps_rad

    return SphericalUnitCoord(
        elevation=relative_clock_rotation(
            fov_vert_start_adj, coord.elevation, SpinningDirection.CLOCKWISE
        ),
        azimuth=relative_angle(
            fov_horiz_start_adj, coord.azimuth, lidar.spinning_direction
        ),
    )


def valid_relative_sensor_angles(
    lidar: LidarCameraParameters, relcoord: SphericalUnitCoord
) -> Tensor:
    """Checks if relative sensor angles are within the FOV of the sensor.
    Relative angle inputs should be computed with relative_sensor_angles to include eps corrections"""

    # Also account for accumulated numerical errors via some epsilon in FOV check
    # (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
    #  so effectively this checks 1x eps on the end of the FOV)
    return (relcoord.elevation <= lidar.fov_vert_rad.span + lidar.fov_eps_rad * 2) & (
        relcoord.azimuth <= lidar.fov_horiz_rad.span + lidar.fov_eps_rad * 2
    )


def valid_sensor_angles(
    lidar: LidarCameraParameters, coord: SphericalUnitCoord
) -> Tensor:
    relcoord = relative_sensor_angles(lidar, coord)
    return valid_relative_sensor_angles(lidar, relcoord)


@dataclass
class SensorRayReturn:
    """
    Contains
        - sensor rays [float] (n,3)
        - valid_flag [bool] (n,)
    """

    sensor_rays: torch.Tensor
    valid_flag: torch.Tensor


def sensor_angles_to_rays(
    lidar: LidarCameraParameters, sensor_angles: SphericalUnitCoord
) -> SensorRayReturn:
    """Computes the sensor rays for elevation/azimuth angles."""

    sensor_angles = sensor_angles.to(device=lidar.device, dtype=lidar.dtype)

    # Check if relative sensor angles are within the FOV of the sensor
    valid = valid_sensor_angles(lidar, sensor_angles)

    # Compute the 3D unit rays
    x = torch.cos(sensor_angles.azimuth) * (
        cos_elevations := torch.cos(sensor_angles.elevation)
    )
    y = torch.sin(sensor_angles.azimuth) * cos_elevations
    z = torch.sin(sensor_angles.elevation)
    sensor_rays = torch.stack([x, y, z], dim=-1)

    return SensorRayReturn(sensor_rays=sensor_rays, valid_flag=valid)


# This can be slow to generate due to the kdtree generation and query.
# We cache the results using functools facilities, so that subsequent calls
# with the same parameters returns the cached result.
@lru_cache(maxsize=10)
def compute_angles_to_columns_map(
    lidar: LidarCameraParameters,
    resolution_factor: float = 4,
    dtype: torch.dtype = torch.int32,
):
    """Computes angles to column map as a 2D array of shape resolution_factor * (n_rows, n_columns)."""

    # The idea here is to create a regular high-resolution grid spanning the whole FOV
    # and store in each grid cell the projected azimuth of the closest lidar ray to it.
    # The lidar (unitary) ray chosen is the one closest (L^2)
    # to the top-left corner of the grid cell, when 'un-projected' to an unitary ray.

    assert (
        torch.iinfo(dtype).max >= lidar.n_columns - 1
    ), "The dtype for the angles to columns map must be able to store the maximum column index, consider increasing angles_to_columns_map_dtype"

    assert (
        not dtype.is_floating_point and not dtype.is_complex
    ), "The dtype for the angles to columns map must be an integer type"

    # Create all element indices [relative to the static model]
    # Element: each elevation/azimuth pair in the quantized grid, i.e. each cell.
    elements = torch.stack(
        torch.meshgrid(
            torch.arange(lidar.n_rows, dtype=torch.long),
            torch.arange(lidar.n_columns, dtype=torch.long),
            indexing="ij",
        ),
        dim=-1,
    )  # row-major (rows,columns)

    # Create regular, high-density angle grid spanning the sensor's FOV.
    grid_elevations_rad, grid_azimuths_rad = torch.meshgrid(
        # Elevations
        torch.linspace(
            lidar.fov_vert_rad.start,
            lidar.fov_vert_rad.end,
            resolution_factor * lidar.n_rows,
            device=lidar.row_elevations_rad.device,
            dtype=lidar.row_elevations_rad.dtype,
        ),
        # Azimuths
        torch.linspace(
            lidar.fov_horiz_rad.start,
            lidar.fov_horiz_rad.end,
            resolution_factor * lidar.n_columns,
            device=lidar.column_azimuths_rad.device,
            dtype=lidar.column_azimuths_rad.dtype,
        ),
        indexing="ij",
    )  # row-major

    grid_angles = SphericalUnitCoord(
        elevation=grid_elevations_rad, azimuth=grid_azimuths_rad
    )

    # Convert grid angles to unit norm rays
    # TODO: if the fov_horiz_rad.span == 2*pi, the first and last columns (0 and 2*pi)
    # will correspond to the same rays. This leads to the same sensor angle being assigned
    # to a cell in either the first or the last column, depending on small floating-point errors.
    # This needs to be fixed.
    grid_rays = sensor_angles_to_rays(lidar, grid_angles)

    assert torch.all(
        grid_rays.valid_flag
    ), "Bug: grid rays must be valid in the FOV of the sensor"

    def normalize_angle_mpi_pi(angle: Tensor) -> Tensor:
        """In-place normalization of the angle, to be between [-pi,pi)"""
        angle[angle > torch.pi] -= 2 * torch.pi
        angle[angle <= torch.pi] += 2 * torch.pi
        return angle

    # Reconstruct angles from model parameterization
    element_azimuths_rad = normalize_angle_mpi_pi(
        lidar.column_azimuths_rad[elements[:, :, 1]]
        + lidar.row_azimuth_offsets_rad[elements[:, :, 0]]
    )

    # Compute a column-major list of all sensor coordinates
    sensor_angles = SphericalUnitCoord(
        elevation=torch.tile(lidar.row_elevations_rad, [lidar.n_columns, 1]).reshape(
            -1
        ),
        azimuth=element_azimuths_rad.transpose(0, 1).reshape(-1),
    )

    assert sensor_angles.shape == (lidar.n_rows * lidar.n_columns,), sensor_angles.shape

    # Now convert the sensor angles to unit rays.
    sensor_rays = sensor_angles_to_rays(lidar, sensor_angles)

    assert torch.all(
        sensor_rays.valid_flag
    ), "Bug: sensor rays must be valid in the FOV of the sensor"

    # Compute the NN of each grid ray in the sensor rays
    # TODO: this is quite slow and should be moved to CUDA.
    #       We can get away with it for now because we're caching the results,
    #       and usually this function would be called only once per lidar model needed.
    kdtree = scipy_spatial.cKDTree(
        sensor_rays.sensor_rays.cpu().numpy()
    )  # ty:ignore[unresolved-attribute]
    _, idxs = kdtree.query(grid_rays.sensor_rays.cpu().numpy())
    idxs = torch.from_numpy(idxs).to(device=lidar.device, dtype=torch.int32)

    # Map the indices to the columns by dividing with the total number of rows and (implicit) flooring
    return (idxs / lidar.n_rows).to(dtype).reshape(grid_angles.shape)
