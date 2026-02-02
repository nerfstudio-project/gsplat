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
from dataclasses import dataclass, field, InitVar
from typing import Optional
from types import SimpleNamespace
from typing_extensions import override
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
    angle_ref: Tensor, angle: Tensor, direction: SpinningDirection, scale: float = 1
) -> float:
    rel_angle = relative_clock_rotation(angle_ref, angle, direction)
    # Normalize between [0,2*pi*scale)
    return normalize_angle(rel_angle, start=0, scale=scale)


def angle_range_wrap_around(start_angle, end_angle, scale: float = 1) -> bool:
    return torch.abs(end_angle - start_angle) >= 2 * math.pi * scale


def normalize_angle(angle: Tensor, *, start: float, scale: float = 1) -> Tensor:
    period = 2 * math.pi * scale
    return (angle - start) % period + start


def normalize_azimuth(angle: Tensor, scale: float = 1) -> Tensor:
    return normalize_angle(angle, start=0, scale=scale)


def normalize_elevation(angle: Tensor, scale: float = 1) -> Tensor:
    return torch.clamp(
        normalize_angle(angle, start=-scale * math.pi, scale=scale),
        -scale * math.pi / 2,
        scale * math.pi / 2,
    )


@dataclass(kw_only=True)
class SphericalUnitCoord:
    elevation: Tensor
    azimuth: Tensor

    def __post_init__(self):
        assert self.elevation.dtype == self.azimuth.dtype, (
            self.elevation.dtype,
            self.azimuth.dtype,
        )
        assert self.elevation.device == self.azimuth.device, (
            self.elevation.device,
            self.azimuth.device,
        )
        assert self.elevation.shape == self.azimuth.shape, (
            self.elevation.shape,
            self.azimuth.shape,
        )

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

    def reshape(self, *args) -> "SphericalUnitCoord":
        return SphericalUnitCoord(
            elevation=self.elevation.reshape(*args), azimuth=self.azimuth.reshape(*args)
        )

    def __getitem__(self, key) -> "SphericalUnitCoord":
        return SphericalUnitCoord(
            elevation=self.elevation[key], azimuth=self.azimuth[key]
        )


@dataclass(kw_only=True, frozen=True)
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
class LidarModelParameters:
    """Represents parameters common to all lidar models"""

    fov_vert_rad: FOV
    fov_horiz_rad: FOV

    fov_eps_rad: float


@dataclass(frozen=True, kw_only=True)
class SpinningLidarModelParameters(LidarModelParameters):
    """Represents parameters common to all spinning lidar models"""

    spinning_frequency_hz: float
    spinning_direction: SpinningDirection


@dataclass(frozen=True, kw_only=True)
class StructuredLidarModelParameters(LidarModelParameters):
    """Represents parameters for a structured spinning lidar model.

    A structured lidar model consists of a fixed number of rows x columns point measurements per frame
    """

    n_rows: int
    n_columns: int


@dataclass(frozen=True, kw_only=True)
class StructuredSpinningLidarModelParameters(
    StructuredLidarModelParameters, SpinningLidarModelParameters
):
    pass


@dataclass(frozen=True, kw_only=True)
class RowOffsetStructuredSpinningLidarModelParameters(
    StructuredSpinningLidarModelParameters
):
    """Represents parameters for a structured spinning lidar model that is using a per-row azimuth-offset (compatible with, e.g., Hesai P128 sensors)"""

    # elevation angle of each row,
    # constant for each column [clockwise around y axis, relative to x axis] [(n_rows,) radians]
    row_elevations_rad: Tensor

    # azimuth angle of each column,
    # starting at first element of the spin [clockwise / counter-clockwise around z axis
    # depending on sensors spin direction, relative to x axis] [(n_columns,) radians]
    column_azimuths_rad: Tensor

    # azimuth angle offsets for each row [around z axis, relative to x axis] [(n_rows,) radians]
    row_azimuth_offsets_rad: Tensor

    fov_eps_factor: InitVar[int]

    def __init__(
        self,
        *,
        row_elevations_rad: Tensor,
        column_azimuths_rad: Tensor,
        row_azimuth_offsets_rad: Tensor,
        spinning_frequency_hz: float,
        spinning_direction: SpinningDirection,
        fov_eps_factor: int = 4,
    ) -> None:

        object.__setattr__(self, "row_elevations_rad", row_elevations_rad)
        object.__setattr__(self, "column_azimuths_rad", column_azimuths_rad)
        object.__setattr__(self, "row_azimuth_offsets_rad", row_azimuth_offsets_rad)

        params = SimpleNamespace()
        params.fov_vert_rad = self._compute_fov_vert_rad()
        params.fov_horiz_rad = self._compute_fov_horiz_rad(
            spinning_direction=spinning_direction
        )
        params.fov_eps_rad = fov_eps_factor * torch.finfo(self.dtype).eps
        params.n_rows = row_elevations_rad.shape[0]
        params.n_columns = column_azimuths_rad.shape[0]
        params.spinning_frequency_hz = spinning_frequency_hz
        params.spinning_direction = spinning_direction

        super().__init__(**vars(params))

    def __post_init__(self, fov_eps_factor: int):
        super().__post_init__()

        assert fov_eps_factor > 0

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

    @property
    def device(self):
        return self.row_elevations_rad.device

    @property
    def dtype(self):
        return self.row_elevations_rad.dtype

    def __hash__(self) -> int:
        return hash(
            (
                super().__hash__(),
                int(torch.hash_tensor(self.row_elevations_rad).item()),
                int(torch.hash_tensor(self.column_azimuths_rad).item()),
                int(torch.hash_tensor(self.row_azimuth_offsets_rad).item()),
            )
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, RowOffsetStructuredSpinningLidarModelParameters):
            return NotImplemented
        else:
            return (
                super().__eq__(other)
                and torch.equal(self.row_elevations_rad, other.row_elevations_rad)
                and torch.equal(self.column_azimuths_rad, other.column_azimuths_rad)
                and torch.equal(
                    self.row_azimuth_offsets_rad, other.row_azimuth_offsets_rad
                )
            )

    def create_elements(self) -> SphericalUnitCoord:
        """Create all element indices [relative to the static model]
        Element: each elevation/azimuth pair in the quantized grid, i.e. each cell."""

        idxelevations = torch.arange(self.n_rows, device=self.device, dtype=torch.int32)
        idxazimuths = torch.arange(
            self.n_columns, device=self.device, dtype=torch.int32
        )

        return SphericalUnitCoord(
            elevation=torch.repeat_interleave(idxelevations, self.n_columns),
            azimuth=idxazimuths.repeat(self.n_rows),
        )

    def _compute_fov_vert_rad(self) -> FOV:
        """Returns the vertical field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        start_rad = self.row_elevations_rad[0].item()
        span_rad = relative_angle(
            start_rad, self.row_elevations_rad[-1], SpinningDirection.CLOCKWISE
        ).item()
        assert span_rad >= 0

        return FOV(
            start=float(start_rad),
            span=float(span_rad),
            direction=SpinningDirection.CLOCKWISE,
        )

    def _compute_fov_horiz_rad(self, spinning_direction: SpinningDirection) -> FOV:
        """Returns the horizontal field-of-view of the lidar model (starting at first element) in the requested dtype precision"""

        # Reconstruct first and last (wrapped) element azimuths per elevation once to obtain FoV bounds
        azimuth_extremes_rad = (
            self.column_azimuths_rad[None, [0, self.column_azimuths_rad.shape[0] - 1]]
            + self.row_azimuth_offsets_rad[:, None]
        )

        # Determine extremum in first element
        if spinning_direction == SpinningDirection.COUNTER_CLOCKWISE:
            start_rad = azimuth_extremes_rad[:, 0].min().item()
        else:
            start_rad = azimuth_extremes_rad[:, 0].max().item()

        end_rad = azimuth_extremes_rad[:, -1]

        # Check if the azimuth angles of last element wrap around over the start element
        if torch.any(angle_range_wrap_around(start_rad, end_rad)):
            span_rad = 2 * torch.pi
        else:
            span_rad = (
                relative_angle(start_rad, end_rad, spinning_direction).max().item()
            )

        return FOV(
            start=float(start_rad), span=float(span_rad), direction=spinning_direction
        )

    def elements_to_sensor_angles(
        self, elements: SphericalUnitCoord
    ) -> SphericalUnitCoord:
        return SphericalUnitCoord(
            elevation=normalize_elevation(self.row_elevations_rad[elements.elevation]),
            azimuth=normalize_azimuth(
                self.column_azimuths_rad[elements.azimuth]
                + self.row_azimuth_offsets_rad[elements.elevation]
            ),
        )


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParameters
):
    """Lidar camera parameters extended with acceleration structures"""

    angles_to_columns_map: Tensor

    def __init__(self, angles_to_columns_map: Tensor, **kwargs) -> None:
        self.angles_to_columns_map = angles_to_columns_map
        super().__init__(**kwargs)

        assert (
            self.angles_to_columns_map.shape[0] % self.row_elevations_rad.shape[0] == 0
        )
        assert (
            self.angles_to_columns_map.shape[1] % self.column_azimuths_rad.shape[0] == 0
        )
        a2cmap_resfactor = (
            self.angles_to_columns_map.shape[0] / self.row_elevations_rad.shape[0]
        )
        assert (
            a2cmap_resfactor
            == self.angles_to_columns_map.shape[1] / self.column_azimuths_rad.shape[0]
        )


# --------------------- Computation of angles_to_columns_map ---------------------------


def relative_sensor_angles(
    lidar: SpinningLidarModelParameters, coord: SphericalUnitCoord, scale: float = 1
) -> SphericalUnitCoord:
    return SphericalUnitCoord(
        elevation=relative_clock_rotation(
            lidar.fov_vert_rad.start * scale,
            coord.elevation,
            SpinningDirection.CLOCKWISE,
        ),
        azimuth=relative_angle(
            lidar.fov_horiz_rad.start * scale,
            coord.azimuth,
            lidar.spinning_direction,
            scale,
        ),
    )


def valid_sensor_angles(
    lidar: SpinningLidarModelParameters, coord: SphericalUnitCoord, scale: float = 1
) -> Tensor:
    # Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
    fov_vert_start_adj = lidar.fov_vert_rad.start + lidar.fov_eps_rad
    if lidar.spinning_direction == SpinningDirection.CLOCKWISE:
        # angles increase from right->left  bottom->top
        fov_horiz_start_adj = lidar.fov_horiz_rad.start + lidar.fov_eps_rad
    else:
        # angles increase from left->right or top->bottom.
        fov_horiz_start_adj = lidar.fov_horiz_rad.start - lidar.fov_eps_rad

    rel_elevation = relative_clock_rotation(
        fov_vert_start_adj * scale, coord.elevation, SpinningDirection.CLOCKWISE
    )
    rel_azimuth = relative_angle(
        fov_horiz_start_adj * scale, coord.azimuth, lidar.spinning_direction, scale
    )

    # Also account for accumulated numerical errors via some epsilon in FOV check
    # (using 2x eps as 1x eps is "inherited" from the start of the FOV in the relative angles,
    #  so effectively this checks 1x eps on the end of the FOV)
    return (
        rel_elevation <= (lidar.fov_vert_rad.span + lidar.fov_eps_rad * 2) * scale
    ) & (rel_azimuth <= (lidar.fov_horiz_rad.span + lidar.fov_eps_rad * 2) * scale)


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
    lidar: SpinningLidarModelParameters, sensor_angles: SphericalUnitCoord
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
    lidar: RowOffsetStructuredSpinningLidarModelParameters,
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

    # Compute a column-major list of all sensor coordinates
    elements = lidar.create_elements()
    sensor_angles = lidar.elements_to_sensor_angles(elements).reshape(-1)

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
        sensor_rays.sensor_rays.contiguous().cpu().numpy()
    )  # ty:ignore[unresolved-attribute]
    _, idxs = kdtree.query(grid_rays.sensor_rays.contiguous().cpu().numpy())
    idxs = torch.from_numpy(idxs).to(device=lidar.device, dtype=torch.int32)

    # Map the indices to the columns by dividing with the total number of rows and (implicit) flooring
    return (idxs / lidar.n_rows).to(dtype).reshape(grid_angles.shape)
