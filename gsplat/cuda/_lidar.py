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
import hashlib
import math
from dataclasses import dataclass, field, InitVar
from enum import Enum
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
from typing_extensions import override


def _tensor_hash(tensor: Tensor) -> int:
    """Stable content hash for a tensor (shape, dtype, and values), compatible with PyTorch versions without torch.hash_tensor."""
    hash_fn = getattr(torch, "hash_tensor", None)
    if hash_fn is not None:
        return int(hash_fn(tensor).item())
    # Contiguous so the same logical content hashes identically regardless of strides/view
    t = tensor.detach().cpu().contiguous()
    header = (str(tuple(t.shape)) + str(t.dtype)).encode()
    data = header + t.numpy().tobytes()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


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
                _tensor_hash(self.row_elevations_rad),
                _tensor_hash(self.column_azimuths_rad),
                _tensor_hash(self.row_azimuth_offsets_rad),
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

    def create_elements(self) -> Tensor:
        """Create all element indices [relative to the static model].
        Element: each elevation/azimuth pair in the quantized grid, i.e. each cell.
        Flat order is row-major (flat_idx = row * n_columns + col)
        Returns a [N, 2] tensor with [..., 0]=azimuth, [..., 1]=elevation.
        """
        idxelevations = torch.arange(self.n_rows, device=self.device, dtype=torch.int32)
        idxazimuths = torch.arange(
            self.n_columns, device=self.device, dtype=torch.int32
        )

        azimuth = idxazimuths.repeat(self.n_rows)
        elevation = torch.repeat_interleave(idxelevations, self.n_columns)
        return torch.stack([azimuth, elevation], dim=-1)

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

    def elements_to_sensor_angles(self, elements: Tensor) -> Tensor:
        """elements: [..., 2] with [0]=azimuth index, [1]=elevation index."""
        elem_az = elements[..., 0]
        elem_el = elements[..., 1]
        azimuth = normalize_azimuth(
            self.column_azimuths_rad[elem_az] + self.row_azimuth_offsets_rad[elem_el]
        )
        elevation = normalize_elevation(self.row_elevations_rad[elem_el])
        return torch.stack([azimuth, elevation], dim=-1)


# TODO: merge this class into RowOffsetStructuredSpinningLidarModelParametersExt
@dataclass
class LidarTiling:
    n_bins_azimuth: int
    n_bins_elevation: int
    cdf_elevation: torch.Tensor
    cdf_dense_ray_mask: torch.Tensor  # [elevation, azimuth]

    # tiles_pack_info maps each tileid to the range of elements in it,
    # expressed as .x=offset into tiles_to_elements_map and .y=number of
    # elements (rays) in that tile.
    tiles_pack_info: torch.Tensor
    # List of (ray) elements, ordered by tile.
    # Each element is represented as the (row,col) of the ray
    # in the lidar sensor space.
    tiles_to_elements_map: torch.Tensor

    def __post_init__(self):
        assert self.cdf_elevation.dtype == torch.int32, self.cdf_elevation.dtype
        assert self.cdf_elevation.ndim == 1, f"{self.cdf_elevation.shape=}"
        assert (
            self.cdf_elevation[-1].item() == self.n_bins_elevation
        ), "cdf_elevation[-1] must be equal to n_bins_elevation"

        assert (
            self.cdf_dense_ray_mask.dtype == torch.int32
        ), self.cdf_dense_ray_mask.dtype
        assert self.cdf_dense_ray_mask.ndim == 2, self.cdf_dense_ray_mask.ndim
        assert (
            self.cdf_dense_ray_mask.shape[-2] == self.cdf_elevation.shape[0]
        ), f"{self.cdf_dense_ray_mask.shape=} {self.cdf_elevation.shape=}"

        assert self.tiles_pack_info.ndim == 2, self.tiles_pack_info.ndim
        assert self.tiles_pack_info.shape == (
            self.n_bins_azimuth * self.n_bins_elevation,
            2,
        ), self.tiles_pack_info.shape
        assert self.tiles_pack_info.dtype == torch.int32, self.tiles_pack_info.dtype

        assert self.tiles_to_elements_map.ndim == 2, self.tiles_to_elements_map.ndim
        assert (
            self.tiles_to_elements_map.dtype == torch.int32
        ), self.tiles_to_elements_map.dtype

        # TODO: These asserts are important, but we don't have the needed variables at this point.
        # These checks need improvement.
        # assert self.tiles_to_elements_map.shape == (self.n_rows*self.n_columns,2)
        # assert self.tiles_to_elements_map.shape[0] <= self.max_pts_per_tile*self.tiles_pack_info.shape[0]

    @property
    def cdf_resolution_elevation(self) -> int:
        return self.cdf_dense_ray_mask.shape[-2] - 1

    @property
    def cdf_resolution_azimuth(self) -> int:
        return self.cdf_dense_ray_mask.shape[-1] - 1


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParameters
):
    """Lidar camera parameters extended with acceleration structures"""

    angles_to_columns_map: Tensor
    tiling: LidarTiling

    def __init__(
        self,
        params: RowOffsetStructuredSpinningLidarModelParameters,
        angles_to_columns_map: Tensor,
        tiling: LidarTiling,
    ) -> None:
        assert angles_to_columns_map is not None
        assert tiling is not None

        # Copy-construct the base class from an already-validated params instance
        self.__dict__.update(params.__dict__)
        self.angles_to_columns_map = angles_to_columns_map
        self.tiling = tiling

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
    lidar: SpinningLidarModelParameters, coord: Tensor, scale: float = 1
) -> Tensor:
    """coord: [..., 2] with [0]=azimuth, [1]=elevation. Returns same layout."""
    azimuth = relative_angle(
        lidar.fov_horiz_rad.start * scale,
        coord[..., 0],
        lidar.spinning_direction,
        scale,
    )
    elevation = relative_clock_rotation(
        lidar.fov_vert_rad.start * scale,
        coord[..., 1],
        SpinningDirection.CLOCKWISE,
    )
    return torch.stack([azimuth, elevation], dim=-1)


def valid_sensor_angles(
    lidar: SpinningLidarModelParameters, coord: Tensor, scale: float = 1
) -> Tensor:
    # Account for accumulated numerical errors via some epsilon in FOV check (1x eps on the start of the FOV)
    fov_vert_start_adj = lidar.fov_vert_rad.start + lidar.fov_eps_rad
    if lidar.spinning_direction == SpinningDirection.CLOCKWISE:
        # angles increase from right->left  bottom->top
        fov_horiz_start_adj = lidar.fov_horiz_rad.start + lidar.fov_eps_rad
    else:
        # angles increase from left->right or top->bottom.
        fov_horiz_start_adj = lidar.fov_horiz_rad.start - lidar.fov_eps_rad

    coord_az = coord[..., 0]
    coord_el = coord[..., 1]
    rel_elevation = relative_clock_rotation(
        fov_vert_start_adj * scale, coord_el, SpinningDirection.CLOCKWISE
    )
    rel_azimuth = relative_angle(
        fov_horiz_start_adj * scale, coord_az, lidar.spinning_direction, scale
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
    lidar: SpinningLidarModelParameters, sensor_angles: Tensor
) -> SensorRayReturn:
    """Computes the sensor rays for elevation/azimuth angles.
    sensor_angles: [..., 2] with [0]=azimuth, [1]=elevation.
    """

    sensor_angles = sensor_angles.to(device=lidar.device, dtype=lidar.dtype)

    # Check if relative sensor angles are within the FOV of the sensor
    valid = valid_sensor_angles(lidar, sensor_angles)

    # Compute the 3D unit rays
    azimuth = sensor_angles[..., 0]
    elevation = sensor_angles[..., 1]
    cos_elevation = torch.cos(elevation)
    x = torch.cos(azimuth) * cos_elevation
    y = torch.sin(azimuth) * cos_elevation
    z = torch.sin(elevation)
    sensor_rays = torch.stack([x, y, z], dim=-1)

    return SensorRayReturn(sensor_rays=sensor_rays, valid_flag=valid)


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

    grid_angles = torch.stack([grid_azimuths_rad, grid_elevations_rad], dim=-1)

    # Convert grid angles to unit norm rays
    # TODO: if the fov_horiz_rad.span == 2*pi, the first and last columns (0 and 2*pi)
    # will correspond to the same rays. This leads to the same sensor angle being assigned
    # to a cell in either the first or the last column, depending on small floating-point errors.
    # This needs to be fixed.
    grid_rays = sensor_angles_to_rays(lidar, grid_angles)

    assert torch.all(
        grid_rays.valid_flag
    ), "Bug: grid rays must be valid in the FOV of the sensor"

    elements = lidar.create_elements()
    elem_az = elements[..., 0]
    elem_el = elements[..., 1]
    raw_elevation = lidar.row_elevations_rad[elem_el]
    raw_azimuth = (
        lidar.column_azimuths_rad[elem_az] + lidar.row_azimuth_offsets_rad[elem_el]
    )
    raw_azimuth = torch.where(
        raw_azimuth > torch.pi, raw_azimuth - 2 * torch.pi, raw_azimuth
    )
    raw_azimuth = torch.where(
        raw_azimuth <= -torch.pi, raw_azimuth + 2 * torch.pi, raw_azimuth
    )
    sensor_angles = torch.stack([raw_azimuth, raw_elevation], dim=-1).reshape(-1, 2)

    assert sensor_angles.shape == (
        lidar.n_rows * lidar.n_columns,
        2,
    ), sensor_angles.shape

    # Now convert the sensor angles to unit rays.
    sensor_rays = sensor_angles_to_rays(lidar, sensor_angles)

    assert torch.all(
        sensor_rays.valid_flag
    ), "Bug: sensor rays must be valid in the FOV of the sensor"

    # Compute the NN of each grid ray in the sensor rays
    # TODO: this is quite slow and should be moved to CUDA.
    #       We can get away with it for now because we're caching the results,
    #       and usually this function would be called only once per lidar model needed.
    from scipy import spatial as scipy_spatial  # requires pip install gsplat[lidar]

    kdtree = scipy_spatial.cKDTree(
        sensor_rays.sensor_rays.contiguous().cpu().numpy()
    )  # ty:ignore[unresolved-attribute]
    _, idxs = kdtree.query(grid_rays.sensor_rays.contiguous().cpu().numpy())
    idxs = torch.from_numpy(idxs).to(device=lidar.device, dtype=torch.int32)

    # Elements are in row-major order (flat_idx = row * n_columns + col), so column index = idx % n_columns.
    return (idxs % lidar.n_columns).to(dtype).reshape(grid_angles.shape[:-1])


# --------------------- Computation of LidarTiling structures ---------------------------


@torch.no_grad()
def angles_to_dense_ray_mask_cdf(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    angles: torch.Tensor,
    *,
    resolution_elevation: int,
    resolution_azimuth: int,
) -> Tensor:  # [res_elev+1, res_azim+1]

    # dense tile indices
    def uniform_quantization(x: torch.Tensor, n_bins: int):
        return (x * n_bins).int() % n_bins

    relative_angles = relative_sensor_angles(parameters, angles)

    normalized_azimuth = relative_angles[..., 0] / parameters.fov_horiz_rad.span
    normalized_elevation = relative_angles[..., 1] / parameters.fov_vert_rad.span

    elevations_indices = uniform_quantization(
        normalized_elevation, resolution_elevation
    )
    azimuths_indices = uniform_quantization(normalized_azimuth, resolution_azimuth)
    indices = azimuths_indices + elevations_indices * resolution_azimuth

    masks = torch.zeros(
        resolution_elevation * resolution_azimuth,
        device=angles.device,
        dtype=torch.int32,
    )
    masks[indices] = 1

    masks2d = masks.reshape(resolution_elevation, resolution_azimuth)
    masks2d_padded = torch.zeros(
        resolution_elevation + 1,
        resolution_azimuth + 1,
        device=angles.device,
        dtype=torch.int32,
    )

    masks2d_padded[1:, 1:] = masks2d
    masks2d_integral = masks2d_padded.cumsum(dim=0).cumsum(dim=1)

    # Return (elevation+1, azimuth+1)
    return masks2d_integral.int()


def angles_to_tile_indices(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    angles: Tensor,
    *,
    n_bins_azimuth: int,
    n_bins_elevation: int,
    cdf_elevation: torch.Tensor,
):
    # the length of cdf_elevation is one plus the number of bins, so we need to subtract one here
    resolution = len(cdf_elevation) - 1

    relative_angles = relative_sensor_angles(parameters, angles)

    normalized_azimuth = (
        relative_angles[..., 0] / parameters.fov_horiz_rad.span * n_bins_azimuth
    )
    normalized_elevation = (
        relative_angles[..., 1] / parameters.fov_vert_rad.span * resolution
    )

    # compute the azimuth tile indices directly
    azimuths_indices = normalized_azimuth.int() % n_bins_azimuth

    # remap the elevations
    elevations_indices_cdf = torch.clamp(normalized_elevation, 0, resolution - 1).int()
    elevations_indices = cdf_elevation[elevations_indices_cdf].int()

    # NOTE: tile indices are row-major: tile_id = elevation * n_bins_azimuth + azimuth
    return azimuths_indices + elevations_indices * n_bins_azimuth


def compute_tiles_to_elements_map(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_azimuth: int,
    densification_factor_azimuth: float,
    cdf_elevation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the mapping from tiles to elements for the given lidar model and parameters"""

    # compute the number of bins for elevation
    n_bins_elevation = cdf_elevation[-1].item()
    assert n_bins_elevation.is_integer(), "CDF elevation must be an integer"
    n_bins_elevation = int(n_bins_elevation)

    # create all element indices [relative to the static model]
    elements = parameters.create_elements()
    angles = parameters.elements_to_sensor_angles(elements)

    tile_indices = angles_to_tile_indices(
        parameters,
        angles,
        n_bins_azimuth=n_bins_azimuth,
        n_bins_elevation=n_bins_elevation,
        cdf_elevation=cdf_elevation,
    )

    tile_counts = torch.bincount(
        tile_indices, minlength=n_bins_azimuth * n_bins_elevation
    )
    tile_starts = torch.cumsum(tile_counts, dim=0) - tile_counts
    tiles_pack_info = (
        torch.stack([tile_starts, tile_counts], dim=-1)
        .int()
        .to(device=elements.device)
        .contiguous()
    )

    # sort the elements by tile indices
    sorted_element_indices = torch.argsort(tile_indices)
    sorted_elements = elements[sorted_element_indices].contiguous()

    # compute the dense mask
    cdf_dense_ray_mask = angles_to_dense_ray_mask_cdf(
        parameters,
        angles,
        resolution_elevation=len(cdf_elevation) - 1,
        resolution_azimuth=n_bins_azimuth * densification_factor_azimuth,
    )

    return sorted_elements, tiles_pack_info, cdf_dense_ray_mask


def compute_histogram_equalization(
    parameters: RowOffsetStructuredSpinningLidarModelParameters,
    *,
    n_bins_elevation: int,
    max_pts_per_tile: int,
    resolution_elevation: int,
) -> tuple[int, torch.Tensor]:

    elements = parameters.create_elements()
    angles = parameters.elements_to_sensor_angles(elements).reshape(
        parameters.n_rows, parameters.n_columns, 2
    )
    angles = relative_sensor_angles(parameters, angles)
    angles_az = angles[..., 0]
    angles_el = angles[..., 1]

    # TODO: use parameters.fov_eps_rad
    fov_eps_rad = 2 * torch.finfo(torch.float32).eps
    ranges_azimuth = (-fov_eps_rad, parameters.fov_horiz_rad.span + fov_eps_rad)
    ranges_elevation = (-fov_eps_rad, parameters.fov_vert_rad.span + fov_eps_rad)
    assert torch.all(
        (angles_el >= ranges_elevation[0]) & (angles_el <= ranges_elevation[1])
    ), f"elevations are out of bounds, {angles_el.min()} < {ranges_elevation[0]},  {angles_el.max()} > {ranges_elevation[1]}"

    assert torch.all(
        (angles_az >= ranges_azimuth[0]) & (angles_az <= ranges_azimuth[1])
    ), f"azimuths are out of bounds, {angles_az.min()} < {ranges_azimuth[0]},  {angles_az.max()} > {ranges_azimuth[1]}"

    # --------------------------------
    # Histogram Equalization
    # --------------------------------
    def compute_hist1d(
        data: torch.Tensor,
        bins: int | np.ndarray | torch.Tensor,
        range: tuple[float, float],
    ):
        if isinstance(bins, torch.Tensor):
            bins = bins.cpu().numpy()
        hist, *others = np.histogram(data.cpu().numpy(), bins=bins, range=range)
        return torch.tensor(hist, device=data.device, dtype=torch.int32), *others

    # 1. compute the prefix sum of data
    hist, _ = compute_hist1d(
        angles_el, bins=resolution_elevation, range=ranges_elevation
    )

    tot = torch.sum(hist)
    cdf_elevation = torch.zeros((len(hist) + 1), device=parameters.device)
    cdf_elevation[1:] = torch.cumsum(hist, dim=0)
    cdf_elevation = cdf_elevation / tot * (n_bins_elevation)

    # 2. interpolate the new values
    edges_list = [0]
    curr = 1
    for i in range(len(cdf_elevation)):
        if cdf_elevation[i] >= curr:
            edges_list.append(i)
            curr += 1
    edges_list[-1] = len(cdf_elevation) - 1
    edges = torch.tensor(edges_list, device=parameters.device, dtype=torch.float32)

    # recompute the histograms
    edges_elevation = edges / resolution_elevation * parameters.fov_vert_rad.span
    hist_elevations, _ = compute_hist1d(
        angles_el, bins=edges_elevation, range=ranges_elevation
    )

    n_bins_azimuth = int(
        torch.ceil(hist_elevations.float().mean() / max_pts_per_tile)
    )  # estimate the number of bins for azimuths
    assert n_bins_azimuth > 0

    # now we need to find the smallest number of bins that satisfies the max_pts_per_tile
    angles_azimuth_np = angles_az.flatten().cpu().numpy()
    angles_elevation_np = angles_el.flatten().cpu().numpy()
    edges_elevation_np = edges_elevation.cpu().numpy()

    def compute_hist2d(n_bins_azimuth):
        return np.histogram2d(
            angles_azimuth_np,
            angles_elevation_np,
            bins=[n_bins_azimuth, edges_elevation_np],
            range=[ranges_azimuth, ranges_elevation],
        )

    hist2d, _, _ = compute_hist2d(n_bins_azimuth)
    while hist2d.max() > max_pts_per_tile:
        n_bins_azimuth += 1
        hist2d, _, _ = compute_hist2d(n_bins_azimuth)

    assert isinstance(cdf_elevation, torch.Tensor)
    assert (
        cdf_elevation[-1].item() == n_bins_elevation
    ), f"{cdf_elevation[-1]=} {n_bins_elevation=}"
    assert cdf_elevation.shape == (
        resolution_elevation + 1,
    ), f"{cdf_elevation.shape=} {(resolution_elevation+1,)=}"

    return (
        n_bins_azimuth,
        cdf_elevation,
    )


def compute_tiling(
    lidar_params: RowOffsetStructuredSpinningLidarModelParameters,
    n_bins_elevation: int = 16,
    max_pts_per_tile=16 * 16,
    resolution_elevation: int = 1600,
    densification_factor_azimuth: int = 8,
) -> LidarTiling:
    params = SimpleNamespace()
    params.n_bins_elevation = n_bins_elevation

    (params.n_bins_azimuth, params.cdf_elevation,) = compute_histogram_equalization(
        lidar_params,
        n_bins_elevation=n_bins_elevation,
        max_pts_per_tile=max_pts_per_tile,
        resolution_elevation=resolution_elevation,
    )

    (
        params.tiles_to_elements_map,
        params.tiles_pack_info,
        params.cdf_dense_ray_mask,
    ) = compute_tiles_to_elements_map(
        lidar_params,
        n_bins_azimuth=params.n_bins_azimuth,
        densification_factor_azimuth=densification_factor_azimuth,
        cdf_elevation=params.cdf_elevation,
    )

    params.cdf_elevation = params.cdf_elevation.int()

    return LidarTiling(**vars(params))
