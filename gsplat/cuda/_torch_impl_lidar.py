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
import torch
from torch import Tensor
from dataclasses import dataclass
import math
from typing import Tuple, Union, Literal
from ._lidar import SpinningDirection, relative_sensor_angles, SphericalUnitCoord
from ._wrapper import RowOffsetStructuredSpinningLidarModelParametersExt
from ._torch_lidars import _RowOffsetStructuredSpinningLidarModel
import struct

ANGLE_TO_PIXEL_SCALING_FACTOR: int = 1024


@dataclass
class LidarSampleTileIdReturn:
    idx_el: Tensor
    idx_az: Tensor
    idxdense_el: Tensor
    idxdense_az: Tensor


def lidar_sample_tileid(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    rel_pix_el: Tensor,
    rel_pix_az: Tensor,
) -> LidarSampleTileIdReturn:
    kToPixel = ANGLE_TO_PIXEL_SCALING_FACTOR

    fov_span_pix_az = kToPixel * lidar.fov_horiz_rad.span
    fov_span_pix_el = kToPixel * lidar.fov_vert_rad.span

    norm_az = rel_pix_az / fov_span_pix_az
    norm_el = rel_pix_el / fov_span_pix_el

    idxdense_az = torch.floor(norm_az * lidar.tiling.cdf_resolution_azimuth).to(
        dtype=torch.int32
    )
    idxdense_el = torch.floor(norm_el * lidar.tiling.cdf_resolution_elevation).to(
        dtype=torch.int32
    )
    # Clamp to valid range
    idxdense_el = torch.clamp(
        idxdense_el, min=0, max=lidar.tiling.cdf_resolution_elevation - 1
    )

    idx_az = torch.floor(norm_az * lidar.tiling.n_bins_azimuth).to(dtype=torch.int32)
    idx_el = lidar.tiling.cdf_elevation[idxdense_el]
    assert idx_el.dtype == torch.int32

    return LidarSampleTileIdReturn(
        idx_el=idx_el,
        idx_az=idx_az,
        idxdense_el=idxdense_el,
        idxdense_az=idxdense_az,
    )


def has_any_rays_in_tile(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    min_el: Tensor,
    max_el: Tensor,
    min_az: Tensor,
    max_az: Tensor,
) -> Tensor:
    """Filter Gaussian particles based on the precomputed ray mask and the range
    of tiles it overlaps with. If no ray intercepts it, remove this Gaussian particle."""

    raycdf = lidar.tiling.cdf_dense_ray_mask

    raycdf_size_az = lidar.tiling.cdf_resolution_azimuth
    raycdf_size_el = lidar.tiling.cdf_resolution_elevation

    assert min_el.shape == max_el.shape
    assert min_az.shape == max_az.shape
    assert torch.all((0 <= min_el) & (min_el <= max_el))
    assert torch.all(max_el <= raycdf_size_el)
    assert torch.all(min_az <= max_az)

    N = min_az.shape[0]

    # -1 -> element not initialized.
    #  0 -> False
    #  1 -> True
    has_rays = torch.full((N,), -1, dtype=torch.int32, device=min_az.device)

    # Gaussian's azimuth covers the whole fov horiz?
    has_rays[(has_rays < 0) & (min_az <= 0) & (max_az >= raycdf_size_az)] = True
    # Entirely to the left of the FOV start
    has_rays[(has_rays < 0) & (max_az <= 0)] = False
    # Entirely to the right of the FOV end
    has_rays[(has_rays < 0) & (min_az >= raycdf_size_az)] = False

    def cdf_region_sum(min_az: Tensor, max_az: Tensor) -> Tensor:
        min_az = torch.clamp(min_az, 0, raycdf_size_az)
        max_az = torch.clamp(max_az, 0, raycdf_size_az)

        return (
            raycdf[max_az, max_el]
            - raycdf[min_az, max_el]
            - raycdf[max_az, min_el]
            + raycdf[min_az, min_el]
        )

    # Fully inside FOV
    region_sum_fully_inside = cdf_region_sum(min_az=min_az, max_az=max_az)
    has_rays = torch.where(
        (has_rays < 0) & ((min_az >= 0) & (max_az <= raycdf_size_az)),
        region_sum_fully_inside > 0,
        has_rays,
    )

    zero_az = torch.zeros_like(min_az)
    full_az = torch.full_like(min_az, raycdf_size_az)

    # Wraps at left edge
    region_sum_wraps_left = cdf_region_sum(
        min_az=zero_az, max_az=max_az
    ) + cdf_region_sum(min_az=min_az + raycdf_size_az, max_az=full_az)
    # max_az is inclusive, so if it's == 0, the range falls into FOV.
    has_rays = torch.where(
        (has_rays < 0) & ((min_az < 0) & (max_az > 0)),
        region_sum_wraps_left > 0,
        has_rays,
    )

    # Wraps at right edge
    region_sum_wraps_right = cdf_region_sum(
        min_az=min_az, max_az=full_az
    ) + cdf_region_sum(min_az=zero_az, max_az=max_az - full_az)
    has_rays = torch.where(
        (has_rays < 0) & ((min_az < raycdf_size_az) & (max_az >= raycdf_size_az)),
        region_sum_wraps_right > 0,
        has_rays,
    )
    has_rays = torch.where(
        (has_rays < 0) & ((min_az < raycdf_size_az) & (max_az > raycdf_size_az)),
        region_sum_wraps_right > 0,
        has_rays,
    )

    assert torch.all(has_rays >= 0)

    return has_rays.to(torch.bool)


@torch.no_grad()
def _isect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: Tensor,  # [..., N, 2]
    radii: Tensor,  # [..., N, 2]
    depths: Tensor,  # [..., N]
    *,
    sort: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Pytorch implementation of `gsplat.cuda._wrapper.isect_tiles_lidar()`.

    This function expects `means2d` and `radii` to be in angular pixel space
    (angle * ANGLE_TO_PIXEL_SCALING_FACTOR).
    """

    sensor = _RowOffsetStructuredSpinningLidarModel(lidar)

    image_dims = means2d.shape[:-2]
    N = means2d.shape[-2]  # Total number of gaussians
    assert means2d.shape == (*image_dims, N, 2), means2d.shape
    assert radii.shape == (*image_dims, N, 2), radii.shape
    assert depths.shape == (*image_dims, N), depths.shape
    assert depths.dtype == torch.float32
    assert lidar.tiling.n_bins_azimuth > 0, lidar.tiling.n_bins_azimuth
    assert lidar.tiling.n_bins_elevation > 0, lidar.tiling.n_bins_elevation
    assert (
        lidar.tiling.cdf_dense_ray_mask.dtype == torch.int32
    ), lidar.tiling.cdf_dense_ray_mask.dtype
    assert lidar.tiling.cdf_dense_ray_mask.device == means2d.device, (
        lidar.tiling.cdf_dense_ray_mask.device,
        means2d.device,
    )

    device = means2d.device
    I = math.prod(image_dims)  # image count

    # Retrieve projection data, already in pixel space
    # TODO: assuming x==elevation, y==azimuth. This must be transposed in a future update.
    angles_pix = SphericalUnitCoord(
        elevation=means2d.reshape(-1, 2)[:, 0], azimuth=means2d.reshape(-1, 2)[:, 1]
    )
    extent_x = radii.reshape(-1, 2)[:, 0]
    extent_y = radii.reshape(-1, 2)[:, 1]

    # 1. Go through all gaussians and compute which and how many tiles they intersect with.

    # Compute the relative gaussian projection center (in pixel space) wrt. FOV start
    rel_angles_pix = sensor.relative_sensor_angles(
        angles_pix, scale=ANGLE_TO_PIXEL_SCALING_FACTOR
    )
    # Compute the projected gaussian extents in pixel space.
    min_pix_el = rel_angles_pix.elevation - extent_x
    min_pix_az = rel_angles_pix.azimuth - extent_y
    max_pix_el = rel_angles_pix.elevation + extent_x
    max_pix_az = rel_angles_pix.azimuth + extent_y

    # Compute which tiles the projected gaussian bbox falls into
    min_sample_tileid = lidar_sample_tileid(lidar, min_pix_el, min_pix_az)
    max_sample_tileid = lidar_sample_tileid(lidar, max_pix_el, max_pix_az)

    min_dense_el = min_sample_tileid.idxdense_el
    min_dense_az = min_sample_tileid.idxdense_az
    max_dense_el = torch.min(
        max_sample_tileid.idxdense_el + 1,
        min_dense_el + lidar.tiling.cdf_resolution_elevation,
    )
    max_dense_az = torch.min(
        max_sample_tileid.idxdense_az + 1,
        min_dense_az + lidar.tiling.cdf_resolution_azimuth,
    )
    assert torch.all(min_dense_el <= max_dense_el)
    assert torch.all(min_dense_az <= max_dense_az)

    # There are no rays if the gaussian area is 0 or
    has_rays = ((extent_x != 0) & (extent_y != 0)) & has_any_rays_in_tile(
        lidar,
        min_el=min_dense_el,
        max_el=max_dense_el,
        min_az=min_dense_az,
        max_az=max_dense_az,
    )

    # Save the tiles that overlap with the projected gaussian
    tile_min_el = torch.where(has_rays, min_sample_tileid.idx_el, 0)
    tile_min_az = torch.where(has_rays, min_sample_tileid.idx_az, 0)
    tile_max_el = torch.where(
        has_rays,
        torch.min(
            max_sample_tileid.idx_el + 1, tile_min_el + lidar.tiling.n_bins_elevation
        ),
        0,
    )
    tile_max_az = torch.where(
        has_rays,
        torch.min(
            max_sample_tileid.idx_az + 1, tile_min_az + lidar.tiling.n_bins_azimuth
        ),
        0,
    )

    assert torch.all(tile_min_el <= tile_max_el)
    assert torch.all(tile_min_az <= tile_max_az)

    # Save how many tiles this gaussian covers
    tiles_per_gauss = (tile_max_el - tile_min_el) * (tile_max_az - tile_min_az)

    # 2. Enumerate all gaussian x tile intersections.
    # TODO: this is the same code as in _isect_tiles, we should reuse it.

    assert len(tiles_per_gauss) == I * N
    n_isects = int(tiles_per_gauss.sum().item())
    cum_tiles_per_gauss = torch.cumsum(tiles_per_gauss, dim=0)
    assert cum_tiles_per_gauss[-1] == n_isects

    isect_ids_lo = torch.empty(n_isects, dtype=torch.int32, device=device)
    isect_ids_hi = torch.empty(n_isects, dtype=torch.int32, device=device)
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device=device)

    image_n_bits = I.bit_length()
    tile_n_bits = (
        lidar.tiling.n_bins_elevation * lidar.tiling.n_bins_azimuth
    ).bit_length()
    assert image_n_bits + tile_n_bits <= 32, "Too many tiles"

    depths = depths.reshape(-1)

    def kernel(image_id, gauss_id):
        index = image_id * N + gauss_id
        if extent_x[index] <= 0.0 or extent_y[index] <= 0.0:
            return
        curr_idx = cum_tiles_per_gauss[index - 1] if index > 0 else 0

        # Reinterpret float bits as int32 (preserving bit pattern)
        depth_f32 = depths[index]
        depth_id = struct.unpack("i", struct.pack("f", depth_f32))[0]
        # Store in a 64-bit int, zero-extending to lower 32 bits
        depth_id = int(depth_id) & 0xFFFFFFFF  # Ensures upper 32 bits are zero

        for y in range(tile_min_az[index], tile_max_az[index]):
            mod_y = y % lidar.tiling.n_bins_azimuth
            for x in range(tile_min_el[index], tile_max_el[index]):
                tile_id = mod_y * lidar.tiling.n_bins_elevation + x
                # isect_ids[curr_idx] = (
                #     (image_id << (tile_n_bits + 32))
                #     | (tile_id << 32)
                #     | depth_id
                # )
                isect_ids_lo[curr_idx] = depth_id
                isect_ids_hi[curr_idx] = (image_id << tile_n_bits) | tile_id
                flatten_ids[curr_idx] = index  # flattened index
                curr_idx += 1

    for image_id in range(I):
        for gauss_id in range(N):
            kernel(image_id, gauss_id)

    isect_ids = (isect_ids_hi.to(torch.int64) << 32) | (
        isect_ids_lo.to(torch.int64) & 0xFFFFFFFF
    )

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (N,))

    assert tiles_per_gauss.dtype == torch.int32
    assert isect_ids.dtype == torch.int64
    assert flatten_ids.dtype == torch.int32

    return tiles_per_gauss, isect_ids, flatten_ids
