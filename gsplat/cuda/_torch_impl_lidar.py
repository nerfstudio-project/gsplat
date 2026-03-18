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
from ._lidar import SpinningDirection, relative_sensor_angles
from ._wrapper import RowOffsetStructuredSpinningLidarModelParametersExt
import struct

ANGLE_TO_PIXEL_SCALING_FACTOR: int = 1024


@dataclass
class LidarSampleTileIdReturn:
    idx: Tensor
    idxdense: Tensor


def lidar_sample_tileid(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    rel_pix: Tensor,
    round_fn,
) -> LidarSampleTileIdReturn:
    assert round_fn is torch.floor or round_fn is torch.ceil

    kToPixel = ANGLE_TO_PIXEL_SCALING_FACTOR

    fov_span_pix_az = kToPixel * lidar.fov_horiz_rad.span
    fov_span_pix_el = kToPixel * lidar.fov_vert_rad.span

    norm_az = rel_pix[..., 0] / fov_span_pix_az
    norm_el = rel_pix[..., 1] / fov_span_pix_el

    idxdense_az = round_fn(norm_az * lidar.tiling.cdf_resolution_azimuth).to(
        dtype=torch.int32
    )
    idxdense_el = round_fn(norm_el * lidar.tiling.cdf_resolution_elevation).to(
        dtype=torch.int32
    )

    assert torch.all(
        (0 <= idxdense_el) & (idxdense_el <= lidar.tiling.cdf_resolution_elevation)
    )
    assert torch.all(idxdense_el < lidar.tiling.cdf_elevation.shape[0])

    if round_fn is torch.ceil:
        # [min_dense_el, max_dense_el) is a half-open range, so the coarse
        # tile range [tile_min_el, tile_max_el) must be too.
        # CDF[max_dense_el] gives the coarse bin that dense bin max_dense_el
        # *belongs to*, not the exclusive upper bound.
        # Use CDF[max_dense_el - 1] + 1 instead.
        tile_el = torch.where(
            idxdense_el >= 1,
            torch.clamp(
                lidar.tiling.cdf_elevation[torch.clamp(idxdense_el - 1, min=0)] + 1,
                max=lidar.tiling.n_bins_elevation,
            ),
            lidar.tiling.cdf_elevation[idxdense_el],
        )
    else:
        tile_el = lidar.tiling.cdf_elevation[idxdense_el]

    idx_az = round_fn(norm_az * lidar.tiling.n_bins_azimuth).to(dtype=torch.int32)

    return LidarSampleTileIdReturn(
        idx=torch.stack([idx_az, tile_el], dim=-1),
        idxdense=torch.stack([idxdense_az, idxdense_el], dim=-1),
    )


def has_any_rays_in_tile(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    beg: Tensor,
    end: Tensor,
) -> Tensor:
    """Check whether any rays fall within the dense tile range [beg, end).
    The full-cover case (beg=0, end=N) forces has_rays=True as a conservative
    optimization implemented in CUDA to avoid I/O.
    We do it here as well for results to match.
    """

    raycdf = lidar.tiling.cdf_dense_ray_mask

    raycdf_size_az = lidar.tiling.cdf_resolution_azimuth
    raycdf_size_el = lidar.tiling.cdf_resolution_elevation

    beg_az = beg[..., 0]
    beg_el = beg[..., 1]
    end_az = end[..., 0]
    end_el = end[..., 1]

    assert torch.all((0 <= beg_el) & (beg_el <= raycdf_size_el))
    assert torch.all((0 <= end_el) & (end_el <= raycdf_size_el))
    assert torch.all((0 <= beg_az) & (beg_az <= raycdf_size_az))
    assert torch.all((0 <= end_az) & (end_az <= raycdf_size_az))

    num_rays = (
        raycdf[end_el, end_az]
        - raycdf[beg_el, end_az]
        - raycdf[end_el, beg_az]
        + raycdf[beg_el, beg_az]
    )

    has_rays = (num_rays > 0) | ((beg_az <= 0) & (end_az >= raycdf_size_az))

    return has_rays


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

    angles_pix = means2d.reshape(-1, 2)
    extent_pix = radii.reshape(-1, 2)

    nonzero_extent = (extent_pix[..., 0] > 0) & (extent_pix[..., 1] > 0)
    full_circle_pix = 2 * math.pi * ANGLE_TO_PIXEL_SCALING_FACTOR
    fov_span_pix_az = ANGLE_TO_PIXEL_SCALING_FACTOR * lidar.fov_horiz_rad.span
    fov_span_pix_el = ANGLE_TO_PIXEL_SCALING_FACTOR * lidar.fov_vert_rad.span

    # 1. Compute relative angles once on the mean, then +/- extent.
    mean_rel_pix = relative_sensor_angles(
        lidar, angles_pix, ANGLE_TO_PIXEL_SCALING_FACTOR
    )

    beg_pix = mean_rel_pix - extent_pix
    end_pix = mean_rel_pix + extent_pix

    beg_az = beg_pix[..., 0]
    beg_el = beg_pix[..., 1]
    end_az = end_pix[..., 0]
    end_el = end_pix[..., 1]

    # 2. Computation of gaussian/fov intersection regions A and B.

    # Check full_cover before capping to 2*pi, since the cap can push end
    # below fov_span for wide FOVs even though the gaussian covers the full FOV.
    full_cover = (beg_az <= 0) & (end_az >= fov_span_pix_az)

    # Don't let gaussian size to be more than 2*pi
    end_az = torch.minimum(end_az, beg_az + full_circle_pix)

    overflows = end_az > full_circle_pix
    underflows = beg_az < 0

    # Definition of the two regions depending on gaussian and hfov range
    # full_cover:  A = [0, fov_span), B = empty
    # underflows:  A = [0, end),      B = [beg+2pi, 2pi)
    # overflows:   A = [beg, fc),     B = [0, end-2pi)
    # normal:      A = [beg, end),    B = empty

    begA_az = torch.where(full_cover | underflows, 0, beg_az)
    endA_az = torch.where(
        full_cover, fov_span_pix_az, torch.where(overflows, full_circle_pix, end_az)
    )
    begB_az = torch.where(underflows & ~full_cover, beg_az + full_circle_pix, 0)
    endB_az = torch.where(
        overflows & ~full_cover,
        end_az - full_circle_pix,
        torch.where(underflows & ~full_cover, full_circle_pix, 0),
    )

    begA_rel_pix = torch.stack([begA_az, beg_el], dim=-1)
    endA_rel_pix = torch.stack([endA_az, end_el], dim=-1)
    begB_rel_pix = torch.stack([begB_az, beg_el], dim=-1)
    endB_rel_pix = torch.stack([endB_az, end_el], dim=-1)

    # 3. Compute which tile (dense and regular) each boundary of the range fall into.
    # Recall that the range is half-open [min,max), so is the sampled range.

    # Clamp pixel regions to [0, fov_span] so that normalized values land in
    # [0, 1] and the resulting dense/tile indices are guaranteed in-range.
    def clamp_az(v):
        return torch.clamp(v, 0, fov_span_pix_az)

    def clamp_el(v):
        return torch.clamp(v, 0, fov_span_pix_el)

    begA_rel_pix = torch.stack([clamp_az(begA_az), clamp_el(beg_el)], dim=-1)
    endA_rel_pix = torch.stack([clamp_az(endA_az), clamp_el(end_el)], dim=-1)
    begB_rel_pix = torch.stack([clamp_az(begB_az), clamp_el(beg_el)], dim=-1)
    endB_rel_pix = torch.stack([clamp_az(endB_az), clamp_el(end_el)], dim=-1)

    begA_sample = lidar_sample_tileid(lidar, begA_rel_pix, torch.floor)
    endA_sample = lidar_sample_tileid(lidar, endA_rel_pix, torch.ceil)
    begB_sample = lidar_sample_tileid(lidar, begB_rel_pix, torch.floor)
    endB_sample = lidar_sample_tileid(lidar, endB_rel_pix, torch.ceil)

    # Make sure all indices are within bounds
    for s in [begA_sample, endA_sample, begB_sample, endB_sample]:
        idxdense_az = s.idxdense[..., 0]
        idx_az = s.idx[..., 0]
        assert torch.all(
            (0 <= idxdense_az) & (idxdense_az <= lidar.tiling.cdf_resolution_azimuth)
        )
        assert torch.all((0 <= idx_az) & (idx_az <= lidar.tiling.n_bins_azimuth))

    # Elevation ranges of Regions A and B must be the same.
    begA_idx_el = begA_sample.idx[..., 1]
    endA_idx_el = endA_sample.idx[..., 1]
    assert torch.all(begA_idx_el == begB_sample.idx[..., 1])
    assert torch.all(endA_idx_el == endB_sample.idx[..., 1])

    # 4. Check if there's any ray in each one of the ranges.
    # If any of them has none, we don't need to process it.
    has_raysA = nonzero_extent & has_any_rays_in_tile(
        lidar, begA_sample.idxdense, endA_sample.idxdense
    )
    has_raysB = nonzero_extent & has_any_rays_in_tile(
        lidar, begB_sample.idxdense, endB_sample.idxdense
    )
    has_rays = has_raysA | has_raysB

    # 5. Calculate the final tile ranges that have at least 1 ray in it.
    begA_idx_az = begA_sample.idx[..., 0]
    endA_idx_az = endA_sample.idx[..., 0]
    begB_idx_az = begB_sample.idx[..., 0]
    endB_idx_az = endB_sample.idx[..., 0]

    tile_range_el = (
        torch.where(has_rays, begA_idx_el, 0),
        torch.where(has_rays, endA_idx_el, 0),
    )
    assert torch.all(tile_range_el[0] <= tile_range_el[1])

    n_bins_az = lidar.tiling.n_bins_azimuth
    periodic_az = fov_span_pix_az >= full_circle_pix

    begA_tile_az = torch.where(has_raysA, begA_idx_az, 0)
    endA_tile_az = torch.where(has_raysA, endA_idx_az, 0)
    begB_tile_az = torch.where(has_raysB, begB_idx_az, 0)
    endB_tile_az = torch.where(has_raysB, endB_idx_az, 0)

    if periodic_az:
        # For periodic azimuth, tiles wrap around (tile n_bins-1 is adjacent
        # to tile 0). Merge B into A by extending A across the 0/n_bins seam:
        #   underflows: A=[0, endA), B=[begB, n_bins) -> merged=[begB-n_bins, endA)
        #   overflows:  A=[begA, n_bins), B=[0, endB) -> merged=[begA, endB+n_bins)
        # The kernel then uses az % n_bins to map back to valid tile indices.
        begA_tile_az = torch.where(
            has_raysB & underflows, begB_tile_az - n_bins_az, begA_tile_az
        )
        endA_tile_az = torch.where(
            has_raysB & overflows, endB_tile_az + n_bins_az, endA_tile_az
        )
        # Cap to at most n_bins wide to prevent double-counting tiles at the
        # seam when ceil(endA) and floor(begB) produce overlapping tile indices.
        begA_tile_az = torch.maximum(begA_tile_az, endA_tile_az - n_bins_az)
        endA_tile_az = torch.minimum(endA_tile_az, begA_tile_az + n_bins_az)
        begB_tile_az = torch.zeros_like(begB_tile_az)
        endB_tile_az = torch.zeros_like(endB_tile_az)
    else:
        # For non-periodic azimuth, A and B could generally disjoint (e.g.,
        # behind-sensor gaussians with tips at both edges of the FOV).
        # However, when extent is very close to pi, ceil(endA) and floor(begB)
        # can produce overlapping tile indices. Since one range always anchors
        # at 0 and the other at n_bins, any overlap means the gaussian covers
        # the entire FOV. Merge into [0, n_bins) in that case.
        tile_overlap = (
            has_raysA
            & has_raysB
            & (begB_tile_az < endA_tile_az)
            & (begA_tile_az < endB_tile_az)
        )
        begA_tile_az[tile_overlap] = 0
        endA_tile_az[tile_overlap] = n_bins_az
        begB_tile_az[tile_overlap] = 0
        endB_tile_az[tile_overlap] = 0

    tile_ranges_az = [
        (begA_tile_az, endA_tile_az),
        (begB_tile_az, endB_tile_az),
    ]
    assert torch.all(tile_ranges_az[0][0] <= tile_ranges_az[0][1])
    assert torch.all(tile_ranges_az[1][0] <= tile_ranges_az[1][1])

    # 6. Enumerate all gaussian x tile intersections.
    az_count = sum((e - s) for s, e in tile_ranges_az).to(torch.int32)
    el_count = tile_range_el[1] - tile_range_el[0]
    tiles_per_gauss = el_count * az_count

    assert len(tiles_per_gauss) == I * N
    n_isects = int(tiles_per_gauss.sum().item())
    accum_tiles_per_gauss = torch.cumsum(tiles_per_gauss, dim=0)
    assert accum_tiles_per_gauss[-1] == n_isects

    isect_ids_lo = torch.empty(n_isects, dtype=torch.int32, device="cpu")
    isect_ids_hi = torch.empty(n_isects, dtype=torch.int32, device="cpu")
    flatten_ids = torch.empty(n_isects, dtype=torch.int32, device="cpu")

    image_n_bits = I.bit_length()
    tile_n_bits = (
        lidar.tiling.n_bins_elevation * lidar.tiling.n_bins_azimuth
    ).bit_length()
    assert image_n_bits + tile_n_bits <= 32, "Too many tiles"

    depths = depths.reshape(-1)

    def kernel(image_id, gauss_id):
        index = image_id * N + gauss_id

        if not nonzero_extent[index]:
            return

        curr_idx = accum_tiles_per_gauss[index - 1] if index > 0 else 0

        depth_f32 = depths[index]
        depth_id = struct.unpack("i", struct.pack("f", depth_f32))[0]
        depth_id = int(depth_id) & 0xFFFFFFFF

        for el in range(int(tile_range_el[0][index]), int(tile_range_el[1][index])):
            for az_s, az_e in tile_ranges_az:
                for az in range(int(az_s[index]), int(az_e[index])):
                    actual_az = az % n_bins_az if periodic_az else az
                    tile_id = el * lidar.tiling.n_bins_azimuth + actual_az
                    isect_ids_lo[curr_idx] = depth_id
                    isect_ids_hi[curr_idx] = (image_id << tile_n_bits) | tile_id
                    flatten_ids[curr_idx] = index
                    curr_idx += 1

    for image_id in range(I):
        for gauss_id in range(N):
            kernel(image_id, gauss_id)

    isect_ids_lo = isect_ids_lo.to(device=device)
    isect_ids_hi = isect_ids_hi.to(device=device)
    flatten_ids = flatten_ids.to(device=device)

    isect_ids = (isect_ids_hi.to(torch.int64) << 32) | (
        isect_ids_lo.to(torch.int64) & 0xFFFFFFFF
    )

    if sort:
        isect_ids, sort_indices = torch.sort(isect_ids)
        flatten_ids = flatten_ids[sort_indices]

    tiles_per_gauss = tiles_per_gauss.reshape(image_dims + (N,))

    assert tiles_per_gauss.dtype == torch.int32, tiles_per_gauss.dtype
    assert isect_ids.dtype == torch.int64, isect_ids.dtype
    assert flatten_ids.dtype == torch.int32, flatten_ids.dtype

    return tiles_per_gauss, isect_ids, flatten_ids
