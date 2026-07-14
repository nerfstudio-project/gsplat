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
"""Unit tests for the sparse top-contributing-gaussian-ids op.

Validates ``rasterize_top_contributing_gaussian_ids_sparse`` against the dense
``rasterize_top_contributing_gaussian_ids`` gathered at the requested pixels:
the top-K contributor ids (exact) and weights must match the dense result at
each requested pixel. Shared scene/pixel/gather helpers come from the sibling
``sparse_test_helpers`` module.
"""

import pytest
import torch

from tests._cuda import cuda_is_available

from .sparse_test_helpers import gather, grid, make_scene, subset_pixels

pytestmark = pytest.mark.skipif(
    not cuda_is_available(),
    reason="rasterize_top_contributing_gaussian_ids_sparse is CUDA-only",
)


def _dense(scene, width, height, tile_size, num_depth_samples):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_top_contributing_gaussian_ids,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = grid(width, height, tile_size)
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)
    return rasterize_top_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        num_depth_samples,
    )


def _sparse(scene, pixels, image_ids, width, height, tile_size, num_depth_samples):
    from gsplat import (
        build_sparse_tile_layout,
        isect_tiles_sparse,
        rasterize_top_contributing_gaussian_ids_sparse,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = grid(width, height, tile_size)
    (
        active_tiles,
        active_tile_mask,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
    ) = build_sparse_tile_layout(pixels, image_ids, C, tile_size, tw, th)
    tile_offsets, flatten_ids = isect_tiles_sparse(
        means2d, radii, depths, active_tile_mask, active_tiles, C, tile_size, tw, th
    )
    return rasterize_top_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        width,
        height,
        tile_size,
        tw,
        th,
        num_depth_samples,
    )


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
@pytest.mark.parametrize("num_depth_samples", [1, 4])
def test_top_contributing(C, N, frac, num_depth_samples):
    W, H = 48, 32
    ts = 16
    scene = make_scene(C, N, W, H, seed=7 + N + num_depth_samples)
    d_ids, d_w = _dense(scene, W, H, ts, num_depth_samples)
    pixels, image_ids = subset_pixels(C, W, H, frac, seed=N)
    s_ids, s_w = _sparse(scene, pixels, image_ids, W, H, ts, num_depth_samples)

    torch.testing.assert_close(s_ids, gather(d_ids, pixels, image_ids), rtol=0, atol=0)
    torch.testing.assert_close(
        s_w, gather(d_w, pixels, image_ids), rtol=1e-5, atol=1e-5
    )
