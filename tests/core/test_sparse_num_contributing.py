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
"""Unit tests for the sparse num-contributing-gaussians op.

Validates ``rasterize_num_contributing_gaussians_sparse`` against the dense
``rasterize_num_contributing_gaussians`` gathered at the requested pixels -- the
sparse op must reproduce, at each requested pixel, exactly the dense per-pixel
contributor count and accumulated alpha. Shared scene/pixel/gather helpers come
from the sibling ``sparse_test_helpers`` module.
"""

import pytest
import torch

from .sparse_test_helpers import (
    all_pixels,
    gather,
    grid,
    make_scene,
    subset_pixels,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rasterize_num_contributing_gaussians_sparse is CUDA-only",
)


def _dense(scene, width, height, tile_size):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_num_contributing_gaussians,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = grid(width, height, tile_size)
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)
    return rasterize_num_contributing_gaussians(
        means2d, conics, opacities, isect_offsets, flatten_ids, width, height, tile_size
    )


def _sparse(scene, pixels, image_ids, width, height, tile_size):
    from gsplat import (
        build_sparse_tile_layout,
        isect_tiles_sparse,
        rasterize_num_contributing_gaussians_sparse,
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
    return rasterize_num_contributing_gaussians_sparse(
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
    )


@pytest.mark.parametrize("C,N", [(1, 32), (2, 128), (3, 256)])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_all_pixels(C, N, tile_size):
    W, H = 40, 28
    scene = make_scene(C, N, W, H, seed=100 + N + tile_size)
    dc, da = _dense(scene, W, H, tile_size)
    pixels, image_ids = all_pixels(C, W, H)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, tile_size)
    # contributor counts are exact integers
    torch.testing.assert_close(sc, gather(dc, pixels, image_ids), rtol=0, atol=0)
    torch.testing.assert_close(sa, gather(da, pixels, image_ids), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
def test_subset(C, N, frac):
    W, H = 48, 32
    ts = 16
    scene = make_scene(C, N, W, H, seed=7 + N)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = subset_pixels(C, W, H, frac, seed=N)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, ts)
    torch.testing.assert_close(sc, gather(dc, pixels, image_ids), rtol=0, atol=0)
    torch.testing.assert_close(sa, gather(da, pixels, image_ids), rtol=1e-5, atol=1e-5)


def test_non_square():
    C, N = 2, 200
    W, H = 73, 41
    ts = 16
    scene = make_scene(C, N, W, H, seed=11)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = subset_pixels(C, W, H, 0.4, seed=3)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, ts)
    torch.testing.assert_close(sc, gather(dc, pixels, image_ids), rtol=0, atol=0)
    torch.testing.assert_close(sa, gather(da, pixels, image_ids), rtol=1e-5, atol=1e-5)
