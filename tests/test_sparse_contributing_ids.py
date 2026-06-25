# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sparse contributing-gaussian-ids op.

Validates ``rasterize_contributing_gaussian_ids_sparse`` against the dense
``rasterize_contributing_gaussian_ids`` gathered at the requested pixels: the
recorded contributor ids (exact) and weights must match the dense result at each
requested pixel, up to the per-pixel contributor count (the rest is padding).
Shared scene/pixel/gather helpers live in ``tests.sparse_test_helpers``.
"""

import pytest
import torch

from tests.sparse_test_helpers import gather, grid, make_scene, subset_pixels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rasterize_contributing_gaussian_ids_sparse is CUDA-only",
)


def _dense(scene, width, height, tile_size):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = grid(width, height, tile_size)
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)
    ncg, _ = rasterize_num_contributing_gaussians(
        means2d, conics, opacities, isect_offsets, flatten_ids, width, height, tile_size
    )
    return rasterize_contributing_gaussian_ids(
        means2d,
        conics,
        opacities,
        isect_offsets,
        flatten_ids,
        width,
        height,
        tile_size,
        ncg,
    )


def _sparse(scene, pixels, image_ids, width, height, tile_size):
    from gsplat import (
        build_sparse_tile_layout,
        isect_tiles_sparse,
        rasterize_contributing_gaussian_ids_sparse,
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
    ncg, _ = rasterize_num_contributing_gaussians_sparse(
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
    return rasterize_contributing_gaussian_ids_sparse(
        means2d,
        conics,
        opacities,
        active_tiles,
        tile_offsets,
        flatten_ids,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map,
        ncg,
        width,
        height,
        tile_size,
        tw,
        th,
    )


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_contributing_ids(C, N, frac, tile_size):
    W, H = 48, 32
    scene = make_scene(C, N, W, H, seed=7 + N + tile_size)
    d_ids, d_w = _dense(scene, W, H, tile_size)
    pixels, image_ids = subset_pixels(C, W, H, frac, seed=N)
    s_ids, s_w = _sparse(scene, pixels, image_ids, W, H, tile_size)

    d_ids_g = gather(d_ids, pixels, image_ids)  # [P, K_dense]
    d_w_g = gather(d_w, pixels, image_ids)

    Ks = s_ids.shape[1]  # K_sparse <= K_dense (subset of pixels)
    assert Ks <= d_ids_g.shape[1]
    # ids are exact; weights match to fp tolerance. Both are front-to-back with
    # -1 / 0 padding beyond each pixel's contributor count.
    torch.testing.assert_close(s_ids, d_ids_g[:, :Ks], rtol=0, atol=0)
    torch.testing.assert_close(s_w, d_w_g[:, :Ks], rtol=1e-5, atol=1e-5)
