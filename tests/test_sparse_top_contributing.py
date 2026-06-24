# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sparse top-contributing-gaussian-ids op.

Validates ``rasterize_top_contributing_gaussian_ids_sparse`` against the dense
``rasterize_top_contributing_gaussian_ids`` gathered at the requested pixels:
the top-K contributor ids (exact) and weights must match the dense result at
each requested pixel.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rasterize_top_contributing_gaussian_ids_sparse is CUDA-only",
)


def _grid(width, height, tile_size):
    return math.ceil(height / tile_size), math.ceil(width / tile_size)


def _make_scene(C, N, width, height, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*s, device=device, generator=gen)
    means2d = torch.empty(C, N, 2, device=device)
    means2d[..., 0] = u(C, N) * width
    means2d[..., 1] = u(C, N) * height
    r = u(C, N) * 6.0 + 2.0
    inv = 1.0 / (r * r)
    conics = torch.stack([inv, torch.zeros_like(inv), inv], dim=-1).contiguous()
    rb = (r * 3.0).ceil().to(torch.int32)
    radii = torch.stack([rb, rb], dim=-1).contiguous()
    depths = (u(C, N) * 9.9 + 0.1).contiguous()
    opacities = (u(C, N) * 0.9 + 0.05).contiguous()
    return means2d, conics, opacities, radii, depths


def _subset_pixels(C, width, height, frac, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    pix_list, img_list = [], []
    for c in range(C):
        k = max(1, int(width * height * frac))
        perm = torch.randperm(width * height, device=device, generator=gen)[:k]
        pix_list.append(torch.stack([perm // width, perm % width], dim=-1))
        img_list.append(torch.full((k,), c, device=device, dtype=torch.int32))
    return (
        torch.cat(pix_list).to(torch.int32).contiguous(),
        torch.cat(img_list).contiguous(),
    )


def _dense(scene, width, height, tile_size, num_depth_samples):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_top_contributing_gaussian_ids,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = _grid(width, height, tile_size)
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
    th, tw = _grid(width, height, tile_size)
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
    scene = _make_scene(C, N, W, H, seed=7 + N + num_depth_samples)
    d_ids, d_w = _dense(scene, W, H, ts, num_depth_samples)
    pixels, image_ids = _subset_pixels(C, W, H, frac, seed=N)
    s_ids, s_w = _sparse(scene, pixels, image_ids, W, H, ts, num_depth_samples)

    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    d_ids_g = d_ids[img, rows, cols]
    d_w_g = d_w[img, rows, cols]

    torch.testing.assert_close(s_ids, d_ids_g, rtol=0, atol=0)
    torch.testing.assert_close(s_w, d_w_g, rtol=1e-5, atol=1e-5)
