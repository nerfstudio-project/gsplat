# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sparse num-contributing-gaussians op.

Validates ``gsplat.cuda._wrapper.rasterize_num_contributing_gaussians_sparse``
against the dense ``rasterize_num_contributing_gaussians`` gathered at the
requested pixels -- the sparse op must reproduce, at each requested pixel,
exactly the dense per-pixel contributor count and accumulated alpha. It reuses
the dense traversal harness/accumulator with sparse addressing, so the dense op
is the natural oracle.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rasterize_num_contributing_gaussians_sparse is CUDA-only",
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


def _all_pixels(C, width, height):
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    one = torch.stack([rr.flatten(), cc.flatten()], dim=-1)
    pixels = one.repeat(C, 1).to(torch.int32).contiguous()
    image_ids = torch.arange(C, device=device, dtype=torch.int32).repeat_interleave(
        height * width
    )
    return pixels, image_ids


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


def _dense(scene, width, height, tile_size):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_num_contributing_gaussians,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = _grid(width, height, tile_size)
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


def _gather(ncg_img, alphas_img, pixels, image_ids):
    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    return ncg_img[img, rows, cols], alphas_img[img, rows, cols]


@pytest.mark.parametrize("C,N", [(1, 32), (2, 128), (3, 256)])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_all_pixels(C, N, tile_size):
    W, H = 40, 28
    scene = _make_scene(C, N, W, H, seed=100 + N + tile_size)
    dc, da = _dense(scene, W, H, tile_size)
    pixels, image_ids = _all_pixels(C, W, H)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, tile_size)
    rc, ra = _gather(dc, da, pixels, image_ids)
    # contributor counts are exact integers
    torch.testing.assert_close(sc, rc, rtol=0, atol=0)
    torch.testing.assert_close(sa, ra, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
def test_subset(C, N, frac):
    W, H = 48, 32
    ts = 16
    scene = _make_scene(C, N, W, H, seed=7 + N)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = _subset_pixels(C, W, H, frac, seed=N)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, ts)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(sc, rc, rtol=0, atol=0)
    torch.testing.assert_close(sa, ra, rtol=1e-5, atol=1e-5)


def test_non_square():
    C, N = 2, 200
    W, H = 73, 41
    ts = 16
    scene = _make_scene(C, N, W, H, seed=11)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = _subset_pixels(C, W, H, 0.4, seed=3)
    sc, sa = _sparse(scene, pixels, image_ids, W, H, ts)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(sc, rc, rtol=0, atol=0)
    torch.testing.assert_close(sa, ra, rtol=1e-5, atol=1e-5)
