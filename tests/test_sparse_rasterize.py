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
"""Unit tests for the sparse 3DGS rasterizer.

Validates ``gsplat.cuda._wrapper.rasterize_to_pixels_sparse`` against the dense
rasterizer ``rasterize_to_pixels``: the sparse op must reproduce, at each
requested pixel, exactly the dense rendering -- in both the forward outputs and
the input gradients. Both paths share the same per-gaussian blending math
(``RasterizeToPixels3DGSDevice.cuh``); only tile iteration and pixel addressing
differ, so the dense op is the natural oracle. The dense and sparse tile
intersections (``isect_tiles`` / ``isect_tiles_sparse``) feed the same gaussians
per active tile, so an exact match is expected up to floating-point noise.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="rasterize_to_pixels_sparse is CUDA-only"
)


def _grid(width, height, tile_size):
    return math.ceil(height / tile_size), math.ceil(width / tile_size)


def _make_scene(C, N, width, height, channels, seed, requires_grad=False):
    """A synthetic 2D gaussian scene. conics are isotropic with extent set by a
    per-gaussian radius; radii are the (looser) integer bounding boxes used by
    tile intersection."""
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*s, device=device, generator=gen)

    means2d = torch.empty(C, N, 2, device=device)
    means2d[..., 0] = u(C, N) * width  # x (col)
    means2d[..., 1] = u(C, N) * height  # y (row)
    r = u(C, N) * 6.0 + 2.0  # gaussian std in pixels
    inv = 1.0 / (r * r)
    conics = torch.stack([inv, torch.zeros_like(inv), inv], dim=-1).contiguous()
    rb = (r * 3.0).ceil().to(torch.int32)
    radii = torch.stack([rb, rb], dim=-1).contiguous()
    depths = (u(C, N) * 9.9 + 0.1).contiguous()
    colors = u(C, N, channels).contiguous()
    opacities = (u(C, N) * 0.9 + 0.05).contiguous()

    if requires_grad:
        means2d.requires_grad_(True)
        conics.requires_grad_(True)
        colors.requires_grad_(True)
        opacities.requires_grad_(True)
    return means2d, conics, colors, opacities, radii, depths


def _all_pixels(C, width, height):
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    one = torch.stack([rr.flatten(), cc.flatten()], dim=-1)  # [H*W, 2]
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
        rows = perm // width
        cols = perm % width
        pix_list.append(torch.stack([rows, cols], dim=-1))
        img_list.append(torch.full((k,), c, device=device, dtype=torch.int32))
    pixels = torch.cat(pix_list).to(torch.int32).contiguous()
    image_ids = torch.cat(img_list).contiguous()
    return pixels, image_ids


def _dense(scene, width, height, tile_size, backgrounds=None, masks=None):
    from gsplat import isect_offset_encode, isect_tiles, rasterize_to_pixels

    means2d, conics, colors, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = _grid(width, height, tile_size)
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)
    return rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
    )


def _sparse(
    scene, pixels, image_ids, width, height, tile_size, backgrounds=None, masks=None
):
    from gsplat import (
        build_sparse_tile_layout,
        isect_tiles_sparse,
        rasterize_to_pixels_sparse,
    )

    means2d, conics, colors, opacities, radii, depths = scene
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
    return rasterize_to_pixels_sparse(
        means2d,
        conics,
        colors,
        opacities,
        image_ids,
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
        backgrounds=backgrounds,
        masks=masks,
    )


def _gather(colors_img, alphas_img, pixels, image_ids):
    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    return colors_img[img, rows, cols], alphas_img[img, rows, cols]


@pytest.mark.parametrize("C,N", [(1, 32), (2, 128), (3, 256)])
@pytest.mark.parametrize("tile_size", [4, 16])
@pytest.mark.parametrize("channels", [3, 32])
def test_forward_all_pixels(C, N, tile_size, channels):
    W, H = 40, 28
    scene = _make_scene(C, N, W, H, channels, seed=100 + N + tile_size + channels)
    dc, da = _dense(scene, W, H, tile_size)
    pixels, image_ids = _all_pixels(C, W, H)
    oc, oa = _sparse(scene, pixels, image_ids, W, H, tile_size)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(oc, rc, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(oa, ra, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
def test_forward_subset(C, N, frac):
    W, H = 48, 32
    ts = 16
    scene = _make_scene(C, N, W, H, 3, seed=7 + N)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = _subset_pixels(C, W, H, frac, seed=N)
    oc, oa = _sparse(scene, pixels, image_ids, W, H, ts)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(oc, rc, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(oa, ra, rtol=1e-4, atol=1e-4)


def test_forward_non_square():
    C, N = 2, 200
    W, H = 73, 41
    ts = 16
    scene = _make_scene(C, N, W, H, 3, seed=11)
    dc, da = _dense(scene, W, H, ts)
    pixels, image_ids = _subset_pixels(C, W, H, 0.4, seed=3)
    oc, oa = _sparse(scene, pixels, image_ids, W, H, ts)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(oc, rc, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(oa, ra, rtol=1e-4, atol=1e-4)


def test_forward_backgrounds():
    C, N, channels = 2, 128, 3
    W, H = 40, 28
    ts = 16
    scene = _make_scene(C, N, W, H, channels, seed=21)
    gen = torch.Generator(device=device).manual_seed(21)
    backgrounds = torch.rand(C, channels, device=device, generator=gen).contiguous()
    dc, da = _dense(scene, W, H, ts, backgrounds=backgrounds)
    pixels, image_ids = _all_pixels(C, W, H)
    oc, oa = _sparse(scene, pixels, image_ids, W, H, ts, backgrounds=backgrounds)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(oc, rc, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(oa, ra, rtol=1e-4, atol=1e-4)


def test_forward_masks():
    C, N = 2, 128
    W, H = 48, 32
    ts = 16
    th, tw = _grid(W, H, ts)
    scene = _make_scene(C, N, W, H, 3, seed=33)
    gen = torch.Generator(device=device).manual_seed(33)
    masks = (torch.rand(C, th, tw, device=device, generator=gen) < 0.5).contiguous()
    dc, da = _dense(scene, W, H, ts, masks=masks)
    pixels, image_ids = _all_pixels(C, W, H)
    oc, oa = _sparse(scene, pixels, image_ids, W, H, ts, masks=masks)
    rc, ra = _gather(dc, da, pixels, image_ids)
    torch.testing.assert_close(oc, rc, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(oa, ra, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("use_backgrounds", [False, True])
def test_backward_matches_dense(use_backgrounds):
    C, N, channels = 2, 96, 3
    W, H = 40, 28
    ts = 16
    scene = _make_scene(C, N, W, H, channels, seed=5, requires_grad=True)
    means2d, conics, colors, opacities, _radii, _depths = scene
    inputs = (means2d, conics, colors, opacities)

    backgrounds = None
    if use_backgrounds:
        gen0 = torch.Generator(device=device).manual_seed(50)
        backgrounds = torch.rand(
            C, channels, device=device, generator=gen0, requires_grad=True
        )

    pixels, image_ids = _subset_pixels(C, W, H, 0.5, seed=5)

    grad_inputs = inputs + ((backgrounds,) if use_backgrounds else ())

    # Sparse forward + grads w.r.t. a random cotangent.
    oc, oa = _sparse(scene, pixels, image_ids, W, H, ts, backgrounds=backgrounds)
    gen = torch.Generator(device=device).manual_seed(99)
    v_colors = torch.rand(oc.shape, device=device, generator=gen)
    v_alphas = torch.rand(oa.shape, device=device, generator=gen)
    g_sparse = torch.autograd.grad((oc, oa), grad_inputs, (v_colors, v_alphas))

    # Dense forward + grads w.r.t. the same cotangent scattered into the image.
    dc, da = _dense(scene, W, H, ts, backgrounds=backgrounds)
    v_dc = torch.zeros_like(dc)
    v_da = torch.zeros_like(da)
    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    v_dc[img, rows, cols] = v_colors
    v_da[img, rows, cols] = v_alphas
    g_dense = torch.autograd.grad((dc, da), grad_inputs, (v_dc, v_da))

    for gs, gd in zip(g_sparse, g_dense):
        torch.testing.assert_close(gs, gd, rtol=1e-3, atol=1e-3)
