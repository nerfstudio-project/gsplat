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
"""Unit tests for the sparse contributing-gaussian-ids op.

Validates ``rasterize_contributing_gaussian_ids_sparse`` against the dense
``rasterize_contributing_gaussian_ids`` gathered at the requested pixels: the
recorded contributor ids (exact) and weights must match the dense result at each
requested pixel, up to the per-pixel contributor count (the rest is padding).
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="rasterize_contributing_gaussian_ids_sparse is CUDA-only",
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


def _dense(scene, width, height, tile_size):
    from gsplat import (
        isect_offset_encode,
        isect_tiles,
        rasterize_contributing_gaussian_ids,
        rasterize_num_contributing_gaussians,
    )

    means2d, conics, opacities, radii, depths = scene
    C = means2d.shape[0]
    th, tw = _grid(width, height, tile_size)
    _tpg, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tw, th, n_images=C
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tw, th)
    ncg, _ = rasterize_num_contributing_gaussians(
        means2d, conics, opacities, isect_offsets, flatten_ids, width, height, tile_size
    )
    ids, weights = rasterize_contributing_gaussian_ids(
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
    return ids, weights


def _sparse(scene, pixels, image_ids, width, height, tile_size):
    from gsplat import (
        build_sparse_tile_layout,
        isect_tiles_sparse,
        rasterize_contributing_gaussian_ids_sparse,
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
    ids, weights = rasterize_contributing_gaussian_ids_sparse(
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
    return ids, weights


@pytest.mark.parametrize("C,N", [(1, 64), (2, 128)])
@pytest.mark.parametrize("frac", [0.1, 0.5])
@pytest.mark.parametrize("tile_size", [4, 16])
def test_contributing_ids(C, N, frac, tile_size):
    W, H = 48, 32
    scene = _make_scene(C, N, W, H, seed=7 + N + tile_size)
    d_ids, d_w = _dense(scene, W, H, tile_size)
    pixels, image_ids = _subset_pixels(C, W, H, frac, seed=N)
    s_ids, s_w = _sparse(scene, pixels, image_ids, W, H, tile_size)

    # gather dense at the requested pixels: [P, K_dense]
    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    d_ids_g = d_ids[img, rows, cols]
    d_w_g = d_w[img, rows, cols]

    Ks = s_ids.shape[1]  # K_sparse <= K_dense (subset of pixels)
    assert Ks <= d_ids_g.shape[1]
    # ids are exact; weights match to fp tolerance. Both are front-to-back with
    # -1 / 0 padding beyond each pixel's contributor count.
    torch.testing.assert_close(s_ids, d_ids_g[:, :Ks], rtol=0, atol=0)
    torch.testing.assert_close(s_w, d_w_g[:, :Ks], rtol=1e-5, atol=1e-5)
