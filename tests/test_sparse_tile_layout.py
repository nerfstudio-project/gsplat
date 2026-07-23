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
"""Unit tests for the sparse tile-layout op.

Compares the CUDA ``gsplat.cuda._wrapper.build_sparse_tile_layout`` against the
pure-torch reference ``gsplat.cuda._torch_impl._build_sparse_tile_layout``.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="build_sparse_tile_layout is CUDA-only"
)


def _grid(width, height, tile_size):
    return math.ceil(height / tile_size), math.ceil(width / tile_size)


def _make_pixels(per_image_counts, width, height, seed):
    """Packed (pixels [P,2] (row,col), image_ids [P]) with no duplicate pixels
    within an image (the op's dedup precondition)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pix_list, img_list = [], []
    for c, count in enumerate(per_image_counts):
        assert count <= width * height
        perm = torch.randperm(width * height, device=device, generator=gen)[:count]
        rows = perm // width
        cols = perm % width
        pix_list.append(torch.stack((rows, cols), dim=-1))
        img_list.append(torch.full((count,), c, device=device, dtype=torch.int32))
    pixels = torch.cat(pix_list).to(torch.int32).contiguous()
    image_ids = torch.cat(img_list).contiguous()
    return pixels, image_ids


def _check_layout(pixels, image_ids, n_images, ts, tw, th):
    from gsplat.cuda._torch_impl import _build_sparse_tile_layout
    from gsplat.cuda._wrapper import build_sparse_tile_layout

    g_at, g_mask, g_bm, g_cs, g_pm = build_sparse_tile_layout(
        pixels, image_ids, n_images, ts, tw, th
    )
    r_at, r_mask, r_bm, r_cs, r_pm = _build_sparse_tile_layout(
        pixels, image_ids, n_images, ts, tw, th
    )

    P = pixels.shape[0]
    AT = r_at.numel()
    words = (ts * ts + 63) // 64

    # dtypes
    assert g_at.dtype == torch.int32
    assert g_mask.dtype == torch.bool
    assert g_bm.dtype == torch.uint64
    assert g_cs.dtype == torch.int64
    assert g_pm.dtype == torch.int64
    # shapes
    assert g_at.shape == (AT,)
    assert g_mask.shape == (n_images, th, tw)
    assert g_bm.shape == (AT, words)
    assert g_cs.shape == (AT,)
    assert g_pm.shape == (P,)

    # exact match against the reference (keys are unique given the dedup
    # precondition, so the sort order -- and thus pixel_map -- is deterministic).
    torch.testing.assert_close(g_at, r_at, rtol=0, atol=0)
    torch.testing.assert_close(g_mask, r_mask, rtol=0, atol=0)
    torch.testing.assert_close(g_bm, r_bm, rtol=0, atol=0)
    torch.testing.assert_close(g_cs, r_cs, rtol=0, atol=0)
    torch.testing.assert_close(g_pm, r_pm, rtol=0, atol=0)

    # invariants (independent of the reference)
    assert torch.equal(g_at, torch.nonzero(g_mask.flatten()).flatten().to(torch.int32))
    if AT:
        assert g_cs[-1].item() == P  # every pixel lands in some active tile
    # the bitmask sets exactly P bits across all active tiles
    total_bits = sum(
        bin(int(v) & 0xFFFFFFFFFFFFFFFF).count("1") for v in g_bm.flatten().tolist()
    )
    assert total_bits == P
    return g_at, g_mask, g_bm, g_cs, g_pm


@pytest.mark.parametrize("counts", [[16], [64, 32], [100, 0, 50], [8, 8, 8, 8]])
@pytest.mark.parametrize("tile_size", [4, 8, 16])
def test_layout_matches_reference(counts, tile_size):
    W, H = 64, 48
    th, tw = _grid(W, H, tile_size)
    pixels, image_ids = _make_pixels(counts, W, H, seed=100 + sum(counts) + tile_size)
    _check_layout(pixels, image_ids, len(counts), tile_size, tw, th)


def test_layout_non_square():
    W, H = 73, 41
    th, tw = _grid(W, H, 16)
    pixels, image_ids = _make_pixels([200, 150], W, H, seed=11)
    _check_layout(pixels, image_ids, 2, 16, tw, th)


def test_layout_single_tile():
    # image exactly one tile: every pixel maps to tile 0 of its image.
    ts = 16
    pixels, image_ids = _make_pixels([50, 40], ts, ts, seed=3)
    _check_layout(pixels, image_ids, 2, ts, 1, 1)


def test_layout_dense_tile():
    # a fully-packed single tile (all 256 pixels active) exercises every bit of
    # the bitmask, including bit 63 of each word.
    ts = 16
    rows = torch.arange(ts, device=device).repeat_interleave(ts)
    cols = torch.arange(ts, device=device).repeat(ts)
    pixels = torch.stack((rows, cols), dim=-1).to(torch.int32).contiguous()
    image_ids = torch.zeros(ts * ts, device=device, dtype=torch.int32)
    _check_layout(pixels, image_ids, 1, ts, 1, 1)


def test_layout_trailing_empty_image():
    # n_images larger than the max image_id present (trailing image has no
    # pixels) -- active_tile_mask must still be sized [n_images, th, tw].
    W, H = 32, 32
    th, tw = _grid(W, H, 16)
    pixels, image_ids = _make_pixels([20, 0, 0], W, H, seed=7)
    _check_layout(pixels, image_ids, 4, 16, tw, th)


def test_layout_empty():
    from gsplat.cuda._wrapper import build_sparse_tile_layout

    ts, tw, th = 16, 4, 3
    pixels = torch.empty(0, 2, dtype=torch.int32, device=device)
    image_ids = torch.empty(0, dtype=torch.int32, device=device)
    at, mask, bm, cs, pm = build_sparse_tile_layout(pixels, image_ids, 2, ts, tw, th)
    words = (ts * ts + 63) // 64
    assert at.shape == (0,) and at.dtype == torch.int32
    assert mask.shape == (2, th, tw) and not mask.any()
    assert bm.shape == (0, words) and bm.dtype == torch.uint64
    assert cs.shape == (1,) and cs[0].item() == 0  # empty case returns a [1] zero
    assert pm.shape == (0,)


def test_layout_deterministic():
    from gsplat.cuda._wrapper import build_sparse_tile_layout

    W, H = 64, 48
    th, tw = _grid(W, H, 16)
    pixels, image_ids = _make_pixels([300, 200, 100], W, H, seed=9)
    a = build_sparse_tile_layout(pixels, image_ids, 3, 16, tw, th)
    b = build_sparse_tile_layout(pixels, image_ids, 3, 16, tw, th)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
