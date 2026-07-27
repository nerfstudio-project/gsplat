# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the sparse tile-intersection op.

Compares the CUDA ``gsplat.cuda._wrapper.isect_tiles_sparse`` against the
pure-torch reference ``gsplat.cuda._torch_impl._isect_tiles_sparse``, an
obviously-correct looped implementation.
"""

import math

import pytest
import torch

device = torch.device("cuda:0")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="isect_tiles_sparse is CUDA-only"
)


def _make_inputs(
    C,
    N,
    seed,
    *,
    width=64,
    height=48,
    radius=(2.0, 12.0),
    oob_frac=0.1,
    dtype=torch.float32,
):
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*s, device=device, generator=gen)
    means2d = torch.empty(C, N, 2, device=device, dtype=dtype)
    means2d[..., 0] = u(C, N) * (width + 32) - 16
    means2d[..., 1] = u(C, N) * (height + 24) - 12
    n_oob = int(N * oob_frac)
    if n_oob:
        means2d[:, :n_oob] = -1000.0
    r_lo, r_hi = radius
    rx = (u(C, N) * (r_hi - r_lo) + r_lo).floor().clamp(min=1).to(torch.int32)
    ry = (u(C, N) * (r_hi - r_lo) + r_lo).floor().clamp(min=1).to(torch.int32)
    radii = torch.stack((rx, ry), dim=-1).contiguous()
    depths = (u(C, N) * 99.9 + 0.1).to(dtype).contiguous()
    return means2d.contiguous(), radii, depths, width, height


def _grid(width, height, tile_size):
    return math.ceil(height / tile_size), math.ceil(width / tile_size)


def _make_mask(C, th, tw, pattern, seed):
    if pattern == "all":
        mask = torch.ones(C, th, tw, dtype=torch.bool, device=device)
    elif pattern == "quadrant":
        mask = torch.zeros(C, th, tw, dtype=torch.bool, device=device)
        mask[:, : th // 2, : tw // 2] = True
    elif pattern == "random":
        gen = torch.Generator(device=device).manual_seed(seed)
        mask = torch.rand(C, th, tw, device=device, generator=gen) < 0.5
    else:
        raise ValueError(pattern)
    active = torch.nonzero(mask.flatten()).flatten().to(torch.int32)
    return mask.contiguous(), active


def _check(means2d, radii, depths, mask, active, C, ts, tw, th, image_ids=None):
    from gsplat.cuda._torch_impl import _isect_tiles_sparse
    from gsplat.cuda._wrapper import isect_tiles_sparse

    g_off, g_ids = isect_tiles_sparse(
        means2d, radii, depths, mask, active, C, ts, tw, th, image_ids=image_ids
    )
    r_off, r_ids = _isect_tiles_sparse(
        means2d, radii, depths, mask, active, C, ts, tw, th, image_ids=image_ids
    )
    assert g_off.dtype == torch.int32 and g_ids.dtype == torch.int32
    assert g_off.shape == (active.numel() + 1,)
    torch.testing.assert_close(g_off, r_off, rtol=0, atol=0)
    torch.testing.assert_close(g_ids, r_ids, rtol=0, atol=0)
    # last offset is the sentinel == n_isects
    assert g_off[-1].item() == g_ids.numel()


@pytest.mark.parametrize("C,N", [(1, 16), (2, 64), (3, 256), (4, 512)])
@pytest.mark.parametrize("pattern", ["all", "quadrant", "random"])
def test_sparse_intersect_matches_reference(C, N, pattern):
    means2d, radii, depths, W, H = _make_inputs(C, N, seed=100 + N)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(C, th, tw, pattern, seed=N)
    _check(means2d, radii, depths, mask, active, C, 16, tw, th)


@pytest.mark.parametrize("tile_size", [4, 16])
def test_isect_tiles_sparse_sizes(tile_size):
    means2d, radii, depths, W, H = _make_inputs(2, 128, seed=7, width=64, height=48)
    th, tw = _grid(W, H, tile_size)
    mask, active = _make_mask(2, th, tw, "random", seed=tile_size)
    _check(means2d, radii, depths, mask, active, 2, tile_size, tw, th)


def test_sparse_intersect_non_square():
    means2d, radii, depths, W, H = _make_inputs(2, 200, seed=11, width=73, height=41)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(2, th, tw, "quadrant", seed=3)
    _check(means2d, radii, depths, mask, active, 2, 16, tw, th)


def test_sparse_intersect_packed():
    means2d, radii, depths, W, H = _make_inputs(3, 128, seed=21)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(3, th, tw, "random", seed=21)
    M = 3 * 128
    image_ids = torch.arange(3, device=device, dtype=torch.int32).repeat_interleave(128)
    _check(
        means2d.reshape(M, 2).contiguous(),
        radii.reshape(M, 2).contiguous(),
        depths.reshape(M).contiguous(),
        mask,
        active,
        3,
        16,
        tw,
        th,
        image_ids=image_ids,
    )


def test_sparse_intersect_zero_radii():
    means2d, radii, depths, W, H = _make_inputs(2, 64, seed=13)
    radii = torch.zeros_like(radii)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(2, th, tw, "all", seed=13)
    from gsplat.cuda._wrapper import isect_tiles_sparse

    off, ids = isect_tiles_sparse(means2d, radii, depths, mask, active, 2, 16, tw, th)
    assert ids.numel() == 0
    assert torch.all(off == 0)


def test_sparse_intersect_empty_active():
    means2d, radii, depths, W, H = _make_inputs(2, 64, seed=14)
    th, tw = _grid(W, H, 16)
    mask = torch.zeros(2, th, tw, dtype=torch.bool, device=device)
    active = torch.empty(0, dtype=torch.int32, device=device)
    from gsplat.cuda._wrapper import isect_tiles_sparse

    off, ids = isect_tiles_sparse(means2d, radii, depths, mask, active, 2, 16, tw, th)
    assert ids.numel() == 0
    assert off.shape == (1,) and off[0].item() == 0


def _assert_tile_sets_match(g_off, g_ids, r_off, r_ids):
    """Offsets must match exactly; the per-active-tile id *sets* must match.

    Within-tile order is undefined when depths tie (radix sort vs argsort are
    not stably ordered on equal keys), so compare the set of ids in each tile
    slice rather than element-wise.
    """
    torch.testing.assert_close(g_off, r_off, rtol=0, atol=0)
    assert g_ids.numel() == r_ids.numel()
    bounds = g_off.tolist()
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        assert sorted(g_ids[s:e].tolist()) == sorted(r_ids[s:e].tolist()), f"tile {i}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_isect_tiles_sparse_dtypes(dtype):
    # The kernel dispatches on means2d's dtype (float/double) and narrows the
    # depth sort key to float32; the torch reference does the same. The CUDA op
    # must agree with the reference for both dtypes.
    means2d, radii, depths, W, H = _make_inputs(3, 256, seed=5, dtype=dtype)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(3, th, tw, "random", seed=5)
    _check(means2d, radii, depths, mask, active, 3, 16, tw, th)


def test_isect_tiles_sparse_ties():
    # Quantize depths to a few discrete values so many gaussians within a tile
    # share an exact depth. Tie order is undefined, so the CUDA op and the torch
    # reference need only agree on the per-tile id *set*; the offset table must
    # still match exactly.
    from gsplat.cuda._torch_impl import _isect_tiles_sparse
    from gsplat.cuda._wrapper import isect_tiles_sparse

    means2d, radii, depths, W, H = _make_inputs(2, 512, seed=8, radius=(8.0, 20.0))
    depths = torch.randint(1, 4, depths.shape, device=device).to(depths).contiguous()
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(2, th, tw, "random", seed=8)

    g_off, g_ids = isect_tiles_sparse(
        means2d, radii, depths, mask, active, 2, 16, tw, th
    )
    r_off, r_ids = _isect_tiles_sparse(
        means2d, radii, depths, mask, active, 2, 16, tw, th
    )
    _assert_tile_sets_match(g_off, g_ids, r_off, r_ids)

    # Guardrail: confirm the scenario actually produces within-tile depth ties,
    # otherwise the set-vs-elementwise distinction is untested.
    dep_flat = depths.reshape(-1)
    bounds = g_off.tolist()
    has_tie = any(
        (seg := dep_flat[g_ids[bounds[i] : bounds[i + 1]].long()]).numel()
        > seg.unique().numel()
        for i in range(len(bounds) - 1)
    )
    assert has_tie, "expected within-tile depth ties in this scenario"


def test_isect_tiles_sparse_deterministic():
    # Each run on identical inputs must produce identical outputs (no
    # nondeterministic sort/atomics) — cross-checks would be unstable otherwise.
    from gsplat.cuda._wrapper import isect_tiles_sparse

    means2d, radii, depths, W, H = _make_inputs(3, 512, seed=9)
    th, tw = _grid(W, H, 16)
    mask, active = _make_mask(3, th, tw, "random", seed=9)
    a_off, a_ids = isect_tiles_sparse(
        means2d, radii, depths, mask, active, 3, 16, tw, th
    )
    b_off, b_ids = isect_tiles_sparse(
        means2d, radii, depths, mask, active, 3, 16, tw, th
    )
    torch.testing.assert_close(a_off, b_off, rtol=0, atol=0)
    torch.testing.assert_close(a_ids, b_ids, rtol=0, atol=0)
