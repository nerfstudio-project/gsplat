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

"""Parity + edge-case tests for the macro-tile (MT) intersection pipeline.

The MT path should match the existing flat intersection path at rendered-output
level for the standard 3DGS EWA cases covered here. Tolerances match what the
existing `test_rasterize_to_pixels` accepts for `GSPLAT_FAST_MATH=ON` builds.
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

import gsplat
from gsplat.cuda._wrapper import isect_tiles, isect_tiles_macro_binning

pytest.importorskip("gsplat.csrc", exc_type=ImportError)


# --------------------------------------------------------------------------
# Scene helpers
# --------------------------------------------------------------------------


def _make_scene(
    n_gauss: int,
    width: int,
    height: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, ...]:
    """Synthetic 3DGS scene placed in front of the camera. Returns the
    standard tuple expected by ``rasterization``.

    ``viewmats`` is identity, so means are already in camera space. Positive
    ``z`` is forward and is forced to [1, 2]; ``tan(fov_y / 2)`` is effectively
    0.4, giving visible vertical range about [-0.4z, 0.4z]. With ``fx == fy``,
    square pixels are preserved while wider images see more horizontally, so
    sampling x/y from [-1, 1] intentionally mixes visible, partially visible,
    and offscreen Gaussians.
    """
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)

    means = torch.empty(n_gauss, 3, device=device).uniform_(
        -1.0, 1.0, generator=generator
    )
    means[:, 2] = means[:, 2].abs() + 1.0  # all gaussians in front

    quats = torch.nn.functional.normalize(
        torch.randn(n_gauss, 4, device=device, generator=generator), dim=-1
    )
    scales = torch.empty(n_gauss, 3, device=device).uniform_(
        0.02, 0.10, generator=generator
    )
    opacities = torch.empty(n_gauss, device=device).uniform_(
        0.4, 0.95, generator=generator
    )
    colors = torch.rand(n_gauss, 3, device=device, generator=generator)

    fy = float(height) / (2.0 * 0.4)
    K = torch.tensor(
        [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]
    viewmats = torch.eye(4, device=device)[None]
    return means, quats, scales, opacities, colors, viewmats, K


def _render(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    K,
    width: int,
    height: int,
    *,
    # Keep render options keyword-only to avoid ambiguous positional arguments.
    tile_size: int = 16,
    macro_tile: bool,
):
    return gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        width,
        height,
        packed=False,
        tile_size=tile_size,
        macro_tile=macro_tile,
    )


# --------------------------------------------------------------------------
# Parity: macro_tile=True vs macro_tile=False produce matching pixels
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "n_gauss,image_size",
    [(64, (128, 128)), (512, (256, 192)), (2048, (480, 320))],
    ids=["small", "mid", "wide"],
)
def test_rasterize_macro_tile_parity_ewa(n_gauss: int, image_size: Tuple[int, int]):
    """Rendered-output parity between macro_tile=True and macro_tile=False on
    the standard 3DGS EWA path."""

    width, height = image_size
    tile_size = 16
    scene = _make_scene(n_gauss, width, height)
    out_ref, alphas_ref, _ = _render(
        *scene,
        width=width,
        height=height,
        macro_tile=False,
        tile_size=tile_size,
    )
    out_mt, alphas_mt, _ = _render(
        *scene,
        width=width,
        height=height,
        macro_tile=True,
        tile_size=tile_size,
    )
    torch.testing.assert_close(out_mt, out_ref, rtol=1e-4, atol=2e-3)
    torch.testing.assert_close(alphas_mt, alphas_ref, rtol=1e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_segmented_sort_overflow_parity():
    """A single 8,193-item segment exercises segmented-sort overflow merging."""
    device = "cuda"
    n_gauss = 8_193
    width = height = 64

    generator = torch.Generator(device=device).manual_seed(42)
    depths = torch.linspace(1.0, 2.0, n_gauss, device=device)
    depths = depths[torch.randperm(n_gauss, device=device, generator=generator)]
    means = torch.zeros(n_gauss, 3, device=device)
    means[:, 2] = depths
    quats = torch.zeros(n_gauss, 4, device=device)
    quats[:, 0] = 1.0
    scales = torch.full((n_gauss, 3), 0.01, device=device)
    opacities = torch.full((n_gauss,), 0.01, device=device)
    colors = torch.stack(
        (depths - 1.0, 2.0 - depths, torch.full_like(depths, 0.25)), dim=-1
    )
    viewmats = torch.eye(4, device=device)[None]
    K = torch.tensor(
        [[80.0, 0.0, 32.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]

    scene = means, quats, scales, opacities, colors, viewmats, K
    out_ref, alphas_ref, _ = _render(
        *scene, width=width, height=height, macro_tile=False
    )
    out_mt, alphas_mt, _ = _render(*scene, width=width, height=height, macro_tile=True)

    torch.testing.assert_close(out_mt, out_ref, rtol=1e-4, atol=2e-3)
    torch.testing.assert_close(alphas_mt, alphas_ref, rtol=1e-4, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "n_gauss,image_size",
    [(128, (128, 128)), (512, (256, 192))],
    ids=["small", "mid"],
)
def test_rasterize_macro_tile_parity_ewa_two_images(
    n_gauss: int, image_size: Tuple[int, int]
):
    """Two-camera parity covers cumulative MT offsets across images."""

    width, height = image_size
    tile_size = 16
    means, quats, scales, opacities, colors, viewmats, K = _make_scene(
        n_gauss, width, height
    )
    viewmats = viewmats.repeat(2, 1, 1)
    K = K.repeat(2, 1, 1)
    yaw = torch.tensor(0.25, device=viewmats.device)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    viewmats[1, 0, 0] = cos_yaw
    viewmats[1, 0, 2] = sin_yaw
    viewmats[1, 2, 0] = -sin_yaw
    viewmats[1, 2, 2] = cos_yaw
    viewmats[1, 0, 3] = 0.35
    viewmats[1, 1, 3] = -0.15

    out_ref, alphas_ref, _ = _render(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        width=width,
        height=height,
        macro_tile=False,
        tile_size=tile_size,
    )
    out_mt, alphas_mt, _ = _render(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        width=width,
        height=height,
        macro_tile=True,
        tile_size=tile_size,
    )

    assert out_mt.shape == (2, height, width, 3)
    assert alphas_mt.shape == (2, height, width, 1)
    torch.testing.assert_close(out_mt, out_ref, rtol=1e-4, atol=2e-3)
    torch.testing.assert_close(alphas_mt, alphas_ref, rtol=1e-4, atol=2e-3)


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize(
    "conic",
    [
        pytest.param((1.0, 1.0, 1.0), id="zero-disc"),
        pytest.param((1.0, 2.0, 1.0), id="positive-disc"),
        pytest.param((1.0, float("nan"), 1.0), id="nan-disc"),
    ],
)
def test_ellipse_intersections_reject_invalid_discriminant(conic):
    """Both production intersection paths reject invalid conics before
    dividing by their discriminant."""

    device = "cuda"
    means2d = torch.tensor([[[32.0, 32.0]]], device=device)
    radii = torch.tensor([[[8, 8]]], device=device, dtype=torch.int32)
    depths = torch.tensor([[1.0]], device=device)
    conics = torch.tensor([[conic]], device=device)
    opacities = torch.tensor([[0.9]], device=device)
    tile_size = 16
    tile_width = tile_height = 4

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        conics=conics,
        opacities=opacities,
    )
    torch.testing.assert_close(tiles_per_gauss, torch.zeros_like(tiles_per_gauss))
    assert isect_ids.numel() == 0
    assert flatten_ids.numel() == 0

    isect_offsets, flatten_ids = isect_tiles_macro_binning(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        tile_size,
        tile_width,
        tile_height,
    )
    torch.testing.assert_close(isect_offsets, torch.zeros_like(isect_offsets))
    assert flatten_ids.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_zero_visible_gaussians():
    """Positive-depth Gaussians project offscreen and hit no render tiles."""

    device = "cuda"
    n = 4
    means = torch.tensor([[100.0, 100.0, 1.5]] * n, device=device)
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.full((n, 3), 0.05, device=device)
    opacities = torch.full((n,), 0.9, device=device)
    colors = torch.rand(n, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    # K is required by the rasterization API; this focal length still projects
    # the positive-depth Gaussians far outside the image.
    K = torch.tensor(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]

    out, alphas, _ = gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        128,
        64,
        packed=False,
        macro_tile=True,
    )
    assert out.shape == (1, 64, 128, 3)
    assert alphas.shape == (1, 64, 128, 1)
    assert torch.all(alphas == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_image_smaller_than_one_macrotile():
    """Image smaller than one macro-tile (8x4 render-tiles with
    tile_size=16 -> 128x64 px). Exercises the rt_col/rt_row
    out-of-range mask in the render-tile kernels."""
    tile_size = 16
    macro_tile_width = 8
    macro_tile_height = 4
    width, height = 96, 48
    tile_cols = (width + tile_size - 1) // tile_size
    tile_rows = (height + tile_size - 1) // tile_size
    assert tile_cols < macro_tile_width
    assert tile_rows < macro_tile_height
    scene = _make_scene(128, width, height)
    out_mt, _, _ = _render(
        *scene, width=width, height=height, tile_size=tile_size, macro_tile=True
    )
    out_ref, _, _ = _render(
        *scene, width=width, height=height, tile_size=tile_size, macro_tile=False
    )
    torch.testing.assert_close(out_mt, out_ref, rtol=1e-4, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_all_gaussians_behind_camera():
    """Every Gaussian behind the camera (z < near_plane) — projection
    produces radii=0 for all, MT visibility filter culls everything,
    output is the background."""
    device = "cuda"
    n = 16
    means = torch.empty(n, 3, device=device).uniform_(-1.0, 1.0)
    means[:, 2] = -1.0  # all behind camera
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
    scales = torch.empty(n, 3, device=device).uniform_(0.02, 0.10)
    opacities = torch.full((n,), 0.9, device=device)
    colors = torch.rand(n, 3, device=device)
    viewmats = torch.eye(4, device=device)[None]
    K = torch.tensor(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        device=device,
    )[None]
    out, alphas, _ = gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        128,
        64,
        packed=False,
        macro_tile=True,
    )
    assert torch.all(alphas == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_image_not_multiple_of_macrotile_block():
    """Image dimensions are NOT a multiple of MACRO_TILE_WIDTH/HEIGHT *
    tile_size. The main parity parametrization already covers a partial right
    edge (480x320); this focused case adds partial bottom-edge coverage."""

    tile_size = 16
    macro_tile_width = 8
    macro_tile_height = 4
    macro_tile_px_width = tile_size * macro_tile_width
    macro_tile_px_height = tile_size * macro_tile_height
    width, height = 153, 97
    assert width % macro_tile_px_width != 0
    assert height % macro_tile_px_height != 0
    scene = _make_scene(256, width, height)
    out_mt, _, _ = _render(
        *scene, width=width, height=height, tile_size=tile_size, macro_tile=True
    )
    out_ref, _, _ = _render(
        *scene, width=width, height=height, tile_size=tile_size, macro_tile=False
    )
    torch.testing.assert_close(out_mt, out_ref, rtol=1e-4, atol=2e-3)


# --------------------------------------------------------------------------
# Out-of-scope error gating
# --------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_macro_tile_with_packed_raises():
    """packed=True is not supported in this stage — raise so callers
    don't silently fall through."""
    scene = _make_scene(32, 64, 64)
    with pytest.raises(RuntimeError, match="packed"):
        gsplat.rasterization(*scene, width=64, height=64, packed=True, macro_tile=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.skipif(not gsplat.has_3dgut(), reason="3DGUT support isn't built in")
def test_macro_tile_with_ut_raises():
    """3DGUT (with_ut) is not yet supported — raise."""
    scene = _make_scene(32, 64, 64)
    with pytest.raises(RuntimeError, match="3DGUT"):
        gsplat.rasterization(
            *scene,
            width=64,
            height=64,
            packed=False,
            with_ut=True,
            macro_tile=True,
        )
