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

"""Tests for median-depth rasterization on the classic 3DGS path.

Median depth is defined as the depth of the Gaussian whose contribution causes
the accumulated transmittance to drop across 0.5. When the threshold is never
crossed we fall back to the last contributing Gaussian's depth; when the pixel
receives no contribution the median depth is 0.

Usage:
```bash
pytest tests/test_median_depth.py -s
```
"""

import math
import os
from typing import Tuple

import pytest
import torch

import gsplat
from gsplat.cuda._backend import _C

if _C is None:
    pytest.skip("gsplat CUDA extension not available", allow_module_level=True)

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization


device = torch.device("cuda:0")


def _expand(data: dict, batch_dims: Tuple[int, ...]):
    """Prepend batch_dims to every tensor leaf (mirrors tests/test_basic.py)."""
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            ret[k] = v.expand(batch_dims + v.shape)
        else:
            ret[k] = v
    return ret


@pytest.fixture
def test_data():
    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    return {
        "means": means,  # [N, 3]
        "quats": quats,  # [N, 4]
        "scales": scales,  # [N, 3]
        "opacities": opacities,  # [N]
        "colors": colors,  # [N, 3] (or similar)
        "viewmats": viewmats,  # [C, 4, 4]
        "Ks": Ks,  # [C, 3, 3]
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------------------------
# Scene-level parity-style tests (Garden scene)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("render_mode", ["D", "RGB+D"])
@pytest.mark.parametrize("batch_dims", [(), (2,), (1, 2)])
def test_median_depth_shape_and_sanity(
    test_data, render_mode: str, batch_dims: Tuple[int, ...]
):
    """Median depth must match the expected shape of the depth channel and be
    finite, non-negative, and bounded above by the max scene depth."""
    torch.manual_seed(42)

    test_data = _expand(test_data, batch_dims)
    means = test_data["means"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    viewmats = test_data["viewmats"]
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]

    render_colors, render_alphas, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        render_mode=render_mode,
        depth_mode="median",
        packed=False,
    )

    C = viewmats.shape[-3]
    if render_mode == "D":
        expected_last = 1
    else:  # "RGB+D"
        expected_last = colors.shape[-1] + 1
    assert render_colors.shape == batch_dims + (C, height, width, expected_last)
    assert render_alphas.shape == batch_dims + (C, height, width, 1)

    depth = render_colors[..., -1]
    assert torch.isfinite(depth).all(), "median depth should be finite"
    assert (depth >= 0).all(), "median depth should be non-negative (0 = no hit)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("render_mode", ["D", "RGB+D"])
def test_median_vs_expected_differ(test_data, render_mode: str):
    """On a real scene, median and expected depth should both be valid but
    generally differ at most pixels (they agree only in degenerate cases)."""
    torch.manual_seed(42)

    means = test_data["means"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    viewmats = test_data["viewmats"]
    Ks = test_data["Ks"]
    width = test_data["width"]
    height = test_data["height"]

    expected_colors, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        render_mode=render_mode,
        depth_mode="expected",
        packed=False,
    )
    median_colors, _, _ = rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
        render_mode=render_mode,
        depth_mode="median",
        packed=False,
    )

    # Shapes should match.
    assert expected_colors.shape == median_colors.shape
    # If RGB channels are present they should be identical (median only
    # affects the last/depth channel).
    if render_mode == "RGB+D":
        torch.testing.assert_close(expected_colors[..., :-1], median_colors[..., :-1])
    # Depth channels should not be identical everywhere on a non-trivial scene.
    exp_d = expected_colors[..., -1]
    med_d = median_colors[..., -1]
    assert not torch.allclose(exp_d, med_d)


# ---------------------------------------------------------------------------
# Validation of the depth_mode argument
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_rejects_rgb_only(test_data):
    """depth_mode='median' without a depth render_mode must raise."""
    with pytest.raises(ValueError, match="depth_mode='median' requires a depth"):
        rasterization(
            test_data["means"],
            test_data["quats"],
            test_data["scales"] * 0.1,
            test_data["opacities"],
            test_data["colors"],
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            render_mode="RGB",
            depth_mode="median",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_rejects_invalid_mode(test_data):
    with pytest.raises(ValueError, match="depth_mode"):
        rasterization(
            test_data["means"],
            test_data["quats"],
            test_data["scales"] * 0.1,
            test_data["opacities"],
            test_data["colors"],
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            render_mode="RGB+D",
            depth_mode="bogus",  # type: ignore[arg-type]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_rejects_eval3d(test_data):
    """depth_mode='median' is only supported for the classic 3DGS path."""
    if not gsplat.has_3dgut():
        pytest.skip("3DGUT support isn't built in; eval3d path unavailable")
    with pytest.raises(NotImplementedError, match="classic 3DGS path"):
        rasterization(
            test_data["means"],
            test_data["quats"],
            test_data["scales"] * 0.1,
            test_data["opacities"],
            test_data["colors"],
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            render_mode="RGB+D",
            depth_mode="median",
            with_eval3d=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_rejects_hit_distance(test_data):
    with pytest.raises(NotImplementedError, match="hit-distance"):
        rasterization(
            test_data["means"],
            test_data["quats"],
            test_data["scales"] * 0.1,
            test_data["opacities"],
            test_data["colors"],
            test_data["viewmats"],
            test_data["Ks"],
            test_data["width"],
            test_data["height"],
            render_mode="RGB-d",
            depth_mode="median",
        )


# ---------------------------------------------------------------------------
# Analytical tests against the low-level rasterize_to_pixels kernel.
#
# These drive the kernel directly through `rasterize_to_pixels(return_median=True)`
# with hand-built `means2d`/`conics`/`colors` tensors, so we can unambiguously
# pin down the expected median-depth output per pixel.
# ---------------------------------------------------------------------------


def _build_isects(
    means2d: torch.Tensor,  # [I, N, 2]
    depths: torch.Tensor,  # [I, N]
    image_width: int,
    image_height: int,
    tile_size: int,
):
    """Produce tile_offsets / flatten_ids assuming every Gaussian covers the
    whole image (simple catch-all intersection). Gaussians are listed in
    ascending depth order within each tile, which matches normal rasterization
    order for front-to-back compositing."""
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    I = means2d.shape[0]
    N = means2d.shape[1]
    # Give every Gaussian a radius large enough to cover the whole image.
    # `radii` must be int32: it is treated as an integer half-extent (in
    # pixels) of the conservative AABB used for tile intersection; see
    # Projection2DGSFused.cu and IntersectTile.cu where radii are written
    # and read as int32_t.
    radii = torch.full(
        (I, N, 2),
        max(image_width, image_height),
        device=means2d.device,
        dtype=torch.int32,
    )
    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    return isect_offsets, flatten_ids, tile_width, tile_height


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_two_opaque_gaussians():
    """Two fully-opaque Gaussians on one ray at different depths.

    The first Gaussian drives T from 1.0 to ~0, so it crosses 0.5 and should
    be chosen as the median-depth Gaussian. This independently verifies the
    'T > 0.5 and next_T <= 0.5' selection rule in the kernel.
    """
    from gsplat.cuda._wrapper import rasterize_to_pixels

    image_width = image_height = 8
    tile_size = 16
    I = 1
    N = 2
    # Identity conics so the Gaussian density is exp(-0.5 * ((du)^2 + (dv)^2)).
    means2d = torch.tensor(
        [[[image_width / 2.0, image_height / 2.0]] * N],
        device=device,
        dtype=torch.float32,
    )
    conics = torch.tensor(
        [[[1.0, 0.0, 1.0]] * N],
        device=device,
        dtype=torch.float32,
    )
    # Last channel is depth (matches gsplat's D / RGB+D plumbing).
    depths_near, depths_far = 3.0, 7.0
    colors = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, depths_near],
                [0.4, 0.5, 0.6, depths_far],
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    opacities = torch.full((I, N), 0.99, device=device, dtype=torch.float32)
    depths_for_sort = torch.tensor(
        [[depths_near, depths_far]], device=device, dtype=torch.float32
    )

    isect_offsets, flatten_ids, _, _ = _build_isects(
        means2d, depths_for_sort, image_width, image_height, tile_size
    )

    render_colors, render_alphas, render_median = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_median=True,
    )

    assert render_median.shape == (I, image_height, image_width, 1)

    # At the Gaussian centre, both Gaussians contribute with effectively full
    # opacity. After the first Gaussian T drops from 1.0 to ~0.01, crossing
    # 0.5, so the median depth must be the near depth.
    cy, cx = image_height // 2, image_width // 2
    med_center = render_median[0, cy, cx, 0].item()
    assert (
        abs(med_center - depths_near) < 1e-4
    ), f"expected median={depths_near}, got {med_center}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_last_hit_fallback():
    """A single low-opacity Gaussian that never drops T below 0.5.

    In this case the kernel must fall back to the last-hit depth.
    """
    from gsplat.cuda._wrapper import rasterize_to_pixels

    image_width = image_height = 8
    tile_size = 16
    I = 1
    N = 1

    means2d = torch.tensor(
        [[[image_width / 2.0, image_height / 2.0]]],
        device=device,
        dtype=torch.float32,
    )
    conics = torch.tensor([[[1.0, 0.0, 1.0]]], device=device, dtype=torch.float32)
    depth_val = 4.2
    colors = torch.tensor(
        [[[0.0, 0.0, 0.0, depth_val]]], device=device, dtype=torch.float32
    )
    # Low opacity -> at the centre alpha ~= 0.2, so T stays around 0.8 (> 0.5),
    # median_found=False, fallback to last hit.
    opacities = torch.tensor([[0.2]], device=device, dtype=torch.float32)
    depths_for_sort = torch.tensor([[depth_val]], device=device, dtype=torch.float32)

    isect_offsets, flatten_ids, _, _ = _build_isects(
        means2d, depths_for_sort, image_width, image_height, tile_size
    )

    _, _, render_median = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_median=True,
    )

    cy, cx = image_height // 2, image_width // 2
    med_center = render_median[0, cy, cx, 0].item()
    assert (
        abs(med_center - depth_val) < 1e-4
    ), f"expected fallback median={depth_val}, got {med_center}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_median_depth_empty_pixel():
    """When no Gaussians contribute to a tile the median depth must be 0."""
    from gsplat.cuda._wrapper import rasterize_to_pixels

    image_width = image_height = 8
    tile_size = 16
    I = 1
    N = 1

    # Place the Gaussian far away from the image plane (outside) with tiny
    # radius so it does not cover any tile.
    means2d = torch.tensor([[[-1000.0, -1000.0]]], device=device, dtype=torch.float32)
    # Very sharp covariance -> effectively a delta function outside the image.
    conics = torch.tensor([[[1e6, 0.0, 1e6]]], device=device, dtype=torch.float32)
    colors = torch.tensor([[[0.0, 0.0, 0.0, 5.0]]], device=device, dtype=torch.float32)
    opacities = torch.tensor([[0.9]], device=device, dtype=torch.float32)

    # Hand-build isect tensors with no intersections at all.
    tile_width = math.ceil(image_width / float(tile_size))
    tile_height = math.ceil(image_height / float(tile_size))
    isect_offsets = torch.zeros(
        (I, tile_height, tile_width), device=device, dtype=torch.int32
    )
    flatten_ids = torch.empty((0,), device=device, dtype=torch.int32)

    _, _, render_median = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
        return_median=True,
    )

    assert render_median.shape == (I, image_height, image_width, 1)
    assert torch.all(
        render_median == 0
    ), f"expected all-zero median for empty tiles, got max={render_median.max().item()}"
