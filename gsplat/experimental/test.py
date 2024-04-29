"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


@pytest.fixture
def test_data():
    torch.manual_seed(42)

    # TODO: move it to somewhere in the repo?
    data_path = "/workspace/gsplat2/assets/data.npz"
    data = np.load(data_path)  # 3 cameras
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"]).to(device)
    Ks = torch.from_numpy(data["Ks"]).to(device)
    means = torch.from_numpy(data["means3d"]).to(device)
    scales = torch.rand((len(means), 3), device=device) * 0.1 + 0.1
    quats = F.normalize(torch.randn(len(means), 4), dim=-1).to(device)
    opacities = torch.rand((len(means),), device=device)
    colors = torch.rand((len(Ks), len(means), 3), device=device)

    return {
        "height": height,
        "width": width,
        "viewmats": viewmats,
        "Ks": Ks,
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }


@pytest.mark.parametrize("triu", [False, True])
def test_quat_scale_to_covar_perci(test_data, triu: bool):
    from gsplat.experimental.cuda import (
        _quat_scale_to_covar_perci,
        quat_scale_to_covar_perci,
    )

    quats = test_data["quats"]
    scales = test_data["scales"]
    N = len(quats)
    quats.requires_grad = True
    scales.requires_grad = True

    if triu:
        v_covars = torch.randn(N, 6).to(device)
        v_percis = torch.randn(N, 6).to(device)
    else:
        v_covars = torch.randn(N, 3, 3).to(device)
        v_percis = torch.randn(N, 3, 3).to(device)

    covars, percis = quat_scale_to_covar_perci(quats, scales, triu=triu)
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + percis * v_percis).sum(),
        (quats, scales),
        retain_graph=True,
    )

    _covars, _percis = _quat_scale_to_covar_perci(quats, scales, triu=triu)
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _percis * v_percis).sum(),
        (quats, scales),
        retain_graph=True,
    )

    torch.testing.assert_close(covars, _covars)
    torch.testing.assert_close(percis, _percis, rtol=1e-3, atol=1e-3)
    if not triu:
        I = torch.eye(3, device=device).expand(N, 3, 3)
        torch.testing.assert_close(torch.bmm(covars, percis), I)
        torch.testing.assert_close(torch.bmm(percis, covars), I)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-3, atol=1e-3)


def test_world_to_cam(test_data):
    from gsplat.experimental.cuda import (
        _world_to_cam,
        quat_scale_to_covar_perci,
        world_to_cam,
    )

    viewmats = test_data["viewmats"]
    means = test_data["means"]
    scales = test_data["scales"]
    quats = test_data["quats"]
    covars, _ = quat_scale_to_covar_perci(quats, scales)
    N = len(means)
    C = len(viewmats)

    means.requires_grad = True
    covars.requires_grad = True
    viewmats.requires_grad = True

    v_means_c = torch.randn(C, N, 3, device=device)
    v_covars_c = torch.randn(C, N, 3, 3, device=device)

    means_c, covars_c = world_to_cam(means, covars, viewmats)
    v_means, v_covars, v_viewmats = torch.autograd.grad(
        (means_c * v_means_c).sum() + (covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
        retain_graph=True,
    )

    _means_c, _covars_c = _world_to_cam(means, covars, viewmats)
    _v_means, _v_covars, _v_viewmats = torch.autograd.grad(
        (_means_c * v_means_c).sum() + (_covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
        retain_graph=True,
    )

    torch.testing.assert_close(means_c, _means_c)
    torch.testing.assert_close(covars_c, _covars_c)
    torch.testing.assert_close(v_means, _v_means)
    torch.testing.assert_close(v_covars, _v_covars)
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-4, atol=1e-5)


def test_persp_proj(test_data):
    from gsplat.experimental.cuda import (
        _persp_proj,
        persp_proj,
        quat_scale_to_covar_perci,
        world_to_cam,
    )

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    C = len(Ks)

    covars, _ = quat_scale_to_covar_perci(test_data["quats"], test_data["scales"])
    means, covars = world_to_cam(test_data["means"], covars, viewmats)
    C, N, _ = means.shape

    means.requires_grad = True
    covars.requires_grad = True

    v_means2d = torch.randn(C, N, 2, device=device)
    v_covars2d = torch.randn(C, N, 2, 2, device=device)

    means2d, covars2d = persp_proj(means, covars, Ks, width, height)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(),
        (means, covars),
        retain_graph=True,
    )

    _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(),
        (means, covars),
        retain_graph=True,
    )

    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-2, atol=2e-2)


def test_projection(test_data):
    from gsplat.experimental.cuda import (
        _projection,
        projection,
        quat_scale_to_covar_perci,
    )

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    covars, _ = quat_scale_to_covar_perci(quats, scales, triu=True)  # [N, 6]
    radii, means2d, depths, conics = projection(
        means, covars, viewmats, Ks, width, height
    )

    _covars, _ = quat_scale_to_covar_perci(quats, scales, triu=False)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics = _projection(
        means, _covars, viewmats, Ks, width, height
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0) & (_radii > 0)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)

    v_means2d = torch.randn_like(means2d) * radii[..., None]
    v_depths = torch.randn_like(depths) * radii
    v_conics = torch.randn_like(conics) * radii[..., None]

    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )

    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


def test_isect(test_data):
    from gsplat.experimental.cuda import (
        _isect_offset_encode,
        _isect_tiles,
        isect_offset_encode,
        isect_tiles,
    )

    torch.manual_seed(42)

    C = 3
    N = 1000

    width = 40
    height = 60
    means2d = torch.randn(C, N, 2, device=device) * width
    radii = torch.randint(0, width, (C, N), device=device, dtype=torch.int32)
    depths = torch.rand(C, N, device=device)

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)

    tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    _tiles_per_gauss, _isect_ids, _gauss_ids = _isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    _isect_offsets = _isect_offset_encode(_isect_ids, C, tile_width, tile_height)

    torch.testing.assert_close(tiles_per_gauss, _tiles_per_gauss)
    torch.testing.assert_close(isect_ids, _isect_ids)
    torch.testing.assert_close(gauss_ids, _gauss_ids)
    torch.testing.assert_close(isect_offsets, _isect_offsets)


def test_rasterize_to_pixels(test_data):
    from gsplat.experimental.cuda import _rendering, rendering

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"]
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    C = len(Ks)

    viewmats.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    means.requires_grad = True

    render_colors, render_alphas = rendering(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height
    )
    _render_colors, _render_alphas = _rendering(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height
    )
    assert torch.allclose(render_colors, _render_colors, atol=1e-2, rtol=1e-3)
    assert torch.allclose(render_alphas, _render_alphas, atol=1e-4, rtol=1e-4)
