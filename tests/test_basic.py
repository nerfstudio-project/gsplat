"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


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
    ) = load_test_data(device=device)
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


@pytest.mark.parametrize("triu", [False, True])
def test_quat_scale_to_covar_preci(test_data, triu: bool):
    from gsplat.cuda._torch_impl import _quat_scale_to_covar_preci
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci

    torch.manual_seed(42)

    quats = test_data["quats"]
    scales = test_data["scales"]
    quats.requires_grad = True
    scales.requires_grad = True

    # forward
    covars, precis = quat_scale_to_covar_preci(quats, scales, triu=triu)
    _covars, _precis = _quat_scale_to_covar_preci(quats, scales, triu=triu)
    torch.testing.assert_close(covars, _covars)
    torch.testing.assert_close(precis, _precis, rtol=1e-2, atol=1e-2)
    # if not triu:
    #     I = torch.eye(3, device=device).expand(len(covars), 3, 3)
    #     torch.testing.assert_close(torch.bmm(covars, precis), I)
    #     torch.testing.assert_close(torch.bmm(precis, covars), I)

    # backward
    v_covars = torch.randn_like(covars)
    v_precis = torch.randn_like(precis) * 0.01
    v_quats, v_scales = torch.autograd.grad(
        (covars * v_covars + precis * v_precis).sum(),
        (quats, scales),
    )
    _v_quats, _v_scales = torch.autograd.grad(
        (_covars * v_covars + _precis * v_precis).sum(),
        (quats, scales),
    )
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-1, atol=1e-1)
    torch.testing.assert_close(v_scales, _v_scales, rtol=3e-2, atol=1e-3)


def test_world_to_cam(test_data):
    from gsplat.cuda._torch_impl import _world_to_cam
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    viewmats = test_data["viewmats"]
    means = test_data["means"]
    scales = test_data["scales"]
    quats = test_data["quats"]
    covars, _ = quat_scale_to_covar_preci(quats, scales)
    means.requires_grad = True
    covars.requires_grad = True
    viewmats.requires_grad = True

    # forward
    means_c, covars_c = world_to_cam(means, covars, viewmats)
    _means_c, _covars_c = _world_to_cam(means, covars, viewmats)
    torch.testing.assert_close(means_c, _means_c)
    torch.testing.assert_close(covars_c, _covars_c)

    # backward
    v_means_c = torch.randn_like(means_c)
    v_covars_c = torch.randn_like(covars_c)
    v_means, v_covars, v_viewmats = torch.autograd.grad(
        (means_c * v_means_c).sum() + (covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )
    _v_means, _v_covars, _v_viewmats = torch.autograd.grad(
        (_means_c * v_means_c).sum() + (_covars_c * v_covars_c).sum(),
        (means, covars, viewmats),
    )
    torch.testing.assert_close(v_means, _v_means)
    torch.testing.assert_close(v_covars, _v_covars)
    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)


def test_persp_proj(test_data):
    from gsplat.cuda._torch_impl import _persp_proj
    from gsplat.cuda._wrapper import persp_proj, quat_scale_to_covar_preci, world_to_cam

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    covars, _ = quat_scale_to_covar_preci(test_data["quats"], test_data["scales"])
    means, covars = world_to_cam(test_data["means"], covars, viewmats)
    means.requires_grad = True
    covars.requires_grad = True

    # forward
    means2d, covars2d = persp_proj(means, covars, Ks, width, height)
    _means2d, _covars2d = _persp_proj(means, covars, Ks, width, height)
    torch.testing.assert_close(means2d, _means2d, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(covars2d, _covars2d, rtol=1e-1, atol=3e-2)

    # backward
    v_means2d = torch.randn_like(means2d)
    v_covars2d = torch.randn_like(covars2d)
    v_means, v_covars = torch.autograd.grad(
        (means2d * v_means2d).sum() + (covars2d * v_covars2d).sum(),
        (means, covars),
    )
    _v_means, _v_covars = torch.autograd.grad(
        (_means2d * v_means2d).sum() + (_covars2d * v_covars2d).sum(),
        (means, covars),
    )
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_covars, _v_covars, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("fused", [False, True])
def test_projection(test_data, fused: bool):
    from gsplat.cuda._torch_impl import _projection
    from gsplat.cuda._wrapper import projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

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

    # forward
    if fused:
        radii, means2d, depths, conics = projection(
            means, None, quats, scales, viewmats, Ks, width, height
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [N, 6]
        radii, means2d, depths, conics = projection(
            means, covars, None, None, viewmats, Ks, width, height
        )
    _covars, _ = quat_scale_to_covar_preci(quats, scales, triu=False)  # [N, 3, 3]
    _radii, _means2d, _depths, _conics = _projection(
        means, _covars, viewmats, Ks, width, height
    )

    # radii is integer so we allow for 1 unit difference
    valid = (radii > 0) & (_radii > 0)
    torch.testing.assert_close(radii, _radii, rtol=0, atol=1)
    torch.testing.assert_close(means2d[valid], _means2d[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(depths[valid], _depths[valid], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(conics[valid], _conics[valid], rtol=1e-4, atol=1e-4)

    # backward
    v_means2d = torch.randn_like(means2d) * radii[..., None]
    v_depths = torch.randn_like(depths) * radii
    v_conics = torch.randn_like(conics) * radii[..., None]
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d).sum()
        + (depths * v_depths).sum()
        + (conics * v_conics).sum(),
        (viewmats, quats, scales, means),
    )
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
    )

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-1, atol=2e-1)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-2, atol=6e-2)


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("sparse_grad", [False, True])
def test_projection_packed(test_data, fused: bool, sparse_grad: bool):
    from gsplat.cuda._wrapper import projection, quat_scale_to_covar_preci

    torch.manual_seed(42)

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

    # forward
    if fused:
        rindices, cindices, radii, means2d, depths, conics = projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
        )
        _radii, _means2d, _depths, _conics = projection(
            means, None, quats, scales, viewmats, Ks, width, height, packed=False
        )
    else:
        covars, _ = quat_scale_to_covar_preci(quats, scales, triu=True)  # [N, 6]
        rindices, cindices, radii, means2d, depths, conics = projection(
            means,
            covars,
            None,
            None,
            viewmats,
            Ks,
            width,
            height,
            packed=True,
            sparse_grad=sparse_grad,
        )
        _radii, _means2d, _depths, _conics = projection(
            means, covars, None, None, viewmats, Ks, width, height, packed=False
        )

    # recover packed tensors to full matrices for testing
    __radii = torch.sparse_coo_tensor(
        torch.stack([rindices, cindices]), radii, _radii.shape
    ).to_dense()
    __means2d = torch.sparse_coo_tensor(
        torch.stack([rindices, cindices]), means2d, _means2d.shape
    ).to_dense()
    __depths = torch.sparse_coo_tensor(
        torch.stack([rindices, cindices]), depths, _depths.shape
    ).to_dense()
    __conics = torch.sparse_coo_tensor(
        torch.stack([rindices, cindices]), conics, _conics.shape
    ).to_dense()
    sel = (__radii > 0) & (_radii > 0)
    torch.testing.assert_close(__radii[sel], _radii[sel], rtol=0, atol=1)
    torch.testing.assert_close(__means2d[sel], _means2d[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__depths[sel], _depths[sel], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(__conics[sel], _conics[sel], rtol=1e-4, atol=1e-4)

    # backward
    v_means2d = torch.randn_like(_means2d) * sel[..., None]
    v_depths = torch.randn_like(_depths) * sel
    v_conics = torch.randn_like(_conics) * sel[..., None]
    _v_viewmats, _v_quats, _v_scales, _v_means = torch.autograd.grad(
        (_means2d * v_means2d).sum()
        + (_depths * v_depths).sum()
        + (_conics * v_conics).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    v_viewmats, v_quats, v_scales, v_means = torch.autograd.grad(
        (means2d * v_means2d[sel]).sum()
        + (depths * v_depths[sel]).sum()
        + (conics * v_conics[sel]).sum(),
        (viewmats, quats, scales, means),
        retain_graph=True,
    )
    if sparse_grad:
        v_quats = v_quats.to_dense()
        v_scales = v_scales.to_dense()
        v_means = v_means.to_dense()

    torch.testing.assert_close(v_viewmats, _v_viewmats, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(v_quats, _v_quats, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_scales, _v_scales, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_means, _v_means, rtol=1e-3, atol=1e-3)


def test_isect(test_data):
    from gsplat.cuda._torch_impl import _isect_offset_encode, _isect_tiles
    from gsplat.cuda._wrapper import isect_offset_encode, isect_tiles

    torch.manual_seed(42)

    C, N = 3, 1000
    width, height = 40, 60
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
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        isect_offset_encode,
        isect_tiles,
        persp_proj,
        projection,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    colors = test_data["colors"]
    C = len(Ks)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    # Project Gaussians to 2D
    radii, means2d, depths, conics = projection(
        means, covars, None, None, viewmats, Ks, width, height
    )

    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, gauss_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        gauss_ids,
        backgrounds=backgrounds,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        opacities,
        width,
        height,
        tile_size,
        isect_offsets,
        gauss_ids,
        backgrounds=backgrounds,
    )
    assert torch.allclose(render_colors, _render_colors)
    assert torch.allclose(render_alphas, _render_alphas)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_opacities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, opacities, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_opacities, _v_opacities, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sh_degree", [0, 1, 2, 3])
def test_sh(test_data, sh_degree: int):
    from gsplat.cuda._torch_impl import _spherical_harmonics
    from gsplat.cuda._wrapper import spherical_harmonics

    torch.manual_seed(42)

    N = 1000
    coeffs = torch.randn(N, (3 + 1) ** 2, 3, device=device)
    dirs = F.normalize(torch.randn(N, 3, device=device), dim=-1)
    coeffs.requires_grad = True

    colors = spherical_harmonics(sh_degree, dirs, coeffs)
    _colors = _spherical_harmonics(sh_degree, dirs, coeffs)
    torch.testing.assert_close(colors, _colors, rtol=1e-4, atol=1e-4)

    v_colors = torch.randn_like(colors)

    v_coeffs = torch.autograd.grad(
        (colors * v_colors).sum(), coeffs, retain_graph=True
    )[0]
    _v_coeffs = torch.autograd.grad(
        (_colors * v_colors).sum(), coeffs, retain_graph=True
    )[0]
    torch.testing.assert_close(v_coeffs, _v_coeffs, rtol=1e-4, atol=1e-4)


#     # print("Mean Grad Diff: CUDA2 v.s. PyTorch")
#     # print("v_viewmats", torch.abs(v_viewmats - _v_viewmats).abs().mean())  # 0.0037
#     # print("v_quats", torch.abs(v_quats - _v_quats).abs().mean())  # 2e-7
#     # print("v_scales", torch.abs(v_scales - _v_scales).abs().mean())  # 5e-6
#     # print("v_means", torch.abs(v_means - _v_means).abs().mean())  # 3e-6

#     # __render_colors, __render_alphas = _rendering_gsplat(
#     #     means, quats, scales, opacities, colors, viewmats, Ks, width, height
#     # )
#     # __v_viewmats, __v_quats, __v_scales, __v_means = torch.autograd.grad(
#     #     (__render_colors * v_render_colors).sum()
#     #     + (__render_alphas * v_render_alphas).sum(),
#     #     (viewmats, quats, scales, means),
#     #     retain_graph=True,
#     # )

#     # print("Mean Grad Diff: Gsplat v.s. PyTorch")
#     # print("v_viewmats", torch.abs(__v_viewmats - _v_viewmats).abs().mean())  # 42.8834
#     # print("v_quats", torch.abs(__v_quats - _v_quats).abs().mean())  # 0.0002
#     # print("v_scales", torch.abs(__v_scales - _v_scales).abs().mean())  # 0.0054
#     # print("v_means", torch.abs(__v_means - _v_means).abs().mean())  # 0.0041
