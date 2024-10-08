"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch
from typing_extensions import Literal, assert_never

from gsplat._helper import load_test_data

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.fixture
def test_data():
    torch.manual_seed(42)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_opacity_integral(test_data):
    from gsplat import quat_scale_to_covar_preci
    from gsplat.cuda._torch_impl import integral_opacity, integral_opacity_numerical
    import torch.nn.functional as F

    torch.manual_seed(42)
    N = 100
    Ks = test_data["Ks"][:N]
    viewmats = test_data["viewmats"][:N]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"][:N]
    scales = test_data["scales"][:N] * 100.0
    means = test_data["means"][:N]
    C = len(Ks)
    # gaussian attributes
    _, precis = quat_scale_to_covar_preci(
        quats, scales, compute_covar=False, compute_preci=True, triu=False
    )
    densities = torch.rand(N, device=device)
    # ray attributes
    camera_ids = torch.randint(0, C, (N,), device=device)
    pixel_ids = torch.randint(0, width * height, (N,), device=device)
    pixel_ids_x = pixel_ids % width
    pixel_ids_y = pixel_ids // width
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1) + 0.5  # [N, 2]
    camtoworlds = torch.linalg.inv(viewmats)[camera_ids]  # [N, 4, 4]
    Ks = Ks[camera_ids]  # [N, 3, 3]
    uvs = torch.stack(
        [
            (pixel_coords[:, 0] - Ks[:, 0, 2]) / Ks[:, 0, 0],
            (pixel_coords[:, 1] - Ks[:, 1, 2]) / Ks[:, 1, 1],
        ],
        dim=-1,
    )  # [N, 2]
    camera_dirs = F.pad(uvs, (0, 1), value=1.0)  # [N, 3]
    camera_dirs = F.normalize(camera_dirs, dim=-1)
    viewdirs = (camera_dirs[:, None, :] * camtoworlds[:, :3, :3]).sum(dim=-1)  # [N, 3]
    origins = camtoworlds[:, :3, -1]  # [N, 3]
    # analytical vs numerical integration
    tmaxs = torch.rand(N, device=device) * 1e3
    tmins = -torch.rand(N, device=device) * 1e3
    opacities = integral_opacity(
        means, precis, densities, origins, viewdirs, inf=False, tmins=tmins, tmaxs=tmaxs
    )
    _opacities = integral_opacity_numerical(
        means, precis, densities, origins, viewdirs, tmins=tmins, tmaxs=tmaxs
    )
    torch.testing.assert_close(opacities, _opacities, rtol=1e-3, atol=1e-3)
    # test inf integral
    tmins = torch.full((N,), -1e5, device=device)
    tmaxs = torch.full((N,), 1e5, device=device)
    opacities = integral_opacity(
        means, precis, densities, origins, viewdirs, inf=False, tmins=tmins, tmaxs=tmaxs
    )
    _opacities = integral_opacity(means, precis, densities, origins, viewdirs, inf=True)
    torch.testing.assert_close(opacities, _opacities, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_density_to_opacity(test_data):
    from gsplat.cuda._wrapper import _make_lazy_cuda_func

    torch.manual_seed(42)

    N = 10000
    densities = torch.rand(N, device=device)
    rays_o = torch.randn(N, 3, device=device)
    rays_d = torch.randn(N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
    means = torch.randn(N, 3, device=device)
    # covars_tmp = torch.rand(N, 3, 3, device=device)
    # covars = torch.matmul(covars_tmp, covars_tmp.transpose(1, 2))
    # precisions = torch.inverse(covars).contiguous()
    precisions = torch.eye(3, device=device).expand(N, 3, 3).contiguous()
    print(precisions.shape)

    densities.requires_grad = True
    means.requires_grad = True
    precisions.requires_grad = True

    opacities = _make_lazy_cuda_func("density_to_opacity_fwd")(
        densities, rays_o, rays_d, means, precisions
    )
    v_opacities = torch.randn_like(opacities)
    v_densities, v_means, v_precisions = _make_lazy_cuda_func("density_to_opacity_bwd")(
        densities, rays_o, rays_d, means, precisions, v_opacities
    )

    def _impl(densities, rays_o, rays_d, means, precisions):
        mu = means - rays_o
        rr = torch.einsum("mi,mij,mj->m", rays_d, precisions, rays_d)  # [M]
        # rr.register_hook(lambda x: print("rr", x))
        dr = torch.einsum("mi,mij,mj->m", mu, precisions, rays_d)  # [M]
        # dr.register_hook(lambda x: print("dr", x))
        dd = torch.einsum("mi,mij,mj->m", mu, precisions, mu)  # [M]
        # dd.register_hook(lambda x: print("dd", x))
        rr_safe = torch.clamp(rr, min=1e-6)
        # bb = -dr / rr  # [M]
        cc = dd - dr / rr_safe * dr  # [M]
        # cc.register_hook(lambda x: print("cc", x))
        transmittance = torch.exp(
            -densities * torch.exp(-0.5 * cc) * torch.sqrt(0.5 * math.pi / rr_safe) * 2
        )  # [M]
        opacities = 1.0 - transmittance
        opacities = torch.where(rr > 1e-6, opacities, torch.zeros_like(opacities))
        return opacities

    _opacities = _impl(densities, rays_o, rays_d, means, precisions)
    torch.testing.assert_close(opacities, _opacities)

    _v_densities, _v_means, _v_precisions = torch.autograd.grad(
        (v_opacities * _opacities).sum(), (densities, means, precisions)
    )
    torch.testing.assert_close(v_densities, _v_densities, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v_means, _v_means, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v_precisions, _v_precisions, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3])
def test_density_rasterize_to_pixels(test_data, channels: int):
    from gsplat.cuda._torch_impl import _rasterize_to_pixels
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
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
    densities = torch.rand(len(means), device=device) * 1.0

    C = len(Ks)
    colors = torch.randn(C, len(means), channels, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    covars, precis = quat_scale_to_covar_preci(
        quats, scales, compute_preci=True, triu=True
    )

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    densities = densities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    means2d.requires_grad = True
    conics.requires_grad = True
    colors.requires_grad = True
    densities.requires_grad = True
    backgrounds.requires_grad = True
    
    # forward
    render_colors, render_alphas = rasterize_to_pixels(
        means2d,
        conics,
        colors,
        torch.tensor([], device=device),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        backgrounds=backgrounds,
        densities=densities,
    )
    _render_colors, _render_alphas = _rasterize_to_pixels(
        means2d,
        conics,
        colors,
        torch.tensor([], device=device),
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        backgrounds=backgrounds,
        densities=densities,
    )
    torch.testing.assert_close(render_colors, _render_colors)
    torch.testing.assert_close(render_alphas, _render_alphas)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means2d, v_conics, v_colors, v_densities, v_backgrounds = torch.autograd.grad(
        (render_colors * v_render_colors).sum()
        + (render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, densities, backgrounds),
    )
    (
        _v_means2d,
        _v_conics,
        _v_colors,
        _v_densities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means2d, conics, colors, densities, backgrounds),
    )
    torch.testing.assert_close(v_means2d, _v_means2d, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(v_conics, _v_conics, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(v_densities, _v_densities, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)
