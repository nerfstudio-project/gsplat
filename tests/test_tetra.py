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
def test_ray_tetra_intersection():
    from gsplat.cuda._torch_impl import _ray_tetra_intersection
    from gsplat.cuda._wrapper import ray_tetra_intersection

    torch.manual_seed(42)

    N = 100000
    rays_o = torch.randn(N, 3, device=device)
    rays_d = torch.randn(N, 3, device=device)
    rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)
    vertices = torch.randn(N, 4, 3, device=device)
    vertices.requires_grad = True

    entry_face_ids, exit_face_ids, t_entrys, t_exits = ray_tetra_intersection(
        rays_o, rays_d, vertices
    )
    _entry_face_ids, _exit_face_ids, _t_entrys, _t_exits, hit = _ray_tetra_intersection(
        rays_o, rays_d, vertices
    )

    torch.testing.assert_close(entry_face_ids, _entry_face_ids)
    torch.testing.assert_close(exit_face_ids, _exit_face_ids)

    # if intersection happens, check if the t values are correct
    isect_ids = torch.where((entry_face_ids >= 0) & (exit_face_ids >= 0))[0]
    torch.testing.assert_close(
        t_entrys[isect_ids], _t_entrys[isect_ids], atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        t_exits[isect_ids], _t_exits[isect_ids], atol=2e-5, rtol=1e-6
    )

    # if intersection happens, check if the gradient of vertices are correct
    v_t_entrys = torch.randn_like(t_entrys)
    v_t_exits = torch.randn_like(t_exits)
    v_vertices = torch.autograd.grad(
        (
            t_entrys[isect_ids] * v_t_entrys[isect_ids]
            + t_exits[isect_ids] * v_t_exits[isect_ids]
        ).sum(),
        vertices,
        retain_graph=True,
        allow_unused=True,
    )[0]
    _v_vertices = torch.autograd.grad(
        (
            _t_entrys[isect_ids] * v_t_entrys[isect_ids]
            + _t_exits[isect_ids] * v_t_exits[isect_ids]
        ).sum(),
        vertices,
        retain_graph=True,
        allow_unused=True,
    )[0]
    torch.testing.assert_close(v_vertices, _v_vertices, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("channels", [3])
def test_rasterize_to_pixels(test_data, channels: int):
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
    opacities = test_data["opacities"]
    C = len(Ks)
    colors = torch.randn(C, len(means), channels, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    means.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    colors.requires_grad = True
    opacities.requires_grad = True
    backgrounds.requires_grad = True

    covars, precis = quat_scale_to_covar_preci(
        quats, scales, compute_preci=True, triu=True
    )

    # Project Gaussians to 2D
    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, width, height
    )
    opacities = opacities.repeat(C, 1)

    # Identify intersecting tiles
    tile_size = 16 if channels <= 32 else 4
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # means2d.requires_grad = True
    # conics.requires_grad = True
    # colors.requires_grad = True
    # opacities.requires_grad = True
    # backgrounds.requires_grad = True

    enable_culling = True
    tquats = torch.rand_like(quats)
    tscales = torch.mean(scales, dim=1)

    from gsplat.cuda._torch_impl import _quat_scale_to_matrix

    tvertices = torch.tensor(
        [
            [math.sqrt(8 / 9), 0, -1 / 3],
            [-math.sqrt(2 / 9), math.sqrt(2 / 3), -1 / 3],
            [-math.sqrt(2 / 9), -math.sqrt(2 / 3), -1 / 3],
            [0, 0, 1],
        ],
        device=means.device,
        dtype=means.dtype,
    )  # [4, 3]
    rotmats = _quat_scale_to_matrix(tquats, tscales[:, None])  # [N, 3, 3]
    tvertices = torch.einsum("nij,kj->nki", rotmats, tvertices)  # [N, 4, 3]
    tvertices = tvertices + means[:, None, :]  # [N, 4, 3]

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
        flatten_ids,
        backgrounds=backgrounds,
        enable_culling=enable_culling,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        tvertices=tvertices,
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
        flatten_ids,
        backgrounds=backgrounds,
        enable_culling=enable_culling,
        camtoworlds=torch.linalg.inv(viewmats),
        Ks=Ks,
        means3d=means,
        precis=precis,
        tvertices=tvertices,
    )
    # torch.testing.assert_close(render_colors, _render_colors)
    # torch.testing.assert_close(render_alphas, _render_alphas)

    # backward
    v_render_colors = torch.randn_like(render_colors)
    v_render_alphas = torch.randn_like(render_alphas)

    v_means, v_quats, v_scales, v_colors, v_opacities, v_backgrounds = (
        torch.autograd.grad(
            (render_colors * v_render_colors).sum()
            + (render_alphas * v_render_alphas).sum(),
            (means, quats, scales, colors, opacities, backgrounds),
            retain_graph=True,
        )
    )
    (
        _v_means,
        _v_quats,
        _v_scales,
        _v_colors,
        _v_opacities,
        _v_backgrounds,
    ) = torch.autograd.grad(
        (_render_colors * v_render_colors).sum()
        + (_render_alphas * v_render_alphas).sum(),
        (means, quats, scales, colors, opacities, backgrounds),
        retain_graph=True,
    )

    # torch.testing.assert_close(v_means, _v_means, rtol=5e-3, atol=5e-3)
    # torch.testing.assert_close(v_quats, _v_quats, rtol=5e-3, atol=5e-3)
    # torch.testing.assert_close(v_scales, _v_scales, rtol=5e-3, atol=5e-3)
    # torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
    # torch.testing.assert_close(v_opacities, _v_opacities, rtol=2e-3, atol=2e-3)
    # torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-3, atol=1e-3)


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
