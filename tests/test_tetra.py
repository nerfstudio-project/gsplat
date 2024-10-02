"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing_extensions import Literal, assert_never
import math

import pytest
import torch

from gsplat._helper import load_test_data

device = torch.device("cuda:0")

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
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
    
    entry_face_ids, exit_face_ids, t_entrys, t_exits = ray_tetra_intersection(rays_o, rays_d, vertices)
    _entry_face_ids, _exit_face_ids, _t_entrys, _t_exits = _ray_tetra_intersection(rays_o, rays_d, vertices)
    
    torch.testing.assert_close(entry_face_ids, _entry_face_ids)
    torch.testing.assert_close(exit_face_ids, _exit_face_ids)
    
    # if intersection happens, check if the t values are correct
    isect_ids = torch.where((entry_face_ids >= 0) & (exit_face_ids >= 0))[0]
    torch.testing.assert_close(t_entrys[isect_ids], _t_entrys[isect_ids], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(t_exits[isect_ids], _t_exits[isect_ids], atol=2e-5, rtol=1e-6)
    
    # if intersection happens, check if the gradient of vertices are correct
    v_t_entrys = torch.randn_like(t_entrys)
    v_t_exits = torch.randn_like(t_exits)
    v_vertices = torch.autograd.grad(
        (t_entrys[isect_ids] * v_t_entrys[isect_ids] + t_exits[isect_ids] * v_t_exits[isect_ids]).sum(), vertices, retain_graph=True, allow_unused=True
    )[0]
    _v_vertices = torch.autograd.grad(
        (_t_entrys[isect_ids] * v_t_entrys[isect_ids] + _t_exits[isect_ids] * v_t_exits[isect_ids]).sum(), vertices, retain_graph=True, allow_unused=True
    )[0]
    torch.testing.assert_close(v_vertices, _v_vertices, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("camera_model", ["pinhole", "ortho"])
def test_radius_culling(test_data, camera_model: Literal["pinhole", "ortho", "fisheye"]):
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._wrapper import (
        fully_fused_projection,
        isect_offset_encode,
        isect_tiles,
        quat_scale_to_covar_preci,
        rasterize_to_pixels,
    )

    torch.manual_seed(42)

    N = len(test_data["means"])
    Ks = test_data["Ks"]
    viewmats = test_data["viewmats"]
    height = test_data["height"]
    width = test_data["width"]
    quats = test_data["quats"]
    scales = test_data["scales"] * 0.1
    means = test_data["means"]
    opacities = test_data["opacities"]
    tquats = torch.randn(N, 4, device=device)
    tscales = torch.rand((N,), device=device) * 100.0
    
    C = len(Ks)
    colors = torch.randn(C, len(means), 3, device=device)
    backgrounds = torch.rand((C, colors.shape[-1]), device=device)

    covars, _ = quat_scale_to_covar_preci(quats, scales, compute_preci=False, triu=True)

    radii, means2d, depths, conics, compensations = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, tquats, tscales, width, height
    )
    _radii, _, _, _, _ = fully_fused_projection(
        means, covars, None, None, viewmats, Ks, tquats, tscales, width, height
    )
    torch.testing.assert_close(radii, _radii, atol=1e-6, rtol=1e-6)