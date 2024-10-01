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
