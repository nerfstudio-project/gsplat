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
def test_density_to_opacity():
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
    print (precisions.shape)

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