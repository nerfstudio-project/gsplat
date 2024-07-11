"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_strategy():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy, NullStrategy

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means3d": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
        }
    ).to(device)
    activations = {
        "means3d": lambda x: x,
        "scales": lambda x: torch.exp(x),
        "quats": lambda x: F.normalize(x, p=2, dim=-1),
        "opacities": lambda x: torch.sigmoid(x),
        "colors": lambda x: torch.sigmoid(x),
    }
    lrs = {
        "means3d": 1e-3,
        "scales": 1e-3,
        "quats": 1e-3,
        "opacities": 1e-3,
        "colors": 1e-3,
    }
    optimizers = [
        torch.optim.Adam(
            [{"params": v, "lr": lrs[k], "name": k} for k, v in params.items()]
        )
    ]

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=activations["means3d"](params["means3d"]),
        quats=activations["quats"](params["quats"]),
        scales=activations["scales"](params["scales"]),
        opacities=activations["opacities"](params["opacities"]),
        colors=activations["colors"](params["colors"]),
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )
    print(render_colors.shape, render_alphas.shape)

    # Test NullStrategy
    strategy = NullStrategy(params, activations, optimizers)
    strategy.step_pre_backward(step=0, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(step=0, info=info)

    # Test DefaultStrategy
    strategy = DefaultStrategy(params, activations, optimizers, verbose=True)
    strategy.step_pre_backward(step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(step=600, info=info)

    # Test MCMCStrategy
    strategy = MCMCStrategy(params, activations, optimizers, verbose=True)
    strategy.step_pre_backward(step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(step=600, info=info, lr=1e-3)


if __name__ == "__main__":
    test_strategy()
