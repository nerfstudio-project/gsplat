# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math

import pytest
import torch
import gsplat

device = torch.device("cuda:0")


def test_mcmc_strategy_positional_constructor():
    from gsplat.strategy import MCMCStrategy

    strategy = MCMCStrategy(123, 4.5, 6, 7, 8, 9, 0.1, True, 0.2, 30.0)

    assert strategy.cap_max == 123
    assert strategy.noise_lr == 4.5
    assert strategy.refine_start_iter == 6
    assert strategy.refine_stop_iter == 7
    assert strategy.noise_injection_stop_iter == 8
    assert strategy.refine_every == 9
    assert strategy.min_opacity == 0.1
    assert strategy.verbose is True
    assert strategy.noise_opacity_t == 0.2
    assert strategy.noise_opacity_k == 30.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_strategy():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
        }
    ).to(device)
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)

    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
def test_strategy_requires_grad():
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    def assert_consistent_sizes(params):
        sizes = [v.shape[0] for v in params.values()]
        assert all([s == sizes[0] for s in sizes])

    torch.manual_seed(42)

    # Prepare Gaussians
    N = 100
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": torch.rand(N, 3),
            "non_trainable_features": torch.rand(N, 3),
        }
    ).to(device)
    params["non_trainable_features"].requires_grad = False
    requires_grad_map = {k: v.requires_grad for k, v in params.items()}
    optimizers = {
        k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items() if v.requires_grad
    }

    # A dummy rendering call
    render_colors, render_alphas, info = rasterization(
        means=params["means"],
        quats=params["quats"],  # F.normalize is fused into the kernel
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=torch.eye(4).unsqueeze(0).to(device),
        Ks=torch.eye(3).unsqueeze(0).to(device),
        width=10,
        height=10,
        packed=False,
    )

    # Test DefaultStrategy
    strategy = DefaultStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info)
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert params["non_trainable_features"].grad is None
    assert_consistent_sizes(params)
    # Test MCMCStrategy
    strategy = MCMCStrategy(verbose=True)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    render_colors.mean().backward(retain_graph=True)
    strategy.step_post_backward(params, optimizers, state, step=600, info=info, lr=1e-3)
    assert params["non_trainable_features"].grad is None
    for k, v in params.items():
        assert v.requires_grad == requires_grad_map[k]
    assert_consistent_sizes(params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not gsplat.has_3dgs(), reason="3DGS support isn't built in")
@pytest.mark.parametrize("strategy_name", ["default", "mcmc"])
@pytest.mark.parametrize("sh_degree", [None, 1])
def test_strategy_with_spherical_beta(strategy_name: str, sh_degree):
    """Spherical Beta lobes survive densification under either strategy.

    `sbN` is [N, M, 6] rather than [N, D], so this guards the assumption that
    the densification ops index every parameter generically along dim 0.
    """
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy, MCMCStrategy

    torch.manual_seed(42)

    N, M = 200, 2
    sb = torch.zeros(N, M, 6)
    sb[..., :3] = torch.rand(N, M, 3) * 0.5
    sb[..., 3] = torch.rand(N, M) * math.pi
    sb[..., 4] = torch.rand(N, M) * 2 * math.pi
    params = torch.nn.ParameterDict(
        {
            "means": torch.randn(N, 3),
            "scales": torch.rand(N, 3),
            "quats": torch.randn(N, 4),
            "opacities": torch.rand(N),
            "colors": (
                torch.rand(N, 3)
                if sh_degree is None
                else torch.randn(N, (sh_degree + 1) ** 2, 3) * 0.3
            ),
            "sbN": sb,
        }
    ).to(device)
    optimizers = {k: torch.optim.Adam([v], lr=1e-3) for k, v in params.items()}

    viewmats = torch.eye(4, device=device).unsqueeze(0)
    viewmats[:, 2, 3] = 4.0
    Ks = torch.tensor(
        [[60.0, 0.0, 30.0], [0.0, 60.0, 30.0], [0.0, 0.0, 1.0]], device=device
    ).unsqueeze(0)

    render_colors, _, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=params["colors"],
        viewmats=viewmats,
        Ks=Ks,
        width=60,
        height=60,
        sh_degree=sh_degree,
        sb_params=params["sbN"],
        sb_number=M,
        packed=False,
    )

    if strategy_name == "default":
        strategy = DefaultStrategy()
        step_kwargs = {}
    else:
        strategy = MCMCStrategy(cap_max=2 * N)
        step_kwargs = {"lr": 1e-3}
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()
    strategy.step_pre_backward(params, optimizers, state, step=600, info=info)
    render_colors.mean().backward()
    assert params["sbN"].grad is not None
    assert params["sbN"].grad.abs().max() > 0

    # step 600 lands inside the refinement window of both strategies, so this
    # call actually grows or relocates rather than being a no-op.
    strategy.step_post_backward(
        params, optimizers, state, step=600, info=info, **step_kwargs
    )

    counts = {k: v.shape[0] for k, v in params.items()}
    assert len(set(counts.values())) == 1, counts
    assert params["sbN"].shape[1:] == (M, 6), params["sbN"].shape
    if strategy_name == "default":
        assert counts["sbN"] > N, "DefaultStrategy should have densified"
    # The optimizer state must be resized in lockstep or the next step throws.
    optimizers["sbN"].step()


if __name__ == "__main__":
    test_strategy()
    test_strategy_requires_grad()
