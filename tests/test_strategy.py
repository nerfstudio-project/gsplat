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

import pytest
import torch
import gsplat

device = torch.device("cuda:0")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("growth_op", ["duplicate", "split"])
def test_growth_ops_preserve_optimizer_state_dtype(dtype, growth_op):
    from gsplat.strategy import ops

    p = torch.nn.Parameter(
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype)
    )
    params = {"colors": p}
    if growth_op == "split":
        params["scales"] = torch.nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        params["quats"] = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1), requires_grad=False
        )
    optimizer = torch.optim.Adam([p], eps=1e-4)
    p.square().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    old_state = {
        key: optimizer.state[p][key].clone() for key in ("exp_avg", "exp_avg_sq")
    }

    getattr(ops, growth_op)(
        params, {"colors": optimizer}, {}, torch.tensor([True, False])
    )
    p = params["colors"]
    assert p.dtype == dtype
    assert p.device == old_state["exp_avg"].device
    retained = [0, 1] if growth_op == "duplicate" else [1]
    for key, previous in old_state.items():
        current = optimizer.state[p][key]
        assert current.dtype == previous.dtype
        assert current.device == previous.device
        torch.testing.assert_close(
            current[: len(retained)], previous[retained], rtol=0, atol=0
        )
        assert torch.count_nonzero(current[len(retained) :]) == 0

    p.square().sum().backward()
    optimizer.step()
    assert torch.isfinite(p).all()


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


if __name__ == "__main__":
    test_strategy()
    test_strategy_requires_grad()
