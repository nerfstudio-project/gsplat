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


def _dense_info(grads, radii, width=40, height=30):
    means2d = torch.zeros_like(grads, requires_grad=True)
    means2d.grad = grads
    means2d.absgrad = grads.abs() * 2.0
    return {
        "width": width,
        "height": height,
        "n_cameras": grads.shape[0],
        "radii": radii,
        "gaussian_ids": None,
        "means2d": means2d,
    }


def _expected_state(infos, n, absgrad):
    """Per-camera, per-Gaussian reference for DefaultStrategy._update_state."""
    grad2d, count, radii_max = torch.zeros(n), torch.zeros(n), torch.zeros(n)
    for info in infos:
        g = info["means2d"].absgrad if absgrad else info["means2d"].grad
        for c in range(info["n_cameras"]):
            for i in range(n):
                r = info["radii"][c, i]
                if (r > 0).all():
                    gx = g[c, i, 0] * info["width"] / 2.0 * info["n_cameras"]
                    gy = g[c, i, 1] * info["height"] / 2.0 * info["n_cameras"]
                    grad2d[i] += torch.stack([gx, gy]).norm()
                    count[i] += 1
                    radius = r.max() / float(max(info["width"], info["height"]))
                    radii_max[i] = max(radii_max[i], radius)
    return grad2d, count, radii_max


@pytest.mark.parametrize("n_cameras", [1, 3])
@pytest.mark.parametrize("absgrad", [False, True])
def test_default_strategy_update_state_dense(n_cameras, absgrad):
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(0)
    n = 50
    params = {"means": torch.zeros(n, 3)}
    strategy = DefaultStrategy(absgrad=absgrad, refine_scale2d_stop_iter=1000)
    state = strategy.initialize_state()

    infos = []
    for _ in range(2):  # two steps, so the radii are a running maximum
        grads = torch.randn(n_cameras, n, 2)
        radii = torch.randint(0, 20, (n_cameras, n, 2), dtype=torch.int32)
        radii[:, :5] = 0  # invisible in every camera
        radii[:, 5:10, 1] = 0  # one radius zero is invisible too
        infos.append(_dense_info(grads, radii))
        strategy._update_state(params, state, infos[-1], packed=False)

    grad2d, count, radii_max = _expected_state(infos, n, absgrad)
    torch.testing.assert_close(state["grad2d"], grad2d)
    torch.testing.assert_close(state["count"], count)
    torch.testing.assert_close(state["radii"], radii_max)
    assert (state["count"][:10] == 0).all()
    assert (state["grad2d"][:10] == 0).all()
    assert (state["radii"][:10] == 0).all()


def test_default_strategy_update_state_dense_matches_packed_one_camera():
    from gsplat.strategy import DefaultStrategy

    torch.manual_seed(0)
    n = 50
    params = {"means": torch.zeros(n, 3)}
    grads = torch.randn(1, n, 2)
    radii = torch.randint(0, 20, (1, n, 2), dtype=torch.int32)
    dense = _dense_info(grads, radii)

    sel = (radii > 0).all(dim=-1)
    packed = dict(dense)
    packed["gaussian_ids"] = torch.where(sel)[1]
    packed["radii"] = radii[sel]
    packed["means2d"] = torch.zeros(int(sel.sum()), 2, requires_grad=True)
    packed["means2d"].grad = grads[sel]

    strategy = DefaultStrategy(refine_scale2d_stop_iter=1000)
    state_dense = strategy.initialize_state()
    state_packed = strategy.initialize_state()
    strategy._update_state(params, state_dense, dense, packed=False)
    strategy._update_state(params, state_packed, packed, packed=True)
    for key in ["grad2d", "count", "radii"]:
        assert torch.equal(state_dense[key], state_packed[key]), key


if __name__ == "__main__":
    test_strategy()
    test_strategy_requires_grad()
