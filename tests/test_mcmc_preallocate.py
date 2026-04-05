# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Tests for MCMCStrategy preallocate=True and the grow_active_params helper.

Usage:
```bash
pytest tests/test_mcmc_preallocate.py -s
```
"""

import pytest
import torch

try:
    import gsplat as _gsplat

    _HAS_RELOC = _gsplat.has_reloc()
    _HAS_3DGS = _gsplat.has_3dgs()
except Exception:
    _HAS_RELOC = False
    _HAS_3DGS = False

_CUDA = torch.cuda.is_available()
_SKIP_CUDA = pytest.mark.skipif(not _CUDA, reason="No CUDA device")
_SKIP_RELOC = pytest.mark.skipif(not _HAS_RELOC, reason="Relocation not built")
_SKIP_3DGS = pytest.mark.skipif(not _HAS_3DGS, reason="3DGS not built")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_splats_and_active(N_max, N_init, device):
    """Return (splats ParameterDict, active_params dict, optimizers dict)."""
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(N_max, 3)),
            "scales": torch.nn.Parameter(torch.rand(N_max, 3)),
            "quats": torch.nn.Parameter(torch.randn(N_max, 4)),
            "opacities": torch.nn.Parameter(torch.rand(N_max)),
        }
    ).to(device)

    active_params = {
        name: torch.nn.Parameter(
            splats[name].data[:N_init], requires_grad=True
        )
        for name in splats.keys()
    }

    optimizers = {
        name: torch.optim.Adam([active_params[name]], lr=1e-3)
        for name in active_params.keys()
    }
    return splats, active_params, optimizers


def _populate_optimizer_state(active_params, optimizers):
    """Run one forward+backward+step so Adam state tensors are allocated."""
    loss = sum(p.sum() for p in active_params.values())
    loss.backward()
    for opt in optimizers.values():
        opt.step()
    for p in active_params.values():
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# grow_active_params unit tests (CPU, no CUDA required)
# ---------------------------------------------------------------------------

def test_grow_active_params_sizes():
    """After grow, active_params and optimizer state have the new size."""
    from gsplat.strategy.ops import grow_active_params

    N_max, N_init, N_new = 100, 20, 35
    device = torch.device("cpu")
    splats, active_params, optimizers = _make_splats_and_active(N_max, N_init, device)
    _populate_optimizer_state(active_params, optimizers)

    grow_active_params(splats, active_params, optimizers, N_new)

    for name, param in active_params.items():
        assert param.shape[0] == N_new, f"{name}: expected {N_new}, got {param.shape[0]}"

    for name, opt in optimizers.items():
        tracked = opt.param_groups[0]["params"][0]
        assert tracked is active_params[name], f"{name}: param_group not updated"
        for k, v in opt.state[tracked].items():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                assert v.shape[0] == N_new, (
                    f"{name} state[{k}]: expected {N_new}, got {v.shape[0]}"
                )


def test_grow_active_params_state_preserved():
    """Optimizer momentum for the original rows is preserved after grow."""
    from gsplat.strategy.ops import grow_active_params

    N_max, N_init, N_new = 100, 20, 35
    device = torch.device("cpu")
    splats, active_params, optimizers = _make_splats_and_active(N_max, N_init, device)
    _populate_optimizer_state(active_params, optimizers)

    # Snapshot momentum tensors before grow
    before = {}
    for name, opt in optimizers.items():
        param = active_params[name]
        before[name] = {
            k: v.clone()
            for k, v in opt.state[param].items()
            if isinstance(v, torch.Tensor) and v.dim() > 0
        }

    grow_active_params(splats, active_params, optimizers, N_new)

    for name, opt in optimizers.items():
        tracked = opt.param_groups[0]["params"][0]
        for k, v in opt.state[tracked].items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and k in before[name]:
                torch.testing.assert_close(
                    v[:N_init], before[name][k],
                    msg=f"{name} state[{k}]: old rows changed after grow",
                )
                assert v[N_init:N_new].abs().max().item() == 0.0, (
                    f"{name} state[{k}]: new rows should be zero after grow"
                )


def test_grow_active_params_shared_storage():
    """active_params data is a view into splats — writes propagate both ways."""
    from gsplat.strategy.ops import grow_active_params

    N_max, N_init, N_new = 100, 20, 35
    device = torch.device("cpu")
    splats, active_params, optimizers = _make_splats_and_active(N_max, N_init, device)
    _populate_optimizer_state(active_params, optimizers)

    grow_active_params(splats, active_params, optimizers, N_new)

    for name in active_params:
        p = active_params[name]
        idx = (0,) * p.dim()  # works for both 1-D and 2-D params

        # Writing into active_params should appear in splats
        sentinel = 999.0
        p.data[idx] = sentinel
        assert splats[name].data[idx].item() == pytest.approx(sentinel), (
            f"{name}: active_params is not a view of splats after grow"
        )

        # Writing into splats should appear in active_params
        sentinel2 = -999.0
        splats[name].data[idx] = sentinel2
        assert p.data[idx].item() == pytest.approx(sentinel2), (
            f"{name}: splats update not visible in active_params after grow"
        )


def test_grow_active_params_no_state_before_step():
    """grow works before any optimizer step (empty state dict)."""
    from gsplat.strategy.ops import grow_active_params

    N_max, N_init, N_new = 50, 10, 20
    device = torch.device("cpu")
    splats, active_params, optimizers = _make_splats_and_active(N_max, N_init, device)
    # No backward/step — optimizer.state is empty

    grow_active_params(splats, active_params, optimizers, N_new)

    for name, param in active_params.items():
        assert param.shape[0] == N_new
        assert optimizers[name].param_groups[0]["params"][0] is param


# ---------------------------------------------------------------------------
# MCMCStrategy preallocate integration tests (CUDA required)
# ---------------------------------------------------------------------------

@_SKIP_CUDA
@_SKIP_RELOC
@_SKIP_3DGS
def test_mcmc_preallocate_optimizer_wiring():
    """After wiring, optimizer tracks the active slice, not the full buffer."""
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    N_init, cap_max = 30, 200
    strategy = MCMCStrategy(cap_max=cap_max, preallocate=True)

    splats, active_params, optimizers = _make_splats_and_active(cap_max, N_init, device)

    # Rewire (mirrors what simple_trainer does)
    for name, opt in optimizers.items():
        full_param = splats[name]
        for group in opt.param_groups:
            for i, p in enumerate(group["params"]):
                if p is full_param:
                    group["params"][i] = active_params[name]

    state = strategy.initialize_state(n_initial=N_init)
    state["active_params"] = active_params

    for name, opt in optimizers.items():
        tracked = opt.param_groups[0]["params"][0]
        assert tracked is active_params[name], (
            f"{name}: optimizer should track active_params, not splats"
        )
        assert tracked.shape[0] == N_init, (
            f"{name}: optimizer param size should be {N_init}, got {tracked.shape[0]}"
        )
        # Must NOT be tracking the full buffer
        assert tracked.shape[0] != cap_max


@_SKIP_CUDA
@_SKIP_RELOC
@_SKIP_3DGS
def test_mcmc_preallocate_step_runs():
    """strategy.step_post_backward runs without error in preallocate mode."""
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    N_init, cap_max = 50, 500
    strategy = MCMCStrategy(
        cap_max=cap_max,
        preallocate=True,
        refine_start_iter=0,
        refine_stop_iter=10000,
        refine_every=100,
        verbose=True,
    )

    splats, active_params, optimizers = _make_splats_and_active(cap_max, N_init, device)

    # Rewire optimizers to active slice
    for name, opt in optimizers.items():
        full_param = splats[name]
        for group in opt.param_groups:
            for i, p in enumerate(group["params"]):
                if p is full_param:
                    group["params"][i] = active_params[name]

    state = strategy.initialize_state(n_initial=N_init)
    state["active_params"] = active_params
    strategy.check_sanity(splats, optimizers)

    def _render(ap):
        return rasterization(
            means=ap["means"],
            quats=ap["quats"],
            scales=torch.exp(ap["scales"]),
            opacities=torch.sigmoid(ap["opacities"]),
            colors=torch.rand(ap["means"].shape[0], 3, device=device),
            viewmats=torch.eye(4, device=device).unsqueeze(0),
            Ks=torch.eye(3, device=device).unsqueeze(0),
            width=16,
            height=16,
            packed=False,
        )

    # Run several steps including a refinement step (step=100 is divisible by refine_every=100)
    for step in [50, 100, 150, 200]:
        render_colors, _, info = _render(active_params)
        render_colors.mean().backward()
        strategy.step_post_backward(
            splats, optimizers, state, step=step, info=info, lr=1e-3
        )
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()


@_SKIP_CUDA
@_SKIP_RELOC
@_SKIP_3DGS
def test_mcmc_preallocate_grows_on_refinement():
    """n_active and active_params size increase at a refinement step."""
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    N_init, cap_max = 50, 500
    strategy = MCMCStrategy(
        cap_max=cap_max,
        preallocate=True,
        refine_start_iter=0,
        refine_stop_iter=10000,
        refine_every=100,
    )

    splats, active_params, optimizers = _make_splats_and_active(cap_max, N_init, device)
    for name, opt in optimizers.items():
        full_param = splats[name]
        for group in opt.param_groups:
            for i, p in enumerate(group["params"]):
                if p is full_param:
                    group["params"][i] = active_params[name]

    state = strategy.initialize_state(n_initial=N_init)
    state["active_params"] = active_params

    # Warm-up step (no refinement)
    render_colors, _, info = rasterization(
        means=active_params["means"],
        quats=active_params["quats"],
        scales=torch.exp(active_params["scales"]),
        opacities=torch.sigmoid(active_params["opacities"]),
        colors=torch.rand(N_init, 3, device=device),
        viewmats=torch.eye(4, device=device).unsqueeze(0),
        Ks=torch.eye(3, device=device).unsqueeze(0),
        width=16,
        height=16,
        packed=False,
    )
    render_colors.mean().backward()
    strategy.step_post_backward(splats, optimizers, state, step=50, info=info, lr=1e-3)
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()

    n_before = state["n_active"]
    size_before = active_params["means"].shape[0]

    # Refinement step
    render_colors, _, info = rasterization(
        means=active_params["means"],
        quats=active_params["quats"],
        scales=torch.exp(active_params["scales"]),
        opacities=torch.sigmoid(active_params["opacities"]),
        colors=torch.rand(active_params["means"].shape[0], 3, device=device),
        viewmats=torch.eye(4, device=device).unsqueeze(0),
        Ks=torch.eye(3, device=device).unsqueeze(0),
        width=16,
        height=16,
        packed=False,
    )
    render_colors.mean().backward()
    strategy.step_post_backward(splats, optimizers, state, step=100, info=info, lr=1e-3)
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad()

    n_after = state["n_active"]
    size_after = active_params["means"].shape[0]

    assert n_after >= n_before, "n_active should not decrease after refinement"
    assert size_after == n_after, (
        f"active_params size ({size_after}) should equal n_active ({n_after})"
    )

    # Optimizer tracks the grown active_params
    for name, opt in optimizers.items():
        tracked = opt.param_groups[0]["params"][0]
        assert tracked is active_params[name], f"{name}: param_group stale after grow"
        assert tracked.shape[0] == n_after, (
            f"{name}: optimizer param size {tracked.shape[0]} != n_active {n_after}"
        )
        if tracked in opt.state:
            for k, v in opt.state[tracked].items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    assert v.shape[0] == n_after, (
                        f"{name} state[{k}] size {v.shape[0]} != n_active {n_after}"
                    )


@_SKIP_CUDA
@_SKIP_RELOC
@_SKIP_3DGS
def test_mcmc_preallocate_active_params_is_view_of_splats():
    """active_params always shares storage with splats throughout training."""
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy

    torch.manual_seed(0)
    device = torch.device("cuda:0")

    N_init, cap_max = 40, 400
    strategy = MCMCStrategy(
        cap_max=cap_max,
        preallocate=True,
        refine_start_iter=0,
        refine_stop_iter=10000,
        refine_every=100,
    )

    splats, active_params, optimizers = _make_splats_and_active(cap_max, N_init, device)
    for name, opt in optimizers.items():
        full_param = splats[name]
        for group in opt.param_groups:
            for i, p in enumerate(group["params"]):
                if p is full_param:
                    group["params"][i] = active_params[name]

    state = strategy.initialize_state(n_initial=N_init)
    state["active_params"] = active_params

    def _check_shared_storage():
        n = state["n_active"]
        for name in active_params:
            ap_data = active_params[name].data
            sp_data = splats[name].data[:n]
            torch.testing.assert_close(
                ap_data, sp_data,
                msg=f"{name}: active_params data diverged from splats[:n_active]",
            )

    for step in [50, 100]:
        n = active_params["means"].shape[0]
        render_colors, _, info = rasterization(
            means=active_params["means"],
            quats=active_params["quats"],
            scales=torch.exp(active_params["scales"]),
            opacities=torch.sigmoid(active_params["opacities"]),
            colors=torch.rand(n, 3, device=device),
            viewmats=torch.eye(4, device=device).unsqueeze(0),
            Ks=torch.eye(3, device=device).unsqueeze(0),
            width=16,
            height=16,
            packed=False,
        )
        render_colors.mean().backward()
        strategy.step_post_backward(
            splats, optimizers, state, step=step, info=info, lr=1e-3
        )
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

        _check_shared_storage()


@_SKIP_CUDA
@_SKIP_RELOC
@_SKIP_3DGS
def test_mcmc_preallocate_time_independent_of_cap_max():
    """Optimizer step time must not grow with cap_max.

    Both setups start with the same n_initial Gaussians.  The only difference
    is how large the pre-allocated buffer is.  Because the optimizer only ever
    sees the active slice, wall-clock time per step should be ~equal regardless
    of cap_max.

    Pass criterion: the large-cap_max run is no more than MAX_SLOWDOWN_FACTOR
    times slower than the small-cap_max run (per-step median, optimizer only).
    """
    import time
    from gsplat.rendering import rasterization
    from gsplat.strategy import MCMCStrategy

    device = torch.device("cuda:0")
    N_INIT = 200          # active Gaussians — identical for both runs
    CAP_SMALL = 500       # just big enough to hold N_INIT
    CAP_LARGE = 100_000   # 200× larger buffer — must NOT slow things down
    N_WARMUP = 5
    N_MEASURE = 20
    MAX_SLOWDOWN_FACTOR = 2.0  # generous; should be ~1.0 if fix is correct

    def _setup(cap_max):
        torch.manual_seed(0)
        strategy = MCMCStrategy(
            cap_max=cap_max,
            preallocate=True,
            # Disable refinement so n_active stays fixed during timing
            refine_start_iter=10_000,
        )
        splats, active_params, optimizers = _make_splats_and_active(
            cap_max, N_INIT, device
        )
        for name, opt in optimizers.items():
            full_param = splats[name]
            for group in opt.param_groups:
                for i, p in enumerate(group["params"]):
                    if p is full_param:
                        group["params"][i] = active_params[name]
        state = strategy.initialize_state(n_initial=N_INIT)
        state["active_params"] = active_params
        return splats, active_params, optimizers, state, strategy

    def _measure_optimizer_step_ms(cap_max):
        splats, active_params, optimizers, state, strategy = _setup(cap_max)

        def _render():
            return rasterization(
                means=active_params["means"],
                quats=active_params["quats"],
                scales=torch.exp(active_params["scales"]),
                opacities=torch.sigmoid(active_params["opacities"]),
                colors=torch.rand(N_INIT, 3, device=device),
                viewmats=torch.eye(4, device=device).unsqueeze(0),
                Ks=torch.eye(3, device=device).unsqueeze(0),
                width=16,
                height=16,
                packed=False,
            )

        # Warm-up: forward + backward + strategy + optimizer
        for _ in range(N_WARMUP):
            render_colors, _, info = _render()
            render_colors.mean().backward()
            strategy.step_post_backward(
                splats, optimizers, state, step=1, info=info, lr=1e-3
            )
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad()
        torch.cuda.synchronize()

        # Measure only the optimizer.step() calls
        times = []
        for _ in range(N_MEASURE):
            render_colors, _, info = _render()
            render_colors.mean().backward()
            strategy.step_post_backward(
                splats, optimizers, state, step=1, info=info, lr=1e-3
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for opt in optimizers.values():
                opt.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            for opt in optimizers.values():
                opt.zero_grad()

        # Return median in milliseconds
        times.sort()
        return times[len(times) // 2] * 1000.0

    ms_small = _measure_optimizer_step_ms(CAP_SMALL)
    ms_large = _measure_optimizer_step_ms(CAP_LARGE)

    print(
        f"\n  cap_max={CAP_SMALL:>7}: {ms_small:.3f} ms/step  |  "
        f"cap_max={CAP_LARGE:>7}: {ms_large:.3f} ms/step  |  "
        f"ratio={ms_large / ms_small:.2f}x  (limit: {MAX_SLOWDOWN_FACTOR:.1f}x)"
    )

    assert ms_large < ms_small * MAX_SLOWDOWN_FACTOR, (
        f"Optimizer step time scaled with cap_max! "
        f"small={ms_small:.3f} ms, large={ms_large:.3f} ms, "
        f"ratio={ms_large / ms_small:.2f}x > {MAX_SLOWDOWN_FACTOR}x"
    )


if __name__ == "__main__":
    test_grow_active_params_sizes()
    test_grow_active_params_state_preserved()
    test_grow_active_params_shared_storage()
    test_grow_active_params_no_state_before_step()
    if torch.cuda.is_available():
        test_mcmc_preallocate_optimizer_wiring()
        test_mcmc_preallocate_step_runs()
        test_mcmc_preallocate_grows_on_refinement()
        test_mcmc_preallocate_active_params_is_view_of_splats()
        test_mcmc_preallocate_time_independent_of_cap_max()
