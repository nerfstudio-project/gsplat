"""Tests for age-gated opacity reset (DefaultStrategy.reset_max_age).

CPU-only: exercises the strategy/ops layer, not the rasterizer.
"""

import torch

from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import reset_opa


def _make_params(n: int = 8, opacity: float = 0.9):
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(torch.randn(n, 3)),
            "scales": torch.nn.Parameter(torch.randn(n, 3)),
            "quats": torch.nn.Parameter(torch.randn(n, 4)),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((n,), opacity))
            ),
        }
    )
    optimizers = {
        name: torch.optim.Adam([p], lr=1e-3) for name, p in params.items()
    }
    # populate optimizer state so reset_opa has something to zero
    loss = sum(p.sum() for p in params.values())
    loss.backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    return params, optimizers


def test_reset_opa_mask_none_matches_previous_behavior():
    params, optimizers = _make_params()
    before = params["opacities"].detach().clone()
    reset_opa(params, optimizers, state={}, value=0.01)
    after = torch.sigmoid(params["opacities"].detach())
    assert (after <= 0.01 + 1e-6).all()
    assert not torch.equal(before, params["opacities"].detach())


def test_reset_opa_mask_selects_subset():
    params, optimizers = _make_params(n=8)
    before = params["opacities"].detach().clone()
    mask = torch.zeros(8, dtype=torch.bool)
    mask[:3] = True
    reset_opa(params, optimizers, state={}, value=0.01, mask=mask)
    after = params["opacities"].detach()
    assert (torch.sigmoid(after[:3]) <= 0.01 + 1e-6).all(), "masked must reset"
    assert torch.equal(after[3:], before[3:]), "unmasked must be untouched"
    # optimizer state zeroed only where masked
    opt = optimizers["opacities"]
    st = opt.state[params["opacities"]]
    assert (st["exp_avg"][:3] == 0).all()
    assert not (st["exp_avg"][3:] == 0).all()


def test_default_strategy_initializes_and_uses_birth_step():
    strategy = DefaultStrategy(reset_max_age=500)
    state = strategy.initialize_state(scene_scale=1.0)
    assert "birth_step" in state and state["birth_step"] is None

    # simulate the lazily-initialized state after some training
    n = 6
    params, optimizers = _make_params(n=n)
    state["birth_step"] = torch.zeros(n, dtype=torch.long)
    state["birth_step"][-2:] = 2800  # two young Gaussians
    step = 3000

    reset_mask = (step - state["birth_step"]) <= strategy.reset_max_age
    before = params["opacities"].detach().clone()
    reset_opa(params, optimizers, state=state, value=0.01, mask=reset_mask)
    after = params["opacities"].detach()
    assert torch.equal(after[:-2], before[:-2]), "mature Gaussians preserved"
    assert (torch.sigmoid(after[-2:]) <= 0.01 + 1e-6).all(), "young reset"


def test_default_strategy_default_is_backward_compatible():
    strategy = DefaultStrategy()
    state = strategy.initialize_state(scene_scale=1.0)
    assert "birth_step" not in state
