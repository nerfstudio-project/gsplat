# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gsplat.contrib.dynamic.DynamicStrategy``.

Covers the deformable-aware densification strategy ported from G-SHARP v0.2.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from gsplat.contrib.dynamic.strategy import DynamicStrategy


def _make_params_optimizers(num_gaussians: int = 10):
    """Build the per-Gaussian params + optimizers pair expected by ``check_sanity``."""
    params: dict[str, torch.nn.Parameter] = {
        "means": torch.nn.Parameter(torch.randn(num_gaussians, 3)),
        "scales": torch.nn.Parameter(torch.randn(num_gaussians, 3)),
        "quats": torch.nn.Parameter(torch.randn(num_gaussians, 4)),
        "opacities": torch.nn.Parameter(torch.randn(num_gaussians, 1)),
    }
    optimizers = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.items()}
    return params, optimizers


# ---------------------------------------------------------------------------
# check_sanity
# ---------------------------------------------------------------------------


def test_dynamic_strategy_check_sanity_passes_on_well_formed_params():
    """DynamicStrategy.check_sanity has the same per-Gaussian requirements
    as the parent — means/scales/quats/opacities must be present and
    optimizer keys must match. HexPlane / DeformNet trainables live
    outside *params* (see DynamicStrategy class docstring)."""
    strategy = DynamicStrategy()
    params, optimizers = _make_params_optimizers()
    strategy.check_sanity(params, optimizers)  # must not raise


def test_dynamic_strategy_check_sanity_fails_on_missing_per_gaussian_keys():
    """The parent's check still fires if a per-Gaussian key is missing."""
    strategy = DynamicStrategy()
    params, optimizers = _make_params_optimizers()
    # Drop means (required by DefaultStrategy.check_sanity).
    del params["means"]
    del optimizers["means"]
    with pytest.raises(AssertionError, match="means"):
        strategy.check_sanity(params, optimizers)


# ---------------------------------------------------------------------------
# initialize_state
# ---------------------------------------------------------------------------


def test_dynamic_strategy_initialize_state_creates_dynamic_mask_tensor():
    """``state["dynamic_mask"]`` is a plain bool tensor (not a wrapper).
    Default is all-True so every Gaussian routes through DeformNet."""
    strategy = DynamicStrategy()
    state = strategy.initialize_state(scene_scale=1.0, num_gaussians=10)
    assert "dynamic_mask" in state
    mask = state["dynamic_mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (10,)
    assert mask.all()  # default: every Gaussian dynamic


def test_dynamic_strategy_initialize_state_honours_init_dynamic_false():
    strategy = DynamicStrategy()
    state = strategy.initialize_state(
        scene_scale=1.0, num_gaussians=4, init_dynamic=False
    )
    assert not state["dynamic_mask"].any()


def test_dynamic_strategy_initialize_state_preserves_default_keys():
    """Make sure we don't drop any of the parent's state keys."""
    strategy = DynamicStrategy()
    state = strategy.initialize_state(scene_scale=2.5, num_gaussians=4)
    assert state["scene_scale"] == 2.5
    assert state["grad2d"] is None
    assert state["count"] is None


# ---------------------------------------------------------------------------
# step_post_backward guard rails
# ---------------------------------------------------------------------------


def test_dynamic_strategy_post_backward_without_init_raises():
    strategy = DynamicStrategy()
    params, optimizers = _make_params_optimizers()
    state = {"grad2d": None, "count": None, "scene_scale": 1.0}  # no dynamic_mask
    info: dict = {}
    with pytest.raises(RuntimeError, match="initialize_state"):
        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=0,
            info=info,
        )


# ---------------------------------------------------------------------------
# Full one-step training pass — covered by the integration test once the
# trainer (TDD step 9) lands. Skipped here because rasterization is
# CUDA-only and the strategy on its own can't drive a forward pass.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Activated as tests/test_dynamic_surgical_trainer.py::"
        "test_trainer_one_step_train_no_nan (CUDA + ENDONERF_DATA_DIR-gated). "
        "Kept here as a marker so anyone scanning DynamicStrategy tests can "
        "find the integration counterpart."
    )
)
def test_dynamic_strategy_one_step_train_no_nan():
    pass
