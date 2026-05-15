# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gsplat.contrib.dynamic.DynamicStrategy`` (branch: vnath_gsharp).

Covers the deformable-aware densification strategy ported from G-SHARP v0.2.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from gsplat.contrib.dynamic.deformation import DeformationTable
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


def test_dynamic_strategy_initialize_state_creates_table():
    strategy = DynamicStrategy()
    state = strategy.initialize_state(scene_scale=1.0, num_gaussians=10)
    assert "deformation_table" in state
    table = state["deformation_table"]
    assert isinstance(table, DeformationTable)
    assert len(table) == 10
    # All flags start static (False).
    assert not table.mask.any()


def test_dynamic_strategy_initialize_state_preserves_default_keys():
    """Make sure we don't drop any of the parent's state keys."""
    strategy = DynamicStrategy()
    state = strategy.initialize_state(scene_scale=2.5, num_gaussians=4)
    assert state["scene_scale"] == 2.5
    assert state["grad2d"] is None
    assert state["count"] is None


# ---------------------------------------------------------------------------
# Resize hook (densify lock-step)
# ---------------------------------------------------------------------------


def test_dynamic_strategy_densify_resizes_deformation_table_grow():
    table = DeformationTable(num_gaussians=5)
    table.set_indices(torch.tensor([0, 2]))  # [T, F, T, F, F]

    DynamicStrategy._resize_table(table, n_before=5, n_after=8)

    assert len(table) == 8
    # Original five preserved.
    assert table.mask[:5].tolist() == [True, False, True, False, False]
    # Three new Gaussians flagged dynamic.
    assert table.mask[5:].all()


def test_dynamic_strategy_densify_resizes_deformation_table_shrink():
    table = DeformationTable(num_gaussians=8)
    table.set_indices(torch.tensor([0, 2, 6]))

    DynamicStrategy._resize_table(table, n_before=8, n_after=4)

    assert len(table) == 4
    # Truncation: prefix preserved.
    assert table.mask[:4].tolist() == [True, False, True, False]


def test_dynamic_strategy_resize_table_no_change_is_noop():
    table = DeformationTable(num_gaussians=3)
    table.set_indices(torch.tensor([1]))
    snapshot = table.mask.clone()

    DynamicStrategy._resize_table(table, n_before=3, n_after=3)

    assert (table.mask == snapshot).all()


# ---------------------------------------------------------------------------
# step_post_backward guard rails
# ---------------------------------------------------------------------------


def test_dynamic_strategy_post_backward_without_init_raises():
    strategy = DynamicStrategy()
    params, optimizers = _make_params_optimizers()
    state = {"grad2d": None, "count": None, "scene_scale": 1.0}  # no deformation_table
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
