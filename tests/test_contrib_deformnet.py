# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gsplat.contrib.dynamic.DeformNetwork`` and ``DeformationTable``.

Branch: vnath_gsharp. Seed is set to 42 by the autouse fixture in
``conftest.py``.
"""

import pytest
import torch

from gsplat._helper import expect_grad_reference_close
from gsplat.contrib.dynamic.deformation import DeformationTable, DeformNetwork


# ---------------------------------------------------------------------------
# DeformNetwork — construction
# ---------------------------------------------------------------------------


def test_deform_net_invalid_construction_raises():
    with pytest.raises(ValueError, match="num_layers"):
        DeformNetwork(feature_dim=32, num_layers=0)
    with pytest.raises(ValueError, match="feature_dim"):
        DeformNetwork(feature_dim=0)


# ---------------------------------------------------------------------------
# DeformNetwork — forward / identity
# ---------------------------------------------------------------------------


def _make_inputs(n: int, feature_dim: int, dtype: torch.dtype = torch.float32):
    means = torch.randn(n, 3, dtype=dtype)
    quats = torch.randn(n, 4, dtype=dtype)
    opacities = torch.randn(n, 1, dtype=dtype)
    t = torch.zeros(n, 1, dtype=dtype)
    features = torch.randn(n, feature_dim, dtype=dtype)
    return means, quats, opacities, t, features


def test_deform_net_zero_init_is_identity():
    """Heads are zero-initialised → forward returns inputs unchanged."""
    feature_dim = 64
    net = DeformNetwork(feature_dim=feature_dim)
    means, quats, opacities, t, features = _make_inputs(16, feature_dim)

    out_m, out_q, out_o = net(means, quats, opacities, t, features)

    assert torch.allclose(out_m, means, atol=1e-6)
    assert torch.allclose(out_q, quats, atol=1e-6)
    assert torch.allclose(out_o, opacities, atol=1e-6)


# ---------------------------------------------------------------------------
# DeformNetwork — gradient flow
# ---------------------------------------------------------------------------


def test_deform_net_gradients_flow_to_mlp_weights():
    """Backward graph is correctly wired across trunk + heads.

    Zero-init head weights are an intentional design choice (see
    ``test_deform_net_zero_init_is_identity``), but they short-circuit
    gradient flow to the trunk at init: ``dL/d(trunk_out) = dL/d(out) ·
    head_weight`` is zero everywhere when ``head_weight == 0``. The first
    backward pass sets head-weight gradients to non-zero, so after one
    optimiser step the trunk would receive gradients. To test the
    backward graph itself without doing an opt step, perturb the heads
    off zero before forward.
    """
    feature_dim = 32
    net = DeformNetwork(feature_dim=feature_dim, hidden_dim=16, num_layers=2)
    with torch.no_grad():
        for head in (net.pos_head, net.quat_head, net.opacity_head):
            torch.nn.init.normal_(head.weight, std=0.01)

    means, quats, opacities, t, features = _make_inputs(8, feature_dim)
    out_m, out_q, out_o = net(means, quats, opacities, t, features)
    (out_m.sum() + out_q.sum() + out_o.sum()).backward()

    has_trunk_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in net.trunk.parameters()
    )
    assert has_trunk_grad, "No trunk parameter received a gradient."

    for head in (net.pos_head, net.quat_head, net.opacity_head):
        assert head.weight.grad is not None
        assert (
            head.weight.grad.abs().sum() > 0
        ), f"Head {head} weight received zero gradient."


def test_deform_net_zero_init_blocks_trunk_gradient_at_init():
    """Companion to the above: at init (zero-init heads), the trunk gradient
    is exactly zero. This locks the design choice — if a future change
    flips heads to non-zero init, this test must be updated alongside the
    `_zero_init_is_identity` test.
    """
    feature_dim = 32
    net = DeformNetwork(feature_dim=feature_dim, hidden_dim=16, num_layers=2)

    means, quats, opacities, t, features = _make_inputs(8, feature_dim)
    out_m, out_q, out_o = net(means, quats, opacities, t, features)
    (out_m.sum() + out_q.sum() + out_o.sum()).backward()

    for param_idx, p in enumerate(net.trunk.parameters()):
        materialized_grad = torch.zeros_like(p) if p.grad is None else p.grad
        expect_grad_reference_close(
            materialized_grad,
            torch.zeros_like(p),
            rtol=0.0,
            atol=0.0,
            max_rel_l2=0.0,
            max_rel_l1=0.0,
            min_cosine=1.0,
            max_signed_bias=0.0,
            msg=(
                "DeformNetwork zero-init trunk gradient " f"for parameter {param_idx}"
            ),
        )
    # Head weight gradients ARE non-zero (they receive trunk_out as input).
    assert net.pos_head.weight.grad is not None
    assert net.pos_head.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# DeformNetwork — input validation
# ---------------------------------------------------------------------------


def test_deform_net_dtype_mismatch_raises():
    feature_dim = 32
    net = DeformNetwork(feature_dim=feature_dim)
    means, quats, opacities, t, _ = _make_inputs(4, feature_dim, dtype=torch.float32)
    bad_features = torch.randn(4, feature_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        net(means, quats, opacities, t, bad_features)


def test_deform_net_batch_dim_mismatch_raises():
    feature_dim = 32
    net = DeformNetwork(feature_dim=feature_dim)
    means = torch.randn(4, 3)
    quats = torch.randn(5, 4)  # mismatch
    opacities = torch.randn(4, 1)
    t = torch.zeros(4, 1)
    features = torch.randn(4, feature_dim)
    with pytest.raises(ValueError, match="batch dim"):
        net(means, quats, opacities, t, features)


def test_deform_net_feature_dim_mismatch_raises():
    feature_dim = 32
    net = DeformNetwork(feature_dim=feature_dim)
    means, quats, opacities, t, _ = _make_inputs(4, feature_dim)
    bad_features = torch.randn(4, feature_dim + 1)  # wrong feature dim
    with pytest.raises(ValueError, match="last dim"):
        net(means, quats, opacities, t, bad_features)


# ---------------------------------------------------------------------------
# DeformationTable — basics
# ---------------------------------------------------------------------------


def test_deformation_table_init_all_static():
    table = DeformationTable(num_gaussians=10)
    assert len(table) == 10
    assert not table.mask.any()


def test_deformation_table_negative_count_raises():
    with pytest.raises(ValueError, match="num_gaussians"):
        DeformationTable(num_gaussians=-1)


def test_deformation_table_set_indices():
    table = DeformationTable(num_gaussians=10)
    table.set_indices(torch.tensor([1, 3, 5]))
    assert bool(table.mask[1])
    assert bool(table.mask[3])
    assert bool(table.mask[5])
    assert not bool(table.mask[0])
    assert not bool(table.mask[2])


# ---------------------------------------------------------------------------
# DeformationTable — densify ops (lock-step resize)
# ---------------------------------------------------------------------------


def test_deformation_table_prune_resizes_correctly():
    table = DeformationTable(num_gaussians=5)
    table.set_indices(torch.tensor([0, 2, 4]))  # mask = [T, F, T, F, T]
    keep = torch.tensor([True, False, True, False, True])
    table.prune(keep)
    assert len(table) == 3
    # All survivors were dynamic (indices 0, 2, 4).
    assert table.mask.all()


def test_deformation_table_prune_shape_mismatch_raises():
    table = DeformationTable(num_gaussians=5)
    bad_keep = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="shape"):
        table.prune(bad_keep)


def test_deformation_table_duplicate_appends_children():
    table = DeformationTable(num_gaussians=5)
    table.set_indices(torch.tensor([0, 2]))  # [T, F, T, F, F]
    table.duplicate(torch.tensor([0, 1]))
    assert len(table) == 7
    expected = torch.tensor([True, False, True, False, False, True, False])
    assert (table.mask == expected).all()


def test_deformation_table_split_removes_parents_appends_children():
    table = DeformationTable(num_gaussians=5)
    table.set_indices(torch.tensor([0, 2]))  # [T, F, T, F, F]
    table.split(torch.tensor([0, 2]), factor=2)
    # Survivors after removing indices 0 and 2: [F, F, F]
    # Two children per split, each inheriting parent's flag (both T): [T, T, T, T]
    expected = torch.tensor([False, False, False, True, True, True, True])
    assert len(table) == 7
    assert (table.mask == expected).all()


def test_deformation_table_split_invalid_factor_raises():
    table = DeformationTable(num_gaussians=5)
    with pytest.raises(ValueError, match="factor"):
        table.split(torch.tensor([0]), factor=0)
