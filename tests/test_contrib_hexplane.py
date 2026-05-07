# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gsplat.contrib.dynamic.HexPlaneField`` (branch: vnath_gsharp).

Covers the multi-resolution 6-plane spatio-temporal feature field ported
from G-SHARP v0.2 (`training/scene/hexplane.py`).

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from gsplat.contrib.dynamic.hexplane import HexPlaneField


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_hexplane_default_construction_has_expected_feat_dim():
    """Default config: 32 features per plane × 2 scales = 64-d output."""
    field = HexPlaneField()
    assert field.feat_dim == 32 * 2


def test_hexplane_custom_planes_config_changes_feat_dim():
    """Smaller plane resolution + 3 scales: feat_dim = 16 * 3 = 48."""
    cfg = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "resolution": [16, 16, 16, 8],
    }
    field = HexPlaneField(bounds=1.0, planes_config=cfg, multires=(1, 2, 4))
    assert field.feat_dim == 16 * 3


# ---------------------------------------------------------------------------
# Forward shape
# ---------------------------------------------------------------------------


def test_hexplane_forward_shape():
    field = HexPlaneField()
    n = 16
    xyzt = torch.rand(n, 4) * 2 - 1
    feat = field(xyzt)
    assert feat.shape == (n, field.feat_dim)


def test_hexplane_forward_finite_for_in_bounds_input():
    field = HexPlaneField()
    xyzt = torch.rand(16, 4) * 2 - 1
    feat = field(xyzt)
    assert torch.isfinite(feat).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_hexplane_gradients_reach_plane_params():
    field = HexPlaneField()
    xyzt = torch.rand(16, 4) * 2 - 1
    feat = field(xyzt)
    feat.sum().backward()

    has_grad = False
    for grids in field.grids:
        for plane in grids:
            if plane.grad is not None and plane.grad.abs().sum() > 0:
                has_grad = True
                break
        if has_grad:
            break
    assert has_grad, "No HexPlane parameter received a gradient."


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_hexplane_out_of_bounds_time_handled():
    """Out-of-range t values are clamped via ``padding_mode='border'`` in
    grid_sample; output must stay finite."""
    field = HexPlaneField()
    xyzt = torch.tensor(
        [
            [0.0, 0.0, 0.0, -2.0],  # t below grid
            [0.0, 0.0, 0.0, 2.0],   # t above grid
            [0.0, 0.0, 0.0, 0.0],   # in-bounds
        ]
    )
    feat = field(xyzt)
    assert feat.shape == (3, field.feat_dim)
    assert torch.isfinite(feat).all()


def test_hexplane_invalid_input_shape_raises():
    field = HexPlaneField()
    bad = torch.rand(16, 3)  # 3D, missing time
    with pytest.raises(ValueError, match="last dim"):
        field(bad)
