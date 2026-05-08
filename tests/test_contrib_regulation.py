# SPDX-License-Identifier: Apache-2.0
"""Tests for HexPlane regularizers in ``gsplat.contrib.dynamic.regulation``.

Branch: vnath_gsharp. Seed is set to 42 by the autouse fixture in
``conftest.py``.
"""

import pytest
import torch

from gsplat.contrib.dynamic.regulation import (
    plane_smoothness,
    time_l1,
    time_smoothness,
)


# ---------------------------------------------------------------------------
# plane_smoothness / time_smoothness (share the same math)
# ---------------------------------------------------------------------------


def test_plane_smoothness_constant_input_is_zero():
    plane = torch.full((1, 4, 8, 8), 0.5)
    out = plane_smoothness([plane])
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


def test_plane_smoothness_linear_input_is_zero():
    """Linear ramp along H → constant first-difference → zero second-difference."""
    h, w = 8, 8
    plane = torch.arange(h, dtype=torch.float32).view(1, 1, h, 1).expand(1, 4, h, w)
    out = plane_smoothness([plane])
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-6)


def test_plane_smoothness_curved_input_positive():
    """Quadratic ramp along H → constant non-zero second-difference → positive."""
    h, w = 8, 8
    coords = torch.arange(h, dtype=torch.float32)
    plane = (coords**2).view(1, 1, h, 1).expand(1, 4, h, w).contiguous()
    out = plane_smoothness([plane])
    assert out.item() > 0


def test_plane_smoothness_sums_across_planes():
    plane_a = torch.full((1, 1, 4, 4), 0.5)  # constant → 0 contribution
    plane_b = (torch.arange(4 * 4, dtype=torch.float32) ** 2).view(
        1, 1, 4, 4
    )  # nonzero
    out_b_only = plane_smoothness([plane_b])
    out_both = plane_smoothness([plane_a, plane_b])
    assert torch.allclose(out_both, out_b_only, atol=1e-6)


def test_plane_smoothness_invalid_rank_raises():
    bad = torch.rand(8, 8)  # 2D, missing batch + channel
    with pytest.raises(ValueError, match="4D"):
        plane_smoothness([bad])


def test_time_smoothness_same_math_as_plane_smoothness():
    """The two functions are intentionally interchangeable on the math level."""
    plane = torch.randn(1, 3, 6, 6)
    assert torch.allclose(plane_smoothness([plane]), time_smoothness([plane]))


def test_smoothness_on_too_short_axis_skips_plane():
    """H = 2 is too short for a second difference; the plane is silently skipped."""
    plane = torch.randn(1, 3, 2, 8)  # H = 2
    out = plane_smoothness([plane])
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


# ---------------------------------------------------------------------------
# time_l1 — penalises deviation from 1.0
# ---------------------------------------------------------------------------


def test_time_l1_at_one_is_zero():
    plane = torch.ones(1, 4, 8, 8)
    out = time_l1([plane])
    assert torch.allclose(out, torch.tensor(0.0), atol=1e-7)


def test_time_l1_constant_offset_matches_offset():
    """Plane at constant 0.7 → mean |1 - 0.7| = 0.3 per plane."""
    plane = torch.full((1, 1, 4, 4), 0.7)
    out = time_l1([plane])
    assert torch.allclose(out, torch.tensor(0.3), atol=1e-6)


def test_time_l1_sums_across_planes():
    p1 = torch.full((1, 1, 2, 2), 0.5)  # mean |1 - 0.5| = 0.5
    p2 = torch.full((1, 1, 2, 2), 0.8)  # mean |1 - 0.8| = 0.2
    out = time_l1([p1, p2])
    assert torch.allclose(out, torch.tensor(0.7), atol=1e-6)


def test_time_l1_gradient_flows():
    plane = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    out = time_l1([plane])
    out.backward()
    assert plane.grad is not None
    # |1 - 0.5| = 0.5, derivative w.r.t. plane is -1 / numel = -1/16.
    assert torch.allclose(plane.grad, torch.full_like(plane, -1.0 / 16.0), atol=1e-6)
