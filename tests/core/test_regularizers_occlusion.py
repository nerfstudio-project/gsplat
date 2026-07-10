# SPDX-License-Identifier: Apache-2.0
"""Tests for the G-SHARP v0.2 regularizers in ``gsplat.regularizers`` (branch: vnath_gsharp).

Covers the three ported functions:

- :func:`gsplat.regularizers.compute_tv_loss_targeted` — anisotropic TV with
  optional binary mask weighting.
- :func:`gsplat.regularizers.dilate_mask` — torch max-pool dilation.
- :func:`gsplat.regularizers.create_invisible_mask` — union of per-frame masks.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from gsplat.regularizers import (
    compute_tv_loss_targeted,
    create_invisible_mask,
    dilate_mask,
)


# ---------------------------------------------------------------------------
# compute_tv_loss_targeted
# ---------------------------------------------------------------------------


def test_tv_targeted_constant_image_is_zero():
    img = torch.full((1, 3, 8, 8), 0.5)
    loss = compute_tv_loss_targeted(img)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_tv_targeted_step_edge_matches_analytic_value():
    """3x3 image with a horizontal step at row 1.

    tv_h sums |row1 - row0| + |row2 - row1| = 1 + 0 across each of 3 columns = 3.
    tv_w is 0 (rows are constant horizontally).
    Without a mask: (3 + 0) / image.numel() = 3 / 9 = 1/3.
    """
    img = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]])
    loss = compute_tv_loss_targeted(img)
    assert torch.allclose(loss, torch.tensor(1.0 / 3.0), atol=1e-6)


def test_tv_targeted_empty_mask_returns_zero_no_div_by_zero():
    img = torch.rand(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    loss = compute_tv_loss_targeted(img, mask)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_tv_targeted_non_binary_mask_rejected(monkeypatch):
    """Binary-mask check is gated behind ENFORCE_CONTRACTS to avoid a CPU sync
    in the hot training loop; this test opts in so it can verify the check
    still fires when asked."""
    monkeypatch.setattr("gsplat.regularizers.ENFORCE_CONTRACTS", True)
    img = torch.rand(2, 3, 8, 8)
    mask = torch.full((2, 1, 8, 8), 0.5)
    with pytest.raises(ValueError, match="binary"):
        compute_tv_loss_targeted(img, mask)


def test_tv_targeted_non_binary_mask_silent_when_contracts_disabled(monkeypatch):
    """When ENFORCE_CONTRACTS is off (default), a non-binary mask should NOT
    raise — production hot-path behaviour."""
    monkeypatch.setattr("gsplat.regularizers.ENFORCE_CONTRACTS", False)
    img = torch.rand(2, 3, 8, 8)
    mask = torch.full((2, 1, 8, 8), 0.5)
    loss = compute_tv_loss_targeted(img, mask)
    assert torch.isfinite(loss).all()


def test_tv_targeted_non_4d_image_raises():
    img = torch.rand(3, 8, 8)  # missing batch dim
    with pytest.raises(ValueError, match="4D"):
        compute_tv_loss_targeted(img)


def test_tv_targeted_gradient_flows():
    img = torch.rand(2, 3, 8, 8, requires_grad=True)
    mask = torch.ones(2, 1, 8, 8)
    loss = compute_tv_loss_targeted(img, mask)
    loss.backward()
    assert img.grad is not None
    assert img.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# dilate_mask
# ---------------------------------------------------------------------------


def test_dilate_mask_kernel_one_is_identity():
    mask = (torch.rand(8, 8) > 0.5).float()
    dilated = dilate_mask(mask, kernel_size=1)
    assert torch.equal(dilated, mask)


def test_dilate_mask_kernel_three_grows_by_one_pixel():
    mask = torch.zeros(5, 5)
    mask[2, 2] = 1.0
    dilated = dilate_mask(mask, kernel_size=3)
    expected = torch.zeros(5, 5)
    expected[1:4, 1:4] = 1.0
    assert torch.equal(dilated, expected)


def test_dilate_mask_invalid_kernel_size_raises():
    mask = torch.zeros(4, 4)
    with pytest.raises(ValueError, match="kernel_size"):
        dilate_mask(mask, kernel_size=2)
    with pytest.raises(ValueError, match="kernel_size"):
        dilate_mask(mask, kernel_size=0)


def test_dilate_mask_preserves_input_rank():
    mask_2d = torch.zeros(4, 4)
    mask_2d[1, 1] = 1.0
    assert dilate_mask(mask_2d).ndim == 2

    mask_3d = mask_2d.unsqueeze(0)
    assert dilate_mask(mask_3d).ndim == 3

    mask_4d = mask_3d.unsqueeze(0)
    assert dilate_mask(mask_4d).ndim == 4


# ---------------------------------------------------------------------------
# create_invisible_mask
# ---------------------------------------------------------------------------


def test_create_invisible_mask_union_of_inputs():
    m1 = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    m2 = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    m3 = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    invisible = create_invisible_mask([m1, m2, m3])
    expected = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    assert torch.equal(invisible, expected)


def test_create_invisible_mask_empty_iterable_raises():
    with pytest.raises(ValueError, match="at least one"):
        create_invisible_mask([])


def test_create_invisible_mask_shape_mismatch_raises():
    m1 = torch.zeros(4, 4)
    m2 = torch.zeros(4, 5)
    with pytest.raises(ValueError, match="shape"):
        create_invisible_mask([m1, m2])


def test_create_invisible_mask_invalid_element_type_raises():
    with pytest.raises(TypeError, match="Tensor or str"):
        create_invisible_mask([torch.zeros(4, 4), 42])
