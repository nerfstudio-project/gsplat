# SPDX-License-Identifier: Apache-2.0
"""Tests for the G-SHARP v0.2 additions to ``gsplat.losses`` (branch: vnath_gsharp).

Covers the four ported functions:

- :func:`gsplat.losses.binocular_disparity_l1` — L1 in inverse-depth space, mask-aware.
- :func:`gsplat.losses.pearson_depth_loss` — ``1 - Pearson r`` on (masked) flattened depth pairs.
- :func:`gsplat.losses.masked_l1` — L1 over only the ``mask != 0`` region.
- :func:`gsplat.losses.masked_ssim` — SSIM loss after zeroing both inputs in the masked region.

Seed is set to 42 by the autouse fixture in ``conftest.py``.
"""

import pytest
import torch

from gsplat.losses import (
    binocular_disparity_l1,
    masked_l1,
    masked_ssim,
    pearson_depth_loss,
)


# ---------------------------------------------------------------------------
# binocular_disparity_l1
# ---------------------------------------------------------------------------


def test_binocular_disparity_l1_identical_depths_is_zero():
    depth = torch.rand(2, 4, 8) + 0.1  # strictly positive
    loss = binocular_disparity_l1(depth, depth)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_binocular_disparity_l1_gradient_flows():
    pred = (torch.rand(2, 4, 8) + 0.1).requires_grad_(True)
    gt = torch.rand(2, 4, 8) + 0.1
    loss = binocular_disparity_l1(pred, gt)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_binocular_disparity_l1_mask_slices_correct():
    """Errors in mask==0 region must not contribute to the loss."""
    pred = torch.full((2, 4, 8), 0.5)
    gt = pred.clone()
    # Inject a large error only where the mask is zero.
    pred_with_bad_left = pred.clone()
    pred_with_bad_left[..., :4] = 100.0

    mask = torch.zeros(2, 4, 8)
    mask[..., 4:] = 1.0  # only the right half counts

    loss = binocular_disparity_l1(pred_with_bad_left, gt, mask=mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_binocular_disparity_l1_shape_mismatch_raises():
    pred = torch.rand(2, 4, 8) + 0.1
    gt = torch.rand(2, 4, 7) + 0.1
    with pytest.raises(ValueError, match="shape"):
        binocular_disparity_l1(pred, gt)


def test_binocular_disparity_l1_all_masked_returns_zero_no_nan():
    pred = torch.rand(2, 4, 8) + 0.1
    gt = torch.rand(2, 4, 8) + 0.1
    mask = torch.zeros(2, 4, 8)
    loss = binocular_disparity_l1(pred, gt, mask=mask)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_binocular_disparity_l1_zero_depth_handled_via_eps():
    """Zero (sentinel for ``no depth``) must not produce NaN / Inf in the inverse step."""
    pred = torch.tensor([[1.0, 0.0, 2.0, 0.5]])
    gt = torch.tensor([[1.0, 0.0, 2.0, 0.5]])
    loss = binocular_disparity_l1(pred, gt)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_binocular_disparity_l1_one_side_invalid_excludes_pair():
    """If only one side of a pair is invalid (depth ~ 0), that pair must not
    leak ``|0 - 1/other|`` into the loss — the docstring promises pair-level
    validity, not per-side."""
    # Pair 0: both valid, identical → contributes 0.
    # Pair 1: pred invalid (0.0), gt = 2.0 → excluded.
    # Pair 2: pred = 4.0, gt invalid (0.0) → excluded.
    # Pair 3: both valid, identical → contributes 0.
    pred = torch.tensor([[1.0, 0.0, 4.0, 0.5]])
    gt = torch.tensor([[1.0, 2.0, 0.0, 0.5]])
    loss = binocular_disparity_l1(pred, gt)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


# ---------------------------------------------------------------------------
# pearson_depth_loss
# ---------------------------------------------------------------------------


def test_pearson_depth_loss_correlated_inputs_zero():
    gt = torch.rand(2, 4, 8)
    pred = gt.clone()  # perfect positive correlation -> loss = 0
    loss = pearson_depth_loss(pred, gt)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_pearson_depth_loss_anticorrelated_inputs_two():
    gt = torch.rand(2, 4, 8)
    pred = -gt  # perfect negative correlation -> loss = 1 - (-1) = 2
    loss = pearson_depth_loss(pred, gt)
    assert torch.allclose(loss, torch.tensor(2.0), atol=1e-5)


def test_pearson_depth_loss_shape_mismatch_raises():
    pred = torch.rand(2, 4, 8)
    gt = torch.rand(2, 4, 7)
    with pytest.raises(ValueError, match="shape"):
        pearson_depth_loss(pred, gt)


def test_pearson_depth_loss_constant_input_no_nan():
    """Pearson is undefined when one input has zero variance; loss must stay finite."""
    pred = torch.zeros(2, 4, 8)
    gt = torch.rand(2, 4, 8)
    loss = pearson_depth_loss(pred, gt)
    assert torch.isfinite(loss).all()


def test_pearson_depth_loss_fully_masked_returns_zero_no_nan():
    pred = torch.rand(2, 4, 8)
    gt = torch.rand(2, 4, 8)
    mask = torch.zeros(2, 4, 8)
    loss = pearson_depth_loss(pred, gt, mask=mask)
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# masked_l1
# ---------------------------------------------------------------------------


def test_masked_l1_ignores_masked_out_pixels():
    pred = torch.zeros(2, 3, 8, 8)
    gt = torch.zeros(2, 3, 8, 8)
    pred[..., :4] = 1.0  # error only on the masked-out (left) half

    mask = torch.zeros(2, 1, 8, 8)
    mask[..., 4:] = 1.0  # right half is the active region

    loss = masked_l1(pred, gt, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_masked_l1_empty_mask_returns_zero_no_nan():
    pred = torch.rand(2, 3, 8, 8)
    gt = torch.rand(2, 3, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    loss = masked_l1(pred, gt, mask)
    assert torch.isfinite(loss).all()
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_masked_l1_shape_mismatch_raises():
    pred = torch.rand(2, 3, 8, 8)
    gt = torch.rand(2, 3, 8, 7)
    mask = torch.ones(2, 1, 8, 8)
    with pytest.raises(ValueError, match="shape"):
        masked_l1(pred, gt, mask)


def test_masked_l1_gradient_flows():
    pred = torch.rand(2, 3, 8, 8, requires_grad=True)
    gt = torch.rand(2, 3, 8, 8)
    mask = torch.ones(2, 1, 8, 8)
    loss = masked_l1(pred, gt, mask)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# masked_ssim
# ---------------------------------------------------------------------------


def test_masked_ssim_identical_inputs_loss_near_zero():
    img = torch.rand(2, 3, 32, 32)  # values already in [0, 1]
    mask = torch.ones(2, 1, 32, 32)
    loss = masked_ssim(img, img, mask)
    assert loss.item() < 1e-3


def test_masked_ssim_shape_mismatch_raises():
    pred = torch.rand(2, 3, 32, 32)
    gt = torch.rand(2, 3, 32, 31)
    mask = torch.ones(2, 1, 32, 32)
    with pytest.raises(ValueError, match="shape"):
        masked_ssim(pred, gt, mask)


def test_masked_ssim_gradient_flows():
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    gt = torch.rand(2, 3, 32, 32)
    mask = torch.ones(2, 1, 32, 32)
    loss = masked_ssim(pred, gt, mask)
    loss.backward()
    assert pred.grad is not None


def test_masked_ssim_partial_mask_loss_near_zero():
    """Partial mask: identical inputs in the active half, mismatch in the masked-out half.

    `masked_ssim` zeros both inputs in the masked-out region before SSIM, so
    pred*mask and gt*mask are identical everywhere (matching values in the
    active half; both zeroed in the inactive half). SSIM ≈ 1 → loss ≈ 0.

    This pins the "masked-out region is excluded" semantic against any
    future drift to a crop-then-SSIM alternative.
    """
    pred = torch.rand(2, 3, 32, 32)
    gt = pred.clone()
    # Inject a deliberate mismatch only in the left half (masked-out).
    gt[..., :16] = torch.rand(2, 3, 32, 16)

    mask = torch.zeros(2, 1, 32, 32)
    mask[..., 16:] = 1.0  # active region: right half only

    loss = masked_ssim(pred, gt, mask)
    assert loss.item() < 1e-3


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No CUDA device"
            ),
        ),
    ]
)
def masked_loss_device(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def masked_loss_dtype(request):
    return request.param


def _assert_differentiable_zero(loss, pred, gt):
    assert loss.shape == torch.Size([])
    assert loss.dtype == pred.dtype
    assert loss.device == pred.device
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred), rtol=0, atol=0)
    assert gt.grad is None


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), -float("inf"), "overflow"],
    ids=["nan", "inf", "negative-inf", "overflow"],
)
@pytest.mark.parametrize(
    "loss_fn,valid_samples,layout",
    [
        (masked_l1, 0, "full"),
        (masked_l1, 0, "broadcast"),
        (masked_l1, 0, "noncontiguous"),
        (pearson_depth_loss, 0, "full"),
        (pearson_depth_loss, 1, "full"),
        (pearson_depth_loss, 0, "noncontiguous"),
        (pearson_depth_loss, 1, "noncontiguous"),
    ],
)
def test_masked_losses_degenerate_nonfinite(
    masked_loss_device, masked_loss_dtype, bad_value, loss_fn, valid_samples, layout
):
    if bad_value == "overflow":
        bad_value = torch.finfo(masked_loss_dtype).max * 0.75
    pred = torch.tensor(
        [[bad_value, bad_value], [2.0, 4.0]],
        device=masked_loss_device,
        dtype=masked_loss_dtype,
    )
    gt = torch.tensor(
        [[bad_value, bad_value], [3.0, 7.0]],
        device=masked_loss_device,
        dtype=masked_loss_dtype,
    )
    mask = torch.zeros_like(pred, dtype=torch.bool)
    if valid_samples:
        mask[1, 0] = True
    if layout == "broadcast":
        pred = pred.repeat(2, 3, 1, 1)
        gt = gt.repeat(2, 3, 1, 1)
        mask = mask.repeat(2, 1, 1, 1)
    elif layout == "noncontiguous":
        pred, gt, mask = pred.t(), gt.t(), mask.t()
        assert not pred.is_contiguous()
    pred.requires_grad_()
    gt.requires_grad_()
    _assert_differentiable_zero(loss_fn(pred, gt, mask), pred, gt)


@pytest.mark.parametrize(
    "loss_fn,masked",
    [(masked_l1, True), (pearson_depth_loss, True), (pearson_depth_loss, False)],
)
def test_masked_losses_empty_inputs(
    masked_loss_device, masked_loss_dtype, loss_fn, masked
):
    pred = torch.empty(
        2, 0, device=masked_loss_device, dtype=masked_loss_dtype, requires_grad=True
    )
    gt = torch.empty_like(pred, requires_grad=True)
    mask = torch.zeros_like(pred, dtype=torch.bool) if masked else None
    _assert_differentiable_zero(loss_fn(pred, gt, mask), pred, gt)


@pytest.mark.parametrize("value", [2.0, float("nan"), float("inf"), -float("inf")])
def test_pearson_depth_loss_unmasked_singleton(
    masked_loss_device, masked_loss_dtype, value
):
    pred = torch.tensor(
        [value], device=masked_loss_device, dtype=masked_loss_dtype, requires_grad=True
    )
    gt = torch.ones_like(pred, requires_grad=True)
    _assert_differentiable_zero(pearson_depth_loss(pred, gt), pred, gt)


@pytest.mark.parametrize("loss_fn", [masked_l1, pearson_depth_loss])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_masked_losses_nonfinite_excluded_reference(
    masked_loss_device, masked_loss_dtype, loss_fn, bad_value
):
    pred = torch.tensor(
        [bad_value, 1.0, 2.0, bad_value, 4.0, 6.0],
        device=masked_loss_device,
        dtype=masked_loss_dtype,
        requires_grad=True,
    )
    gt = torch.tensor(
        [bad_value, 2.0, 4.0, bad_value, 3.0, 8.0],
        device=masked_loss_device,
        dtype=masked_loss_dtype,
        requires_grad=True,
    )
    mask = torch.tensor(
        [False, True, True, False, True, True], device=masked_loss_device
    )
    reference_pred = pred[mask].detach().clone().requires_grad_()
    reference_gt = gt[mask].detach().clone().requires_grad_()
    if loss_fn is masked_l1:
        expected = (reference_pred - reference_gt).abs().mean()
    else:
        expected = (
            1.0 - torch.corrcoef(torch.stack([reference_pred, reference_gt]))[0, 1]
        )
    actual = loss_fn(pred, gt, mask)
    assert actual.dtype == pred.dtype and actual.device == pred.device
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.backward()
    expected.backward()
    for tensor, reference in ((pred, reference_pred), (gt, reference_gt)):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        torch.testing.assert_close(
            tensor.grad[mask], reference.grad, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            tensor.grad[~mask], torch.zeros_like(tensor.grad[~mask]), rtol=0, atol=0
        )
