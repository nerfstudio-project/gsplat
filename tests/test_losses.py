"""Tests for gsplat.losses module."""

import math

import pytest
import torch
import torch.nn.functional as F

from gsplat.losses import (
    create_ssim_window,
    depth_l1_loss,
    l1_loss,
    mse_loss,
    opacity_reg_loss,
    scale_reg_loss,
    ssim_loss,
    torch_ssim_loss,
    total_variation_loss,
)


# ---------------------------------------------------------------------------
# L1 / MSE
# ---------------------------------------------------------------------------


class TestL1Loss:
    def test_identical_inputs(self):
        x = torch.rand(4, 3)
        assert (l1_loss(x, x) == 0).all()

    def test_shape_preserved(self):
        x = torch.rand(2, 5, 3)
        y = torch.rand(2, 5, 3)
        assert l1_loss(x, y).shape == x.shape

    def test_known_value(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 1.0, 4.0])
        expected = torch.tensor([0.5, 1.0, 1.0])
        assert torch.allclose(l1_loss(pred, target), expected)

    def test_matches_pytorch(self):
        pred = torch.tensor([0.2, 0.8, -0.5, 1.3])
        target = torch.tensor([0.0, 1.0, 0.5, 0.3])
        assert torch.allclose(
            l1_loss(pred, target),
            F.l1_loss(pred, target, reduction="none"),
        )

    def test_gradient_values(self):
        pred = torch.tensor([2.0, 0.5], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        l1_loss(pred, target).sum().backward()
        # L1 gradient is sign(pred - target): [+1, -1]
        assert torch.allclose(pred.grad, torch.tensor([1.0, -1.0]))


class TestMSELoss:
    def test_identical_inputs(self):
        x = torch.rand(4, 3)
        assert (mse_loss(x, x) == 0).all()

    def test_shape_preserved(self):
        x = torch.rand(2, 5, 3)
        y = torch.rand(2, 5, 3)
        assert mse_loss(x, y).shape == x.shape

    def test_known_value(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 2.0, 5.0])
        expected = torch.tensor([1.0, 0.0, 4.0])
        assert torch.allclose(mse_loss(pred, target), expected)

    def test_matches_pytorch(self):
        pred = torch.tensor([0.2, 0.8, -0.5, 1.3])
        target = torch.tensor([0.0, 1.0, 0.5, 0.3])
        assert torch.allclose(
            mse_loss(pred, target),
            F.mse_loss(pred, target, reduction="none"),
        )

    def test_gradient_values(self):
        pred = torch.tensor([3.0, 1.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        mse_loss(pred, target).sum().backward()
        # MSE gradient is 2*(pred - target): [4.0, 0.0]
        assert torch.allclose(pred.grad, torch.tensor([4.0, 0.0]))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


class TestSSIM:
    def test_create_window_shape(self):
        w = create_ssim_window(11, 3)
        assert w.shape == (3, 1, 11, 11)

    def test_create_window_normalized(self):
        w = create_ssim_window(11, 1)
        # Each channel slice should sum to ~1
        assert torch.isclose(w.sum(), torch.tensor(1.0), atol=1e-5)

    def test_torch_ssim_loss_identical(self):
        img = torch.rand(2, 3, 64, 64)
        window = create_ssim_window(11, 3)
        ssim_map = torch_ssim_loss(img, img, window, 11, 3)
        # Identical images should have SSIM ~1 everywhere (except borders)
        center = ssim_map[:, :, 5:-5, 5:-5]
        assert (center > 0.99).all()

    def test_torch_ssim_loss_different(self):
        img1 = torch.zeros(1, 1, 32, 32)
        img2 = torch.ones(1, 1, 32, 32)
        window = create_ssim_window(11, 1)
        ssim_map = torch_ssim_loss(img1, img2, window, 11, 1)
        # Very different images should have low SSIM
        assert ssim_map.mean() < 0.5

    def test_ssim_loss_identical(self):
        img = torch.rand(1, 3, 64, 64)
        loss = ssim_loss(img, img, window_size=11)
        assert loss.item() < 0.01  # ~0 for identical

    def test_ssim_loss_range(self):
        img1 = torch.rand(1, 3, 64, 64)
        img2 = torch.rand(1, 3, 64, 64)
        loss = ssim_loss(img1, img2, window_size=11)
        assert 0.0 <= loss.item() <= 2.0  # 1-SSIM in [0, 2]

    def test_ssim_loss_gradient_nonzero(self):
        img1 = torch.rand(1, 3, 32, 32, requires_grad=True)
        img2 = torch.rand(1, 3, 32, 32)
        loss = ssim_loss(img1, img2, window_size=11)
        loss.backward()
        assert img1.grad is not None
        # Gradient should be nonzero for dissimilar images
        assert img1.grad.abs().sum() > 0

    def test_torch_ssim_loss_constant_images_reference(self):
        # For constant images of value c, SSIM = 1.0 regardless of c
        for c in [0.0, 0.5, 1.0]:
            img = torch.full((1, 1, 32, 32), c)
            window = create_ssim_window(11, 1)
            ssim_map = torch_ssim_loss(img, img, window, 11, 1)
            center = ssim_map[:, :, 5:-5, 5:-5]
            assert (center > 0.999).all(), f"SSIM should be ~1 for constant image c={c}"


# ---------------------------------------------------------------------------
# Depth loss
# ---------------------------------------------------------------------------


class TestDepthL1Loss:
    def test_identical_depths(self):
        d = torch.rand(4, 3).clamp(min=0.1)
        loss = depth_l1_loss(d, d)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        pred = torch.tensor([1.0, 2.0, 4.0])
        gt = torch.tensor([2.0, 2.0, 1.0])
        loss = depth_l1_loss(pred, gt)
        assert loss.item() > 0.0

    def test_scene_scale(self):
        pred = torch.tensor([1.0, 2.0])
        gt = torch.tensor([2.0, 4.0])
        loss1 = depth_l1_loss(pred, gt, scene_scale=1.0)
        loss2 = depth_l1_loss(pred, gt, scene_scale=2.0)
        assert torch.isclose(loss2, loss1 * 2.0, atol=1e-6)

    def test_zero_pred_depth_handling(self):
        pred = torch.tensor([0.0, 1.0])
        gt = torch.tensor([1.0, 1.0])
        # Should not raise; zero pred depth maps to zero disparity
        loss = depth_l1_loss(pred, gt)
        assert torch.isfinite(loss)

    def test_zero_gt_depth_handling(self):
        pred = torch.tensor([1.0, 2.0])
        gt = torch.tensor([0.0, 2.0])
        # Should not raise; zero gt depth maps to zero disparity (no inf/nan)
        loss = depth_l1_loss(pred, gt)
        assert torch.isfinite(loss)

    def test_known_disparity_value(self):
        # pred=2 -> disp=0.5, gt=4 -> disp=0.25, |0.5-0.25|=0.25
        pred = torch.tensor([2.0])
        gt = torch.tensor([4.0])
        loss = depth_l1_loss(pred, gt, scene_scale=1.0)
        assert torch.isclose(loss, torch.tensor(0.25))


# ---------------------------------------------------------------------------
# Total variation
# ---------------------------------------------------------------------------


class TestTotalVariationLoss:
    def test_constant_tensor(self):
        x = torch.ones(2, 3, 4, 5)
        loss = total_variation_loss(x)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_positive_for_random(self):
        x = torch.rand(2, 3, 8, 8)
        loss = total_variation_loss(x)
        assert loss.item() > 0.0

    def test_single_spatial(self):
        x = torch.rand(2, 3, 1, 1)
        loss = total_variation_loss(x)
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_5d_tensor(self):
        x = torch.rand(2, 3, 4, 4, 4)
        loss = total_variation_loss(x)
        assert loss.item() > 0.0

    def test_gradient_nonzero(self):
        x = torch.rand(2, 3, 4, 4, requires_grad=True)
        total_variation_loss(x).backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_known_value_1d(self):
        # 1D spatial: x = [0, 1, 0] along last dim
        x = torch.tensor([[[0.0, 1.0, 0.0]]]).unsqueeze(0)  # (1, 1, 1, 3)
        loss = total_variation_loss(x)
        # diff = [1, -1], squared = [1, 1], sum = 2, count = 2 (1*2), tv = 2/2 = 1, /batch=1
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_manual_variation_5d(self):
        # Manual variation test for 5D tensors with diagonal ones
        batch_size, num_channels, xyz_size = 5, 7, 2
        x = torch.zeros(batch_size, num_channels, xyz_size, xyz_size, xyz_size)
        x[:, :, 0, 0, 0] = 1.0
        x[:, :, 1, 1, 1] = 1.0

        # Diagonally opposite "1"s accumulate TV in all 3 dims, 2 units each.
        # Both channels and batch cancel in the normalized result.
        number_of_deltas_per_dim = (xyz_size - 1) * xyz_size * xyz_size
        expected_tv_per_sample = 3 * 2 / number_of_deltas_per_dim  # = 3.0
        expected_mean = expected_tv_per_sample  # batch cancels

        loss = total_variation_loss(x)
        assert torch.isclose(
            loss, torch.tensor(expected_mean)
        ), f"Expected {expected_mean}, got {loss.item()}"


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------


class TestOpacityRegLoss:
    def test_returns_scalar(self):
        loss = opacity_reg_loss(torch.randn(100))
        assert loss.shape == ()

    def test_range(self):
        loss = opacity_reg_loss(torch.randn(1000))
        assert 0.0 < loss.item() < 1.0  # sigmoid output mean is ~0.5

    def test_high_logits(self):
        loss = opacity_reg_loss(torch.full((10,), 100.0))
        assert torch.isclose(loss, torch.tensor(1.0), atol=1e-4)

    def test_known_value_zero_logits(self):
        # sigmoid(0) = 0.5, mean = 0.5
        loss = opacity_reg_loss(torch.zeros(5))
        assert torch.isclose(loss, torch.tensor(0.5))

    def test_gradient_values(self):
        x = torch.zeros(1, requires_grad=True)
        opacity_reg_loss(x).backward()
        # d/dx sigmoid(x).mean() at x=0: sigmoid(0)*(1-sigmoid(0)) = 0.25
        assert torch.isclose(x.grad, torch.tensor(0.25))


class TestScaleRegLoss:
    def test_returns_scalar(self):
        loss = scale_reg_loss(torch.randn(100, 3))
        assert loss.shape == ()

    def test_zero_log_scales(self):
        loss = scale_reg_loss(torch.zeros(10, 3))
        assert torch.isclose(loss, torch.tensor(1.0))  # exp(0) = 1

    def test_known_value(self):
        # exp(1) = e ≈ 2.7183, mean over 3 elements = e
        loss = scale_reg_loss(torch.ones(1, 3))
        assert torch.isclose(loss, torch.tensor(math.e), atol=1e-4)

    def test_gradient_values(self):
        x = torch.zeros(1, 1, requires_grad=True)
        scale_reg_loss(x).backward()
        # d/dx exp(x).mean() at x=0: exp(0) = 1.0
        assert torch.isclose(x.grad, torch.tensor([[1.0]]))
