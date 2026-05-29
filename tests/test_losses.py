# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for gsplat.losses module."""

import math

import pytest
import torch
import torch.nn.functional as F

import gsplat.losses as _losses_module

_losses_module.ENFORCE_CONTRACTS = True

from gsplat.losses import (
    create_ssim_window,
    depth_l1_loss,
    l1_loss,
    lidar_background_loss,
    lidar_distance_loss,
    lidar_intensity_loss,
    lidar_raydrop_loss,
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
# LiDAR losses
# ---------------------------------------------------------------------------


class TestLidarDistanceLoss:
    def test_identical_distances(self):
        d = torch.rand(100).clamp(min=0.1)
        assert torch.isclose(lidar_distance_loss(d, d), torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        pred = torch.tensor([1.0, 2.0, 4.0])
        gt = torch.tensor([2.0, 3.0, 1.0])
        assert lidar_distance_loss(pred, gt).item() > 0.0

    def test_known_value(self):
        pred = torch.tensor([1.0, 3.0])
        gt = torch.tensor([2.0, 5.0])
        # L1: (|1-2| + |3-5|) / 2 = 1.5
        assert torch.isclose(lidar_distance_loss(pred, gt), torch.tensor(1.5))

    def test_valid_mask(self):
        pred = torch.tensor([1.0, 100.0, 3.0])
        gt = torch.tensor([2.0, 200.0, 5.0])
        mask = torch.tensor([True, False, True])
        # Only indices 0 and 2: (|1-2| + |3-5|) / 2 = 1.5
        assert torch.isclose(
            lidar_distance_loss(pred, gt, valid_mask=mask), torch.tensor(1.5)
        )

    def test_empty_mask_returns_zero(self):
        pred = torch.tensor([1.0, 2.0])
        gt = torch.tensor([3.0, 4.0])
        mask = torch.tensor([False, False])
        # F.l1_loss on empty tensors returns 0
        loss = lidar_distance_loss(pred, gt, valid_mask=mask)
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        pred = torch.tensor([1.0, 3.0], requires_grad=True)
        gt = torch.tensor([2.0, 5.0])
        lidar_distance_loss(pred, gt).backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_loss_fn_mse(self):
        pred = torch.tensor([1.0, 3.0])
        gt = torch.tensor([2.0, 5.0])
        # MSE: ((1-2)^2 + (3-5)^2) / 2 = (1+4)/2 = 2.5
        assert torch.isclose(
            lidar_distance_loss(pred, gt, loss_fn="mse"), torch.tensor(2.5)
        )

    def test_nd_shapes(self):
        pred = torch.rand(2, 64, 1)
        gt = torch.rand(2, 64, 1)
        loss = lidar_distance_loss(pred, gt)
        assert loss.ndim == 0


class TestLidarIntensityLoss:
    def test_identical_intensities(self):
        x = torch.rand(100)
        assert torch.isclose(lidar_intensity_loss(x, x), torch.tensor(0.0), atol=1e-6)

    def test_positive_for_different(self):
        pred = torch.tensor([0.1, 0.5, 0.9])
        gt = torch.tensor([0.2, 0.3, 0.8])
        assert lidar_intensity_loss(pred, gt).item() > 0.0

    def test_known_value_default_l1(self):
        pred = torch.tensor([0.1, 0.3])
        gt = torch.tensor([0.2, 0.5])
        # Default L1: (|0.1-0.2| + |0.3-0.5|) / 2 = (0.1 + 0.2) / 2 = 0.15
        assert torch.isclose(
            lidar_intensity_loss(pred, gt), torch.tensor(0.15), atol=1e-6
        )

    def test_valid_mask(self):
        pred = torch.tensor([0.1, 999.0, 0.3])
        gt = torch.tensor([0.2, 0.0, 0.5])
        mask = torch.tensor([True, False, True])
        expected = F.l1_loss(torch.tensor([0.1, 0.3]), torch.tensor([0.2, 0.5]))
        assert torch.isclose(lidar_intensity_loss(pred, gt, valid_mask=mask), expected)

    def test_loss_fn_mse(self):
        pred = torch.tensor([0.1, 0.3])
        gt = torch.tensor([0.2, 0.5])
        expected = F.mse_loss(pred, gt)
        assert torch.isclose(lidar_intensity_loss(pred, gt, loss_fn="mse"), expected)

    def test_gradient_flows(self):
        pred = torch.tensor([0.5, 0.8], requires_grad=True)
        gt = torch.tensor([0.3, 0.9])
        lidar_intensity_loss(pred, gt).backward()
        assert pred.grad is not None


class TestLidarRaydropLoss:
    def test_confident_prediction_low_loss(self):
        # Default is BCE-with-logits: pred are logits. Large positive → class 1,
        # large negative → class 0.
        pred = torch.tensor([10.0, -10.0])
        gt = torch.tensor([1.0, 0.0])
        loss = lidar_raydrop_loss(pred, gt)
        assert loss.item() < 0.01

    def test_wrong_prediction_high_loss(self):
        # Confident wrong logits → large BCE-with-logits loss
        pred = torch.tensor([-10.0, 10.0])
        gt = torch.tensor([1.0, 0.0])
        loss = lidar_raydrop_loss(pred, gt)
        assert loss.item() > 5.0

    def test_valid_mask(self):
        pred = torch.tensor([10.0, 99.0, -10.0])
        gt = torch.tensor([1.0, 1.0, 0.0])
        mask = torch.tensor([True, False, True])
        loss = lidar_raydrop_loss(pred, gt, valid_mask=mask)
        # Only indices 0, 2: confident correct → near-zero
        assert loss.item() < 0.01

    def test_loss_fn_mse(self):
        pred = torch.tensor([0.0, 1.0])
        gt = torch.tensor([1.0, 0.0])
        # MSE: ((0-1)^2 + (1-0)^2) / 2 = 1.0
        loss = lidar_raydrop_loss(pred, gt, loss_fn="mse")
        assert torch.isclose(loss, torch.tensor(1.0))

    def test_gradient_flows(self):
        pred = torch.tensor([0.5, 0.5], requires_grad=True)
        gt = torch.tensor([1.0, 0.0])
        lidar_raydrop_loss(pred, gt).backward()
        assert pred.grad is not None


class TestLidarBackgroundLoss:
    def test_correct_foreground_opacity(self):
        # High opacity on foreground rays, low on background → small BCE
        pred_opacity = torch.tensor([0.99, 0.99, 0.01, 0.01])
        bg_mask = torch.tensor([False, False, True, True])
        loss = lidar_background_loss(pred_opacity, bg_mask)
        # BCE with confident-correct probabilities is small
        assert loss.item() < 0.02

    def test_wrong_opacity_high_loss(self):
        # High opacity on background rays → large BCE
        pred_opacity = torch.tensor([0.01, 0.01, 0.99, 0.99])
        bg_mask = torch.tensor([False, False, True, True])
        loss = lidar_background_loss(pred_opacity, bg_mask)
        # BCE with confident-wrong probabilities explodes
        assert loss.item() > 3.0

    def test_valid_mask(self):
        pred_opacity = torch.tensor([0.99, 0.5, 0.01])
        bg_mask = torch.tensor([False, False, True])
        mask = torch.tensor([True, False, True])
        loss = lidar_background_loss(pred_opacity, bg_mask, valid_mask=mask)
        # Only indices 0 (fg, 0.99 vs 1) and 2 (bg, 0.01 vs 0) → very small BCE
        assert loss.item() < 0.02

    def test_loss_fn_bce_clipped(self):
        pred = torch.tensor([0.99, 0.99, 0.01, 0.01])
        bg_mask = torch.tensor([False, False, True, True])
        # Should not NaN even with near-boundary values
        loss = lidar_background_loss(pred, bg_mask, loss_fn="bce_clipped")
        assert torch.isfinite(loss)
        assert loss.item() < 0.1

    def test_loss_fn_mse(self):
        pred = torch.tensor([0.99, 0.99, 0.01, 0.01])
        bg_mask = torch.tensor([False, False, True, True])
        # MSE: mean((0.99-1)^2, (0.99-1)^2, (0.01-0)^2, (0.01-0)^2) = 1e-4
        loss = lidar_background_loss(pred, bg_mask, loss_fn="mse")
        assert torch.isclose(loss, torch.tensor(1e-4), atol=1e-6)

    def test_bce_clipped_is_smaller_than_bce_at_boundary(self):
        # At the exact 0/1 boundary, raw BCE saturates to PyTorch's internal
        # clamp (100.0 per F.binary_cross_entropy's log(0) guard), while
        # bce_clipped clips pred further inside [0, 1] and produces a much
        # smaller, numerically-useful gradient signal.
        pred = torch.tensor([1.0, 0.0])
        bg_mask = torch.tensor([True, False])  # target = [0.0, 1.0] — all wrong
        bce_loss = lidar_background_loss(pred, bg_mask, loss_fn="bce")
        bce_clipped_loss = lidar_background_loss(pred, bg_mask, loss_fn="bce_clipped")
        assert torch.isfinite(bce_loss) and torch.isfinite(bce_clipped_loss)
        # Raw BCE saturates at PyTorch's ~100 clamp; bce_clipped stays small.
        assert bce_loss.item() > 50.0
        assert bce_clipped_loss.item() < 20.0
        assert bce_clipped_loss.item() < bce_loss.item()

    def test_gradient_flows(self):
        pred = torch.tensor([0.5, 0.5], requires_grad=True)
        bg_mask = torch.tensor([False, True])
        lidar_background_loss(pred, bg_mask).backward()
        assert pred.grad is not None


# ---------------------------------------------------------------------------
# Shared loss_fn behavior across LiDAR loss functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["l1", "mse", "huber", "smooth_l1"])
def test_distance_all_variants_finite(name):
    pred = torch.rand(8)
    gt = torch.rand(8)
    loss = lidar_distance_loss(pred, gt, loss_fn=name)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_loss_fn_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss_fn"):
        lidar_distance_loss(torch.zeros(4), torch.zeros(4), loss_fn="xyz")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        lidar_distance_loss(torch.zeros(4), torch.zeros(5))


def test_mask_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        lidar_distance_loss(
            torch.zeros(4), torch.zeros(4), valid_mask=torch.zeros(5).bool()
        )


def test_loss_fn_custom_callable():
    # Real callable: 2×L1, unreduced
    def my_loss(p, t):
        return (p - t).abs() * 2.0

    pred = torch.tensor([1.0, 3.0])
    gt = torch.tensor([2.0, 5.0])
    # mean(2*|1-2| + 2*|3-5|) / 2 = 3.0
    assert torch.isclose(
        lidar_distance_loss(pred, gt, loss_fn=my_loss), torch.tensor(3.0)
    )


def test_loss_fn_callable_wrong_shape_raises():
    # A callable that incorrectly reduces to scalar should be caught.
    def scalar_loss(p, t):
        return F.l1_loss(p, t)  # default reduction='mean' → scalar

    with pytest.raises(ValueError, match="per-element"):
        lidar_distance_loss(torch.zeros(4), torch.zeros(4), loss_fn=scalar_loss)


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


# ---------------------------------------------------------------------------
# Standard loss wrappers
# ---------------------------------------------------------------------------

from gsplat.losses import (
    LinearLambdaScheduler,
    bce_clipped,
    bce_loss,
    bce_with_logits_loss,
    cross_entropy_loss,
    depth_inverse_mse,
    huber_loss,
    identity_distance,
    log_l1,
    normal_cosine_loss,
    relu_sum,
    smooth_l1_loss,
    total_variation_temporal,
    weights_reg,
)


class TestHuberLoss:
    def test_identical(self):
        x = torch.rand(4, 3)
        assert (huber_loss(x, x) == 0).all()

    def test_shape(self):
        x, y = torch.rand(2, 5), torch.rand(2, 5)
        assert huber_loss(x, y).shape == x.shape

    def test_reference_value(self):
        # For |error| <= delta: 0.5 * error^2. error=0.5, delta=1 -> 0.125
        pred = torch.tensor([1.5])
        target = torch.tensor([1.0])
        assert torch.isclose(huber_loss(pred, target), torch.tensor(0.125))

    def test_large_error_reference(self):
        # For |error| > delta: delta * (|error| - 0.5*delta). error=2, delta=1 -> 1.5
        pred = torch.tensor([3.0])
        target = torch.tensor([1.0])
        assert torch.isclose(huber_loss(pred, target), torch.tensor(1.5))

    def test_gradient_values(self):
        # In quadratic region: grad = error. pred=1.3, target=1.0, error=0.3
        pred = torch.tensor([1.3], requires_grad=True)
        target = torch.tensor([1.0])
        huber_loss(pred, target).sum().backward()
        assert torch.isclose(pred.grad, torch.tensor([0.3]), atol=1e-6)


class TestSmoothL1Loss:
    def test_identical(self):
        x = torch.rand(4, 3)
        assert (smooth_l1_loss(x, x) == 0).all()

    def test_shape(self):
        x, y = torch.rand(2, 5), torch.rand(2, 5)
        assert smooth_l1_loss(x, y).shape == x.shape

    def test_matches_pytorch(self):
        pred = torch.tensor([0.2, 0.8, 2.5])
        target = torch.tensor([0.0, 1.0, 0.0])
        assert torch.allclose(
            smooth_l1_loss(pred, target),
            F.smooth_l1_loss(pred, target, reduction="none"),
        )

    def test_gradient_nonzero(self):
        x = torch.rand(4, 3, requires_grad=True)
        smooth_l1_loss(x, torch.rand(4, 3)).sum().backward()
        assert x.grad.abs().sum() > 0


class TestBCELoss:
    def test_matches_pytorch(self):
        pred = torch.tensor([0.2, 0.7, 0.9])
        target = torch.tensor([0.0, 1.0, 1.0])
        expected = F.binary_cross_entropy(pred, target, reduction="none")
        assert torch.allclose(bce_loss(pred, target), expected)

    def test_shape(self):
        x = torch.rand(4, 3)
        assert bce_loss(x, torch.rand(4, 3)).shape == x.shape

    def test_known_value(self):
        # BCE(0.5, 1.0) = -log(0.5) = log(2) ≈ 0.6931
        loss = bce_loss(torch.tensor([0.5]), torch.tensor([1.0]))
        assert torch.isclose(loss, torch.tensor([math.log(2)]), atol=1e-4)

    def test_gradient_nonzero(self):
        x = torch.rand(4, 3, requires_grad=True)
        bce_loss(x, torch.rand(4, 3)).sum().backward()
        assert x.grad.abs().sum() > 0


class TestBCEWithLogitsLoss:
    def test_shape(self):
        x = torch.randn(4, 3)
        assert bce_with_logits_loss(x, torch.rand(4, 3)).shape == x.shape

    def test_matches_pytorch(self):
        logits = torch.randn(10)
        targets = torch.rand(10)
        expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        assert torch.allclose(bce_with_logits_loss(logits, targets), expected)

    def test_known_value(self):
        # logit=0 -> sigmoid=0.5 -> BCE(0.5,1) = log(2) ≈ 0.6931
        loss = bce_with_logits_loss(torch.tensor([0.0]), torch.tensor([1.0]))
        assert torch.isclose(loss, torch.tensor([math.log(2)]), atol=1e-4)

    def test_gradient_nonzero(self):
        x = torch.randn(4, 3, requires_grad=True)
        bce_with_logits_loss(x, torch.rand(4, 3)).sum().backward()
        assert x.grad.abs().sum() > 0


class TestCrossEntropyLoss:
    def test_shape(self):
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        assert cross_entropy_loss(logits, targets).shape == (8,)

    def test_matches_pytorch(self):
        logits = torch.randn(4, 3)
        targets = torch.tensor([0, 1, 2, 1])
        expected = F.cross_entropy(logits, targets, reduction="none")
        assert torch.allclose(cross_entropy_loss(logits, targets), expected)

    def test_known_value(self):
        # Uniform logits [0,0,0] with target=0: -log(1/3) = log(3) ≈ 1.0986
        logits = torch.zeros(1, 3)
        loss = cross_entropy_loss(logits, torch.tensor([0]))
        assert torch.isclose(loss, torch.tensor([math.log(3)]), atol=1e-4)

    def test_gradient_nonzero(self):
        logits = torch.randn(4, 3, requires_grad=True)
        cross_entropy_loss(logits, torch.tensor([0, 1, 2, 1])).sum().backward()
        assert logits.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# BCE clipped
# ---------------------------------------------------------------------------


class TestBCEClipped:
    def test_shape(self):
        x = torch.rand(4, 3)
        assert bce_clipped(x, torch.rand(4, 3)).shape == x.shape

    def test_boundary_no_inf_nan(self):
        pred = torch.tensor([0.0, 0.5, 1.0])
        target = torch.tensor([0.0, 0.5, 1.0])
        loss = bce_clipped(pred, target)
        assert torch.isfinite(loss).all()

    def test_matches_bce_interior(self):
        pred = torch.tensor([0.3, 0.7])
        target = torch.tensor([0.0, 1.0])
        clipped = bce_clipped(pred, target, eps=0.001)
        standard = F.binary_cross_entropy(pred, target, reduction="none")
        assert torch.allclose(clipped, standard, atol=1e-4)

    def test_clipping_effect(self):
        # At pred=0.0 with eps=0.1, clipped to 0.1: BCE(0.1, 0.0) = -log(0.9) ≈ 0.1054
        loss = bce_clipped(torch.tensor([0.0]), torch.tensor([0.0]), eps=0.1)
        expected = -math.log(0.9)
        assert torch.isclose(loss, torch.tensor([expected]), atol=1e-4)

    def test_gradient_nonzero(self):
        x = torch.rand(4, 3, requires_grad=True)
        bce_clipped(x, torch.rand(4, 3)).sum().backward()
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Depth inverse MSE
# ---------------------------------------------------------------------------


class TestDepthInverseMSE:
    def test_identical(self):
        d = torch.rand(4, 3).clamp(min=0.1)
        assert torch.allclose(depth_inverse_mse(d, d), torch.zeros_like(d), atol=1e-10)

    def test_shape(self):
        x = torch.rand(2, 5).clamp(min=0.1)
        y = torch.rand(2, 5).clamp(min=0.1)
        assert depth_inverse_mse(x, y).shape == x.shape

    def test_inverse_scaling(self):
        # Closer depths have larger inverse-space error for same absolute diff
        pred = torch.tensor([1.0, 10.0])
        target = torch.tensor([2.0, 11.0])
        loss = depth_inverse_mse(pred, target)
        assert loss[0] > loss[1]

    def test_known_value(self):
        # pred=2 -> 1/2=0.5, target=4 -> 1/4=0.25, (0.5-0.25)^2 = 0.0625
        pred = torch.tensor([2.0])
        target = torch.tensor([4.0])
        assert torch.isclose(depth_inverse_mse(pred, target), torch.tensor([0.0625]))

    def test_gradient_nonzero(self):
        x = torch.rand(4, 3).clamp(min=0.1).requires_grad_(True)
        depth_inverse_mse(x, torch.rand(4, 3).clamp(min=0.1)).sum().backward()
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Log L1
# ---------------------------------------------------------------------------


class TestLogL1:
    def test_identical(self):
        x = torch.rand(4, 3)
        assert torch.allclose(log_l1(x, x), torch.zeros_like(x))

    def test_shape(self):
        x, y = torch.rand(2, 5), torch.rand(2, 5)
        assert log_l1(x, y).shape == x.shape

    def test_monotonic(self):
        target = torch.zeros(3)
        pred = torch.tensor([0.1, 0.5, 1.0])
        loss = log_l1(pred, target)
        assert (loss[1:] > loss[:-1]).all()

    def test_known_value(self):
        # log(1 + |1 - 0|) = log(2) ≈ 0.6931
        loss = log_l1(torch.tensor([1.0]), torch.tensor([0.0]))
        assert torch.isclose(loss, torch.tensor([math.log(2)]))

    def test_gradient_value(self):
        # d/dx log(1+|x|) at x=1: 1/(1+1) * sign(1) = 0.5
        x = torch.tensor([1.0], requires_grad=True)
        log_l1(x, torch.tensor([0.0])).backward()
        assert torch.isclose(x.grad, torch.tensor([0.5]))


# ---------------------------------------------------------------------------
# Normal cosine
# ---------------------------------------------------------------------------


class TestNormalCosineLoss:
    def test_identical(self):
        n = F.normalize(torch.randn(10, 3), dim=-1)
        loss = normal_cosine_loss(n, n)
        assert torch.allclose(loss, torch.zeros(10), atol=1e-6)

    def test_opposing(self):
        n = F.normalize(torch.randn(5, 3), dim=-1)
        loss = normal_cosine_loss(n, -n)
        assert torch.allclose(loss, torch.full((5,), 2.0), atol=1e-5)

    def test_orthogonal_reference(self):
        # Orthogonal normals: cos=0, loss=1
        n1 = torch.tensor([[1.0, 0.0, 0.0]])
        n2 = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.isclose(normal_cosine_loss(n1, n2), torch.tensor([1.0]))

    def test_shape(self):
        n1 = F.normalize(torch.randn(4, 8, 3), dim=-1)
        n2 = F.normalize(torch.randn(4, 8, 3), dim=-1)
        assert normal_cosine_loss(n1, n2).shape == (4, 8)

    def test_gradient_nonzero(self):
        n1 = F.normalize(torch.randn(4, 3), dim=-1).requires_grad_(True)
        n2 = F.normalize(torch.randn(4, 3), dim=-1)
        normal_cosine_loss(n1, n2).sum().backward()
        assert n1.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# ReLU sum
# ---------------------------------------------------------------------------


class TestReluSum:
    def test_below_threshold(self):
        x = torch.tensor([0.1, 0.2, 0.3])
        assert relu_sum(x, eps=0.5).item() == 0.0

    def test_above_threshold(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        loss = relu_sum(x, eps=1.5)
        # relu([1-1.5, 2-1.5, 3-1.5]) = [0, 0.5, 1.5], sum = 2.0
        assert torch.isclose(loss, torch.tensor(2.0))

    def test_scalar_output(self):
        assert relu_sum(torch.rand(10, 3), eps=0.5).shape == ()

    def test_gradient_selective(self):
        # Only elements above eps should have nonzero gradient
        x = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
        relu_sum(x, eps=0.4).backward()
        assert x.grad[0].item() == 0.0  # below eps
        assert x.grad[1].item() == 1.0  # above eps
        assert x.grad[2].item() == 1.0  # above eps


# ---------------------------------------------------------------------------
# Weights reg
# ---------------------------------------------------------------------------


class TestWeightsReg:
    def test_zero_weights(self):
        w = [torch.zeros(3, 5), torch.zeros(2, 5)]
        assert weights_reg(w).item() == 0.0

    def test_known_value(self):
        # ones(2,3): each row sum of squares = 3, mean([3,3]) = 3.0
        w = [torch.ones(2, 3)]
        assert torch.isclose(weights_reg(w), torch.tensor(3.0))

    def test_scalar_output(self):
        w = [torch.rand(4, 5), torch.rand(3, 5)]
        assert weights_reg(w).shape == ()

    def test_gradient_nonzero(self):
        w = [torch.rand(4, 5, requires_grad=True)]
        weights_reg(w).backward()
        assert w[0].grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Identity distance
# ---------------------------------------------------------------------------


class TestIdentityDistance:
    def test_identity_grid(self):
        eye = torch.eye(3, 4).flatten().unsqueeze(0)  # (1, 12)
        assert torch.isclose(identity_distance(eye), torch.tensor([0.0]), atol=1e-6)

    def test_non_identity(self):
        grid = torch.zeros(1, 12)
        dist = identity_distance(grid)
        assert dist.item() > 0.0

    def test_zero_grid_reference(self):
        # Zero matrix vs identity 3x4: Frobenius norm of I_{3x4} = sqrt(3)
        grid = torch.zeros(1, 12)
        dist = identity_distance(grid)
        assert torch.isclose(dist, torch.tensor([math.sqrt(3.0)]), atol=1e-5)

    def test_spatial_dims(self):
        grid = torch.rand(2, 12, 4, 4)
        dist = identity_distance(grid)
        assert dist.shape == (2, 4, 4)

    def test_custom_dims(self):
        grid = torch.eye(2, 3).flatten().unsqueeze(0)  # (1, 6)
        dist = identity_distance(grid, num_rows=2, num_cols=3)
        assert torch.isclose(dist, torch.tensor([0.0]), atol=1e-6)

    def test_gradient_nonzero(self):
        grid = torch.rand(2, 12, requires_grad=True)
        identity_distance(grid).sum().backward()
        assert grid.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Total variation temporal
# ---------------------------------------------------------------------------


class TestTotalVariationTemporal:
    def test_constant(self):
        x = torch.ones(4, 3, 2, 2, 2)
        mask = torch.ones(3)
        tv = total_variation_temporal(x, mask)
        assert torch.allclose(tv, torch.zeros(3))

    def test_single_frame(self):
        x = torch.rand(1, 3, 2, 2, 2)
        mask = torch.ones(1)
        tv = total_variation_temporal(x, mask)
        assert torch.isclose(tv.sum(), torch.tensor(0.0))

    def test_mask_zeroes(self):
        x = torch.rand(3, 3, 2, 2, 2)
        mask = torch.tensor([0.0, 1.0])
        tv = total_variation_temporal(x, mask)
        assert tv[0].item() == 0.0
        assert tv[1].item() >= 0.0

    def test_known_value(self):
        # Two frames: [0,0,...] and [1,1,...]. diff^2 = 1 everywhere, mean = 1.0
        x = torch.zeros(2, 1, 1, 1, 1)
        x[1] = 1.0
        mask = torch.ones(1)
        tv = total_variation_temporal(x, mask)
        assert torch.isclose(tv, torch.tensor([1.0]))

    def test_shape(self):
        x = torch.rand(5, 3, 2, 4, 4)
        mask = torch.ones(4)
        assert total_variation_temporal(x, mask).shape == (4,)

    def test_gradient_nonzero(self):
        x = torch.rand(3, 3, 2, 2, 2, requires_grad=True)
        mask = torch.ones(2)
        total_variation_temporal(x, mask).sum().backward()
        assert x.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Linear lambda scheduler
# ---------------------------------------------------------------------------


class TestLinearLambdaScheduler:
    def test_before_start(self):
        s = LinearLambdaScheduler(start=100, end=200, lambda_init=1.0, lambda_end=0.0)
        assert s(epoch=0, global_step=50) == 1.0

    def test_at_start(self):
        s = LinearLambdaScheduler(start=100, end=200, lambda_init=1.0, lambda_end=0.0)
        assert s(epoch=0, global_step=100) == 1.0

    def test_at_end(self):
        s = LinearLambdaScheduler(start=100, end=200, lambda_init=1.0, lambda_end=0.0)
        assert s(epoch=0, global_step=200) == 0.0

    def test_midpoint(self):
        s = LinearLambdaScheduler(start=0, end=100, lambda_init=1.0, lambda_end=0.0)
        val = s(epoch=0, global_step=50)
        assert abs(val - 0.5) < 1e-6

    def test_quarter_point(self):
        s = LinearLambdaScheduler(start=0, end=100, lambda_init=1.0, lambda_end=0.0)
        assert abs(s(epoch=0, global_step=25) - 0.75) < 1e-6

    def test_after_end(self):
        s = LinearLambdaScheduler(start=0, end=100, lambda_init=1.0, lambda_end=0.0)
        assert s(epoch=0, global_step=200) == 0.0

    def test_epoch_mode(self):
        s = LinearLambdaScheduler(
            start=0,
            end=10,
            lambda_init=2.0,
            lambda_end=0.0,
            update_interval="epoch",
        )
        val = s(epoch=5, global_step=99999)
        assert abs(val - 1.0) < 1e-6

    def test_update_frequency(self):
        s = LinearLambdaScheduler(
            start=0,
            end=100,
            lambda_init=1.0,
            lambda_end=0.0,
            update_frequency=10,
        )
        v1 = s(epoch=0, global_step=5)
        v2 = s(epoch=0, global_step=15)
        assert v1 > v2

    def test_increasing_schedule(self):
        s = LinearLambdaScheduler(start=0, end=100, lambda_init=0.0, lambda_end=1.0)
        assert s(epoch=0, global_step=0) == 0.0
        assert abs(s(epoch=0, global_step=50) - 0.5) < 1e-6
        assert s(epoch=0, global_step=100) == 1.0

    def test_invalid_update_frequency(self):
        with pytest.raises(ValueError, match="update_frequency must be > 0"):
            LinearLambdaScheduler(start=0, end=100, lambda_init=1.0, update_frequency=0)

    def test_invalid_end_before_start(self):
        with pytest.raises(ValueError, match="end must be > start"):
            LinearLambdaScheduler(start=100, end=50, lambda_init=1.0)

    def test_invalid_end_equals_start(self):
        with pytest.raises(ValueError, match="end must be > start"):
            LinearLambdaScheduler(start=100, end=100, lambda_init=1.0)

    def test_invalid_total_stages_zero(self):
        # end - start = 1, update_frequency = 10, total_stages = 0
        with pytest.raises(ValueError, match="total_stages must be > 0"):
            LinearLambdaScheduler(start=0, end=1, lambda_init=1.0, update_frequency=10)


# ---------------------------------------------------------------------------
# Reduction functions
# ---------------------------------------------------------------------------

from gsplat.losses import reduce_mean, reduce_quantile, reduce_sum


class TestReduceMean:
    def test_simple_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.isclose(reduce_mean(x), torch.tensor(2.5))

    def test_known_value(self):
        x = torch.tensor([10.0, 20.0])
        assert torch.isclose(reduce_mean(x), torch.tensor(15.0))

    def test_masked_mean(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        # masked values: [1, 3], weighted sum = 4, mask sum = 2, result = 2.0
        assert torch.isclose(reduce_mean(x, mask), torch.tensor(2.0))

    def test_mask_all_zeros(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.zeros(3, dtype=torch.int32)
        # Denominator clamped to 1, so result = 0/1 = 0
        assert torch.isclose(reduce_mean(x, mask), torch.tensor(0.0))

    def test_mask_partial(self):
        x = torch.tensor([5.0, 10.0, 15.0])
        mask = torch.tensor([False, True, True])
        # (0 + 10 + 15) / 2 = 12.5
        assert torch.isclose(reduce_mean(x, mask), torch.tensor(12.5))

    def test_mask_int(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.tensor([1, 0, 1], dtype=torch.int64)
        assert torch.isclose(reduce_mean(x, mask), torch.tensor(2.0))

    def test_scalar_output(self):
        x = torch.rand(4, 3)
        assert reduce_mean(x).shape == ()

    def test_gradient_through_mask(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        mask = torch.tensor([True, False, True])
        reduce_mean(x, mask).backward()
        # Grad for masked elements should be 1/mask_sum = 0.5
        # Grad for unmasked should be 0
        assert torch.isclose(x.grad[0], torch.tensor(0.5))
        assert torch.isclose(x.grad[1], torch.tensor(0.0))
        assert torch.isclose(x.grad[2], torch.tensor(0.5))


class TestReduceQuantile:
    def test_full_quantile(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # quantile=1.0 keeps all values, mean = 2.5
        assert torch.isclose(reduce_quantile(x, quantile=1.0), torch.tensor(2.5))

    def test_half_quantile(self):
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # quantile=0.5 keeps bottom 2: [1, 2], mean = 1.5
        assert torch.isclose(reduce_quantile(x, quantile=0.5), torch.tensor(1.5))

    def test_quarter_quantile(self):
        x = torch.tensor([10.0, 20.0, 30.0, 40.0])
        # quantile=0.25 keeps bottom 1: [10], mean = 10
        assert torch.isclose(reduce_quantile(x, quantile=0.25), torch.tensor(10.0))

    def test_filters_outliers(self):
        x = torch.tensor([1.0, 1.0, 1.0, 100.0])
        # quantile=0.75 keeps bottom 3: [1, 1, 1], mean = 1.0
        result = reduce_quantile(x, quantile=0.75)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_scalar_output(self):
        x = torch.rand(10, 5)
        assert reduce_quantile(x, quantile=0.5).shape == ()

    def test_very_small_quantile_returns_zero(self):
        x = torch.tensor([1.0, 2.0])
        # quantile so small that k=0
        result = reduce_quantile(x, quantile=0.01)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_multidimensional(self):
        x = torch.tensor([[1.0, 4.0], [2.0, 3.0]])
        # Flattened: [1,4,2,3], sorted: [1,2,3,4], bottom 50%: [1,2], mean=1.5
        assert torch.isclose(reduce_quantile(x, quantile=0.5), torch.tensor(1.5))


class TestReduceSum:
    def test_known_value(self):
        x = torch.tensor([1.0, 2.0, 3.0])
        assert torch.isclose(reduce_sum(x), torch.tensor(6.0))

    def test_matches_torch_sum(self):
        x = torch.rand(4, 3)
        assert torch.isclose(reduce_sum(x), x.sum())

    def test_scalar_output(self):
        x = torch.rand(5, 7)
        assert reduce_sum(x).shape == ()

    def test_gradient_values(self):
        x = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
        reduce_sum(x).backward()
        # Gradient of sum is all ones
        assert torch.allclose(x.grad, torch.ones(3))

    def test_negative_values(self):
        x = torch.tensor([-1.0, 2.0, -3.0])
        assert torch.isclose(reduce_sum(x), torch.tensor(-2.0))

    def test_empty_tensor(self):
        x = torch.tensor([])
        assert torch.isclose(reduce_sum(x), torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Edge cases for reduction functions
# ---------------------------------------------------------------------------


class TestReduceEdgeCases:
    def test_mean_empty_returns_nan(self):
        # Without mask, empty tensor mean is NaN (standard PyTorch behavior).
        # Caller can check value.numel() == 0 on CPU to handle this.
        result = reduce_mean(torch.tensor([]))
        assert torch.isnan(result)

    def test_mean_empty_with_mask_returns_zero(self):
        # With mask, empty tensor returns 0.0 (denominator clamped to 1).
        # This is ambiguous: 0.0 could also mean the true mean is zero.
        # The ambiguity avoids a CPU-GPU sync to check mask emptiness.
        result = reduce_mean(torch.tensor([]), torch.tensor([], dtype=torch.bool))
        assert not torch.isnan(result)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_mean_all_zero_mask_returns_zero(self):
        # All-zero mask on non-zero values still returns 0.0 (ambiguous
        # with a true mean of zero, by design to avoid CPU-GPU sync).
        result = reduce_mean(
            torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3, dtype=torch.int32)
        )
        assert torch.isclose(result, torch.tensor(0.0))

    def test_mean_mismatched_mask_raises(self):
        with pytest.raises(RuntimeError):
            reduce_mean(
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([True, False]),
            )

    def test_mean_float_mask_raises(self):
        # Float masks are rejected to avoid incorrect clamping behavior
        # when mask sum is between 0 and 1.
        with pytest.raises(AssertionError, match="must be bool or integer"):
            reduce_mean(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))

    def test_mean_float_mask_half_raises(self):
        with pytest.raises(AssertionError, match="must be bool or integer"):
            reduce_mean(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0.5, 0.5]),
            )

    def test_quantile_empty_returns_zero(self):
        # Empty tensor: k = int(0 * 0.5) = 0, returns zero
        result = reduce_quantile(torch.tensor([]), quantile=0.5)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_quantile_single_element_half(self):
        # Single element with quantile=0.5: k = int(1 * 0.5) = 0, returns zero
        result = reduce_quantile(torch.tensor([5.0]), quantile=0.5)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_quantile_single_element_full(self):
        # Single element with quantile=1.0: k = 1, returns the value
        result = reduce_quantile(torch.tensor([5.0]), quantile=1.0)
        assert torch.isclose(result, torch.tensor(5.0))

    def test_quantile_zero_raises(self):
        with pytest.raises(AssertionError, match="quantile must be in"):
            reduce_quantile(torch.tensor([1.0]), quantile=0.0)

    def test_quantile_above_one_raises(self):
        with pytest.raises(AssertionError, match="quantile must be in"):
            reduce_quantile(torch.tensor([1.0]), quantile=1.5)

    def test_sum_empty_returns_zero(self):
        result = reduce_sum(torch.tensor([]))
        assert torch.isclose(result, torch.tensor(0.0))


# ---------------------------------------------------------------------------
# Gaussian regularization losses
# ---------------------------------------------------------------------------

from gsplat.losses import (
    bilateral_grid_drift_loss,
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)


class TestGaussianScaleReg:
    def test_zero_scales(self):
        scales = torch.zeros(10, 3)
        assert (gaussian_scale_reg(scales) == 0).all()

    def test_known_value(self):
        scales = torch.tensor([[1.0, 2.0, 3.0]])  # post-activation
        expected = torch.tensor([[1.0, 2.0, 3.0]])
        assert torch.allclose(gaussian_scale_reg(scales), expected)

    def test_visibility_masking(self):
        scales = torch.ones(4, 3)
        vis = torch.tensor([1.0, 0.0, 1.0, 0.0])
        result = gaussian_scale_reg(scales, visibility=vis)
        assert (result[0] == 1.0).all()
        assert (result[1] == 0.0).all()
        assert (result[2] == 1.0).all()
        assert (result[3] == 0.0).all()

    def test_no_visibility(self):
        scales = torch.tensor([[0.5, 0.5, 1.5]])  # post-activation
        expected = torch.tensor([[0.5, 0.5, 1.5]])
        assert torch.allclose(gaussian_scale_reg(scales), expected)

    def test_shape(self):
        scales = torch.rand(100, 3)
        assert gaussian_scale_reg(scales).shape == (100, 3)

    def test_gradient_nonzero(self):
        scales = torch.rand(10, 3, requires_grad=True)  # post-activation
        gaussian_scale_reg(scales).sum().backward()
        assert scales.grad.abs().sum() > 0

    def test_gradient_with_visibility(self):
        scales = torch.rand(4, 3, requires_grad=True)  # post-activation
        vis = torch.tensor([1.0, 0.0, 1.0, 0.0])
        gaussian_scale_reg(scales, visibility=vis).sum().backward()
        # Masked Gaussians should have zero gradient
        assert (scales.grad[1] == 0).all()
        assert (scales.grad[3] == 0).all()
        assert scales.grad[0].abs().sum() > 0


class TestGaussianDensityReg:
    def test_zero_densities(self):
        d = torch.zeros(10)
        assert (gaussian_density_reg(d) == 0).all()

    def test_known_value(self):
        d = torch.tensor([0.3, 0.7, 1.0])  # post-activation
        expected = torch.tensor([0.3, 0.7, 1.0])
        assert torch.allclose(gaussian_density_reg(d), expected)

    def test_visibility_masking(self):
        d = torch.ones(4)  # post-activation
        vis = torch.tensor([1.0, 0.0, 0.0, 1.0])
        result = gaussian_density_reg(d, visibility=vis)
        assert torch.allclose(result, vis)

    def test_shape_1d(self):
        d = torch.rand(50)
        assert gaussian_density_reg(d).shape == (50,)

    def test_shape_2d(self):
        d = torch.rand(50, 1)
        assert gaussian_density_reg(d).shape == (50, 1)

    def test_gradient_nonzero(self):
        d = torch.rand(10, requires_grad=True)  # post-activation
        gaussian_density_reg(d).sum().backward()
        assert d.grad.abs().sum() > 0


class TestGaussianZScaleReg:
    def test_below_threshold(self):
        z = torch.tensor([0.1, 0.2, 0.3])
        result = gaussian_z_scale_reg(z, threshold=0.5)
        assert (result == 0).all()

    def test_above_threshold(self):
        z = torch.tensor([0.6, 0.8, 1.0])
        result = gaussian_z_scale_reg(z, threshold=0.5)
        expected = torch.tensor([0.1, 0.3, 0.5])
        assert torch.allclose(result, expected)

    def test_mixed(self):
        z = torch.tensor([0.3, 0.5, 0.7])
        result = gaussian_z_scale_reg(z, threshold=0.5)
        expected = torch.tensor([0.0, 0.0, 0.2])
        assert torch.allclose(result, expected)

    def test_shape(self):
        z = torch.rand(100)
        assert gaussian_z_scale_reg(z, threshold=0.5).shape == (100,)

    def test_gradient_selective(self):
        z = torch.tensor([0.3, 0.7], requires_grad=True)
        gaussian_z_scale_reg(z, threshold=0.5).sum().backward()
        assert z.grad[0].item() == 0.0  # below threshold
        assert z.grad[1].item() == 1.0  # above threshold


class TestOutOfBoundLoss:
    def test_inside_cuboid(self):
        pos = torch.tensor([[0.0, 0.0, 0.0]])
        dims = torch.tensor([[2.0, 2.0, 2.0]])
        assert (out_of_bound_loss(pos, dims) == 0).all()

    def test_outside_cuboid(self):
        pos = torch.tensor([[2.0, 0.0, 0.0]])
        dims = torch.tensor([[2.0, 2.0, 2.0]])
        # |2| - 2/2 = 1.0 on x, 0 on y,z
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(out_of_bound_loss(pos, dims), expected)

    def test_negative_position(self):
        pos = torch.tensor([[-3.0, 0.0, 0.0]])
        dims = torch.tensor([[4.0, 4.0, 4.0]])
        # |-3| - 4/2 = 1.0 on x
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(out_of_bound_loss(pos, dims), expected)

    def test_known_3d(self):
        pos = torch.tensor([[2.0, -1.5, 0.5]])
        dims = torch.tensor([[2.0, 2.0, 2.0]])
        # x: |2|-1=1, y: |-1.5|-1=0.5, z: |0.5|-1=0 (inside)
        expected = torch.tensor([[1.0, 0.5, 0.0]])
        assert torch.allclose(out_of_bound_loss(pos, dims), expected)

    def test_batch(self):
        pos = torch.rand(100, 3) * 4 - 2  # [-2, 2]
        dims = torch.ones(100, 3) * 3.0
        result = out_of_bound_loss(pos, dims)
        assert result.shape == (100, 3)
        assert (result >= 0).all()

    def test_gradient_nonzero(self):
        pos = torch.tensor([[2.0, 0.0, 0.0]], requires_grad=True)
        dims = torch.tensor([[2.0, 2.0, 2.0]])
        out_of_bound_loss(pos, dims).sum().backward()
        assert pos.grad[0, 0].item() == 1.0  # outside: grad = sign(pos) = 1
        assert pos.grad[0, 1].item() == 0.0  # inside: grad = 0
        assert pos.grad[0, 2].item() == 0.0


class TestBilateralGridDriftLoss:
    def test_identity_grids(self):
        eye = torch.eye(3, 4).flatten().unsqueeze(0)  # (1, 12)
        result = bilateral_grid_drift_loss([eye])
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_zero_grid(self):
        grid = torch.zeros(1, 12)
        result = bilateral_grid_drift_loss([grid])
        assert torch.isclose(result, torch.tensor([math.sqrt(3.0)]), atol=1e-5)

    def test_multiple_grids(self):
        eye = torch.eye(3, 4).flatten().unsqueeze(0)
        zero = torch.zeros(1, 12)
        result = bilateral_grid_drift_loss([eye, zero])
        assert result.shape == (2,)
        assert torch.isclose(result[0], torch.tensor(0.0), atol=1e-6)
        assert result[1] > 0

    def test_spatial_dims(self):
        grid = torch.rand(2, 12, 3, 3)
        result = bilateral_grid_drift_loss([grid])
        assert result.shape == (2 * 3 * 3,)

    def test_gradient_nonzero(self):
        grid = torch.rand(1, 12, requires_grad=True)
        bilateral_grid_drift_loss([grid]).sum().backward()
        assert grid.grad.abs().sum() > 0

    def test_empty_list(self):
        result = bilateral_grid_drift_loss([])
        assert result.shape == (0,)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# Input contract assertions
# ---------------------------------------------------------------------------


class TestInputContracts:
    def test_ssim_rejects_out_of_range(self):
        img = torch.rand(1, 3, 32, 32) * 255
        with pytest.raises(AssertionError, match="must be in"):
            ssim_loss(img, img)

    def test_ssim_rejects_negative(self):
        img = torch.rand(1, 3, 32, 32) - 0.5
        with pytest.raises(AssertionError, match="must be in"):
            ssim_loss(img, torch.rand(1, 3, 32, 32))

    def test_ssim_accepts_valid_range(self):
        img = torch.rand(1, 3, 32, 32)
        loss = ssim_loss(img, img)
        assert torch.isfinite(loss)

    def test_normal_cosine_rejects_unnormalized(self):
        n = torch.randn(5, 3) * 10
        with pytest.raises(AssertionError, match="unit-normalized"):
            normal_cosine_loss(n, F.normalize(torch.randn(5, 3), dim=-1))

    def test_normal_cosine_accepts_normalized(self):
        n1 = F.normalize(torch.randn(5, 3), dim=-1)
        n2 = F.normalize(torch.randn(5, 3), dim=-1)
        loss = normal_cosine_loss(n1, n2)
        assert torch.isfinite(loss).all()

    def test_gaussian_scale_rejects_negative(self):
        scales = torch.tensor([[-1.0, 0.5, 0.3]])
        with pytest.raises(AssertionError, match="post-activation"):
            gaussian_scale_reg(scales)

    def test_gaussian_scale_accepts_positive(self):
        scales = torch.tensor([[0.01, 0.5, 1.0]])
        result = gaussian_scale_reg(scales)
        assert (result >= 0).all()

    def test_gaussian_density_rejects_negative(self):
        densities = torch.tensor([-0.5, 0.3, 0.8])
        with pytest.raises(AssertionError, match="post-activation"):
            gaussian_density_reg(densities)

    def test_gaussian_density_accepts_positive(self):
        densities = torch.tensor([0.0, 0.5, 1.0])
        result = gaussian_density_reg(densities)
        assert (result >= 0).all()

    def test_gaussian_z_scale_rejects_negative(self):
        z = torch.tensor([-0.1, 0.5])
        with pytest.raises(AssertionError, match="post-activation"):
            gaussian_z_scale_reg(z, threshold=0.3)

    def test_gaussian_z_scale_accepts_positive(self):
        z = torch.tensor([0.1, 0.5])
        result = gaussian_z_scale_reg(z, threshold=0.3)
        assert torch.isfinite(result).all()

    def test_out_of_bound_rejects_negative_dims(self):
        pos = torch.zeros(1, 3)
        dims = torch.tensor([[-1.0, 2.0, 3.0]])
        with pytest.raises(AssertionError, match="positive"):
            out_of_bound_loss(pos, dims)

    def test_out_of_bound_accepts_positive_dims(self):
        pos = torch.zeros(1, 3)
        dims = torch.tensor([[2.0, 2.0, 2.0]])
        result = out_of_bound_loss(pos, dims)
        assert (result == 0).all()
