"""Loss functions for Gaussian Splatting training.

This module provides loss functions commonly used in 3D Gaussian Splatting
pipelines. All functions follow a functional API: they accept plain
``torch.Tensor`` inputs and return ``torch.Tensor`` outputs.

Unless noted otherwise, loss functions return **unreduced** (per-element)
values so that callers can apply their own reduction (mean, sum, masking,
etc.).  Returning unreduced values has no measurable overhead compared to
returning reduced ones because the element-wise computation dominates.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Photometric losses
# ---------------------------------------------------------------------------


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Element-wise L1 (absolute error) loss.

    Args:
        pred: Predicted values of any shape.
        target: Target values, same shape as *pred*.

    Returns:
        Per-element L1 loss, same shape as *pred*.
    """
    return F.l1_loss(pred, target, reduction="none")


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Element-wise mean-squared-error loss.

    Args:
        pred: Predicted values of any shape.
        target: Target values, same shape as *pred*.

    Returns:
        Per-element MSE loss, same shape as *pred*.
    """
    return F.mse_loss(pred, target, reduction="none")


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(
    window_size: int, sigma: float, device: torch.device = torch.device("cpu")
) -> Tensor:
    x = torch.arange(window_size, device=device, dtype=torch.float32)
    gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_ssim_window(
    window_size: int, channel: int, device: torch.device = torch.device("cpu")
) -> Tensor:
    """Create a Gaussian window for SSIM computation.

    Args:
        window_size: Size of the square Gaussian window.
        channel: Number of image channels.
        device: Device for the output tensor.

    Returns:
        Window tensor of shape ``(channel, 1, window_size, window_size)``.
    """
    w1d = _gaussian_kernel_1d(window_size, 1.5, device=device).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).float().unsqueeze(0).unsqueeze(0)
    return w2d.expand(channel, 1, window_size, window_size).contiguous()


def torch_ssim_loss(
    img1: Tensor,
    img2: Tensor,
    window: Tensor,
    window_size: int = 11,
    channel: int = 3,
) -> Tensor:
    """Reference SSIM implementation (Wang et al. 2004).

    Args:
        img1: First image batch, shape ``(B, C, H, W)``.
        img2: Second image batch, shape ``(B, C, H, W)``.
        window: Pre-computed Gaussian window from :func:`create_ssim_window`.
        window_size: Window size (must match *window*).
        channel: Number of channels (must match *window*).

    Returns:
        SSIM map of shape ``(B, C, H, W)`` with values in ``[-1, 1]``.
    """
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map


def ssim_loss(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
) -> Tensor:
    """Structural Similarity (SSIM) loss: ``1 - SSIM``.

    Uses ``fused_ssim`` when available for better performance, otherwise falls
    back to :func:`torch_ssim_loss`.

    Args:
        img1: First image batch, shape ``(B, C, H, W)``.
        img2: Second image batch, shape ``(B, C, H, W)``.
        window_size: Size of the Gaussian window (default ``11``).

    Returns:
        Scalar SSIM loss (mean over batch, channels, and spatial dimensions).
    """
    try:
        from fused_ssim import fused_ssim

        if window_size == 11 and not img2.requires_grad and img1.is_cuda:
            return 1.0 - fused_ssim(img1, img2, padding="valid")
    except ImportError:
        pass

    channel = img1.shape[1]
    window = create_ssim_window(window_size, channel, device=img1.device).type_as(img1)
    return 1.0 - torch_ssim_loss(img1, img2, window, window_size, channel).mean()


# ---------------------------------------------------------------------------
# Depth losses
# ---------------------------------------------------------------------------


def depth_l1_loss(
    pred_depth: Tensor, gt_depth: Tensor, scene_scale: float = 1.0
) -> Tensor:
    """L1 loss in disparity (inverse-depth) space.

    Args:
        pred_depth: Predicted depth values.
        gt_depth: Ground-truth depth values.
        scene_scale: Multiplier applied to the disparity L1 (default ``1.0``).

    Returns:
        Scalar disparity L1 loss.
    """
    disp = torch.where(pred_depth > 0.0, 1.0 / pred_depth, torch.zeros_like(pred_depth))
    disp_gt = torch.where(gt_depth > 0.0, 1.0 / gt_depth, torch.zeros_like(gt_depth))
    return F.l1_loss(disp, disp_gt) * scene_scale


# ---------------------------------------------------------------------------
# Total variation
# ---------------------------------------------------------------------------


def total_variation_loss(x: Tensor) -> Tensor:
    """Total variation loss on multi-dimensional tensors.

    Computes the sum of squared finite differences along every spatial
    dimension (all dimensions after the first two), normalized by batch size
    and element count.

    Args:
        x: Input tensor of shape ``(B, C, ...)``, where ``B`` is the batch
           size, ``C`` is the channel dimension, and remaining dimensions are
           spatial.

    Returns:
        Scalar total variation loss.
    """
    batch_size = x.shape[0]
    tv: Tensor | float = 0
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        idx1 = torch.arange(1, n_res, device=x.device)
        idx2 = torch.arange(0, n_res - 1, device=x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = max(torch.prod(torch.tensor(x1.size()[1:]).float()).item(), 1.0)
        tv = tv + torch.pow((x1 - x2), 2).sum() / count
    return tv / batch_size  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------


def opacity_reg_loss(opacities: Tensor) -> Tensor:
    """Opacity regularization: mean of sigmoid activations.

    Penalizes high opacity values to encourage transparency where appropriate.

    Args:
        opacities: Raw (pre-sigmoid) opacity logits, any shape.

    Returns:
        Scalar mean opacity after sigmoid.
    """
    return torch.sigmoid(opacities).mean()


def scale_reg_loss(log_scales: Tensor) -> Tensor:
    """Scale regularization: mean of exponentiated log-scales.

    Penalizes large Gaussian scales.

    Args:
        log_scales: Log-scale parameters, any shape.

    Returns:
        Scalar mean scale.
    """
    return torch.exp(log_scales).mean()
