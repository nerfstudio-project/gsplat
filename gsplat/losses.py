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
import os
import sys
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

# When True, loss functions validate input domains (value ranges,
# normalization, post-activation checks).  These checks cause extra GPU
# kernels and CPU-GPU syncs, so they default to False.
# Enable via GSPLAT_ENFORCE_CONTRACTS=1 or PYTHONOPTIMIZE=0 (debug mode); off when unset.
ENFORCE_CONTRACTS: bool = (
    os.environ.get("GSPLAT_ENFORCE_CONTRACTS") == "1"
    or os.environ.get("PYTHONOPTIMIZE") == "0"
)


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


_ssim_window_cache: dict = {}


def ssim_loss(
    img1: Tensor,
    img2: Tensor,
    window_size: int = 11,
) -> Tensor:
    """Structural Similarity (SSIM) loss: ``1 - SSIM``.

    Uses ``fused_ssim`` when available for better performance, otherwise falls
    back to :func:`torch_ssim_loss`.

    Inputs must be in ``[0, 1]`` range (normalized images).

    Args:
        img1: First image batch, shape ``(B, C, H, W)``, values in ``[0, 1]``.
        img2: Second image batch, shape ``(B, C, H, W)``, values in ``[0, 1]``.
        window_size: Size of the Gaussian window (default ``11``).

    Returns:
        Scalar SSIM loss (mean over batch, channels, and spatial dimensions).
    """
    if ENFORCE_CONTRACTS:
        assert (
            img1.min() >= 0.0 and img1.max() <= 1.0
        ), f"img1 must be in [0, 1], got [{img1.min().item():.3f}, {img1.max().item():.3f}]"
        assert (
            img2.min() >= 0.0 and img2.max() <= 1.0
        ), f"img2 must be in [0, 1], got [{img2.min().item():.3f}, {img2.max().item():.3f}]"
    try:
        from fused_ssim import fused_ssim

        if window_size == 11 and not img2.requires_grad and img1.is_cuda:
            return 1.0 - fused_ssim(img1, img2, padding="valid")
    except ImportError:
        pass

    channel = img1.shape[1]
    key = (window_size, channel, img1.device, img1.dtype)
    if key not in _ssim_window_cache:
        _ssim_window_cache[key] = create_ssim_window(
            window_size, channel, device=img1.device
        ).type_as(img1)
    window = _ssim_window_cache[key]
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
# Loss function dispatch
# ---------------------------------------------------------------------------

# Registry mapping string names to per-element loss callables.
# All callables accept (pred, target) and return unreduced per-element loss
# tensors (same shape as pred).
_LOSS_FN_REGISTRY: dict = {
    "l1": lambda pred, target: F.l1_loss(pred, target, reduction="none"),
    "mse": lambda pred, target: F.mse_loss(pred, target, reduction="none"),
    "huber": lambda pred, target: F.huber_loss(
        pred, target, reduction="none", delta=1.0
    ),
    "smooth_l1": lambda pred, target: F.smooth_l1_loss(
        pred, target, reduction="none", beta=1.0
    ),
    "bce": lambda pred, target: F.binary_cross_entropy(pred, target, reduction="none"),
    "bce_with_logits": lambda pred, target: F.binary_cross_entropy_with_logits(
        pred, target, reduction="none"
    ),
    "bce_clipped": lambda pred, target: bce_clipped(pred, target),
}


def _resolve_loss_fn(loss_fn):
    """Resolve a loss function from a string name or return a callable as-is.

    Args:
        loss_fn: Either a string key (``'l1'``, ``'mse'``, ``'bce'``,
            ``'bce_clipped'``, ``'bce_with_logits'``, ``'huber'``,
            ``'smooth_l1'``) or a callable ``(pred, target) -> Tensor``
            returning unreduced per-element loss.

    Returns:
        A callable ``(pred, target) -> Tensor`` returning unreduced loss.
    """
    if callable(loss_fn):
        return loss_fn
    if loss_fn not in _LOSS_FN_REGISTRY:
        raise ValueError(
            f"Unknown loss_fn '{loss_fn}'. "
            f"Available: {sorted(_LOSS_FN_REGISTRY.keys())}. "
            f"Or pass a callable (pred, target) -> Tensor."
        )
    return _LOSS_FN_REGISTRY[loss_fn]


def _apply_loss_fn(fn, pred: Tensor, target: Tensor) -> Tensor:
    """Call *fn* and assert the result is an unreduced per-element tensor.

    This catches common mistakes like passing ``F.l1_loss`` (which reduces to a
    scalar by default) where the callers expect a per-element tensor they can
    mean-reduce themselves.
    """
    out = fn(pred, target)
    if out.shape != pred.shape:
        raise ValueError(
            "loss_fn must return a per-element tensor with the same shape as "
            f"pred ({tuple(pred.shape)}), got shape {tuple(out.shape)}. Make "
            "sure custom callables pass reduction='none' (or equivalent)."
        )
    return out


# ---------------------------------------------------------------------------
# LiDAR losses
# ---------------------------------------------------------------------------


def _validate_lidar_shapes(
    pred: Tensor, gt: Tensor, mask: Tensor | None, name: str
) -> None:
    """Validate that pred and gt have compatible shapes before flattening."""
    if pred.shape != gt.shape:
        raise ValueError(
            f"{name}: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match before flattening."
        )
    if mask is not None and mask.shape != pred.shape:
        raise ValueError(
            f"{name}: mask shape {mask.shape} != pred shape {pred.shape}. "
            "Mask must have the same shape as pred/gt."
        )


def lidar_distance_loss(
    pred_distance: Tensor,
    gt_distance: Tensor,
    valid_mask: Tensor | None = None,
    loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "l1",
) -> Tensor:
    """Per-element loss on LiDAR hit distance (direct distance space).

    Unlike :func:`depth_l1_loss` which operates in inverse-depth (disparity)
    space, this computes a direct loss on the rendered distance along each
    LiDAR ray vs. the measured ground-truth distance.

    Args:
        pred_distance: Predicted hit distance per ray, ``[..., 1]`` or ``[...]``.
        gt_distance: Ground-truth measured distance per ray, same shape.
        valid_mask: Optional boolean mask for valid (non-dropped, non-invalid)
            rays.  When provided, only masked elements contribute to the loss.
        loss_fn: Per-element loss function — a string name (``'l1'``,
            ``'mse'``, ``'huber'``, ``'smooth_l1'``) or a callable
            ``(pred, target) -> Tensor``.  Default ``'l1'``.

    Returns:
        Scalar distance loss (mean-reduced over valid elements).
    """
    _validate_lidar_shapes(
        pred_distance, gt_distance, valid_mask, "lidar_distance_loss"
    )
    fn = _resolve_loss_fn(loss_fn)
    pred = pred_distance.reshape(-1)
    gt = gt_distance.reshape(-1)
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        pred = pred[mask]
        gt = gt[mask]
    if pred.numel() == 0:
        return pred.sum()  # differentiable zero
    return _apply_loss_fn(fn, pred, gt).mean()


def lidar_intensity_loss(
    pred_intensity: Tensor,
    gt_intensity: Tensor,
    valid_mask: Tensor | None = None,
    loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "l1",
) -> Tensor:
    """Per-element loss on LiDAR return intensity.

    Compares rendered intensity (from ``extra_signals``) against measured
    ground-truth intensity for each valid LiDAR ray.

    Args:
        pred_intensity: Predicted intensity per ray, ``[..., 1]`` or ``[...]``.
        gt_intensity: Ground-truth intensity per ray, same shape.
        valid_mask: Optional boolean mask for valid rays.
        loss_fn: Per-element loss function — a string name (``'l1'``,
            ``'mse'``, ``'huber'``) or a callable.  Default ``'l1'``.

    Returns:
        Scalar intensity loss (mean-reduced over valid elements).
    """
    _validate_lidar_shapes(
        pred_intensity, gt_intensity, valid_mask, "lidar_intensity_loss"
    )
    fn = _resolve_loss_fn(loss_fn)
    pred = pred_intensity.reshape(-1)
    gt = gt_intensity.reshape(-1)
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        pred = pred[mask]
        gt = gt[mask]
    if pred.numel() == 0:
        return pred.sum()  # differentiable zero
    return _apply_loss_fn(fn, pred, gt).mean()


def lidar_raydrop_loss(
    pred_raydrop: Tensor,
    gt_raydrop: Tensor,
    valid_mask: Tensor | None = None,
    loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "bce_with_logits",
) -> Tensor:
    """Per-element loss on LiDAR ray-drop prediction.

    Supervises the predicted raydrop value against the ground-truth drop
    mask for each valid LiDAR ray.

    Args:
        pred_raydrop: Predicted raydrop value per ray, ``[..., 1]`` or ``[...]``.
            Interpretation depends on *loss_fn*: probabilities for ``'mse'``,
            raw logits for ``'bce_with_logits'``.
        gt_raydrop: Ground-truth raydrop labels (0 or 1), same shape.
        valid_mask: Optional boolean mask for valid rays (excludes invalid rays
            but keeps dropped rays, since those are the positive class).
        loss_fn: Per-element loss function — a string name
            (``'bce_with_logits'``, ``'mse'``, ``'l1'``) or a callable.
            Default ``'bce_with_logits'``.

    Returns:
        Scalar raydrop loss (mean-reduced over valid elements).
    """
    _validate_lidar_shapes(pred_raydrop, gt_raydrop, valid_mask, "lidar_raydrop_loss")
    fn = _resolve_loss_fn(loss_fn)
    pred = pred_raydrop.reshape(-1)
    gt = gt_raydrop.reshape(-1).float()
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        pred = pred[mask]
        gt = gt[mask]
    if pred.numel() == 0:
        return pred.sum()  # differentiable zero
    return _apply_loss_fn(fn, pred, gt).mean()


def lidar_background_loss(
    pred_opacity: Tensor,
    background_mask: Tensor,
    valid_mask: Tensor | None = None,
    loss_fn: str | Callable[[Tensor, Tensor], Tensor] = "bce",
) -> Tensor:
    """Per-element loss penalizing Gaussian opacity on background/sky LiDAR rays.

    Encourages the model to keep opacity low along LiDAR rays that correspond
    to sky or background (no physical surface), preventing floater artifacts
    in empty space.

    Args:
        pred_opacity: Predicted accumulated opacity per ray, ``[..., 1]`` or ``[...]``.
            Values are clamped to ``[0, 1]`` before the loss computation, so
            logits-based losses (``'bce_with_logits'``) are not supported here.
        background_mask: Boolean mask where True = background/sky ray.
        valid_mask: Optional boolean mask for valid (non-invalid, non-dropped) rays.
        loss_fn: Per-element loss function — a string name (``'bce'``,
            ``'bce_clipped'``, ``'mse'``, ``'l1'``) or a callable.  Default
            ``'bce'``. Note: ``'bce'`` can produce +inf at the 0/1 boundary
            even with the clamp; ``'bce_clipped'`` avoids this by clipping
            internally and is recommended for numerical safety.

    Returns:
        Scalar background loss (mean-reduced over valid elements).
    """
    if pred_opacity.shape != background_mask.shape:
        raise ValueError(
            f"lidar_background_loss: pred_opacity shape {pred_opacity.shape} != "
            f"background_mask shape {background_mask.shape}."
        )
    if valid_mask is not None and valid_mask.shape != pred_opacity.shape:
        raise ValueError(
            f"lidar_background_loss: valid_mask shape {valid_mask.shape} != "
            f"pred_opacity shape {pred_opacity.shape}."
        )
    fn = _resolve_loss_fn(loss_fn)
    pred = pred_opacity.reshape(-1).clamp(0, 1)
    # Target: 0 for background rays (want low opacity), 1 for foreground
    target = (~background_mask.reshape(-1).bool()).float()
    if valid_mask is not None:
        mask = valid_mask.reshape(-1).bool()
        pred = pred[mask]
        target = target[mask]
    if pred.numel() == 0:
        return pred.sum()  # differentiable zero
    return _apply_loss_fn(fn, pred, target).mean()


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
        count = max(float(math.prod(x1.size()[1:])), 1.0)
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


# ---------------------------------------------------------------------------
# Additional standard losses
# ---------------------------------------------------------------------------


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0) -> Tensor:
    """Element-wise Huber loss.

    Args:
        pred: Predicted values of any shape.
        target: Target values, same shape as *pred*.
        delta: Threshold for switching between L1 and L2 (default ``1.0``).

    Returns:
        Per-element Huber loss, same shape as *pred*.
    """
    return F.huber_loss(pred, target, reduction="none", delta=delta)


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """Element-wise Smooth L1 loss.

    Args:
        pred: Predicted values of any shape.
        target: Target values, same shape as *pred*.
        beta: Threshold for switching between L1 and L2 (default ``1.0``).

    Returns:
        Per-element Smooth L1 loss, same shape as *pred*.
    """
    return F.smooth_l1_loss(pred, target, reduction="none", beta=beta)


def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Element-wise Binary Cross-Entropy loss.

    Inputs must be probabilities in ``[0, 1]``.

    Args:
        pred: Predicted probabilities, any shape.
        target: Target probabilities, same shape as *pred*.

    Returns:
        Per-element BCE loss, same shape as *pred*.
    """
    return F.binary_cross_entropy(pred, target, reduction="none")


def bce_with_logits_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Element-wise Binary Cross-Entropy with logits.

    Inputs are raw logits (sigmoid is applied internally).

    Args:
        pred: Predicted logits, any shape.
        target: Target probabilities in ``[0, 1]``, same shape as *pred*.

    Returns:
        Per-element BCE loss, same shape as *pred*.
    """
    return F.binary_cross_entropy_with_logits(pred, target, reduction="none")


def cross_entropy_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Per-sample Cross-Entropy loss.

    Args:
        pred: Predicted logits of shape ``(N, C)`` where *C* is the number of classes.
        target: Target class indices of shape ``(N,)`` with values in ``[0, C)``.

    Returns:
        Per-sample loss of shape ``(N,)``.
    """
    return F.cross_entropy(pred, target, reduction="none")


def bce_clipped(input: Tensor, target: Tensor, eps: float = 0.001) -> Tensor:
    """Binary cross-entropy with input clipping.

    Clips *input* to ``(eps, 1 - eps)`` before computing BCE.  Disables
    autocast for numerical safety.

    Args:
        input: Predicted probabilities, any shape.
        target: Target probabilities, same shape as *input*.
        eps: Clipping margin (default ``0.001``).

    Returns:
        Per-element BCE loss, same shape as *input*.
    """
    with torch.amp.autocast(device_type="cuda", enabled=False):
        return F.binary_cross_entropy(
            input.float().clip(eps, 1 - eps), target.float(), reduction="none"
        )


def depth_inverse_mse(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """MSE loss in inverse (reciprocal) depth space.

    Args:
        pred: Predicted depth values, any shape.
        target: Target depth values, same shape as *pred*.
        eps: Clamping minimum to avoid division by zero (default ``1e-6``).

    Returns:
        Per-element inverse-depth MSE, same shape as *pred*.
    """
    return (pred.clamp(min=eps).reciprocal() - target.clamp(min=eps).reciprocal()).pow(
        2
    )


def log_l1(pred: Tensor, target: Tensor) -> Tensor:
    """Logarithmic L1 loss: ``log(1 + |pred - target|)``.

    Args:
        pred: Predicted values, any shape.
        target: Target values, same shape as *pred*.

    Returns:
        Per-element log-L1 loss, same shape as *pred*.
    """
    return torch.log(1.0 + (pred - target).abs())


def normal_cosine_loss(pred_normal: Tensor, gt_normal: Tensor) -> Tensor:
    """Cosine distance loss for surface normals: ``1 - cos(pred, gt)``.

    Both inputs must be approximately unit-normalized 3D vectors.

    Args:
        pred_normal: Predicted normals of shape ``(..., 3)``, approximately
            unit length.
        gt_normal: Ground-truth normals of shape ``(..., 3)``, approximately
            unit length.

    Returns:
        Per-element cosine distance of shape ``(...)``.
    """
    assert pred_normal.shape == gt_normal.shape
    assert pred_normal.shape[-1] == 3
    if ENFORCE_CONTRACTS:
        pred_norms = pred_normal.norm(dim=-1)
        gt_norms = gt_normal.norm(dim=-1)
        assert torch.allclose(pred_norms, torch.ones_like(pred_norms), atol=1e-3), (
            f"pred_normal must be unit-normalized, got norms in "
            f"[{pred_norms.min().item():.4f}, {pred_norms.max().item():.4f}]"
        )
        assert torch.allclose(gt_norms, torch.ones_like(gt_norms), atol=1e-3), (
            f"gt_normal must be unit-normalized, got norms in "
            f"[{gt_norms.min().item():.4f}, {gt_norms.max().item():.4f}]"
        )
    return 1.0 - torch.sum(pred_normal * gt_normal, dim=-1)


def relu_sum(value: Tensor, eps: float) -> Tensor:
    """ReLU-thresholded sum: ``relu(value - eps).sum()``.

    Args:
        value: Input tensor, any shape.
        eps: Threshold below which values are zeroed.

    Returns:
        Scalar sum of thresholded values.
    """
    return F.relu(value - eps).sum()


def weights_reg(weights_list: list[Tensor], dim: int = 1) -> Tensor:
    """Weight regularization: mean of squared weight norms.

    Args:
        weights_list: List of weight tensors.
        dim: Dimension along which to sum squares (default ``1``).

    Returns:
        Scalar regularization loss.
    """
    return torch.mean(torch.cat([(w**2).sum(dim) for w in weights_list]))


def identity_distance(grid: Tensor, num_rows: int = 3, num_cols: int = 4) -> Tensor:
    """Frobenius distance of an affine grid from the identity transform.

    Reshapes the channel dimension of *grid* into a ``(num_rows, num_cols)``
    matrix per spatial location and computes the Frobenius norm of the
    difference from the identity.

    Args:
        grid: Tensor of shape ``(B, C, ...)`` where ``C = num_rows * num_cols``.
        num_rows: Rows of the affine matrix (default ``3``).
        num_cols: Columns of the affine matrix (default ``4``).

    Returns:
        Per-batch Frobenius distance of shape ``(B, ...)``.
    """
    reshaped = grid.view(grid.shape[0], num_rows, num_cols, *grid.shape[2:])
    identity = torch.eye(num_rows, num_cols, device=grid.device)
    identity = identity.view(1, num_rows, num_cols, *([1] * len(grid.shape[2:])))
    diff = reshaped - identity
    return torch.norm(diff, p="fro", dim=(1, 2))


def total_variation_temporal(x: Tensor, loss_mask: Tensor) -> Tensor:
    """Total variation loss along the temporal (batch) dimension.

    Computes squared differences between consecutive frames along ``dim=0``,
    weighted by *loss_mask*.

    Args:
        x: Input tensor of shape ``(B, C, D, H, W)``.
        loss_mask: Mask of shape ``(B-1,)`` or broadcastable to the temporal
            differences.

    Returns:
        Per-frame TV values of shape ``(B-1,)``, or scalar ``0`` if ``B <= 1``.
    """
    if x.shape[0] <= 1:
        return torch.zeros(1, device=x.device)
    tv_t = x.diff(dim=0).square().mean(dim=(1, 2, 3, 4))
    return tv_t * loss_mask


# ---------------------------------------------------------------------------
# Lambda scheduling
# ---------------------------------------------------------------------------


class LinearLambdaScheduler:
    """Linear interpolation scheduler for loss weights.

    Linearly interpolates the loss weight from *lambda_init* to *lambda_end*
    over the range ``[start, end]`` in steps (or epochs).

    Args:
        start: Step/epoch at which interpolation begins.
        end: Step/epoch at which interpolation ends.
        lambda_init: Initial lambda value.
        lambda_end: Final lambda value (default ``0.0``).
        update_interval: ``"step"`` or ``"epoch"`` (default ``"step"``).
        update_frequency: How often to update within the interval (default ``1``).
    """

    def __init__(
        self,
        start: int,
        end: int,
        lambda_init: float,
        lambda_end: float = 0.0,
        update_interval: str = "step",
        update_frequency: int = 1,
    ) -> None:
        if update_frequency <= 0:
            raise ValueError(f"update_frequency must be > 0, got {update_frequency}")
        if end <= start:
            raise ValueError(f"end must be > start, got start={start}, end={end}")
        total_stages = (end - start) // update_frequency
        if total_stages <= 0:
            raise ValueError(
                f"total_stages must be > 0 (start={start}, end={end}, "
                f"update_frequency={update_frequency} yields "
                f"total_stages={total_stages})"
            )
        self.start = start
        self.end = end
        self.lambda_init = lambda_init
        self.lambda_end = lambda_end
        self.update_interval = update_interval
        self.update_frequency = update_frequency
        self.total_stages = total_stages

    def __call__(self, epoch: int, global_step: int) -> float:
        counter = global_step if self.update_interval == "step" else epoch
        cur_stage = (counter - self.start) // self.update_frequency
        ratio = min(1.0, max(0.0, cur_stage / self.total_stages))
        return (1 - ratio) * self.lambda_init + ratio * self.lambda_end


# ---------------------------------------------------------------------------
# Reduction functions
# ---------------------------------------------------------------------------


def reduce_mean(value: Tensor, mask: Tensor | None = None) -> Tensor:
    """Mean reduction with optional mask.

    Without *mask*, returns ``value.mean()`` (NaN for empty tensors,
    following standard PyTorch behavior).  The caller can check
    ``value.numel() == 0`` beforehand on CPU to handle this case.

    When *mask* is provided, it must be a bool or integer tensor (not
    floating point).  This ensures ``mask.sum().clamp(min=1)`` is a
    correct denominator — fractional masks would require clamping to the
    smallest representable float, which changes the semantics.  Callers
    with float masks should convert to bool first (e.g. ``mask > 0``).

    The masked mean is ``(value * mask).sum() / mask.sum()``, with the
    denominator clamped to 1 to avoid division by zero.  This avoids a
    CPU-GPU sync that would be required to check whether the mask is
    all-zero on the host side.  The trade-off is that a return value of
    ``0`` is ambiguous: it can mean either that the mask is entirely
    zero (no active elements) or that the true weighted mean is zero.
    The caller cannot distinguish these two cases without an explicit
    sync.

    Args:
        value: Input tensor, any shape.
        mask: Optional bool or integer mask tensor broadcastable to
            *value*.  Raises ``AssertionError`` if floating point.

    Returns:
        Scalar reduced value.
    """
    if mask is not None:
        assert not mask.is_floating_point(), (
            f"mask must be bool or integer dtype, got {mask.dtype}. "
            f"Convert to bool first (e.g. mask > 0)."
        )
        return (value * mask).sum() / mask.sum().clamp(min=1)
    return value.mean()


def reduce_quantile(value: Tensor, quantile: float) -> Tensor:
    """Mean of the bottom *quantile* fraction of values.

    Sorts the flattened values and averages the smallest ``quantile * N``
    elements.

    Args:
        value: Input tensor, any shape.
        quantile: Fraction in ``(0, 1]`` of values to keep.

    Returns:
        Scalar reduced value, or ``0`` if selection is empty.
    """
    assert 0 < quantile <= 1, "quantile must be in (0, 1]"
    flat = value.flatten()
    k = int(len(flat) * quantile)
    if k == 0:
        return value.new_zeros((), requires_grad=value.requires_grad)
    topk = flat.topk(k, largest=False)
    return topk.values.mean()


def reduce_sum(value: Tensor) -> Tensor:
    """Sum reduction.

    Args:
        value: Input tensor, any shape.

    Returns:
        Scalar sum.
    """
    return value.sum()


# ---------------------------------------------------------------------------
# Gaussian regularization losses
# ---------------------------------------------------------------------------


def gaussian_scale_reg(scales: Tensor, visibility: Tensor | None = None) -> Tensor:
    """Penalize large Gaussian scale values.

    Returns the absolute value of each scale component, optionally weighted
    by a per-Gaussian visibility mask.

    Scales must be post-activation (non-negative).  Pass ``exp(log_scales)``,
    not raw log-scale parameters.

    Args:
        scales: Post-activation Gaussian scales ``[N, 3]``, must be ``>= 0``.
        visibility: Optional visibility mask ``[N]`` or ``[N, 1]``.

    Returns:
        Per-element regularization ``[N, 3]`` (apply your own reduction).
    """
    if ENFORCE_CONTRACTS:
        assert (scales >= 0).all(), (
            f"scales must be post-activation (>= 0), got min={scales.min().item():.4f}. "
            f"Pass exp(log_scales), not raw log-scale parameters."
        )
    loss = scales.abs()
    if visibility is not None:
        loss = loss * visibility.view(-1, *[1] * (loss.ndim - 1))
    return loss


def gaussian_density_reg(densities: Tensor, visibility: Tensor | None = None) -> Tensor:
    """Penalize large Gaussian density (opacity) values.

    Returns the absolute value of each density, optionally weighted by a
    per-Gaussian visibility mask.

    Densities must be post-activation (non-negative).  Pass
    ``sigmoid(opacity_logits)``, not raw logit parameters.

    Args:
        densities: Post-activation Gaussian densities ``[N]`` or ``[N, 1]``,
            must be ``>= 0``.
        visibility: Optional visibility mask broadcastable to *densities*.

    Returns:
        Per-element regularization, same shape as *densities*.
    """
    if ENFORCE_CONTRACTS:
        assert (densities >= 0).all(), (
            f"densities must be post-activation (>= 0), got min={densities.min().item():.4f}. "
            f"Pass sigmoid(opacity_logits), not raw logit parameters."
        )
    loss = densities.abs()
    if visibility is not None:
        loss = loss * visibility.view(-1, *[1] * (loss.ndim - 1))
    return loss


def gaussian_z_scale_reg(z_scales: Tensor, threshold: float) -> Tensor:
    """Penalize z-scale values above a threshold.

    Used to constrain the z-axis scale of road-layer Gaussians.

    Z-scales must be post-activation (non-negative).

    Args:
        z_scales: Post-activation z-component of Gaussian scales ``[N]``,
            must be ``>= 0``.
        threshold: Scale value above which penalty is applied.

    Returns:
        Per-element penalty ``[N]``.
    """
    if ENFORCE_CONTRACTS:
        assert (
            z_scales >= 0
        ).all(), f"z_scales must be post-activation (>= 0), got min={z_scales.min().item():.4f}"
    return torch.relu(z_scales - threshold)


def out_of_bound_loss(positions: Tensor, cuboid_dims: Tensor) -> Tensor:
    """Penalize Gaussians outside their bounding cuboids.

    Computes ``relu(|position| - cuboid_dim / 2)`` per axis, giving zero
    loss for Gaussians inside the cuboid and linearly increasing loss
    outside.

    Args:
        positions: Gaussian positions in local cuboid frame ``[N, 3]``.
        cuboid_dims: Cuboid dimensions (full extents) per Gaussian ``[N, 3]``,
            must be positive.

    Returns:
        Per-element penalty ``[N, 3]``.
    """
    if ENFORCE_CONTRACTS:
        assert (
            cuboid_dims > 0
        ).all(), f"cuboid_dims must be positive, got min={cuboid_dims.min().item():.4f}"
    return torch.relu(positions.abs() - cuboid_dims / 2)


def bilateral_grid_drift_loss(
    grids: list[Tensor], num_rows: int = 3, num_cols: int = 4
) -> Tensor:
    """Penalize bilateral grid deviation from identity transform.

    Applies :func:`identity_distance` to each grid and concatenates the
    results.  Grids may have different spatial shapes.

    Args:
        grids: List of grid tensors, each ``(B, C, ...)`` where
            ``C = num_rows * num_cols``.
        num_rows: Rows of the affine matrix (default ``3``).
        num_cols: Columns of the affine matrix (default ``4``).

    Returns:
        Concatenated per-element Frobenius distances from identity,
        or an empty tensor if *grids* is empty.
    """
    if not grids:
        return torch.empty(0, device="cpu", dtype=torch.float32)
    return torch.cat([identity_distance(g, num_rows, num_cols).view(-1) for g in grids])
