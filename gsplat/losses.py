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

import bisect
import math
import os
from typing import Callable, Optional, Sequence
import warnings

import torch
import torch.nn.functional as F

from ._quat_math import quat_slerp_batched
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

        if window_size == 11 and not img2.requires_grad and not img1.is_cpu:
            return 1.0 - fused_ssim(img1, img2, padding="valid")
    except ImportError:
        pass
    except NotImplementedError:
        warnings.warn(
            f"fused_ssim implementation not available for {img1.device}. Falling back to reference implementation.",
            stacklevel=2,
        )

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


def binocular_disparity_l1(
    pred_depth: Tensor,
    gt_depth: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> Tensor:
    """L1 loss in inverse-depth (disparity) space, with optional masking.

    Ported from G-SHARP v0.2's binocular branch of
    ``EndoRunner.compute_depth_loss``. Pixels whose predicted *or*
    ground-truth depth has absolute value at or below *eps* are treated as
    "no depth" and contribute 0 to the inverse-depth tensor (matching the
    ``rendered_depth != 0`` sentinel convention used by the source).

    Args:
        pred_depth: Predicted depth, shape ``(..., H, W)``.
        gt_depth: Ground-truth depth, same shape as *pred_depth*.
        mask: Optional binary mask broadcastable to *pred_depth*; ``!= 0`` =
            include pixel, ``0`` = ignore. Without a mask, the loss is the
            mean over all pixels (the zero-depth ones contribute 0 on both
            sides).
        eps: Threshold below which depth is treated as invalid; also keeps
            the divisor away from zero so the reciprocal stays finite.

    Returns:
        Scalar disparity-space L1 loss.
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"binocular_disparity_l1: pred_depth shape {pred_depth.shape} != "
            f"gt_depth shape {gt_depth.shape}. Shapes must match."
        )
    # A pair (pred_i, gt_i) contributes only when *both* sides are valid.
    # Otherwise the per-side `1/x ↔ 0` substitution leaks `|0 - 1/other|`
    # into the loss when only one side is invalid, which is not what the
    # docstring promises.
    valid_pred = pred_depth.abs() > eps
    valid_gt = gt_depth.abs() > eps
    pair_valid = valid_pred & valid_gt

    ones = torch.ones_like(pred_depth)
    safe_pred = torch.where(valid_pred, pred_depth, ones)
    safe_gt = torch.where(valid_gt, gt_depth, ones)
    pred_inv = 1.0 / safe_pred
    gt_inv = 1.0 / safe_gt

    pair_mask = pair_valid.to(pred_depth.dtype)
    if mask is not None:
        pair_mask = pair_mask * mask.to(pred_depth.dtype)
    return masked_l1(pred_inv, gt_inv, pair_mask)


def pearson_depth_loss(
    pred_depth: Tensor,
    gt_depth: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Monocular depth loss: ``1 - Pearson r`` on (masked) flattened depth pairs.

    Ported from G-SHARP v0.2 (monocular branch). Returns a differentiable 0
    when fewer than two valid samples remain after masking, and clamps the
    denominator to a tiny positive constant when either input has near-zero
    variance — both situations would otherwise yield NaN.

    Args:
        pred_depth: Predicted depth, shape ``(..., H, W)``.
        gt_depth: Ground-truth depth, same shape as *pred_depth*.
        mask: Optional binary mask broadcastable to *pred_depth*; samples
            where ``mask == 0`` are dropped before the correlation.

    Returns:
        Scalar loss in approximately ``[0, 2]``.
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(
            f"pearson_depth_loss: pred_depth shape {pred_depth.shape} != "
            f"gt_depth shape {gt_depth.shape}. Shapes must match."
        )
    pred_flat = pred_depth.reshape(-1)
    gt_flat = gt_depth.reshape(-1)
    if mask is not None:
        mask_flat = (mask != 0).reshape(-1)
        pred_flat = pred_flat[mask_flat]
        gt_flat = gt_flat[mask_flat]
    if pred_flat.numel() < 2:
        return pred_depth.sum() * 0.0

    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    num = (pred_centered * gt_centered).sum()
    denom = (
        (pred_centered.pow(2).sum() * gt_centered.pow(2).sum()).clamp(min=1e-12).sqrt()
    )
    return 1.0 - num / denom


# ---------------------------------------------------------------------------
# Mask-aware photometric wrappers (G-SHARP v0.2)
# ---------------------------------------------------------------------------


def masked_l1(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """L1 over only the ``mask != 0`` region.

    Mask-aware wrapper around :func:`l1_loss`, ported from G-SHARP v0.2 to
    zero out tool / dynamic-object regions. The mask is broadcast to *pred*'s
    shape (so a ``[B, 1, H, W]`` mask works on ``[B, C, H, W]`` inputs) and
    the mean is taken over unmasked elements only. Returns a differentiable 0
    when the mask is all-zero (no NaN).

    Args:
        pred: Predicted tensor.
        gt: Ground-truth tensor, same shape as *pred*.
        mask: Mask broadcastable to *pred*; ``!= 0`` = include element.

    Returns:
        Scalar L1 loss.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"masked_l1: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match."
        )
    abs_diff = (pred - gt).abs()
    bool_mask = mask != 0
    if bool_mask.shape != abs_diff.shape:
        bool_mask = bool_mask.expand_as(abs_diff)
    selected = abs_diff[bool_mask]
    if selected.numel() == 0:
        return pred.sum() * 0.0
    return selected.mean()


def masked_ssim(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """SSIM loss after zeroing both inputs in the masked region.

    Mask-aware wrapper around :func:`ssim_loss`, ported from G-SHARP v0.2.
    Both *pred* and *gt* are multiplied by *mask* before SSIM, matching the
    G-SHARP training-loop convention (mask the rendered output and the GT,
    then take an unmasked SSIM over the zeroed pair).

    Note that the mean is taken over the full image, so the loss magnitude
    scales with mask coverage — sparse masks dilute the reported loss in
    proportion to the masked-out fraction. This bias is intentional and
    matches the upstream G-SHARP convention.

    Args:
        pred: Predicted image batch ``[B, C, H, W]``, values in ``[0, 1]``.
        gt: Ground-truth image batch, same shape as *pred*.
        mask: Mask broadcastable to *pred* (e.g. ``[B, 1, H, W]``).

    Returns:
        Scalar SSIM loss (``1 - SSIM`` of the zeroed pair).
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"masked_ssim: pred shape {pred.shape} != gt shape {gt.shape}. "
            "Shapes must match."
        )
    return ssim_loss(pred * mask, gt * mask)


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


def fold_log_space_weight(preactivation: Tensor, weight: float) -> Tensor:
    """Fold a per-segment weight into log-space pre-activations.

    ``exp(x + log(w)) == w * exp(x)``, so adding ``log(weight)`` to a
    segment's pre-activations makes a downstream fused ``exp()`` (e.g.
    :class:`gsplat.losses_fused.FusedGaussianLosses` in pre-activation mode)
    produce the weighted post-activation values directly.

    A non-positive weight takes an exact-zero branch instead: the segment is
    replaced by a ``-inf`` fill, which ``exp()`` maps to an exact zero for
    both the value and the gradient. Weighting by multiplication *after* the
    ``exp()`` could not guarantee this — ``0 * exp(x)`` is NaN once ``x``
    is large enough for ``exp(x)`` to overflow to infinity.

    .. warning::
        Folding is exact only where the downstream loss is **linear** in the
        activated value ``exp(x)`` — the scale member of
        :class:`~gsplat.losses_fused.FusedGaussianLosses` (densities fold
        their weights by plain multiplication instead). Do **not** fold a
        weight into pre-activations that feed a *thresholded* member such as
        z-scale: there the threshold is subtracted after the ``exp()``, so
        the folded weight lands inside the relu::

            relu(exp(z + log(w)) - T) == relu(w * exp(z) - T)
                                      != w * relu(exp(z) - T)

        e.g. with ``exp(z) = 10`` and ``T = 8``, ``w = 0.5`` gives
        ``relu(5 - 8) = 0`` where the intended weighted loss is
        ``0.5 * relu(10 - 8) = 1``. Only ``w = 1`` (identity) and the
        exact-zero ``w <= 0`` branch are valid for such members — and the
        latter only because thresholds are non-negative by contract
        (``relu(0 - T)`` with ``T < 0`` would be positive; the module and
        the native launchers reject negative ``z_scale_threshold``); a
        fractional weight must be applied to the member's *output* instead
        (or folded into the threshold as well: ``relu(w * exp(z) - w * T)
        == w * relu(exp(z) - T)`` for ``w > 0``).

    Args:
        preactivation: Log-space pre-activations, any shape.
        weight: Non-log-space segment weight. ``weight <= 0`` disables the
            segment exactly (zero values, zero gradients — the returned fill
            is a constant with no autograd edge to *preactivation*).

    Returns:
        Pre-activations with the weight folded in, same shape as
        *preactivation*.
    """
    if weight <= 0.0:
        return torch.full_like(preactivation, float("-inf"))
    return preactivation + math.log(weight)


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
        threshold: Scale value above which penalty is applied. Must be
            non-negative, so that empty or disabled members stay exact
            zeros (``relu(0 - T) == 0``).

    Returns:
        Per-element penalty ``[N]``.
    """
    # `not (T >= 0)` mirrors the native launcher predicate: it also rejects
    # NaN, which `T < 0` would let through (nan < 0 is False).
    if not threshold >= 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}")
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


# ---------------------------------------------------------------------------
# Background-in-track + node-semantic Gaussian losses
#
# Pure-PyTorch reference/fallback for the fused CUDA kernel behind
# gsplat.losses_fused.FusedBgTrackNodeSemanticLosses. The track containment
# test reproduces the CUDA `interpolate_track_boxes` precompute exactly:
# SE(3) pose interpolation (lerp centers, slerp unit quaternions in
# (x, y, z, w) order) at the truncated-mean camera timestamp, followed by an
# inclusive AABB test in each track's local frame.
# ---------------------------------------------------------------------------


# Mirrors gsplat_geometry::QuatNormEps (squared-norm threshold below which the
# safe normalize maps the quaternion to identity).
_QUAT_NORM_EPS = {torch.float32: 1e-7, torch.float64: 1e-12}


def _quat_xyzw_to_rotmat(quat: Tensor) -> Tensor:
    """Rotation matrix from a quaternion in ``(x, y, z, w)`` order.

    Mirrors ``gsplat_geometry::quat_to_matrix_fwd_write`` so the fallback
    matches the CUDA track-box precompute bit-for-bit up to float evaluation
    order: safe-normalizes first (a near-zero quaternion maps to identity)
    and uses the ``1 - 2(y^2 + z^2)`` diagonal form.
    """
    norm_sq = (quat * quat).sum()
    # Other float dtypes reuse the fp32 threshold: the compiled path only
    # instantiates fp32/fp64, but this fallback stays usable for any float.
    eps = _QUAT_NORM_EPS.get(quat.dtype, _QUAT_NORM_EPS[torch.float32])
    if float(norm_sq) < eps:
        quat = quat.new_tensor([0.0, 0.0, 0.0, 1.0])
    else:
        quat = quat / norm_sq.sqrt()
    x, y, z, w = quat.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    row0 = torch.stack([1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)])
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)])
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)])
    return torch.stack([row0, row1, row2])


def _unitquat_slerp(q0: Tensor, q1: Tensor, alpha: float) -> Tensor:
    """Shortest-arc slerp between unit quaternions ``[4]``.

    Delegates to the shared extension-free geometry-convention
    implementation (:mod:`gsplat._quat_math`), which mirrors
    ``gsplat_geometry::quat_slerp_pair_fwd`` — the same helper the CUDA
    track-box kernel calls — so the fallback and the kernel cannot drift.
    """
    return quat_slerp_batched(q0, q1, alpha)


def _interpolate_track_boxes(
    camera_timestamps_startend_us: Tensor,
    tracks_packinfo: Tensor,
    tracks_poses: Tensor,
    tracks_timestamps_us: Tensor,
    cuboids_dims: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Interpolate each cuboid track's oriented box at the camera timestamp.

    Reference implementation of the CUDA ``interpolate_track_boxes`` kernel.
    The query timestamp is ``start + trunc((end - start) / 2)`` from the first
    row of *camera_timestamps_startend_us* (exact int64 arithmetic matching
    the kernel's C truncation, which equals ``torch.mean(float64).to(int64)``
    for positive microsecond Unix timestamps).

    A track is **invalid** (never contains any point) when it has fewer than
    two poses, when the timestamp falls outside its pose-timestamp range
    (time-window gating), or when the bracketing pose timestamps coincide.

    Args:
        camera_timestamps_startend_us: ``[B, 2]`` int64; only row 0 is read.
        tracks_packinfo: ``[T, 2]`` int ``(start_index, n_poses)`` per track
            into the packed pose arrays.
        tracks_poses: ``[P, 7]`` float ``(tx, ty, tz, qx, qy, qz, qw)``.
        tracks_timestamps_us: ``[P]`` int64, sorted within each track.
        cuboids_dims: ``[T, 3]`` float full cuboid extents.

    Returns:
        Tuple ``(rot_w2l [T, 3, 3], centers [T, 3], half_dims [T, 3],
        valid [T] bool)`` where ``rot_w2l`` maps world offsets into the track
        frame (``local = rot_w2l @ (p - center)``).

    Note:
        Host-side control flow: reads the pack info and timestamps to the CPU
        (synchronizes on CUDA tensors). Intended for the fallback path.
    """
    n_tracks = int(tracks_packinfo.shape[0])
    device = tracks_poses.device
    dtype = tracks_poses.dtype
    rot_w2l = torch.zeros((n_tracks, 3, 3), device=device, dtype=dtype)
    centers = torch.zeros((n_tracks, 3), device=device, dtype=dtype)
    half_dims = torch.zeros((n_tracks, 3), device=device, dtype=dtype)
    valid = torch.zeros(n_tracks, dtype=torch.bool, device=device)
    if n_tracks == 0:
        return rot_w2l, centers, half_dims, valid

    timestamps_flat = camera_timestamps_startend_us.reshape(-1)
    t_start = int(timestamps_flat[0].item())
    t_end = int(timestamps_flat[1].item())
    delta = t_end - t_start
    # C-style truncation toward zero (Python // floors), matching the kernel.
    timestamp = t_start + (abs(delta) // 2) * (1 if delta >= 0 else -1)

    packinfo = tracks_packinfo.detach().cpu().tolist()
    pose_timestamps = tracks_timestamps_us.detach().cpu().tolist()

    with torch.no_grad():
        for track_idx in range(n_tracks):
            start_idx = int(packinfo[track_idx][0])
            n_poses = int(packinfo[track_idx][1])
            if n_poses <= 1:
                continue
            local_ts = pose_timestamps[start_idx : start_idx + n_poses]
            if timestamp < local_ts[0] or timestamp > local_ts[-1]:
                continue
            # Mirrors the kernel's direct lower_bound bracketing: timestamp
            # is range-checked above, so bisect_left lands in [0, n - 1]; the
            # only boundary needing adjustment is timestamp == local_ts[0],
            # where the bracket must be (0, 1) with alpha == 0 — max() pins it
            # (t == local_ts[-1] already yields n - 1 by lower_bound).
            end_idx = max(bisect.bisect_left(local_ts, timestamp), 1)
            t0 = local_ts[end_idx - 1]
            t1 = local_ts[end_idx]
            if t1 == t0:
                continue
            alpha = (timestamp - t0) / (t1 - t0)
            pose0 = tracks_poses[start_idx + end_idx - 1]
            pose1 = tracks_poses[start_idx + end_idx]
            centers[track_idx] = (1.0 - alpha) * pose0[:3] + alpha * pose1[:3]
            quat = _unitquat_slerp(pose0[3:7], pose1[3:7], alpha)
            rot_w2l[track_idx] = _quat_xyzw_to_rotmat(quat).transpose(0, 1)
            half_dims[track_idx] = cuboids_dims[track_idx] * 0.5
            valid[track_idx] = True
    return rot_w2l, centers, half_dims, valid


def _points_in_tracks(
    positions: Tensor,
    rot_w2l: Tensor,
    centers: Tensor,
    half_dims: Tensor,
    valid: Tensor,
) -> Tensor:
    """Boolean ``[N]`` mask: point lies inside **any** valid track box.

    Containment is inclusive at the cuboid boundary
    (``|rot_w2l @ (p - center)| <= half_dims`` per axis), matching the CUDA
    ``point_inside_box`` arithmetic (rotate the centered offset, then compare).
    """
    n_points = positions.shape[0]
    inside_any = torch.zeros(n_points, dtype=torch.bool, device=positions.device)
    if n_points == 0 or int(valid.sum().item()) == 0:
        return inside_any
    with torch.no_grad():
        rot = rot_w2l[valid]  # [K, 3, 3]
        center = centers[valid]  # [K, 3]
        half = half_dims[valid]  # [K, 3]
        rel = positions.detach().unsqueeze(1) - center.unsqueeze(0)  # [N, K, 3]
        local = torch.einsum("kij,nkj->nki", rot, rel)  # [N, K, 3]
        inside = (local.abs() <= half.unsqueeze(0)).all(dim=-1)  # [N, K]
        inside_any = inside.any(dim=1)
    return inside_any


def _semantic_class_membership(
    semantic_logits: Tensor, class_ids: Sequence[int]
) -> Tensor:
    """Boolean ``[N]`` mask: per-point semantic argmax is in *class_ids*.

    Class ids outside ``[0, C)`` are ignored (matching the CUDA
    ``make_semantic_class_masks`` behavior); an empty id set matches nothing.
    Ties in the argmax resolve to the first maximal class, as in the kernel's
    strict-greater scan and ``torch.argmax``.
    """
    argmax = semantic_logits.detach().argmax(dim=1)
    n_classes = semantic_logits.shape[1]
    valid_ids = sorted({int(c) for c in class_ids if 0 <= int(c) < n_classes})
    if not valid_ids:
        return torch.zeros_like(argmax, dtype=torch.bool)
    ids = torch.tensor(valid_ids, dtype=argmax.dtype, device=argmax.device)
    return torch.isin(argmax, ids)


def _validate_segment_layout(
    name: str,
    semantic_logits: Sequence[Tensor],
    segment_ends: Sequence[int],
    max_end: int,
) -> None:
    """Check the packed-segment contract shared by both grouped losses."""
    if len(semantic_logits) != len(segment_ends):
        raise ValueError(
            f"{name}: semantic tensors and segments must have the same length, "
            f"got {len(semantic_logits)} vs {len(segment_ends)}."
        )
    prev_end = 0
    for seg_idx, (seg_logits, end) in enumerate(zip(semantic_logits, segment_ends)):
        if end < prev_end or end > max_end:
            raise ValueError(
                f"{name}: segment ends must be nondecreasing and bounded by "
                f"{max_end}, got end={end} after {prev_end} (segment {seg_idx})."
            )
        if seg_logits.dim() != 2 or seg_logits.shape[1] < 1:
            raise ValueError(
                f"{name}: each semantic tensor must be [N_segment, C], got "
                f"{tuple(seg_logits.shape)} (segment {seg_idx})."
            )
        if seg_logits.shape[0] != end - prev_end:
            raise ValueError(
                f"{name}: semantic tensor rows ({seg_logits.shape[0]}) must "
                f"equal the segment length ({end - prev_end}) (segment {seg_idx})."
            )
        prev_end = end


def background_in_track_loss(
    positions: Tensor,
    density_logits: Tensor,
    camera_timestamps_startend_us: Tensor,
    tracks_packinfo: Tensor,
    tracks_poses: Tensor,
    tracks_timestamps_us: Tensor,
    cuboids_dims: Tensor,
    semantic_logits: Optional[Sequence[Tensor]] = None,
    semantic_segments: Optional[Sequence[tuple[int, Sequence[int]]]] = None,
    background_lambda: float = 1.0,
    density_logits_min: float = -20.0,
) -> tuple[Tensor, Tensor]:
    """Background-in-track BCE suppression loss (grouped family, bg member).

    Penalizes the density of background Gaussians that lie inside any dynamic
    cuboid track at the (truncated) mean camera timestamp. A point is
    *selected* when all of the following hold:

    - its density logit is strictly greater than *density_logits_min*;
    - its semantic argmax is one of the allowed class ids for its segment
      (points past the last segment end are **unfiltered** and always pass;
      a segment with an empty id list selects nothing);
    - it lies inside at least one valid track box (inclusive AABB test in the
      track frame after SE(3) interpolation, with time-window gating against
      the track's pose timestamps).

    The raw loss is the mean of ``BCEWithLogits(density_logit, 0)`` (i.e.
    ``softplus``) over selected points — a selected-count denominator, exactly
    zero (still differentiable) when nothing is selected. Only
    *density_logits* receives gradients; the containment and semantic tests
    are non-differentiable selections.

    Args:
        positions: ``[N, 3]`` world-space Gaussian positions.
        density_logits: ``[N]`` pre-activation density logits.
        camera_timestamps_startend_us: ``[B, 2]`` int64; row 0 holds the
            (start, end) capture timestamps in microseconds.
        tracks_packinfo: ``[T, 2]`` int ``(start_index, n_poses)`` per track.
        tracks_poses: ``[P, 7]`` float packed ``(tx, ty, tz, qx, qy, qz, qw)``.
        tracks_timestamps_us: ``[P]`` int64 packed pose timestamps.
        cuboids_dims: ``[T, 3]`` float full cuboid extents.
        semantic_logits: Optional per-segment logits ``[N_i, C_i]`` covering
            the first ``sum(N_i)`` points (filtered layers are packed first).
        semantic_segments: Optional ``(exclusive_end, allowed_class_ids)`` per
            segment, aligned with *semantic_logits*.
        background_lambda: Loss weight for the returned weighted scalar.
        density_logits_min: Selection threshold (strictly-greater comparison);
            defaults to ``-20.0``.

    Returns:
        Tuple ``(raw, weighted)`` of scalar tensors with
        ``weighted = background_lambda * raw``.
    """
    if positions.dim() != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"background_in_track_loss: positions must be [N, 3], got "
            f"{tuple(positions.shape)}."
        )
    if density_logits.dim() != 1 or density_logits.shape[0] != positions.shape[0]:
        raise ValueError(
            f"background_in_track_loss: density_logits must be [N] matching "
            f"positions, got {tuple(density_logits.shape)} vs N={positions.shape[0]}."
        )
    if (semantic_logits is None) != (semantic_segments is None):
        raise ValueError(
            "background_in_track_loss: semantic_logits and semantic_segments "
            "must be provided together."
        )
    n_points = int(density_logits.shape[0])
    semantic_logits = list(semantic_logits) if semantic_logits is not None else []
    semantic_segments = list(semantic_segments) if semantic_segments is not None else []
    segment_ends = [int(end) for end, _ in semantic_segments]
    _validate_segment_layout(
        "background_in_track_loss", semantic_logits, segment_ends, n_points
    )

    rot_w2l, centers, half_dims, valid = _interpolate_track_boxes(
        camera_timestamps_startend_us,
        tracks_packinfo,
        tracks_poses,
        tracks_timestamps_us,
        cuboids_dims,
    )
    in_track = _points_in_tracks(positions, rot_w2l, centers, half_dims, valid)

    # Semantic filter: per-segment allowed ids; the unfiltered suffix (points
    # past the last segment end) always passes.
    allowed = torch.ones(n_points, dtype=torch.bool, device=density_logits.device)
    prev_end = 0
    for (end, class_ids), seg_logits in zip(semantic_segments, semantic_logits):
        allowed[prev_end : int(end)] = _semantic_class_membership(seg_logits, class_ids)
        prev_end = int(end)

    selected = in_track & allowed & (density_logits.detach() > density_logits_min)
    # Gather-then-reduce: only selected logits are ever read, so a non-finite
    # value in an unselected point cannot poison the loss or its gradient
    # (parity with the CUDA kernels, which predicate every read on the
    # selection bit). With nothing selected this is an empty-view sum: an
    # exact, still-differentiable zero.
    selected_logits = density_logits[selected]
    per_element = F.binary_cross_entropy_with_logits(
        selected_logits, torch.zeros_like(selected_logits), reduction="none"
    )
    count = selected.sum()
    raw = per_element.sum() / count.clamp(min=1)
    return raw, background_lambda * raw


def node_semantic_loss(
    density_logits: Tensor,
    semantic_logits: Sequence[Tensor],
    segments: Sequence[tuple[int, Sequence[int], bool]],
    node_lambda: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Node-semantic grouped BCE suppression loss (grouped family, node member).

    Density logits from all configured nodes are packed into one ``[N]``
    tensor; each segment ``(exclusive_end, class_ids, select_matches)`` owns
    a contiguous row range and its own semantic tensor (class counts may
    differ between segments). A point is *selected* — i.e. penalized — when

    ``(argmax(semantic_logits) in class_ids) == select_matches``.

    So ``select_matches=True`` penalizes points whose class **is** in
    *class_ids* (an exclusion list), and ``select_matches=False``
    penalizes points whose class is **not** in *class_ids* (an allow list).
    The predicate polarity matters for the empty-id edge
    cases: empty ids with ``select_matches=True`` penalize **nothing**, empty
    ids with ``select_matches=False`` penalize **everything** — both are legal
    configurations.

    The raw loss is the mean of ``BCEWithLogits(density_logit, 0)`` over the
    union of selected points across **all** segments (one shared
    selected-count denominator), zero when nothing is selected. Only
    *density_logits* receives gradients.

    Args:
        density_logits: ``[N]`` packed pre-activation density logits.
        semantic_logits: Per-segment logits ``[N_i, C_i]``; segments must
            cover every packed row.
        segments: ``(exclusive_end, class_ids, select_matches)`` per segment.
        node_lambda: Loss weight for the returned weighted scalar.

    Returns:
        Tuple ``(raw, weighted)`` of scalar tensors with
        ``weighted = node_lambda * raw``.
    """
    if density_logits.dim() != 1:
        raise ValueError(
            f"node_semantic_loss: density_logits must be [N], got "
            f"{tuple(density_logits.shape)}."
        )
    n_points = int(density_logits.shape[0])
    semantic_logits = list(semantic_logits)
    segments = list(segments)
    segment_ends = [int(end) for end, _, _ in segments]
    _validate_segment_layout(
        "node_semantic_loss", semantic_logits, segment_ends, n_points
    )
    if (segment_ends[-1] if segment_ends else 0) != n_points:
        raise ValueError(
            f"node_semantic_loss: segments must cover every packed density row "
            f"(last end {segment_ends[-1] if segment_ends else 0} != {n_points})."
        )

    selected = torch.zeros(n_points, dtype=torch.bool, device=density_logits.device)
    prev_end = 0
    for (end, class_ids, select_matches), seg_logits in zip(segments, semantic_logits):
        matches = _semantic_class_membership(seg_logits, class_ids)
        selected[prev_end : int(end)] = matches if select_matches else ~matches
        prev_end = int(end)

    # Gather-then-reduce: only selected logits are ever read (see
    # background_in_track_loss), keeping non-finite unselected points out of
    # the loss and its gradient; the no-selection case is an empty-view sum
    # (exact, differentiable zero).
    selected_logits = density_logits[selected]
    per_element = F.binary_cross_entropy_with_logits(
        selected_logits, torch.zeros_like(selected_logits), reduction="none"
    )
    count = selected.sum()
    raw = per_element.sum() / count.clamp(min=1)
    return raw, node_lambda * raw


# ---------------------------------------------------------------------------
# Deform smoothness (scalar, training contract)
# ---------------------------------------------------------------------------


def deform_smoothness_loss(deformation: Tensor, mask: Tensor | None = None) -> Tensor:
    """Masked mean of ``|deformation|`` (scalar) — the deform-smoothness loss.

    Unlike most functions in this module, this is a *training-contract*
    scalar (masked mean), matching the fused CUDA kernels behind the optional
    deform member of :class:`gsplat.losses_fused.FusedGaussianLosses` (and
    serving as their pure-PyTorch reference/fallback). The contract is an
    ``abs`` + masked-mean reduction:

    - The numerator sums ``|deformation| * mask`` with the mask broadcast to
      the deformation's shape. Masked-out elements (``mask == 0``) are fully
      inert: their deformation is never read, so a non-finite residual left
      behind in a masked-out row cannot poison the loss (masking a region is
      often exactly why its residuals are garbage).
    - The denominator sums the **original** (un-broadcast) mask exactly once
      and is clamped to ``>= 1``, so an all-zero mask yields an exact,
      differentiable zero (the clamp also suppresses the mask's denominator
      gradient term while active).
    - Without a mask the loss is the plain mean of ``|deformation|``; an
      empty ``deformation`` yields a differentiable zero rather than ``0/0``.

    Gradients w.r.t. ``deformation`` are ``upstream * mask * sign(d) /
    denominator`` (``sign(0) == 0``; exactly zero at masked-out elements).
    Gradients w.r.t. ``mask`` combine the numerator term (``upstream * |d| /
    denominator``, broadcast-reduced onto the mask element, and exactly zero
    at masked-out elements — an inert lane deliberately stops receiving the
    "this residual is large" reopening signal) and the denominator term
    (``-upstream * loss_sum / denominator**2``, applied only when
    ``mask.sum() >= 1``).

    Args:
        deformation: Deformation residuals ``[N, D]`` (e.g. ``[N, 3]``
            position deltas). Any float dtype here; the fused CUDA path
            supports ``fp32``/``fp64`` only.
        mask: Optional float weights broadcastable to *deformation*'s shape
            (e.g. ``[N, 1]`` per-point weights). One-way broadcast only: the
            mask must not force *deformation* to expand.

    Returns:
        Scalar loss tensor (shape ``[]``).
    """
    if mask is not None:
        # One-way broadcast contract (cheap shape-only check, no GPU sync):
        # a mask that would expand deformation changes the numerator element
        # count and is rejected by the CUDA path as well.
        broadcast_shape = torch.broadcast_shapes(deformation.shape, mask.shape)
        if broadcast_shape != deformation.shape:
            raise ValueError(
                f"mask {tuple(mask.shape)} must broadcast to deformation's "
                f"shape {tuple(deformation.shape)} without expanding it"
            )
        # Masked-out elements are fully inert: gate |d| BEFORE the multiply so
        # neither the loss nor any gradient reads it. Gating the product
        # instead would leak NaN into the mask gradient — the mul's backward
        # computes grad_mask = upstream * |d| and 0 * inf == NaN even when the
        # where() zeroes the upstream.
        gated_abs = torch.where(mask == 0, deformation.new_zeros(()), deformation.abs())
        numerator = (gated_abs * mask).sum()
        # The original mask is reduced exactly once (not its broadcast view);
        # the clamp makes the all-zero-mask case an exact zero instead of a
        # blow-up and zeroes the denominator's gradient while active.
        denominator = mask.sum().clamp(min=1)
        return numerator / denominator
    if deformation.numel() == 0:
        # Differentiable zero connected to the input, matching the CUDA path
        # (clamped denominator) instead of mean's 0/0 = NaN.
        return deformation.sum()
    return deformation.abs().sum() / deformation.numel()
