"""Depth-supervised and mask-aware losses ported from G-SHARP v0.2.

Scaffolding only. Implementation pending — see
``docs/source/proposals/gsharp_v0_2_port.rst`` and the ``vnath_gsharp`` branch.

Planned public API (all to live in ``gsplat.losses`` once merged):

- :func:`binocular_disparity_l1` — L1 in inverse depth (matches the binocular
  branch of ``EndoRunner.compute_depth_loss`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``).
- :func:`pearson_depth_loss` — ``1 - Pearson r`` on masked depth pairs
  (monocular branch).
- :func:`masked_l1` / :func:`masked_ssim` — mask-aware wrappers around the
  existing ``l1_loss`` / ``ssim_loss`` that zero out tool / dynamic-object
  regions.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def binocular_disparity_l1(
    pred_depth: Tensor,
    gt_depth: Tensor,
    mask: Optional[Tensor] = None,
    eps: float = 1e-7,
) -> Tensor:
    """L1 loss in inverse-depth (disparity) space.

    Args:
        pred_depth: Predicted depth, shape ``(..., H, W)``.
        gt_depth: Ground-truth depth, same shape as ``pred_depth``.
        mask: Optional binary mask; 1 = include pixel, 0 = ignore.
        eps: Small constant added before reciprocal for numerical safety.
    """
    raise NotImplementedError("vnath_gsharp: binocular_disparity_l1 pending")


def pearson_depth_loss(
    pred_depth: Tensor,
    gt_depth: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Monocular depth loss: ``1 - Pearson r`` on masked flattened pairs."""
    raise NotImplementedError("vnath_gsharp: pearson_depth_loss pending")


def masked_l1(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """L1 over only the ``mask == 1`` region. Safe when mask is all-zero."""
    raise NotImplementedError("vnath_gsharp: masked_l1 pending")


def masked_ssim(pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
    """SSIM over only the ``mask == 1`` region."""
    raise NotImplementedError("vnath_gsharp: masked_ssim pending")
