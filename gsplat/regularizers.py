"""Regularizers ported from G-SHARP v0.2.

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.

Planned public API:

- :func:`compute_tv_loss_targeted` — total-variation restricted to an
  invisible / occluded region. Ported from ``compute_tv_loss_targeted`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``.
- :func:`dilate_mask` — pure-torch max-pool dilation (no OpenCV dep in
  core library).
- :func:`create_invisible_mask` — union of per-frame instrument masks.
"""

from __future__ import annotations

from typing import Iterable, Union

import torch
from torch import Tensor


def compute_tv_loss_targeted(image: Tensor, mask: Tensor) -> Tensor:
    """Anisotropic TV loss restricted to pixels where ``mask == 1``.

    Args:
        image: ``(C, H, W)`` or ``(N, C, H, W)``.
        mask: Same spatial shape as ``image``, 0/1.
    """
    raise NotImplementedError("vnath_gsharp: compute_tv_loss_targeted pending")


def dilate_mask(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Binary dilation via max-pool. ``kernel_size`` must be odd."""
    raise NotImplementedError("vnath_gsharp: dilate_mask pending")


def create_invisible_mask(masks: Iterable[Union[Tensor, str]]) -> Tensor:
    """Union of per-frame tool / dynamic-object masks.

    Accepts either a sequence of ``(H, W)`` tensors or a sequence of paths to
    binary mask PNGs.
    """
    raise NotImplementedError("vnath_gsharp: create_invisible_mask pending")
