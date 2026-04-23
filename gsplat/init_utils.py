"""SfM-free point-cloud initialization ported from G-SHARP v0.2.

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.

Planned public API:

- :func:`multi_frame_depth_unprojection` — unproject masked depth maps into a
  world-space point cloud + per-point RGB, ported from
  ``accumulate_multiframe_pointcloud`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``.
- :func:`knn_scale_init` — per-Gaussian initial scale based on k-NN spacing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def multi_frame_depth_unprojection(
    images: Tensor,
    depths: Tensor,
    masks: Tensor,
    poses: Tensor,
    intrinsics: Tensor,
    max_points: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Multi-frame depth unprojection into a shared world-space point cloud.

    Args:
        images: ``(N, H, W, 3)`` uint8 or float in ``[0, 1]``.
        depths: ``(N, H, W)`` metric depth.
        masks: ``(N, H, W)`` binary (1 = keep).
        poses: ``(N, 4, 4)`` camera-to-world.
        intrinsics: ``(N, 3, 3)``.
        max_points: Optional random subsample cap.

    Returns:
        ``(xyz, rgb)`` — ``xyz`` of shape ``(P, 3)``, ``rgb`` of shape
        ``(P, 3)``.
    """
    raise NotImplementedError("vnath_gsharp: multi_frame_depth_unprojection pending")


def knn_scale_init(xyz: Tensor, k: int = 3) -> Tensor:
    """Per-point initial scale = log(mean k-NN distance). Pure-torch fallback
    when sklearn is not available.
    """
    raise NotImplementedError("vnath_gsharp: knn_scale_init pending")
