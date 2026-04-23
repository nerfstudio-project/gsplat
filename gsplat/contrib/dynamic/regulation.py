"""HexPlane regularizers (experimental).

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.
Port target: ``holohub/applications/surgical_scene_recon/training/scene/regulation.py``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def plane_smoothness(planes: Sequence[Tensor]) -> Tensor:
    """Spatial 2D Laplacian smoothness over each HexPlane feature plane."""
    raise NotImplementedError("vnath_gsharp: plane_smoothness pending")


def time_smoothness(planes: Sequence[Tensor]) -> Tensor:
    """Temporal smoothness (second-difference) across the time axis of the
    ``xt``, ``yt``, ``zt`` planes.
    """
    raise NotImplementedError("vnath_gsharp: time_smoothness pending")


def time_l1(planes: Sequence[Tensor]) -> Tensor:
    """L1 on the time-only component of ``xt``, ``yt``, ``zt`` planes to
    encourage static tissue to stay put.
    """
    raise NotImplementedError("vnath_gsharp: time_l1 pending")
