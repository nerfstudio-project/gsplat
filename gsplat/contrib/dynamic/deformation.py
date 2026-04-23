"""Deformation network and per-Gaussian deformation table (experimental).

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.
Port targets:
- ``holohub/applications/surgical_scene_recon/training/scene/deformation.py``
- ``_deformation_table`` / ``update_deformation_table_with_tool_masks`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DeformNetwork(nn.Module):
    """MLP that consumes HexPlane features and emits per-Gaussian deltas for
    means, rotations, and opacities at a given time ``t``.

    Zero-initialized output heads make the at-initialization behavior an
    identity map on ``(means, quats, opacities)`` — this invariant is asserted
    in ``tests/test_contrib_deformnet.py``.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        raise NotImplementedError("vnath_gsharp: DeformNetwork pending")

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        opacities: Tensor,
        t: Tensor,
        plane_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError("vnath_gsharp: DeformNetwork.forward pending")


class DeformationTable:
    """Per-Gaussian boolean table marking which Gaussians are dynamic.

    Resized in lock-step with :class:`gsplat.strategy.DefaultStrategy`
    densification ops (duplicate / split / prune) by
    :class:`gsplat.contrib.dynamic.DynamicStrategy`.
    """

    def __init__(
        self, num_gaussians: int, device: Optional[torch.device] = None
    ) -> None:
        self.mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)

    def update_on_densify(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "vnath_gsharp: DeformationTable.update_on_densify pending"
        )
