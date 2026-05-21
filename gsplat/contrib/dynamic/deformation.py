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
# Portions of this file (DeformNetwork architecture, DeformationTable
# bookkeeping) are ported from the G-SHARP v0.2 surgical reconstruction
# application; see holohub/applications/surgical_scene_recon/training.
"""Deformation network and per-Gaussian deformation table (experimental).

Public API:

- :class:`DeformNetwork` — MLP that consumes HexPlane features and emits
  per-Gaussian deltas on ``(means, quats, opacities)`` at a given time. Heads
  are zero-initialised so the at-construction behaviour is the identity map
  on its inputs.
- :class:`DeformationTable` — per-Gaussian boolean flag indicating whether
  each Gaussian is animated by the deform-net. Provides
  :meth:`prune` / :meth:`duplicate` / :meth:`split` for lock-step resize
  with ``DefaultStrategy``-style densification.

Port targets:

- ``holohub/applications/surgical_scene_recon/training/scene/deformation.py``
- ``_deformation_table`` / ``update_deformation_table_with_tool_masks`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``

@vcauxbrisebo's MR-013 question to @shsolanki on the integration approach is
still open. Per @vnath's direction (2026-05-07): proceeding without waiting;
refactor if @shsolanki's response changes the design.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["DeformNetwork", "DeformationTable"]


class DeformNetwork(nn.Module):
    """MLP head emitting per-Gaussian deltas on means / quats / opacities.

    Architecture: a *num_layers*-deep ReLU trunk consuming
    ``plane_features`` (typically the output of
    :class:`gsplat.contrib.dynamic.HexPlaneField`), followed by three linear
    heads — 3-d for the position delta, 4-d for the quaternion delta, and
    1-d for the opacity delta. The three heads are zero-initialised so the
    at-construction forward pass returns ``(means, quats, opacities)``
    unchanged (identity map). Locked by
    ``test_deform_net_zero_init_is_identity``.

    The *t* argument is reserved for future time-aware extensions; the
    current implementation expects time information to already be encoded
    into ``plane_features`` via :class:`HexPlaneField`.

    Args:
        feature_dim: Dimensionality of ``plane_features`` (must match
            the producing :class:`HexPlaneField`'s ``feat_dim``).
        hidden_dim: Trunk width. Default ``64``.
        num_layers: Number of ``Linear + ReLU`` blocks in the trunk
            (must be ``>= 1``). Default ``3``.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = [nn.Linear(feature_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.trunk = nn.Sequential(*layers)

        self.pos_head = nn.Linear(hidden_dim, 3)
        self.quat_head = nn.Linear(hidden_dim, 4)
        self.opacity_head = nn.Linear(hidden_dim, 1)

        # Zero-init the heads so the initial deformation is identity.
        # Gradients still flow through the heads (so the trunk learns).
        for head in (self.pos_head, self.quat_head, self.opacity_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        means: Tensor,
        quats: Tensor,
        opacities: Tensor,
        t: Tensor,  # noqa: ARG002 — reserved for future time-aware extensions
        plane_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the per-Gaussian deformation deltas.

        Args:
            means: ``(N, 3)`` Gaussian centres.
            quats: ``(N, 4)`` rotation quaternions (any layout; deltas are
                added in the same layout).
            opacities: ``(N, 1)`` opacity values (raw or activated; the
                delta is added in the same space).
            t: ``(N, 1)`` (or broadcastable) time stamp. Currently ignored;
                kept in the signature for forward-compatibility with
                time-aware variants.
            plane_features: ``(N, feature_dim)`` features sampled from the
                HexPlane field. Must share dtype with the other tensors.

        Returns:
            ``(means_new, quats_new, opacities_new)`` — same shapes as the
            inputs.

        Raises:
            ValueError: on batch-dim mismatch, plane-feature-dim mismatch,
                or dtype mismatch among the four tensors.
        """
        n = means.shape[0]
        if quats.shape[0] != n or opacities.shape[0] != n or plane_features.shape[0] != n:
            raise ValueError(
                f"DeformNetwork: batch dim mismatch — means {means.shape[0]}, "
                f"quats {quats.shape[0]}, opacities {opacities.shape[0]}, "
                f"plane_features {plane_features.shape[0]}."
            )
        if plane_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"DeformNetwork: plane_features last dim "
                f"{plane_features.shape[-1]} != feature_dim {self.feature_dim}."
            )
        if not (means.dtype == quats.dtype == opacities.dtype == plane_features.dtype):
            raise ValueError(
                f"DeformNetwork: dtype mismatch — means {means.dtype}, "
                f"quats {quats.dtype}, opacities {opacities.dtype}, "
                f"plane_features {plane_features.dtype}."
            )

        h = self.trunk(plane_features)
        d_means = self.pos_head(h)
        d_quats = self.quat_head(h)
        d_opacities = self.opacity_head(h)

        return means + d_means, quats + d_quats, opacities + d_opacities


class DeformationTable:
    """Per-Gaussian boolean table marking which Gaussians are dynamic.

    Used by :class:`gsplat.contrib.dynamic.DynamicStrategy` to decide which
    Gaussians get fed through :class:`DeformNetwork` each step. Resize
    lock-step with the gsplat ``DefaultStrategy`` densification ops via
    :meth:`prune`, :meth:`duplicate`, and :meth:`split`.

    The table is a plain ``torch.bool`` tensor (no autograd, no parameters)
    so it adds zero overhead to the optimiser state.

    Args:
        num_gaussians: Initial Gaussian count.
        device: Optional device (defaults to CPU).
    """

    def __init__(
        self, num_gaussians: int, device: Optional[torch.device] = None
    ) -> None:
        if num_gaussians < 0:
            raise ValueError(
                f"DeformationTable: num_gaussians must be >= 0, got {num_gaussians}."
            )
        self.mask = torch.zeros(num_gaussians, dtype=torch.bool, device=device)

    def __len__(self) -> int:
        return int(self.mask.shape[0])

    def set_indices(self, indices: Tensor, value: bool = True) -> None:
        """Mark the given Gaussian indices as dynamic (or static if *value* is False)."""
        self.mask[indices] = value

    def prune(self, keep_mask: Tensor) -> None:
        """Drop Gaussians where *keep_mask* is False (DefaultStrategy prune op).

        Args:
            keep_mask: ``(N,)`` bool tensor with ``True`` for surviving
                Gaussians. Length must equal current table size.
        """
        if keep_mask.shape != self.mask.shape:
            raise ValueError(
                f"DeformationTable.prune: keep_mask shape {tuple(keep_mask.shape)} "
                f"!= table shape {tuple(self.mask.shape)}."
            )
        self.mask = self.mask[keep_mask]

    def duplicate(self, indices: Tensor) -> None:
        """Append duplicates of the given indices (DefaultStrategy duplicate op).

        Originals stay; one duplicate is appended per index, inheriting
        the parent's dynamic flag.
        """
        self.mask = torch.cat([self.mask, self.mask[indices]], dim=0)

    def split(self, indices: Tensor, factor: int = 2) -> None:
        """Replace each index with *factor* children (DefaultStrategy split op).

        Original indices are removed; ``factor`` children are appended per
        split index, each inheriting the parent's dynamic flag.

        Args:
            indices: ``(S,)`` indices to split.
            factor: Children per split. Default ``2`` (matches the gsplat
                ``DefaultStrategy`` convention).
        """
        if factor < 1:
            raise ValueError(
                f"DeformationTable.split: factor must be >= 1, got {factor}."
            )
        keep = torch.ones(self.mask.shape[0], dtype=torch.bool, device=self.mask.device)
        keep[indices] = False
        children = self.mask[indices].repeat_interleave(factor)
        self.mask = torch.cat([self.mask[keep], children], dim=0)
