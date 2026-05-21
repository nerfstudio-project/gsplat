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
"""DynamicStrategy — deformable extension of :class:`DefaultStrategy`.

Public API:

- :class:`DynamicStrategy` — subclass of :class:`gsplat.strategy.DefaultStrategy`
  that additionally tracks a :class:`DeformationTable` and resizes it
  in lock-step with each densify / prune op so that subsequent forward
  passes can route only the dynamic Gaussians through the deform-net.

Design notes:

- ``gsplat.rasterization`` has no time axis. The deformation pass itself
  (HexPlane → :class:`DeformNetwork` → ``(means, quats, opacities)``) lives
  in the trainer (``examples/dynamic_surgical_trainer.py``) and runs *before*
  ``rasterization(...)`` is called, mirroring G-SHARP's ``rasterize_splats``.
  This strategy class only owns the densification policy + deformation-table
  bookkeeping; it does **not** apply the deform-net itself.
- The ``@dataclass`` decorator is intentionally not applied here (per the
  ``0001_scaffolding_commit.md`` note): no new fields are added with
  defaults, so we inherit the parent's auto-generated ``__init__`` cleanly
  without dataclass-field ordering errors.

@vcauxbrisebo's MR-013 question to @shsolanki on the deformation integration
approach is still open. Per @vnath's direction (2026-05-07): proceeding
without waiting; refactor if @shsolanki's response changes the design.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch

from gsplat.strategy import DefaultStrategy

from .deformation import DeformationTable  # noqa: F401  (re-export for back-compat)

__all__ = ["DynamicStrategy"]


class DynamicStrategy(DefaultStrategy):
    """Deformable-aware densification / pruning strategy.

    Extra invariants on top of :class:`DefaultStrategy`:

    - ``state["dynamic_mask"]`` is a per-Gaussian ``torch.bool`` tensor of
      shape ``(num_gaussians,)`` that flags which Gaussians the trainer
      should route through the DeformNet. Resized in lock-step with
      ``params["means"]`` by gsplat's strategy ops (which iterate every
      tensor in *state* and apply the per-Gaussian split / duplicate /
      prune permutation — see ``gsplat/strategy/ops.py:135, 191-195,
      223-225``). Identity is preserved across split (children inherit
      the parent's flag).

    Note that the HexPlane and DeformNet trainables are **not** part of
    *params*. gsplat's densification ops blindly iterate every entry in
    *params* and split/duplicate/prune them per-Gaussian; non-per-Gaussian
    tensors (HexPlane plane grids, DeformNet MLP weights) would be indexed
    with out-of-bounds per-Gaussian indices. Keep those trainables in
    their own optimizers, wired separately by the trainer (see
    ``examples/dynamic_surgical_trainer.py:build_deform_modules``).

    Historical note (MR-013 → MR-022): an earlier version of this strategy
    stored a :class:`DeformationTable` wrapper at ``state["deformation_table"]``
    and resized it via a custom ``_resize_table`` hook. That hook did not
    preserve survivor identity across split, and the trainer never consulted
    the mask anyway. Wiring it through *state* as a plain tensor (so gsplat's
    ops do the right thing) closes both gaps. The wrapper class is still
    importable for back-compat; the canonical mask is the tensor in *state*.
    """

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        """Sanity-check identical to :meth:`DefaultStrategy.check_sanity`.

        The HexPlane / DeformNet trainables live outside *params* (see the
        class docstring), so this method has no extra requirements beyond
        the parent's "params and optimizers share keys, and per-Gaussian
        keys means/scales/quats/opacities are present" check.
        """
        super().check_sanity(params, optimizers)

    def initialize_state(
        self,
        scene_scale: float = 1.0,
        num_gaussians: int = 0,
        device: Optional[torch.device] = None,
        init_dynamic: bool = True,
    ) -> Dict[str, Any]:
        """Extend :meth:`DefaultStrategy.initialize_state` with a per-Gaussian
        dynamic mask.

        The mask is stored under ``state["dynamic_mask"]`` as a plain bool
        tensor (shape ``(num_gaussians,)``). It is **not** wrapped in a
        :class:`DeformationTable` so that gsplat's densification ops in
        :mod:`gsplat.strategy.ops` (which iterate every tensor in *state*
        and apply the per-Gaussian split / duplicate / prune permutation
        automatically — see ops.py:135, 191-195, 223-225) can resize it in
        lock-step with ``params["means"]`` with identity preservation
        across split. The wrapper class is still exposed for callers that
        want the helper API (see :class:`DeformationTable`), but the
        canonical mask now lives in *state*.

        Args:
            scene_scale: Forwarded to the parent.
            num_gaussians: Initial Gaussian count.
            device: Device for the mask tensor (defaults to CPU).
            init_dynamic: Initial value for every flag. ``True`` matches the
                current trainer behaviour (every Gaussian goes through
                :class:`DeformNetwork`); set ``False`` if you have a
                static-by-default workflow and intend to flip dynamic
                indices manually.

        Returns:
            The strategy state dict with the additional ``dynamic_mask``
            entry (a ``torch.bool`` tensor of shape ``(num_gaussians,)``).
        """
        state = super().initialize_state(scene_scale=scene_scale)
        fill = bool(init_dynamic)
        state["dynamic_mask"] = torch.full(
            (num_gaussians,), fill, dtype=torch.bool, device=device
        )
        return state

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> None:
        """Pre-backward hook — passthrough to the parent."""
        super().step_pre_backward(params, optimizers, state, step, info)

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Post-backward hook — defers to the parent.

        ``state["dynamic_mask"]`` is resized in lock-step with the per-Gaussian
        parameters automatically by gsplat's densification ops (which iterate
        every tensor in *state* and apply the same per-Gaussian permutation
        as the params — see ``gsplat/strategy/ops.py:135``). No extra hook
        needed here.

        Raises:
            RuntimeError: if ``state["dynamic_mask"]`` is missing (i.e.
                :meth:`initialize_state` wasn't called first).
        """
        if "dynamic_mask" not in state:
            raise RuntimeError(
                "DynamicStrategy.step_post_backward called before "
                "initialize_state(...). Call "
                "`state = strategy.initialize_state(scene_scale=..., "
                "num_gaussians=N)` first."
            )

        super().step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            packed=packed,
        )
