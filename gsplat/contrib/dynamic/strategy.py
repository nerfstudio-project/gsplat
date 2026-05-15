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

from .deformation import DeformationTable

__all__ = ["DynamicStrategy"]


class DynamicStrategy(DefaultStrategy):
    """Deformable-aware densification / pruning strategy.

    Extra invariants on top of :class:`DefaultStrategy`:

    - ``state`` carries a :class:`DeformationTable` instance under
      ``state["deformation_table"]`` after :meth:`initialize_state`.

    Note that the HexPlane and DeformNet trainables are **not** part of
    *params*. gsplat's densification ops blindly iterate every entry in
    *params* and split/duplicate/prune them per-Gaussian; non-per-Gaussian
    tensors (HexPlane plane grids, DeformNet MLP weights) would be indexed
    with out-of-bounds per-Gaussian indices. Keep those trainables in
    their own optimizers, wired separately by the trainer (see
    ``examples/dynamic_surgical_trainer.py:build_deform_modules``).

    The current resize policy on densify / prune mirrors G-SHARP:

    - **Grow:** new Gaussians appended at the tail of ``params["means"]``
      are flagged as **dynamic** (``True``) in the table — matches the
      G-SHARP ``new_mask = ones(num_new, dtype=bool)`` convention at
      ``gsplat_train.py:1365``.
    - **Shrink:** the table is truncated to the new length. This does
      **not** preserve exact survivor flag identities (the gsplat ``remove``
      op compacts via boolean indexing; we don't have access to the prune
      mask from outside ``_prune_gs``). G-SHARP makes the same trade-off
      and notes "this shouldn't happen in current strategy" — adequate for
      uniformly-dynamic tables; revisit if exact tracking is needed.
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
    ) -> Dict[str, Any]:
        """Extend :meth:`DefaultStrategy.initialize_state` with a deformation table.

        Args:
            scene_scale: Forwarded to the parent.
            num_gaussians: Initial Gaussian count for the
                :class:`DeformationTable`. All flags start ``False``;
                callers can flip dynamic ones via
                :meth:`DeformationTable.set_indices`.
            device: Device for the deformation-table mask (defaults to CPU).

        Returns:
            The strategy state dict with the additional
            ``deformation_table`` entry.
        """
        state = super().initialize_state(scene_scale=scene_scale)
        state["deformation_table"] = DeformationTable(
            num_gaussians=num_gaussians, device=device
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
        """Post-backward hook — runs the parent then resizes the table.

        Raises:
            RuntimeError: if ``state["deformation_table"]`` is missing
                (i.e. :meth:`initialize_state` wasn't called first).
        """
        if "deformation_table" not in state:
            raise RuntimeError(
                "DynamicStrategy.step_post_backward called before "
                "initialize_state(...). Call "
                "`state = strategy.initialize_state(scene_scale=..., "
                "num_gaussians=N)` first."
            )

        n_before = len(params["means"])
        super().step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            packed=packed,
        )
        n_after = len(params["means"])
        if n_after != n_before:
            self._resize_table(state["deformation_table"], n_before, n_after)

    @staticmethod
    def _resize_table(table: DeformationTable, n_before: int, n_after: int) -> None:
        """Resize *table* to match a Gaussian count delta after a densify op.

        - On grow, appends ``n_after - n_before`` ``True`` flags (new
          Gaussians are dynamic by default).
        - On shrink, truncates to ``n_after`` (see class-level docstring for
          the limitation).
        """
        if n_after > n_before:
            n_new = n_after - n_before
            new_flags = torch.ones(n_new, dtype=torch.bool, device=table.mask.device)
            table.mask = torch.cat([table.mask, new_flags], dim=0)
        elif n_after < n_before:
            table.mask = table.mask[:n_after]
