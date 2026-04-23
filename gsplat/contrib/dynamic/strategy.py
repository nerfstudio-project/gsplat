"""DynamicStrategy — deformable extension of :class:`DefaultStrategy`.

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.

Design notes:

- ``gsplat.rasterization`` has no time axis. :class:`DynamicStrategy` applies
  the HexPlane + :class:`DeformNetwork` to ``(means, quats, opacities)``
  *before* ``rasterization(...)`` is invoked, mirroring the pattern in
  G-SHARP's ``rasterize_splats`` (see
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``).
- Densification ops (duplicate / split / prune) are delegated to
  :class:`DefaultStrategy`; this subclass additionally keeps the
  :class:`DeformationTable` in sync with the Gaussian count after every op.
"""

from __future__ import annotations

from typing import Any, Dict

from gsplat.strategy import DefaultStrategy


class DynamicStrategy(DefaultStrategy):
    """Deformable-aware densification/pruning strategy.

    Extra invariants on top of :class:`DefaultStrategy`:

    - ``params`` must include ``hexplane_params`` and ``deform_mlp_params``
      keys (optimized by separate optimizers with matching keys).
    - ``state`` carries a :class:`DeformationTable` instance under
      ``state["deformation_table"]``.
    """

    def check_sanity(self, params: Dict[str, Any], optimizers: Dict[str, Any]) -> None:
        raise NotImplementedError("vnath_gsharp: DynamicStrategy.check_sanity pending")

    def initialize_state(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            "vnath_gsharp: DynamicStrategy.initialize_state pending"
        )

    def step_pre_backward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "vnath_gsharp: DynamicStrategy.step_pre_backward pending"
        )

    def step_post_backward(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "vnath_gsharp: DynamicStrategy.step_post_backward pending"
        )
