"""Dynamic-scene trainer recipe (example).

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.

Wires together:

- :class:`examples.datasets.endonerf.EndoNeRFParser` +
  :class:`EndoNeRFDataset`
- :func:`gsplat.init_utils.multi_frame_depth_unprojection`
- :func:`gsplat.losses_depth.binocular_disparity_l1` /
  :func:`pearson_depth_loss` / :func:`masked_l1`
- :func:`gsplat.regularizers.compute_tv_loss_targeted`
- :class:`gsplat.training.TwoStageScheduler`
- :class:`gsplat.contrib.dynamic.DynamicStrategy` +
  :class:`HexPlaneField` + :class:`DeformNetwork`
- :func:`gsplat.rasterization`

Mirrors ``examples/simple_trainer.py`` in structure.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("vnath_gsharp: examples/dynamic_trainer.py pending")


if __name__ == "__main__":
    main()
