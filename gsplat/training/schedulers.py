"""Training schedulers.

Scaffolding only. Implementation pending on branch ``vnath_gsharp``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoStageScheduleStep:
    """Output of :meth:`TwoStageScheduler.step`."""

    stage: str
    frame_index: int
    shuffle: bool


class TwoStageScheduler:
    """Coarse -> fine training schedule ported from G-SHARP v0.2.

    Coarse stage: lock on a single fixed frame, no shuffle, deformation off.
    Fine stage: shuffled dataloader, deformation on, random frames.

    Ported from ``EndoRunner._train_stage`` in
    ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``.
    """

    def __init__(
        self,
        coarse_steps: int,
        fine_steps: int,
        coarse_frame_index: int = 0,
    ) -> None:
        if coarse_steps < 0 or fine_steps < 0:
            raise ValueError("step counts must be non-negative")
        self.coarse_steps = coarse_steps
        self.fine_steps = fine_steps
        self.coarse_frame_index = coarse_frame_index

    def step(self, global_step: int, num_frames: int) -> TwoStageScheduleStep:
        raise NotImplementedError("vnath_gsharp: TwoStageScheduler.step pending")
