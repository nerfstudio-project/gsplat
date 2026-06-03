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
"""Training schedulers ported from G-SHARP v0.2.

Public API:

- :class:`TwoStageScheduleStep` — return type of :meth:`TwoStageScheduler.step`.
- :class:`TwoStageScheduler` — coarse → fine two-stage training schedule.
  Ported from the G-SHARP v0.2 reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TwoStageScheduleStep:
    """Output of :meth:`TwoStageScheduler.step`.

    Attributes:
        stage: ``"coarse"`` or ``"fine"``.
        frame_index: Frame index to use this step. In the coarse stage this
            is always ``coarse_frame_index``; in the fine stage it cycles
            ``(global_step - coarse_steps) % num_frames``, giving the caller
            a deterministic index that visits every frame.
        shuffle: ``True`` in the fine stage, ``False`` in the coarse stage —
            mirrors the ``shuffle=(stage == "fine")`` flag G-SHARP passes to
            its ``DataLoader``. Callers building a ``DataLoader`` should
            forward this flag and may then ignore *frame_index* (the loader
            provides its own random order).
    """

    stage: str
    frame_index: int
    shuffle: bool


class TwoStageScheduler:
    """Coarse → fine training schedule ported from G-SHARP v0.2.

    - **Coarse stage** (``global_step < coarse_steps``): lock on a single
      fixed frame (``coarse_frame_index``); ``shuffle=False``. This warms up
      the static Gaussians on one viewpoint before time deformation kicks
      in.
    - **Fine stage** (``global_step >= coarse_steps``): cycle through frames
      with ``shuffle=True``. Saturates as fine indefinitely past
      ``coarse_steps + fine_steps`` — the ``fine_steps`` value is the
      caller's training-budget hint and isn't enforced here so that the
      scheduler stays purely a stateless mapping.

    Ported from the G-SHARP v0.2 reference implementation.

    Args:
        coarse_steps: Number of coarse-stage iterations (must be ``>= 0``).
        fine_steps: Informational fine-stage budget (must be ``>= 0``;
            the scheduler does not gate the fine stage on this).
        coarse_frame_index: Frame to lock on during coarse (default ``0``,
            matching G-SHARP's ``self.trainset[0]``).

    Raises:
        ValueError: at construction if any step count is negative.
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
        """Resolve which stage / frame the given global step falls in.

        Args:
            global_step: Non-negative iteration counter (0-indexed).
            num_frames: Total number of training frames available
                (must be ``> 0``).

        Returns:
            :class:`TwoStageScheduleStep` describing the stage, frame index,
            and shuffle flag for this step.

        Raises:
            ValueError: if *global_step* is negative, *num_frames* is not
                positive, or ``coarse_frame_index`` falls outside
                ``[0, num_frames)`` (validated lazily at call time since
                ``num_frames`` isn't known at construction).
        """
        if global_step < 0:
            raise ValueError(f"global_step must be non-negative, got {global_step}")
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if not 0 <= self.coarse_frame_index < num_frames:
            raise ValueError(
                f"coarse_frame_index={self.coarse_frame_index} out of "
                f"[0, num_frames={num_frames})."
            )

        if global_step < self.coarse_steps:
            return TwoStageScheduleStep(
                stage="coarse",
                frame_index=self.coarse_frame_index,
                shuffle=False,
            )

        fine_offset = global_step - self.coarse_steps
        return TwoStageScheduleStep(
            stage="fine",
            frame_index=fine_offset % num_frames,
            shuffle=True,
        )
