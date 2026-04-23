# SPDX-License-Identifier: Apache-2.0
"""Scaffolding tests for gsplat.training.TwoStageScheduler (branch: vnath_gsharp)."""

import pytest


@pytest.mark.skip(reason="vnath_gsharp: TwoStageScheduler.step not yet implemented")
def test_two_stage_coarse_locks_on_coarse_frame_index():
    pass


@pytest.mark.skip(reason="vnath_gsharp: TwoStageScheduler.step not yet implemented")
def test_two_stage_fine_shuffles_after_boundary():
    pass


def test_two_stage_negative_step_count_raises():
    from gsplat.training import TwoStageScheduler

    with pytest.raises(ValueError):
        TwoStageScheduler(coarse_steps=-1, fine_steps=10)
    with pytest.raises(ValueError):
        TwoStageScheduler(coarse_steps=10, fine_steps=-1)
