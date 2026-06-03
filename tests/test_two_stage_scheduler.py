# SPDX-License-Identifier: Apache-2.0
"""Tests for ``gsplat.training.TwoStageScheduler``.

Covers the coarse → fine schedule ported from G-SHARP v0.2's
``EndoRunner._train_stage``.

Seed is set to 42 by the autouse fixture in ``conftest.py`` (not relevant
here since the scheduler is deterministic, but kept for consistency).
"""

import pytest

from gsplat.training import TwoStageScheduler
from gsplat.training.schedulers import TwoStageScheduleStep


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_two_stage_negative_step_count_raises():
    with pytest.raises(ValueError):
        TwoStageScheduler(coarse_steps=-1, fine_steps=10)
    with pytest.raises(ValueError):
        TwoStageScheduler(coarse_steps=10, fine_steps=-1)


def test_two_stage_default_coarse_frame_index_is_zero():
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=10)
    assert sched.coarse_frame_index == 0


# ---------------------------------------------------------------------------
# Coarse stage
# ---------------------------------------------------------------------------


def test_two_stage_coarse_locks_on_coarse_frame_index():
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=20, coarse_frame_index=2)
    for global_step in range(5):
        s = sched.step(global_step=global_step, num_frames=10)
        assert isinstance(s, TwoStageScheduleStep)
        assert s.stage == "coarse"
        assert s.frame_index == 2
        assert s.shuffle is False


def test_two_stage_coarse_uses_default_frame_zero():
    sched = TwoStageScheduler(coarse_steps=3, fine_steps=10)
    s = sched.step(global_step=0, num_frames=8)
    assert s.frame_index == 0


# ---------------------------------------------------------------------------
# Fine stage
# ---------------------------------------------------------------------------


def test_two_stage_fine_shuffles_after_boundary():
    """At global_step == coarse_steps the schedule transitions to fine."""
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=20)
    last_coarse = sched.step(global_step=4, num_frames=10)
    first_fine = sched.step(global_step=5, num_frames=10)

    assert last_coarse.stage == "coarse"
    assert last_coarse.shuffle is False
    assert first_fine.stage == "fine"
    assert first_fine.shuffle is True
    assert 0 <= first_fine.frame_index < 10


def test_two_stage_fine_frame_index_cycles_through_num_frames():
    """Fine stage cycles `(global_step - coarse_steps) % num_frames`."""
    sched = TwoStageScheduler(coarse_steps=2, fine_steps=20)
    num_frames = 4
    expected = [0, 1, 2, 3, 0, 1, 2, 3]
    actual = [
        sched.step(global_step=2 + i, num_frames=num_frames).frame_index
        for i in range(8)
    ]
    assert actual == expected


def test_two_stage_fine_saturates_past_fine_steps():
    """Past ``coarse_steps + fine_steps`` the schedule still returns fine
    (the budget value is informational; gating is the caller's job)."""
    sched = TwoStageScheduler(coarse_steps=2, fine_steps=3)
    # coarse: 0, 1; fine budget: 2, 3, 4; query 5 (past the budget)
    s = sched.step(global_step=5, num_frames=4)
    assert s.stage == "fine"
    assert s.shuffle is True


def test_two_stage_zero_coarse_steps_starts_in_fine():
    """A scheduler with coarse_steps=0 starts in fine immediately."""
    sched = TwoStageScheduler(coarse_steps=0, fine_steps=10)
    s = sched.step(global_step=0, num_frames=5)
    assert s.stage == "fine"
    assert s.shuffle is True


# ---------------------------------------------------------------------------
# step() argument validation
# ---------------------------------------------------------------------------


def test_two_stage_step_negative_global_step_raises():
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=10)
    with pytest.raises(ValueError, match="global_step"):
        sched.step(global_step=-1, num_frames=10)


def test_two_stage_step_zero_num_frames_raises():
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=10)
    with pytest.raises(ValueError, match="num_frames"):
        sched.step(global_step=0, num_frames=0)


def test_two_stage_step_coarse_frame_index_out_of_range_raises():
    sched = TwoStageScheduler(coarse_steps=5, fine_steps=10, coarse_frame_index=8)
    with pytest.raises(ValueError, match="coarse_frame_index"):
        sched.step(global_step=0, num_frames=4)
