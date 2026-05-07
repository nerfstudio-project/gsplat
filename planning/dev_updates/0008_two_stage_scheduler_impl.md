# 0008 — TDD step 4: TwoStageScheduler implemented

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Fourth TDD chunk from `0001_scaffolding_commit.md` §"What unblocks next" item 4: `TwoStageScheduler.step` is now implemented in `gsplat/training/schedulers.py`, with concrete pytest coverage in `tests/test_two_stage_scheduler.py` (the previous `pytest.mark.skip` placeholders are gone; the existing real assertion for negative-step-count construction validation stays).

### `gsplat/training/schedulers.py` — implementation

`TwoStageScheduler.step(global_step, num_frames) -> TwoStageScheduleStep`:

- **Coarse stage** (`global_step < coarse_steps`): returns `stage="coarse"`, `frame_index=coarse_frame_index`, `shuffle=False` — locks training on a single fixed frame, matching G-SHARP's `data = self.trainset[0]` override at `gsplat_train.py:1043–1049`.
- **Fine stage** (`global_step >= coarse_steps`): returns `stage="fine"`, `frame_index=(global_step - coarse_steps) % num_frames` (deterministic cycle), `shuffle=True` — mirrors G-SHARP's `shuffle=(stage == "fine")` flag for the DataLoader.
- Saturates as fine indefinitely past `coarse_steps + fine_steps` (the `fine_steps` value is the caller's training-budget hint; the scheduler stays a stateless mapping).
- Validates lazily: rejects negative `global_step`, non-positive `num_frames`, and `coarse_frame_index` outside `[0, num_frames)` at call time (since `num_frames` isn't known at construction).

The `TwoStageScheduleStep` dataclass docstring was expanded to clarify the `frame_index` cycling convention and the `shuffle` flag's intended use (forward to `DataLoader`; the caller may then ignore `frame_index`).

### `tests/test_two_stage_scheduler.py` — concrete coverage

11 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| Group | Positive | Negative |
|---|---|---|
| Construction | default-coarse-frame-zero | negative-step-count-raises |
| Coarse stage | locks-on-coarse-frame-index, uses-default-frame-zero | — |
| Fine stage | shuffles-after-boundary, frame-index-cycles, saturates-past-fine-steps, zero-coarse-steps-starts-in-fine | — |
| `step()` validation | — | negative-global-step-raises, zero-num-frames-raises, coarse-frame-index-out-of-range-raises |

The boundary test pins the coarse → fine transition at exactly `global_step == coarse_steps`. The cycle test asserts `[0, 1, 2, 3, 0, 1, 2, 3]` over 8 fine-stage steps with `num_frames=4`, locking the deterministic-cycle contract.

## Verification

- `python -m pytest tests/test_two_stage_scheduler.py -x` → **11 passed** in 2.16s.
- No CUDA dependency in this module; tests do not pay JIT-build cost.

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread; it's the next step in the TDD plan.

## What unblocks next

End of the **core (stable)** TDD work. Next chunk is the **contrib (experimental)** path from `0001_scaffolding_commit.md` step 5:

- `gsplat/contrib/dynamic/hexplane.py`:
  - `HexPlaneField` — 6-plane (3 spatial + 3 spatio-temporal) feature grid, multires, ported from G-SHARP `training/scene/hexplane.py`.
- Un-skip `tests/test_contrib_hexplane.py`.

Step 5 is the first piece of the 4D / deformable Gaussian machinery flagged for the experimental namespace, and the most isolated of the contrib modules — good place to land before the deformation/strategy pieces that depend on it.
