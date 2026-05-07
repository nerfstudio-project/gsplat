# 0006 — Round-3 review feedback (MR-010, MR-011, MR-012; track MR-013)

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Three new should-fix / nit comments arrived after the round-2 push (`a380d40` + `dcb84ef`). All three addressed in this commit; one new question to @shsolanki tracked but not actionable from this side.

### MR-010 — CodeRabbit on `0004_post_review_round2.md` line 59

Paste-error fix: the dev update listed the upcoming function as `create_invisible_mask_from_paths` (the holohub source's name) doing an "intersection of inverse masks". The actual function in our scaffold + impl is `create_invisible_mask` (no suffix), and the operation is a union. One-line fix to both copies of `0004`.

### MR-011 — Vincent on `docs/source/proposals/gsharp_v0_2_port.rst` lines 132–133

Missed-mirror from MR-007 / MR-009: the proposal RST still cross-referenced `planning/gsharp_gsplat_plan.html`, which was dropped in `dcb84ef`. Updated lines 132–133 to drop the HTML clause and add a parenthetical noting the HTML can be rendered locally — same shape as the cleanup that landed in `gsharp_gsplat_plan.md` line 124.

### MR-012 — Vincent on `masked_ssim` (`gsplat/losses.py`)

Two-part nit, both addressed:

1. Added a sentence to `masked_ssim`'s docstring calling out that the mean is taken over the full image, so the loss magnitude scales with mask coverage (sparse masks dilute the reported loss). Bias is intentional and matches the upstream G-SHARP convention.
2. Added `test_masked_ssim_partial_mask_loss_near_zero` to `tests/test_losses_depth.py`: identical inputs in the masked-in (right) half, deliberately mismatched inputs in the masked-out (left) half, asserts `loss < 1e-3`. Pins the "masked-out region is excluded" semantic against any future drift to a crop-then-SSIM convention.

The implementation in `gsplat.losses.masked_ssim` is unchanged — only docstring + tests grew.

### MR-013 — Vincent on `gsplat/contrib/dynamic/deformation.py` (question to @shsolanki)

Tracked in `mr_tracker.md` with `status: deferred`. The thread is `@shsolanki could you advise on how to best get this integrated…` — directed at @shsolanki, no @vnath code change required. Will fold @shsolanki's eventual response into the deformation impl plan when it lands (TDD step 6).

## Tracker status changes (`mr_tracker.md`)

Status updates from the new GitLab state:

- MR-001 / MR-002 / MR-003 / MR-004 / MR-005: `addressed` → `resolved` (Andre [self-resolve]; fix landed in `3c0dd4e`; vnath confirmation replies on file).
- MR-006: stays `addressed` (vnath replied; thread still gating @shsolanki's ack).
- MR-007 / MR-008 / MR-009: `addressed` → `resolved` (Vincent resolved all three on 2026-05-07 after seeing `dcb84ef`).
- MR-010 / MR-011 / MR-012: NEW, `addressed` (fixes in this commit; awaiting reviewer ack).
- MR-013: NEW, `deferred` (directed at @shsolanki; no vnath action).

## Verification

- `python -m pytest tests/test_losses_depth.py -x` → **19 passed** in 3.39s (was 18 + 1 new partial-mask test).
- `grep -n "gsharp_gsplat_plan.html" docs/source/` → no matches (RST clean).
- `grep -n "create_invisible_mask_from_paths" planning/dev_updates/` → no matches in active text (only in MR-010 detail file as a verbatim quote, intentional).

## What unblocks next

Resumption of TDD step 3 (init_utils):

- `gsplat/init_utils.py`:
  - `multi_frame_depth_unprojection` — accumulate point clouds across frames by unprojecting masked depth maps using camera poses + intrinsics.
  - `knn_scale_init` — kNN-based per-Gaussian initial scale (sklearn optional, pure-torch fallback).
- Un-skip `tests/test_init_multiframe.py`.
