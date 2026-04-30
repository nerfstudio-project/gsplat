# 0002 — Post-review changes (MR-001, MR-002, MR-003, MR-005)

Date: 2026-04-30
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Applied the code / plan / scaffold changes coming out of the first round of MR review. Per-thread tracking lives in `dev_updates/mr_tracker.md` and `dev_updates/mr_comments/`.

### MR-001 — wrong path prefix in `0000_init.md` (CodeRabbit + @amaximo)
- `planning/dev_updates/0000_init.md` line 14: dropped the `gsplat/` prefix from the proposal path.
- Mirrored in `gsplat/planning/dev_updates/0000_init.md`.

### MR-002 + MR-003 — drop `gsplat/losses_depth.py`, fold into `gsplat/losses.py` (CodeRabbit + @vcauxbrisebo + @amaximo)
- Deleted `gsplat/losses_depth.py`.
- Added the four stub signatures to `gsplat/losses.py`:
  - `binocular_disparity_l1`, `pearson_depth_loss` — appended to the existing "Depth losses" section, after `depth_l1_loss`.
  - `masked_l1`, `masked_ssim` — added under a new "Mask-aware photometric wrappers (G-SHARP v0.2)" section, before LiDAR losses.
- `gsplat/__init__.py` left unchanged — it never re-exported from `losses_depth`, and the new stubs follow the existing convention (e.g. `gsplat.regularizers`, `gsplat.init_utils`) of being imported via the submodule path rather than via top-level `gsplat`.
- `tests/test_losses_depth.py` kept (filename still describes what it tests); only its module-level docstring updated to point at `gsplat.losses` instead of `gsplat.losses_depth`.
- `gsplat/regularizers.py` kept as drafted, per @amaximo's split: losses operate on `(pred, gt)` pairs; regularizers on a single tensor with no ground truth.
- Plan + doc references updated:
  - `planning/gsharp_gsplat_plan.md` (workspace + in-repo): mermaid `losses` node simplified, component-map row for binocular L1 reworded.
  - `docs/source/proposals/gsharp_v0_2_port.rst`: 3 references switched from `gsplat/losses_depth.py` to `gsplat/losses.py`.
  - `docs/source/examples/dynamic_surgical.rst`: tutorial's scaffolding note updated.
- `python -m py_compile gsplat/losses.py` is clean.

### MR-004 — naming `contrib/` vs `experimental/` (Shikhar Solanki)
- No code change. Decision: keep `gsplat/contrib/` per ecosystem precedent (TF / scikit-learn / jax). Reply queued on the GitLab thread.

### MR-005 — rename `dynamic_trainer.py` → `dynamic_surgical_trainer.py` (Shikhar Solanki + @amaximo)
- `examples/dynamic_trainer.py` → `examples/dynamic_surgical_trainer.py`.
- Internal docstring updated to reference `gsplat.losses.binocular_disparity_l1` (post-MR-002/003) and the new file name in the `NotImplementedError` message.
- Plan + doc references updated:
  - `planning/gsharp_gsplat_plan.md` (workspace + in-repo): mermaid `dynTrain` node + Examples-table path.
  - `docs/source/proposals/gsharp_v0_2_port.rst`: 1 reference.
  - `docs/source/examples/dynamic_surgical.rst`: Quickstart bash example.

### MR-006 — `init_utils.py` consolidation (Shikhar Solanki, awaiting his ack)
- No code change. Decision: keep `gsplat/init_utils.py` as a top-level module — siding with @amaximo. Reply queued; thread still gating @shsolanki's response.

## Stale / to-flag

- **`planning/gsharp_gsplat_plan.html`** (both copies): not regenerated. The HTML still has `losses_depth.py` and `dynamic_trainer.py` strings inside its rendered tables and mermaid SVG. Regenerate from the markdown before the next external share, or delete if it's no longer being shared.
- **`gsplat/planning/gsharp_gsplat_plan.md`** line 123 still says `gsplat/docs/source/proposals/...` — same path-prefix bug as MR-001 (just in a different file CodeRabbit didn't flag). Not fixed here to keep MR-001's scope tight; flagged for the next pass.

## Tracker status changes (`mr_tracker.md`)

- MR-001: `open` → `addressed` (one-char fix landed in both `0000_init.md` copies).
- MR-002 / MR-003 / MR-005: were already `addressed` from the decision-recorded round; the underlying code refactor / rename now actually exists, so the label is no longer aspirational.
- MR-004 and MR-006: no code work; status unchanged at `addressed`. Both still need the reply text posted on GitLab to fully resolve.

## Verification

- `grep -rn "losses_depth"` across active code / docs / active plan markdown is clean (the only remaining hit is `test_losses_depth.py` — the test filename — which is intentionally kept).
- `grep -rn "dynamic_trainer"` across active code / docs / active plan markdown is clean.
- `python -m py_compile gsplat/losses.py` passes.

## What unblocks next

The TDD execution order from `0001_scaffolding_commit.md` still applies; only the destination paths change for the depth/mask losses (now `gsplat/losses.py`) and the trainer (now `examples/dynamic_surgical_trainer.py`). Next steps:

1. Post the queued GitLab replies for MR-004, MR-005, MR-006 (text in the chat history that drove this dev update).
2. Wait for @shsolanki's ack on MR-006.
3. Begin TDD on `binocular_disparity_l1` / `pearson_depth_loss` / `masked_l1` / `masked_ssim` — un-skip `tests/test_losses_depth.py`.
