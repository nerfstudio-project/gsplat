# 0004 — Round-2 MR review feedback (MR-007, MR-008, MR-009)

Date: 2026-04-30
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Three new comments from @vcauxbrisebo. All addressed in this commit.

### MR-007 — Stale module names in shareable HTML plan

Decision (Option A): drop the in-repo HTML; refresh and keep the workspace copy as a local-only preview.

- Workspace `planning/gsharp_gsplat_plan.html` (untracked, outside the gsplat repo) was updated in-place: `losses_depth.py` → `losses.py`, `dynamic_trainer.py` → `dynamic_surgical_trainer.py`, comment `arrows: deps into dynamic_trainer` → `… dynamic_surgical_trainer`, the `losses.py + losses_depth.py` SVG text on line 164 → `losses.py (+ depth/mask)`, plus the path-prefix fix from MR-008 in the footer (line 413): `gsplat/docs/source/proposals/...` → `docs/source/proposals/...`.
- In-repo `planning/gsharp_gsplat_plan.html` deleted via `git rm`.
- Reference at line 124 of `gsharp_gsplat_plan.md` ("Shareable HTML at `planning/gsharp_gsplat_plan.html`") replaced with "Shareable HTML preview is now kept locally (regenerate from this markdown when needed); not tracked in the repo." in both planning copies.

### MR-008 — Lingering `gsplat/` path-prefix bug at multiple sites

Three unique path strings replaced via `replace_all` in both planning copies of `gsharp_gsplat_plan.md`:

- `gsplat/docs/source/examples/dynamic_surgical.rst` → `docs/source/examples/dynamic_surgical.rst`
- `gsplat/docs/source/apis/contrib.rst` → `docs/source/apis/contrib.rst`
- `gsplat/docs/source/proposals/gsharp_v0_2_port.rst` → `docs/source/proposals/gsharp_v0_2_port.rst`

Same paste-error class as MR-001 (which fixed only `0000_init.md` line 14). This round catches the four remaining in-tree occurrences (lines 95, 96, 97, 123 of `gsharp_gsplat_plan.md`) plus the same bug in the workspace HTML footer.

### MR-009 — Don't commit generated HTML

Folded into MR-007's Option A. `.gitignore` at the gsplat repo root now contains:

```
# Generated/shareable HTML preview of the planning markdown — kept locally only.
planning/*.html
```

Future renders of the HTML inside `gsplat/planning/` will be silently ignored, eliminating the drift hazard.

## Tracker status changes (`mr_tracker.md`)

- MR-007: `open` → `addressed`.
- MR-008: `open` → `addressed`.
- MR-009: `open` → `addressed`.

## Verification

- `grep -rn "gsplat/docs/source"` across `planning/` and `gsplat/planning/` → only intentional matches remain (verbatim review quotes inside MR-001 / MR-008 detail files, the `mr_tracker.md` description columns, and the historical 0002 dev update note).
- `grep -rn "losses_depth\|dynamic_trainer"` against the workspace `planning/gsharp_gsplat_plan.html` → clean except `test_losses_depth.py` (intentionally kept; that's still the actual test filename).
- `git status` shows: `.gitignore` modified, `planning/gsharp_gsplat_plan.html` deleted (rename detection should not apply since the workspace copy is outside the gsplat repo), `gsharp_gsplat_plan.md` modified, `mr_tracker.md` modified, three new MR detail files added, this dev update new.

## What unblocks next

Resumption of TDD step 2 (regularizers) — was the planned next chunk before this round of comments arrived:

- `gsplat/regularizers.py`:
  - `compute_tv_loss_targeted` — occlusion-targeted TV regularizer.
  - `dilate_mask` — torch max-pool replacement for OpenCV mask dilation.
  - `create_invisible_mask_from_paths` — intersection of inverse masks across training frames.
- Un-skip `tests/test_regularizers_occlusion.py`.
