---
id: MR-007
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-04-30
mr: <TBD>
thread: planning/gsharp_gsplat_plan.html
status: addressed
labels: [planning, html, should-fix]
---

# MR-007 — Stale module names in shareable HTML plan

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-04-30 [should-fix]:**

> Stale module names in shareable HTML plan
>
> The HTML plan still references the old, now-deleted paths from before the MR-002/MR-003/MR-005 refactor:
>
> - line 164: `losses.py + losses_depth.py` (no longer two files)
> - line 218: `dynamic_trainer.py (NEW)` (renamed to `dynamic_surgical_trainer.py`)
> - lines 267, 268, 269: `gsplat/losses_depth.py` referenced in the component-map table
> - line 295: `examples/dynamic_trainer.py` in the deliverables table
> - line 403: `examples/dynamic_trainer.py` in the prose summary
>
> You flagged this in `planning/dev_updates/0002_post_review_changes.md` ("Stale / to-flag") but did not regenerate before pushing. Suggestion: either regenerate from the updated `gsharp_gsplat_plan.md`, or drop the HTML from the diff if it's no longer being shared externally. Shipping a known-stale "shareable" artifact in the repo is worse than not shipping one.
>
> Label: Verified by reading.

## Summary
The shareable HTML plan still has the pre-refactor module names (`losses_depth.py`, `dynamic_trainer.py`) at six locations. We acknowledged this as stale in the previous dev update but didn't act. Vincent gives two options: regenerate from MD, or drop. Tied to MR-009 (whether to keep generated HTML in-repo at all).

## Scope / affected files
If we **drop**: remove `planning/gsharp_gsplat_plan.html` from both planning copies, plus the ref to it in `planning/gsharp_gsplat_plan.md` line 124 and `gsplat/planning/.../0001_scaffolding_commit.md` (or just leave 0001 as historical).

If we **regenerate**: hand-edit the 6 stale strings in both HTML copies (line 164, 218, 267, 268, 269, 295, 403). Mechanical but easy to drift again.

## Action plan
- [x] Decision (@vnath, 2026-04-30): drop the in-repo HTML; keep the workspace copy as a local-only preview (Option A).
- [x] Workspace `planning/gsharp_gsplat_plan.html` refreshed — `losses_depth.py` → `losses.py`, `dynamic_trainer.py` → `dynamic_surgical_trainer.py`, plus the path-prefix fix from MR-008 in the footer (line 413).
- [x] In-repo `gsplat/planning/gsharp_gsplat_plan.html` deleted via `git rm`.
- [x] Reference to the HTML removed from `gsharp_gsplat_plan.md` (both copies) — line now reads "kept locally; not tracked".
- [x] `.gitignore` updated — see MR-009.

## Resolution
HTML dropped from the gsplat repo (commit pending). The workspace mirror at `planning/gsharp_gsplat_plan.html` (outside the gsplat repo) is updated to current state and survives as a local preview. Folded MR-009 into the same change.
