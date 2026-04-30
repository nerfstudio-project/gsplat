---
id: MR-002
author: CodeRabbit (asked); Andre Maximo (decided)
date: 2026-04-24
mr: <TBD>
thread: planning/gsharp_gsplat_plan.md (component map row for losses_depth.py)
status: addressed
labels: [planning, losses, architecture]
related: [MR-003]
---

# MR-002 — losses_depth.py merge timing into losses.py

## Original comment
> **CodeRabbit (2026-04-24):**
>
> ⚠️ Potential issue | 🟡 Minor — Clarify losses_depth.py file handling. The note says "to be merged into `gsplat/losses.py`" but the architecture diagram (line 27) and other references treat it as a separate file. Should this initially be a separate file and later merged, or should it go directly into `losses.py`? If it's meant to be merged eventually, consider adding a timeline or milestone for when that merge should happen.
>
> **Andre Maximo (2026-04-30) [self-resolve]:**
>
> Cross-referencing my answer in @vcauxbrisebo's thread on `gsplat/losses_depth.py` (note 51825241): merge into `losses.py` from the start, drop the separate file. That removes the future merge milestone CodeRabbit asks for — there's no future merge to plan if it's already in the right place.

## Summary
Decision: don't keep `gsplat/losses_depth.py` as a separate file. Put the four depth-loss functions directly in `gsplat/losses.py` from the start. Removes the need for a future merge milestone.

## Scope / affected files
- `planning/gsharp_gsplat_plan.md` — update component map and architecture diagram so both reference `gsplat/losses.py`
- `gsplat/losses_depth.py` — drop; move the (unimplemented) signatures into `gsplat/losses.py`
- `gsplat/__init__.py` — update if it re-exports from `losses_depth`
- `tests/test_losses_depth.py` — keep tests, optionally rename the file
- See MR-003 for the parallel decision on the same code path

## Action plan
- [x] Update plan markdown (component map + diagram) — both mirrored copies (2026-04-30).
- [x] Move stubs from `gsplat/losses_depth.py` into `gsplat/losses.py` (2026-04-30).
- [x] Delete `gsplat/losses_depth.py` (2026-04-30).
- [x] Verify no leftover imports — `grep -rn "losses_depth"` clean across active code/docs/plan; `py_compile gsplat/losses.py` passes.

## Resolution
Code refactor landed 2026-04-30 (see `dev_updates/0002_post_review_changes.md`). `losses_depth.py` deleted; four stubs moved into `gsplat/losses.py` (binocular + Pearson appended to the existing Depth-losses section; masked variants under a new "Mask-aware photometric wrappers" section). Plan markdown + RST docs updated to match. Tracked together with MR-003.
