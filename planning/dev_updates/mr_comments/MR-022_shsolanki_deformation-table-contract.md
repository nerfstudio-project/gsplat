---
id: MR-022
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: examples/dynamic_surgical_trainer.py:323-326; gsplat/contrib/dynamic/{deformation,strategy}.py
status: open
labels: [contrib, dynamic, design, blocking, back-links-MR-013]
related: MR-013
---

# MR-022 — DeformationTable contract is not honoured end-to-end (answers MR-013)

This thread is shsolanki's empirical answer to the question @vcauxbrisebo asked in **MR-013** (deformation integration approach). Filed as its own ID per vnath direction (keeps MR-013 as the original question, MR-022 owns the empirical analysis and design decision).

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20 — empirical follow-up (21h after initial post):**

> The DeformationTable contract is not honoured end-to-end:
>
> 1. **Routing never consults the table.** `train_step` in the trainer (lines 323-326) feeds *all* Gaussians through `DeformNetwork` regardless of `state["deformation_table"].mask`. So every Gaussian gets routed through the deform path; the static/dynamic split exists only in bookkeeping.
> 2. **Mask init is all-False.** `DynamicStrategy.initialize_state` (strategy.py:171-173) starts the table at all-False, and nothing flips them before the first forward pass. So even if routing did consult the table, no Gaussian would ever take the dynamic path.
> 3. **`_resize_table` loses identity across `_refine_split`.** After 700 steps the mask shows 273 True / 50260 total. The 273 True flags are *new* Gaussians appended by densification (per the "new Gaussians flagged dynamic" policy in strategy.py:63-68 and the resize impl at 155-176). The original 50000 stay False forever.
>
> Two coherent design paths:
>
> - **A — Make the table drive routing.** Gather dynamic indices, route only those through `DeformNetwork`, scatter back. Rewrite `_resize_table` to consume the actual prune / split masks emitted by `gsplat/strategy/ops.py` so identity is preserved.
> - **B — Drop the table from the public API.** Document that every Gaussian is dynamic. Remove `state["deformation_table"]`, `DeformationTable`, and `_resize_table`.
>
> The current half-state is misleading and will trip future contributors.

## Summary
The `DeformationTable` is dead bookkeeping. Routing ignores it; init never flips any flag to True; resize doesn't preserve identity across split. Need to either (A) wire the table through, or (B) delete it from the public API. Per @vnath's direction at MR-013, the trainer wires the deformation provisionally — this is the response to that.

## Scope / affected files
- `examples/dynamic_surgical_trainer.py:323-326` (train_step routing)
- `gsplat/contrib/dynamic/strategy.py:63-68, 155-176, 171-173` (resize policy, mask init)
- `gsplat/contrib/dynamic/deformation.py:171` (DeformationTable API)
- `gsplat/contrib/dynamic/__init__.py:8` (the `__all__` export — overlaps with MR-030)
- Tests: `tests/test_contrib_dynamic_strategy.py`

## Decision required from user (vnath)
- **Option A — wire the table through routing**: matches the spirit of G-SHARP v0.2's static/dynamic split. Needs (a) some Gaussians flipped to dynamic at init (heuristic? all-True for now?), (b) gather/scatter in `train_step`, (c) `_resize_table` that consumes the actual prune/split masks emitted by `gsplat/strategy/ops.py`. More work, preserves intended semantics.
- **Option B — delete the table from the public API**: trainer's existing "route everything through DeformNet" behaviour stays, just stop pretending we have a static/dynamic split. Less code, honest about what the port actually does. Could revisit if a future contributor wants the speedup.

**Recommendation:** Option B for this MR (smallest scope, matches what the port actually does today). Reopen as a follow-up issue if A is needed for a perf push later.

**Decision (vnath, 2026-05-21):** **Option A** — wire the table through routing. Will re-run training after the change to confirm PSNR doesn't regress.

## Action plan (Option B; revise if vnath picks A)
- [ ] Get vnath's decision (A vs B).
- [ ] If B:
  - [ ] Remove `state["deformation_table"]` setup from `DynamicStrategy.initialize_state`.
  - [ ] Remove `_resize_table` from `DynamicStrategy`.
  - [ ] Delete `DeformationTable` class and its tests.
  - [ ] Remove `DeformationTable` from `gsplat/contrib/dynamic/__init__.py:__all__` (overlap with MR-030).
  - [ ] Add a note to `gsplat/contrib/dynamic/strategy.py.__doc__` explaining the explicit decision: "every Gaussian is treated as dynamic; static/dynamic routing was a G-SHARP optimisation we deliberately deferred."
  - [ ] Update `0014_long_run_validation_and_render.md` to reflect the removal.
  - [ ] Re-run `pytest tests/test_contrib_dynamic_strategy.py tests/test_dynamic_surgical_trainer.py -v` to confirm.
- [ ] Mark MR-013 `resolved` with a back-link to this entry's resolution.

## Resolution
<filled after vnath decides and fix lands>

## User-side reply draft
- **If B**: "Agreed — the table was dead bookkeeping. Removed `DeformationTable`, `_resize_table`, and the `state["deformation_table"]` entry; documented in the `DynamicStrategy` docstring that every Gaussian goes through DeformNet. Closes the MR-013 thread too — the routing question is moot now that we're explicit about it."
- **If A**: "Wired the table through `train_step` and rewrote `_resize_table` to consume `gsplat/strategy/ops.py`'s prune/split masks. Static/dynamic split now drives routing; new Gaussians from densification inherit the split decision of their parent."
