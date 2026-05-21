---
id: MR-027
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: examples/datasets/endonerf.py:163-168 (poses_bounds.npy load)
status: addressed (per reviewer; verify in our tree)
labels: [datasets, endonerf, llff, resolved-upstream]
---

# MR-027 — LLFF axis permutation in EndoNeRF parser

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20 — initial flag, then marked Resolved 17h later:**

> `endonerf.py:163-168` reads LLFF `poses_bounds.npy` matrices as-is and treats the result as camera-to-world, but LLFF stores them as `[down, right, backwards, t, (H, W, focal)]`.
>
> Port `poses[:, :, [1, 0, 2, 3]] * [1, -1, 1, 1]` from G-SHARP `endo_loader.py`.
>
> Add a projection unit test.
>
> [21h later] **Resolved.**

## Summary
LLFF stores poses with a `[down, right, backwards]` axis convention; treating the matrix as a camera-to-world transform without permutation gives wrong projections. Reviewer noted this is resolved upstream — we need to verify our tree has the fix.

## Scope / affected files
- `examples/datasets/endonerf.py` — `EndoNeRFParser` `_load_poses`, lines 163-168.
- `tests/test_dataset_endonerf.py` — add a projection unit test if not already present.

## Action plan
- [ ] **Verify**: re-read `examples/datasets/endonerf.py:163-168` in our current tree to confirm the axis permutation is applied (i.e. `poses[:, :, [1, 0, 2, 3]] * [1, -1, 1, 1]` or equivalent).
- [ ] If applied: mark this MR `resolved` and reply confirming verification.
- [ ] If not applied: implement the permutation and add the projection regression test.

## Resolution
<filled after verification>

## User-side reply draft
- **If our tree already has the fix**: "Confirmed — verified `endonerf.py:163-168` applies the LLFF axis permutation per G-SHARP `endo_loader.py`. Closing the thread."
- **If our tree is missing it**: "Good catch — fix wasn't in our tree yet. Applied the permutation and added `test_endonerf_camtoworld_axis_permutation`."
