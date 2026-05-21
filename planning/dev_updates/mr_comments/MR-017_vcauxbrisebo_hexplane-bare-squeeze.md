---
id: MR-017
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-14
mr: !169
thread: gsplat/contrib/dynamic/hexplane.py:65 (_grid_sample_wrapper)
status: open
labels: [contrib, dynamic, hexplane, nit, robustness]
---

# MR-017 — _grid_sample_wrapper bare `.squeeze()` is shape-fragile

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-14 [nit — Owner; feel free to defer]:**

> `_grid_sample_wrapper` bare `.squeeze()` is shape-fragile.
>
> The unparameterised `.squeeze()` collapses every size-1 dim, so if a caller ever evaluates a single point (`n==1`), or if `feature_dim==1`, the output rank silently changes and downstream code breaks in confusing ways.
>
> Suggest `.squeeze(0)` (or whichever specific dim is meant) to make the intent explicit.

## Summary
Replace `.squeeze()` with the explicit `.squeeze(0)` so single-point / single-feature edge cases don't silently change output rank.

## Scope / affected files
- `gsplat/contrib/dynamic/hexplane.py` — `_grid_sample_wrapper`, line 65.

## Action plan
- [ ] Audit the `_grid_sample_wrapper` return path; pin the correct dim and use `.squeeze(0)` (or equivalent).
- [ ] Add a small test that calls `HexPlaneField` with `n=1` to lock the contract.

## Resolution
<filled after fix lands>

## User-side reply draft
> Pinned the squeeze to the batch dim explicitly and added an `n=1` regression test. Thanks.
