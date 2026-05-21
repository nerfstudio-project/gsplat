---
id: MR-030
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: gsplat/contrib/dynamic/__init__.py:8,13-21 (`__all__` exports)
status: open
labels: [contrib, dynamic, api-surface]
---

# MR-030 — `gsplat/contrib/dynamic/__init__.py.__all__` pins API surface we don't want pinned

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> Two API exports in `gsplat/contrib/dynamic/__init__.py` lock surface area we don't yet want pinned:
>
> 1. **`DeformationTable`** is internal bookkeeping. No public op accepts one. Drop it from `__all__`. (Tied to MR-022's design decision: if we keep the table, it stays internal; if we delete it, the entry disappears anyway.)
> 2. **`plane_smoothness` / `time_smoothness` / `time_l1`** depend on the spatial / temporal plane partition `(0,1,3)` / `(2,4,5)` which is hardcoded in `HexPlaneField._init_grid_param`. Exposing the regularizers as top-level fns ties future contributors to that partition. Either hide them behind a `hexplane_regularization(field, weights)` convenience that does the partitioning internally, or lift the partition onto `HexPlaneField` as `spatial_planes()` / `temporal_planes()` accessor methods.

## Summary
The `__all__` of `gsplat/contrib/dynamic` exports internals that shouldn't be pinned: `DeformationTable` (bookkeeping) and three regularizers that hardcode a plane-partition convention. Either hide them, or refactor so the partition lives on `HexPlaneField`.

## Scope / affected files
- `gsplat/contrib/dynamic/__init__.py` — `__all__` at lines 8, 13-21.
- `gsplat/contrib/dynamic/hexplane.py` — `_init_grid_param` (plane partition); possibly add `spatial_planes()` / `temporal_planes()` accessors.
- `gsplat/contrib/dynamic/regulation.py` — possibly add a `hexplane_regularization(field, weights)` convenience.
- Callers: `examples/dynamic_surgical_trainer.py` train_step (regularizer calls).

## Action plan
- [ ] Drop `DeformationTable` from `__all__` (regardless of MR-022 outcome; if MR-022 picks Option B, the class is deleted entirely).
- [ ] Refactor: pick **A — accessor on HexPlaneField** (lightest touch) or **B — convenience wrapper `hexplane_regularization`** (cleaner API). Recommendation: **A**: add `HexPlaneField.spatial_planes()` and `HexPlaneField.temporal_planes()` returning lists of plane tensors; have `regulation.py` consume those accessors; drop the hardcoded indices from public docs.
- [ ] Update `train_step` to call via the new accessor pattern.
- [ ] Re-run relevant tests.

## Resolution
<filled after fix lands>

## User-side reply draft
> Dropped `DeformationTable` from `__all__` and lifted the spatial/temporal plane partition onto `HexPlaneField.spatial_planes()` / `temporal_planes()` accessors. The regularizers now consume those accessors instead of indexing planes by hand, so future HexPlane variants can override the partition.
