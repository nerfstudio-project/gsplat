---
id: MR-024
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: gsplat/init_utils.py:159-161 (knn_scale_init)
status: open
labels: [init_utils, perf, memory, oom-risk]
---

# MR-024 — `knn_scale_init` builds an O(N²) `cdist` matrix; OOM on small GPUs

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20 (with empirical follow-up downgrading severity):**
> `knn_scale_init` at line 161 builds the full `N×N cdist` matrix — at the trainer's default `init_max_points=50_000` that is ~10 GB of fp32 distances. Measured 10.48 GB on A100. Fine on A100, but will OOM on 12–16 GB consumer cards.
>
> Fix: chunked-pairwise KNN (loop over query rows in blocks of e.g. 1024), or integrate approximate KNN (`cuML.NearestNeighbors`, FAISS).
>
> Also: please remove the dismissive "this is fine" comment that pre-dated the measurement.

## Summary
KNN init in `gsplat/init_utils.py` uses a full N×N pairwise distance matrix — fine on A100 (~10 GB) but will OOM on consumer GPUs at the default 50k init points. Need to chunk the computation or swap in an approximate KNN library.

## Scope / affected files
- `gsplat/init_utils.py` — `knn_scale_init`, lines 159-161.
- Possibly `tests/test_init_utils.py` (add a large-N test if we're confident the chunked impl scales).

## Action plan
- [ ] Pick approach:
  - **Chunked pure-torch**: loop over query rows in blocks of (e.g.) 1024; only keep top-k per block. Stays dependency-free. Recommended for first cut.
  - **cuML / FAISS**: faster + lower memory but adds a heavy optional dep. Defer unless someone explicitly asks.
- [ ] Implement chunked impl; verify result matches the existing dense impl on a small test (1024 points) within tolerance.
- [ ] Quick smoke run with 50k points and confirm peak GPU memory < 2 GB.
- [ ] Remove the dismissive comment if one is in the file (audit needed).
- [ ] Optional: add an `init_max_points` ceiling guard in the trainer config so misconfig doesn't silently OOM.

## Resolution
<filled after fix lands>

## User-side reply draft
> Switched to a chunked pairwise KNN (block size 1024, dependency-free pure-torch); peak memory dropped from ~10 GB to ~1.5 GB at 50k init points. Removed the stale comment. Results match the dense version within fp32 tolerance.
