# 0007 — TDD step 3: init utilities implemented in `gsplat.init_utils`

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Third TDD chunk from `0001_scaffolding_commit.md` §"What unblocks next" item 3: the two G-SHARP v0.2 SfM-free initialization utilities are now implemented in `gsplat/init_utils.py`, with concrete pytest coverage in `tests/test_init_multiframe.py` (the previous `pytest.mark.skip` placeholders are gone).

### `gsplat/init_utils.py` — implementations

- `multi_frame_depth_unprojection(images, depths, masks, poses, intrinsics, max_points=None)`
  - For each frame: gates pixels on `mask != 0 AND depth > 0`, unprojects via per-frame pinhole intrinsics (`x = (u - cx) * d / fx`, `y = (v - cy) * d / fy`, `z = d`), then transforms camera-space points to world space using the per-frame camera-to-world pose.
  - Concatenates across frames; optional `max_points` random subsample cap.
  - `uint8` images are normalized to `[0, 1]`; float images pass through.
  - Empty result (`(0, 3)` xyz/rgb) when no pixel passes the mask + valid-depth gate.
  - Generalized version of G-SHARP's `accumulate_multiframe_pointcloud` (`gsplat_train.py:2036–2159`): G-SHARP's "first-seen" running-coverage accumulator is dataset-flow-specific and lives in the example, not the core lib. The core function takes per-frame inputs and returns the union; callers can filter / dedupe to taste.

- `knn_scale_init(xyz, k=3, eps=1e-7)`
  - Pure-torch: `cdist` + `topk(k+1, largest=False)`, drop self-column, RMS distance → clamp_min(eps) → `log`.
  - No sklearn dependency in the core library (per the proposal's "open risks" — sklearn stays optional).
  - Numerically matches the G-SHARP path: `sqrt((knn(points, k+1)[:, 1:] ** 2).mean(dim=-1)).clamp_min(eps).log()` from `gsplat_train.py:1832–1838`.
  - Returns `(N,)` per-point log-distance; caller broadcasts to `(N, 3)` log-scale via `result.unsqueeze(-1).repeat(1, 3)`.
  - Raises `ValueError` if `N <= k` (need at least `k+1` points so each has *k* neighbours after excluding self).

### `tests/test_init_multiframe.py` — concrete coverage

10 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| Function | Positive | Negative |
|---|---|---|
| `multi_frame_depth_unprojection` | recovers-synthetic-grid (analytic), rgb-matches-source-pixels, max-points-subsamples | frame-count-mismatch-raises, all-masked-returns-empty, zero-depth-pixels-excluded |
| `knn_scale_init` | matches-sklearn-reference (skipped if sklearn missing), uniform-grid-constant-scale (analytic) | too-few-points-raises, duplicate-points-no-inf |

The synthetic-grid test pins the unprojection math: with `fx = fy = 4`, `cx = cy = 2`, `depth = 1.0`, and identity pose, every pixel maps to a known camera-space (= world-space) coordinate. The `uniform-grid-constant-scale` test pins `knn_scale_init`'s output to `log(spacing)` for points on a regular 1D grid.

## Verification

- `python -m pytest tests/test_init_multiframe.py -x` → **10 passed** in 2.61s.
- The sklearn-reference test cleanly skips on machines without sklearn (uses `pytest.importorskip`).

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread; it's the next step in the TDD plan.

## What unblocks next

Next TDD chunk from `0001_scaffolding_commit.md` step 4:

- `gsplat/training/schedulers.py`:
  - `TwoStageScheduler.step` — coarse (frame 0 only) → fine (shuffled) two-stage training schedule, ported from `EndoRunner._train_stage` (`gsplat_train.py:1007–1050`).
- Un-skip `tests/test_two_stage_scheduler.py`.

After that, contrib-side: HexPlane, DeformNetwork, DynamicStrategy (steps 5–7).
