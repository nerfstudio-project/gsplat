# 0011 — TDD step 7: DynamicStrategy + HexPlane regularizers (closes contrib path)

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Seventh TDD chunk from `0001_scaffolding_commit.md`, plus a small follow-on that finishes the dangling regularizer stubs. After this commit, the `gsplat/contrib/dynamic/` namespace has no remaining `NotImplementedError` placeholders.

@vcauxbrisebo's MR-013 question to @shsolanki on the deformation integration approach is still open. Per @vnath's direction (2026-05-07): proceeding without waiting; refactor if @shsolanki's response changes the design.

### `gsplat/contrib/dynamic/strategy.py` — `DynamicStrategy`

Subclass of :class:`gsplat.strategy.DefaultStrategy`. The `@dataclass` decorator is intentionally not applied (per the `0001_scaffolding_commit.md` note about field-ordering errors); no new dataclass fields are introduced.

- `check_sanity(params, optimizers)` — extends the parent. Additionally requires `hexplane_params` and `deform_mlp_params` keys in *params* (each tied to an optimizer of the same name).
- `initialize_state(scene_scale=1.0, num_gaussians=0, device=None)` — extends the parent. Adds `state["deformation_table"]` as a freshly-constructed :class:`DeformationTable`. All flags start `False`; the trainer flips them on per its policy (e.g. via tool masks).
- `step_pre_backward(...)` — passthrough to the parent.
- `step_post_backward(...)` — wraps the parent: snapshots the Gaussian count before, calls super, then resizes `state["deformation_table"]` if the count changed. Raises `RuntimeError` if `deformation_table` is missing (i.e. caller forgot `initialize_state`).
- `_resize_table(table, n_before, n_after)` (static helper) — on grow, appends `n_new` `True` flags (matches G-SHARP `new_mask = ones(num_new, dtype=bool)` at `gsplat_train.py:1365`); on shrink, truncates. The truncate is approximate (doesn't preserve exact survivor identities) — same trade-off G-SHARP makes; documented limitation. Adequate for uniformly-dynamic tables.

The deformation pass itself (HexPlane → :class:`DeformNetwork` → `(means, quats, opacities)`) lives in the trainer (the next TDD step), not in this strategy class. The strategy owns the densification policy + table bookkeeping; it does not call the deform-net directly.

### `gsplat/contrib/dynamic/regulation.py` — three regularizers

The scaffold's three stubs are now implemented as faithful ports of `training/scene/regulation.py`:

- `plane_smoothness(planes)` and `time_smoothness(planes)` — share the same math (sum of mean-squared second-difference along the H axis). The distinction is which planes the caller passes:
  - For HexPlane outputs, combo indices `[0, 1, 3]` (planes `xy`, `xz`, `yz`) are smoothed *spatially* via `plane_smoothness`.
  - Combo indices `[2, 4, 5]` (planes `xt`, `yt`, `zt`) are smoothed *in time* via `time_smoothness` (these planes' H axis is the temporal axis per :func:`HexPlaneField._init_grid_param`'s `coo_comb[::-1]` reversed-order layout).
- `time_l1(planes)` — L1 deviation from `1.0`, encouraging spatio-temporal planes to stay at their identity init. Matches G-SHARP's `L1TimePlanes` regularizer.

A private `_second_difference_squared` helper backs the two smoothness functions (both delegate to it). Planes with H < 3 are silently skipped (can't take a second difference); validated 4D shape elsewhere.

### Tests

Two new test files, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| File | Coverage | Pass / Skip |
|---|---|---|
| `tests/test_contrib_dynamic_strategy.py` | check_sanity (good + missing keys), initialize_state (table + parent keys), `_resize_table` grow / shrink / no-op, `post_backward_without_init_raises` | 8 pass + 1 skip |
| `tests/test_contrib_regulation.py` | smoothness on constant / linear (zero), curved (positive), sums across planes, invalid-rank-raises, too-short-axis-skips; time_l1 at-one-zero, constant-offset, sums across planes, gradient-flows | 11 pass |

The `_one_step_train_no_nan` test in `test_contrib_dynamic_strategy.py` is intentionally `pytest.mark.skip`-ped: a real one-step pass requires rasterization (CUDA-only) and the deform-net wired up, which lands with the trainer in TDD step 9. The skip reason documents the deferral.

## Verification

- `python -m pytest tests/test_contrib_dynamic_strategy.py tests/test_contrib_regulation.py` → **19 passed, 1 skipped** in 4.58s.
- `python -c "from gsplat.contrib.dynamic import DynamicStrategy"` → clean import (no leftover `NotImplementedError` triggered at import time).

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread. MR-013 stays `deferred`.

## What unblocks next

End of the **contrib (experimental)** core implementation. Two TDD chunks remain in `0001_scaffolding_commit.md`:

- **Step 8** — `examples/datasets/endonerf.py`: `EndoNeRFParser` + `EndoNeRFDataset` for the EndoNeRF / SCARED on-disk layout. Un-skip `tests/test_dataset_endonerf.py`.
- **Step 9** — `examples/dynamic_surgical_trainer.py`: end-to-end recipe that wires together `EndoNeRFDataset`, `multi_frame_depth_unprojection` for SfM-free init, depth + mask losses, regularizers, `TwoStageScheduler`, `DynamicStrategy` + `HexPlaneField` + `DeformNetwork`, and `gsplat.rasterization`. This is also where the `_one_step_train_no_nan` skipped test gets activated.

After step 9 lands, the only `NotImplementedError` left in the tree should be in the trainer scaffolding (until concrete CLI args + dataset loading are wired up against a real EndoNeRF sample).
