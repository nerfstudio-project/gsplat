# 0012 — TDD step 8: EndoNeRF dataset loader implemented

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Eighth TDD chunk from `0001_scaffolding_commit.md`. `EndoNeRFParser` and `EndoNeRFDataset` are now implemented in `examples/datasets/endonerf.py`, with concrete pytest coverage in `tests/test_dataset_endonerf.py` (the previous `pytest.mark.skip` placeholders are gone).

This is the last piece before the end-to-end trainer (step 9) and lives in `examples/` per the architecture split — domain-specific dataset glue, not part of the core library API.

### `examples/datasets/endonerf.py` — implementations

**`EndoNeRFParser(data_dir, dataset_type="endonerf", test_every=8)`** (dataclass)

- Validates the directory layout up front: `poses_bounds.npy` at root, plus `images/`, `depth/`, `masks/` subdirs with matching frame counts.
- Parses `poses_bounds.npy`'s `(N, 17)` layout: 15 floats per row for the `(3, 5)` pose matrix (`[R | t | (H, W, focal)]`), 2 floats for near/far bounds. Promotes the `(3, 4)` pose to a homogeneous `(4, 4)` camera-to-world.
- Builds `K` from `(H, W, focal)`; sets `times = arange(N) / N`.
- Derives `train_idxs` / `test_idxs` / `video_idxs` using the same `(i - 1) % test_every == 0` convention as G-SHARP `EndoNeRF_Dataset.__init__`.
- `dataset_type="scared"` is recognised but routes to a clear `NotImplementedError` (SCARED has a different on-disk layout — JSON calibrations under `data/frame_data/`; see `endo_loader.py:310 class SCARED_Dataset` for the reference).
- Sanity-checks the first mask is binary on disk (unique values ⊂ `{0, 255}`); raises with a clear error message otherwise. Cheap parse-time check that keeps the tool-mask convention honest.

**`EndoNeRFDataset(parser, split="train")`** (`torch.utils.data.Dataset`)

- Random-access over `parser.{train,test,video}_idxs` based on *split*.
- `__getitem__` loads `(image, depth, mask)` from PNGs and returns the dict:
  - `image`: `(H, W, 3)` float32 in `[0, 1]`.
  - `depth`: `(H, W)` float32 metric depth (zero = no measurement).
  - `mask`: `(H, W)` float32 tool mask — applies the G-SHARP `1 - mask/255` inversion at `endo_loader.py:179` so the output is **tool=1, tissue=0**.
  - `camtoworld` / `K` / `time` from the parser.

### `tests/test_dataset_endonerf.py` — concrete coverage

13 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`. Uses pytest's `tmp_path` to materialise a tiny synthetic EndoNeRF directory (4 frames × 8 × 8 PNGs + `poses_bounds.npy`) — the same fixture is reused across most tests.

| Group | Coverage |
|---|---|
| Parser happy path | `loads_fixture` (K, camtoworlds, times, file counts), `split_indices_obey_test_every` (n=8, every=2 → train={0,2,4,6}, test={1,3,5,7}) |
| Parser errors | `scared_routing_raises`, `missing_data_dir_raises`, `missing_poses_bounds_raises`, `non_binary_mask_rejected` (overwrites first mask with value 127), `unknown_dataset_type_raises`, `count_mismatch_raises` (deletes one mask) |
| Dataset basics | `default_split_is_train`, `unknown_split_raises`, `getitem_returns_expected_keys` (validates all dict keys + shapes) |
| Dataset convention | `mask_is_tool_convention` (on-disk 0 → tool=1; on-disk 255 → tool=0), `image_in_unit_range` |

The test file prepends the gsplat repo root to `sys.path` so `from examples.datasets.endonerf import ...` resolves (examples isn't installed). Matches the pattern used by other example-side tests if they exist; if not, this is a one-off but harmless.

## Verification

- `python -m pytest tests/test_dataset_endonerf.py -v` → **13 passed** in 3.18s.
- No CUDA dependency; pure CPU test path against synthetic PNG fixtures.

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread. MR-013 stays `deferred` (still awaiting @shsolanki on deformation integration; orthogonal to dataset loading).

## What unblocks next

Final TDD chunk in `0001_scaffolding_commit.md`:

- **Step 9** — `examples/dynamic_surgical_trainer.py`: end-to-end recipe wiring `EndoNeRFDataset` (this commit) → `multi_frame_depth_unprojection` for SfM-free init → `HexPlaneField` + `DeformNetwork` for per-step deformation → `gsplat.rasterization` → depth + mask losses (`binocular_disparity_l1`, `pearson_depth_loss`, `masked_l1`, `masked_ssim`) → regularizers (`compute_tv_loss_targeted`, `plane_smoothness`, `time_smoothness`, `time_l1`) → backward → `DynamicStrategy.step_post_backward`. `TwoStageScheduler` drives the coarse → fine schedule across both stages.

This is the integration point that touches all of TDD steps 1–8 together and activates the deferred `test_dynamic_strategy_one_step_train_no_nan`.
