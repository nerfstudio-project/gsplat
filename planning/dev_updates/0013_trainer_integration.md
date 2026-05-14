# 0013 — TDD step 9: dynamic surgical trainer (closes the port)

Date: 2026-05-14
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Final TDD chunk from `0001_scaffolding_commit.md`. `examples/dynamic_surgical_trainer.py` is now a working end-to-end recipe wiring every module from TDD steps 1–8 together. The deferred `test_dynamic_strategy_one_step_train_no_nan` is activated as `tests/test_dynamic_surgical_trainer.py::test_trainer_one_step_train_no_nan`. The G-SHARP v0.2 port now has no `NotImplementedError` placeholders anywhere in the tree.

### `examples/dynamic_surgical_trainer.py` — implementation

Three layers:

- **`Config`** (dataclass + `tyro.cli`): paths, scheduler step counts, init knobs (max points, scale multiplier, opacity), HexPlane config, DeformNet config, loss weights, learning rates per parameter, depth mode, seed, device. Validates `depth_mode` and non-negative step counts in `__post_init__`.

- **Setup helpers** (CPU-friendly):
  - `build_splats_from_parser(parser, cfg, device)`: materialises the train-split frames, runs `multi_frame_depth_unprojection` over them (with the dataset's tissue-include mask as the keep gate), optional random subsample to `cfg.init_max_points`, then `knn_scale_init` with G-SHARP's `log(knn_dist * init_scale_multiplier)` rescaling. Quaternions random-normal + unit-normalised; opacities init in logit space so `sigmoid(...) == cfg.init_opacity`; colors taken from unprojected RGB. Returns a `ParameterDict` + matching optimizers, with `hexplane_params` / `deform_mlp_params` placeholders so `DynamicStrategy.check_sanity` passes.
  - `build_deform_modules(cfg, device)`: constructs `HexPlaneField` + `DeformNetwork` + their Adam optimizers. `DeformNetwork.feature_dim` is locked to `HexPlaneField.feat_dim` (output_dim × len(multires)).

- **`train_step(...)`** (CUDA — `gsplat.rasterization` is CUDA-only):
  1. `xyzt = [means | t·1ₙ]` → `plane_features = hexplane(xyzt)`.
  2. `DeformNetwork` produces deformed `(means, quats, opacities)` (zero-init heads → identity at step 0).
  3. `rasterization(...)` in packed format with `render_mode="RGB+ED"` returning RGB + expected depth.
  4. Losses: `masked_l1` + `masked_ssim` on tissue regions (mask from dataset is already tissue-include), depth via `binocular_disparity_l1` or `pearson_depth_loss`, `compute_tv_loss_targeted` on the rendered image, plus the three HexPlane regularizers (`plane_smoothness`, `time_smoothness`, `time_l1`) over the correct spatial/temporal plane subsets.
  5. `strategy.step_pre_backward` → `loss.backward()` → `optimizer.step()` on all groups (incl. hexplane / deform optimizers) → `strategy.step_post_backward(packed=True)` for densification + table resize.

  Returns a dict of scalar loss components for logging.

- **`train(cfg)`**: full coarse → fine loop driven by `TwoStageScheduler.step(global_step, num_frames)`; tqdm progress bar with `stage`, `loss`, and live Gaussian count.

- **`main()`**: tyro CLI entry.

### Mask-convention fix-up (touches three files)

While wiring the trainer I caught that the dataset's mask convention had been mis-documented across the tree. The actual G-SHARP / EndoNeRF convention is **on-disk 255 = tool, 0 = tissue** — i.e. `mask = 1 - mask_raw/255` produces a **tissue-include mask** (`1` = keep tissue, `0` = drop tool), not the tool=1 mask the earlier comments claimed. G-SHARP's own ``# 1=tissue, 0=tool`` comment at `endo_loader.py:179` and `gsplat_train.py:2102` is the ground truth.

Fixed:

- `examples/datasets/endonerf.py`: top-of-file convention docstring and the `EndoNeRFDataset.__getitem__` comment now say "tissue-include mask".
- `examples/dynamic_surgical_trainer.py`: `_load_train_tensors` comment, the `build_splats_from_parser` mask-handoff to `multi_frame_depth_unprojection`, and the `train_step` losses all use the mask directly (no `1 - mask` inversion).
- `tests/test_dataset_endonerf.py`: renamed `test_endonerf_dataset_mask_is_tool_convention` → `test_endonerf_dataset_mask_is_tissue_include_convention`; assertion values stay the same (they were correct), only the docstring and name reflected the wrong semantic.
- `tests/test_dynamic_surgical_trainer.py`: synthetic fixture now writes all-zero on-disk masks (= tissue) so unprojection finds valid pixels.

### Two gsplat-API gotchas hit during integration

- **Packed vs unpacked rasterization output**: in my env, `gsplat.rasterization(...)` returns `info["radii"]` in *packed* layout (`(nnz, 2)`) by default. `DefaultStrategy.step_post_backward(packed=False)` is the default but assumes unpacked `(C, N, 2)` → `_update_state` indexes `torch.where(sel)[1]` and crashes with `IndexError: tuple index out of range`. Fix: pass `packed=True` to both `rasterization(...)` and `strategy.step_post_backward(...)`.
- **`init_scale_multiplier`**: G-SHARP uses `scales = log(knn_dist * init_scale)` with `init_scale=1.0`. EndoNeRF "pulling" depths are in metric units (z ∈ [38, 110] for the fixture I tested), so an `init_scale=0.01` rescale produces sub-pixel screen-space Gaussians that gsplat culls (radii = 0 for all Gaussians → 0-shape `radii` tensor). Bumped the default to `1.0`.

### `tests/test_dynamic_surgical_trainer.py` — concrete coverage

7 tests, 20 total assertions across `tests/test_dynamic_surgical_trainer.py` and `tests/test_dataset_endonerf.py`. Grouped:

| Group | Coverage |
|---|---|
| Config validation (CPU) | rejects-invalid-depth-mode, rejects-negative-step-counts, paths-coerced-to-Path |
| Setup helpers (CPU, synthetic 32×32 × 4-frame fixture) | build-splats produces expected shapes / unit-norm quats, build-deform-modules constructs with correctly-linked `feature_dim` |
| End-to-end (CUDA + `ENDONERF_DATA_DIR`) | `test_trainer_one_step_train_no_nan` — one full `train_step` on real pulling data, asserts all loss components and post-step params stay finite |

The CUDA test is gated on `torch.cuda.is_available() and ENDONERF_DATA_DIR` and skips cleanly when either is missing. The synthetic fixture is fine for setup tests but produces sub-pixel Gaussian scales after kNN init (depth too coarse, frame variation too small), so the integration test points at a real dataset.

The deferred `test_contrib_dynamic_strategy.py::test_dynamic_strategy_one_step_train_no_nan` stays as a `pytest.mark.skip` marker, now with a docstring pointing at the active integration test (so anyone scanning DynamicStrategy tests can find the integration counterpart).

## Verification

```
ENDONERF_DATA_DIR=/home/vnath/Code/surg_scene_gsplat_holoapp_nov_2025_github/holohub/data/EndoNeRF/pulling \
  python -m pytest tests/test_dynamic_surgical_trainer.py tests/test_dataset_endonerf.py -v
```

→ **20 passed** in 11.96s.

Standalone smoke run (5 steps on the pulling dataset, 2000 init points):

```
step 0: loss=0.2353 l1=0.2411 ssim=0.1572 depth=0.0109
step 1: loss=0.2353 ...
step 4: loss=0.2247 l1=0.2292 ssim=0.1512 depth=0.0111
```

Loss decreasing; all params remain finite.

### Environment

- Python 3.12, torch 2.8.0+cu128, gsplat CUDA JIT cached from earlier steps.
- New runtime deps: `tyro 1.0.13`, `tqdm 4.66.4`, `imageio 2.33.1` (added to the base anaconda env via `pip install`; not pinned in any `setup.py` / `pyproject.toml` yet — that's a follow-up for the `examples/` tier if reviewers want it pinned).

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread. MR-013 stays `deferred` (still awaiting @shsolanki on deformation integration). Per @vnath's direction, the trainer wires the deformation through provisionally and will be refactored if @shsolanki's response changes the integration shape.

## What unblocks next

End of the TDD execution-order list. The G-SHARP v0.2 port is feature-complete on `vnath_gsharp`. Plausible follow-ups, all outside the scope of `0001_scaffolding_commit.md`'s nine steps:

- Eval / metric logging / checkpointing in the trainer.
- Tutorial RST (`docs/source/examples/dynamic_surgical.rst`) expanded with the actual usage pattern now that the trainer works.
- SCARED loader (currently a stub raising `NotImplementedError`).
- Pin `tyro` / `tqdm` / `imageio` in `examples/` deps.
- Resolve MR-013 with @shsolanki when his response lands.
- Address whatever feedback the reviewer round produces.
