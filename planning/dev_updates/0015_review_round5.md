# 0015 — Review round 5: address MR-014..MR-031 (CodeRabbit + @vcauxbrisebo + @shsolanki)

Date: 2026-05-21
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Eighteen reviewer threads from review rounds 4 and 5 (CodeRabbit pass on commits `7ed611e6` and `58bf8d9d`, then a substantive round from @vcauxbrisebo and @shsolanki). All threads have detail files under `planning/dev_updates/mr_comments/MR-014..MR-031`. This dev update summarises what changed in code, grouped by commit.

### Correctness bugs (`99f17a1`)

- **MR-015** — `gsplat/contrib/dynamic/regulation.py` initialised the accumulator on CPU (`torch.zeros((), dtype=torch.float32)`), which crashed when planes lived on CUDA. Switched to lazy-init from the first plane's `device`/`dtype`. Added CUDA-gated regression tests in `tests/test_contrib_regulation.py`.
- **MR-017** — `gsplat/contrib/dynamic/hexplane.py:_grid_sample_wrapper`'s bare `.squeeze()` collapsed every size-1 dim; pinned to `.squeeze(0)`.
- **MR-021** — `tests/test_dynamic_surgical_trainer.py::test_build_splats_from_parser_produces_expected_shapes` still asserted `hexplane_params` / `deform_mlp_params` placeholders that were removed in `5f04a63` (MR-022 follow-up). Synced the test + the `build_splats_from_parser` docstring.
- **MR-023** — `binocular_disparity_l1` zeroed `pred_inv` and `gt_inv` independently, leaking `|0 - 1/other|` into the loss when only one side was invalid. Switched to a `pair_valid = valid_pred & valid_gt` mask; added `test_binocular_disparity_l1_one_side_invalid_excludes_pair`.
- **MR-027** — `examples/datasets/endonerf.py` now applies the LLFF → OpenGL axis permutation `c2w[:, :, [1, 0, 2, 3]] * [1, -1, 1, 1]` (per G-SHARP `endo_loader.py`). Added `test_endonerf_parser_applies_llff_axis_permutation` and updated `test_endonerf_parser_loads_fixture` to expect the new convention.

### Perf nits (`e0f764d`)

- **MR-016** — `compute_tv_loss_targeted`'s binary-mask validity check is now gated behind `ENFORCE_CONTRACTS` (matches `gsplat/losses.py`). Removes a per-step host sync from the hot training loop.
- **MR-024** — `knn_scale_init` now scans query rows in 1024-block chunks; peak memory drops from `O(N²)` (~10 GB at N=50k) to `O(chunk * N)` (~200 MB). Default `chunk_size=1024` exposed as a kwarg.

### MR-026 — HexPlane AABB auto-derive (`9df3a8f` + `a9e269b`)

The hardcoded `Config.hex_bounds = 1.6` was much smaller than the init means' bounding box (post-permutation EndoNeRF "pulling" sits around `[-180, 180]`). With `grid_sample(padding_mode="border")` every plane query clamped to the same edge feature and the deformation field collapsed to a spatial constant — dead HexPlane.

Fix:

- `examples/dynamic_surgical_trainer.py:train()` now derives `derived_bounds = max(cfg.hex_bounds, init_means.abs().max() * 2.0)` and passes it to `build_deform_modules(cfg, device, bounds_override=...)`. 2× margin absorbs gradient drift.
- Follow-up: a small set of sensor-noise outlier depths (one pixel at depth ~4132 on EndoNeRF "pulling") inflated `init_means.abs().max()` to absurd scales. Clipping `depths` to `parser.bounds.max() * 1.5` before unprojection drops `init_means.abs().max()` from 4132 → 180.

### MR-022 — Wire DeformationTable through routing (Option A) (`ba6f989`)

Per @vnath's decision (2026-05-21): closed MR-013 + MR-022 by wiring the dynamic mask through routing for real.

- `state["dynamic_mask"]` is now a plain `torch.bool` tensor instead of a `DeformationTable` wrapper. gsplat's strategy ops (`gsplat/strategy/ops.py:135, 191-195, 223-225`) already iterate every tensor in `state` and apply the per-Gaussian split / duplicate / prune permutation — so resize is automatic and **identity-preserving across split**, which the old `_resize_table` was not.
- `train_step` and `render_frame` gather Gaussians where `dynamic_mask` is True, run HexPlane + DeformNet only on those, scatter results back into full-size tensors via index-copy on a `.clone()` (autograd flows through both the dynamic and static branches).
- Default init flag is all-True, matching the previous "every Gaussian is dynamic" behaviour — the wiring is now honest about it.
- `DeformationTable` wrapper class stays importable for back-compat; canonical storage moves into `state`. Removed from `gsplat/contrib/dynamic/__all__` (overlaps with MR-030).

### API + docs hygiene (`889210a`)

- **MR-029** — Standard NVIDIA SPDX block added to 13 new source files (matches `gsplat/rendering.py`). `hexplane.py` and `deformation.py` carry an extra upstream-attribution block (Nerfstudio K-Planes; G-SHARP v0.2).
- **MR-030** — Dropped `DeformationTable` from `__all__`; added `HexPlaneField.spatial_planes()` and `temporal_planes()` accessors so the regularizers don't hardcode `(0,1,3)` / `(2,4,5)`. Added a `hexplane_regularization(field, **lambdas)` convenience. `train_step` now consumes the accessor pattern.
- **MR-014** — ASCII-ified en-dashes / multiplication signs in `gsplat/regularizers.py` docstrings; reworded a few clarity nits in `0006_round3_doc_fixes.md`; updated `MR-012` detail file front matter (`mr: !169`) + ticked the closure checkbox.
- **MR-025** — `docs/source/examples/dynamic_surgical.rst` quickstart now uses flags that actually exist on `Config` (`--data_dir`, `--coarse_steps`, `--fine_steps`, `--render_gif_after_train`); SCARED mention qualified "not yet implemented"; added a "Getting the data" subsection with the med-air/EndoNeRF Google Drive URL + expected layout.
- **MR-028** — Added `Pillow`, `tqdm`, `tyro`, `imageio>=2.37.2` to `setup.py:extras_require['examples']`. Install with `pip install -e .[examples]` for the trainer.
- **MR-031** — `examples/datasets/download_dataset.py` now recognises `endonerf` and prints documented manual-download instructions (Google Drive folder isn't scriptable without `gdown` + license click-through).

### CI code-format (`ee7302f`)

- **MR-020** — Ran `bash lint/format-code.sh --full` (project's pinned `black==22.3.0`). 14 files reformatted, mostly multi-line call-site reflows in the new modules. Stripped trailing whitespace at `docs/source/index.rst:118` that `git diff --check` was flagging.

## Decisions captured

- **MR-018** — Option A. Drop the in-repo `planning/dev_updates/mr_comments/` and `mr_tracker.md` immediately before the final push. Workspace mirror at `/home/vnath/Code/.../planning/dev_updates/` keeps the audit trail.
- **MR-022** — Option A. Wire the DeformationTable through routing (not delete). Done in `ba6f989`.

## Pending owner actions (not in this dev update)

- **MR-019** — Rebase onto `nv/main` to drop AI co-author / `Made-with` trailers from 14 commits, then force-push. Needs explicit @vnath go-ahead per CLAUDE.md before any destructive git op.
- **MR-018 execution** — Right before the final push, delete the in-repo `planning/dev_updates/mr_comments/` and `mr_tracker.md`. Leaving the deletion for the very last commit so the in-repo audit trail is intact for review.

## Verification

- `bash lint/format-code.sh --check --full` → green, 113 files unchanged.
- `pytest tests/test_contrib_regulation.py tests/test_contrib_dynamic_strategy.py tests/test_losses_depth.py tests/test_init_multiframe.py tests/test_regularizers_occlusion.py tests/test_dynamic_surgical_trainer.py tests/test_dataset_endonerf.py` → **85 passed, 2 skipped**.
- End-to-end EndoNeRF "pulling" run (200 coarse + 2500 fine, 50k init points): **best_train_psnr 39.52 dB**, held-out (masked) **27.85 dB**, full-sequence mean (masked) **28.09 dB** — comparable to / slightly better than the pre-round-5 baseline (27.07 / 26.96).

## Tracker status changes (`mr_tracker.md`)

After this commit:

- MR-013: `deferred` → `resolved` (answered by MR-022 wiring; cross-linked).
- MR-014 / MR-015 / MR-016 / MR-017 / MR-020 / MR-021 / MR-022 / MR-023 / MR-024 / MR-025 / MR-026 / MR-027 / MR-028 / MR-029 / MR-030 / MR-031: `open` → `addressed` (awaiting reviewer ack).
- MR-018: `open` → `addressed` (decision recorded; execution scheduled for the final push).
- MR-019: stays `open` (pending @vnath rebase + force-push).
