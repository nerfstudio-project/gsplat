# 0014 — Long-run validation on EndoNeRF "pulling" + render helpers

Date: 2026-05-14
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

End-to-end smoke test on real EndoNeRF "pulling" data, plus the render plumbing needed to produce a preview artifact and PSNR numbers comparable to the holohub reference application.

### Code changes

- **`examples/dynamic_surgical_trainer.py`** — render plumbing:
  - `Config` gains `render_gif_after_train`, `gif_filename`, `gif_fps`, `gif_frame_indices`.
  - New `render_frame(parser, frame_idx, params, hexplane, deform_net, device)` returns `(H, W, 3) uint8` for a single frame at its native time stamp; `@torch.no_grad()`, runs the same HexPlane → DeformNet → rasterizer chain as the train loop.
  - New `render_gif(...)` loops over `parser.video_idxs` (or a caller-supplied frame list), uses `imageio.v2.mimsave` with `duration=max(1, int(1000/fps))`, `loop=0`.
  - `train(cfg)` now optionally calls `render_gif` at the end when `cfg.render_gif_after_train` is set.
  - `build_splats_from_parser` no longer adds `hexplane_params` / `deform_mlp_params` placeholders to the strategy `params` dict (see "Param-dict contract fix" below).

- **`gsplat/contrib/dynamic/strategy.py`** — `DynamicStrategy.check_sanity` reduces to `super().check_sanity(params, optimizers)`. The class docstring now explicitly explains why HexPlane / DeformNet trainables live outside *params* (gsplat densification ops blindly per-Gaussian-index every entry).

- **`tests/test_contrib_dynamic_strategy.py`** — `_make_params_optimizers` simplified (no `include_dynamic_keys` flag); `test_dynamic_strategy_check_sanity_fails_on_missing_keys` → `test_dynamic_strategy_check_sanity_fails_on_missing_per_gaussian_keys`. The deferred `test_dynamic_strategy_one_step_train_no_nan` stays as a `pytest.mark.skip` marker pointing at the active integration test.

### Param-dict contract fix (root cause of a CUDA assert at step ~600)

While running a longer training pass, `gsplat.strategy.ops._split` crashed at `gsplat/strategy/ops.py:179` (`p_split = p[sel].repeat(repeats)`) with a CUDA assertion. Trace under `CUDA_LAUNCH_BLOCKING=1` pointed at the `hexplane_params` / `deform_mlp_params` placeholder tensors I had added to the `ParameterDict` purely to satisfy the old `DynamicStrategy.check_sanity` check.

Root cause: every entry in a strategy `params` dict is treated as **per-Gaussian** by gsplat's densification ops. They index every tensor with per-Gaussian masks and call `repeat(...)` to grow them. A `(1,)`-shape placeholder violates the contract and crashes at the first `_split` op (typically a few hundred steps in).

Fix:
- Remove the requirement from `DynamicStrategy.check_sanity` (no extra keys needed; parent's check is enough).
- Stop adding placeholders in `build_splats_from_parser`.
- Document the constraint in the class docstring and in the trainer integration plan (`0013_trainer_integration.md`).

The HexPlane / DeformNet trainables continue to be wired through `build_deform_modules` with their own Adam optimizers, stepped alongside the per-Gaussian optimizers in `train_step`. They are not affected by densification.

## Verification

Two demo runs on EndoNeRF "pulling" (single GPU, torch 2.8.0+cu128, gsplat CUDA JIT cached):

| Run | Config | Wall-clock | Final n_GS | Notes |
|---|---|---:|---:|---|
| Short demo  | 200 coarse + 1000 fine, 10k init pts | 27 s  | 15,343 | First end-to-end smoke run that produced a viewable GIF |
| Long demo   | 200 coarse + 2500 fine, 50k init pts | 77 s  | 71,297 | Matches holohub init-density; PSNR comparable to reference |

PSNR formula identical to `holohub/applications/surgical_scene_recon/training/utils/image_utils.py:66` (masked-MSE / single-channel mask branch).

Long-demo PSNR breakdown:

| Metric | This port | Holohub reference |
|---|---:|---:|
| Best train-batch PSNR (fine stage, masked) — i.e. holohub `best_psnr` | **39.43 dB** | 37.98 dB (`PROJECT_RELEASE_EMAIL.md:54`) |
| Test-split mean PSNR (8 held-out frames, masked) | 27.07 dB | — (not reported by holohub) |
| Test-split mean PSNR (unmasked, torchmetrics-equivalent) | 16.85 dB | — (tool occlusion dominates without mask) |
| Full-sequence mean PSNR (63 frames, masked) | 26.96 dB | — |

The headline holohub number is `best_psnr` (line `gsplat_train.py:1289` — peak train-batch PSNR seen at any single fine-stage step). Our port matches and slightly exceeds it. The held-out test PSNR (27 dB) is the proper generalization metric — visually clean tissue surfaces; see the saved GIF artifact for confirmation.

## What unblocks next

Same follow-up list as `0013_trainer_integration.md` — no new blockers from this round. The port is feature-complete on `vnath_gsharp`.
