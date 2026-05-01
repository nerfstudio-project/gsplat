# 0003 — TDD step 1: depth/mask losses implemented in `gsplat.losses`

Date: 2026-04-30
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

First TDD chunk from `0001_scaffolding_commit.md` §"What unblocks next" item 1: the four G-SHARP v0.2 depth / mask loss functions added as stubs in commit `3c0dd4e` are now implemented in `gsplat/losses.py`, with concrete pytest coverage in `tests/test_losses_depth.py` (the previous `pytest.mark.skip` placeholders are gone).

### `gsplat/losses.py` — implementations

- `binocular_disparity_l1(pred_depth, gt_depth, mask=None, eps=1e-7)`
  - L1 in inverse-depth (disparity) space.
  - Pixels with `|depth| <= eps` are treated as "no depth" (matches G-SHARP's `rendered_depth != 0` sentinel) — they contribute 0 to the inverse-depth tensor on both sides.
  - Uses `torch.where` over an eps-floored "safe" depth so the reciprocal stays finite without polluting the autograd graph at zero-depth pixels.
  - Without a mask: `(pred_inv - gt_inv).abs().mean()`. With a mask: delegates to `masked_l1`.

- `pearson_depth_loss(pred_depth, gt_depth, mask=None)`
  - `1 - Pearson r` on flattened (optionally masked) depth pairs.
  - Manually computed in pure PyTorch (no `torchmetrics` dependency added); denom clamped to `1e-12` so a constant input yields a finite loss rather than NaN.
  - Returns a differentiable 0 when fewer than 2 valid samples remain after masking.

- `masked_l1(pred, gt, mask)`
  - L1 over only the `mask != 0` region. The mask is broadcast to the loss-tensor shape via `expand_as`, so a `[B, 1, H, W]` mask works on `[B, C, H, W]` inputs.
  - Returns a differentiable 0 when the mask is all-zero.
  - Matches the G-SHARP `loss[mask != 0].mean()` semantic in `holohub/applications/surgical_scene_recon/training/utils/loss_utils.py:l1_loss`.

- `masked_ssim(pred, gt, mask)`
  - SSIM loss on the zeroed pair `(pred * mask, gt * mask)` — matches the G-SHARP training-loop convention (`gsplat_train.py` lines 1063–1101: mask both GT and rendered output, then take an unmasked SSIM).
  - Delegates the actual SSIM math to the existing `gsplat.losses.ssim_loss`, which already handles the `fused_ssim` GPU fast path.

### `tests/test_losses_depth.py` — concrete coverage

18 tests, all passing on CPU (autouse `seed=42` from `tests/conftest.py`):

| Function | Positive | Negative |
|---|---|---|
| `binocular_disparity_l1` | identical-depths-zero, gradient-flows, mask-slices-correct | shape-mismatch-raises, all-masked-returns-zero-no-NaN, zero-depth-handled-via-eps |
| `pearson_depth_loss` | correlated-zero, anti-correlated-two | shape-mismatch-raises, constant-input-no-NaN, fully-masked-returns-zero-no-NaN |
| `masked_l1` | ignores-masked-pixels, gradient-flows | shape-mismatch-raises, empty-mask-returns-zero-no-NaN |
| `masked_ssim` | identical-inputs-loss-near-zero, gradient-flows | shape-mismatch-raises |

## Verification

- `python -m pytest tests/test_losses_depth.py -x` → **18 passed** in 95s (first run, including CUDA JIT build cache fill).
- `python -m pytest tests/test_losses.py` → **209 passed** — no regression in the existing losses tests.
- The sandbox required `git submodule update --init --recursive gsplat/cuda/csrc/third_party/glm` once to populate the GLM headers before the CUDA JIT build could succeed. This is environment setup, not part of the code change.

## What unblocks next

Next TDD chunk from `0001_scaffolding_commit.md` step 2:

- `gsplat/regularizers.py`:
  - `compute_tv_loss_targeted` — occlusion-targeted TV regularizer.
  - `dilate_mask` — max-pool-based mask dilation (replaces the OpenCV call in G-SHARP).
  - `create_invisible_mask_from_paths` — intersection of inverse masks across training frames.
- Un-skip `tests/test_regularizers_occlusion.py`.
