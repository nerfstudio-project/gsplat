# 0005 — TDD step 2: regularizers implemented in `gsplat.regularizers`

Date: 2026-05-07
Author: vnath (with assistance from Claude Code)
Branch: `vnath_gsharp`

## What landed

Second TDD chunk from `0001_scaffolding_commit.md` §"What unblocks next" item 2: the three G-SHARP v0.2 regularizer functions are now implemented in `gsplat/regularizers.py`, with concrete pytest coverage in `tests/test_regularizers_occlusion.py` (the previous `pytest.mark.skip` placeholders are gone).

### `gsplat/regularizers.py` — implementations

- `compute_tv_loss_targeted(image, mask=None)`
  - Anisotropic TV: ``|image[1:] − image[:-1]|`` along H and W.
  - Without a mask: ``(tv_h.sum() + tv_w.sum()) / image.numel()`` — matches the unmasked branch in G-SHARP `gsplat_train.py:2022–2026`.
  - With a mask: each anisotropic difference is weighted by the mask cropped to the diff shape (`mask[:, :, 1:, :]` / `mask[:, :, :, 1:]`), summed, then divided by ``mask.sum() * C + 1e-8`` per axis — matches G-SHARP lines 2008–2021 exactly.
  - Validates inputs: image must be 4D, and mask values must be in ``{0, 1}`` (rejects fuzzy/antialiased masks with a clear error).
  - Signature changed from the scaffold (mask was required → now optional, default `None`) to mirror the G-SHARP source.

- `dilate_mask(mask, kernel_size=3)`
  - Pure-torch replacement for `cv2.dilate` (G-SHARP `gsplat_train.py:1972–1989` used OpenCV; we keep the core library OpenCV-free).
  - Implementation: `F.max_pool2d` with `padding=kernel_size//2`, `stride=1`. Identity for `kernel_size=1`; one-pixel-margin for `kernel_size=3`; etc.
  - Accepts 2D `(H, W)`, 3D `(C, H, W)`, or 4D `(B, C, H, W)` inputs and returns the same rank.
  - Rejects even or non-positive kernel sizes with a clear error.

- `create_invisible_mask(masks)`
  - Union of per-frame tool / dynamic-object masks ("regions chronically occluded by instruments"), per the G-SHARP semantic at `gsplat_train.py:1925–1957`.
  - Two input modes per element of *masks*:
    - `Tensor` — used as-is, assumed in tool=1 convention.
    - `str` (path) — loaded via PIL, normalized to `[0, 1]`, and inverted (`1 − mask`) per G-SHARP's tissue=1-on-disk convention.
  - Combines via element-wise `amax` then `clamp(0, 1)` (cleaner-binary equivalent of G-SHARP's `np.clip(sum, 0, 1)`).
  - Validates: non-empty input, consistent shape, valid element type.

### `tests/test_regularizers_occlusion.py` — concrete coverage

14 tests, all passing on CPU under the autouse `seed=42` fixture from `tests/conftest.py`:

| Function | Positive | Negative |
|---|---|---|
| `compute_tv_loss_targeted` | constant-image-zero, step-edge-analytic-1/3, gradient-flows | empty-mask-zero-no-div-by-zero, non-binary-mask-rejected, non-4D-image-raises |
| `dilate_mask` | kernel-1-identity, kernel-3-grows-by-one-pixel, preserves-input-rank-{2D,3D,4D} | invalid-kernel-size-raises (even and zero) |
| `create_invisible_mask` | union-of-three-binary-tensors | empty-iterable-raises, shape-mismatch-raises, invalid-element-type-raises |

## Verification

- `python -m pytest tests/test_regularizers_occlusion.py -x` → **14 passed** in 3.18s.
- The previous CUDA JIT build is cached from commit `a380d40`, so the tests no longer pay the multi-minute first-run cost.

## Tracker status changes (`mr_tracker.md`)

No changes — this work doesn't touch any open reviewer thread; it's the next step in the TDD plan.

## What unblocks next

Next TDD chunk from `0001_scaffolding_commit.md` step 3:

- `gsplat/init_utils.py`:
  - `multi_frame_depth_unprojection` — accumulate point clouds across frames by unprojecting masked depth maps using camera poses and intrinsics.
  - `knn_scale_init` — kNN-based per-Gaussian initial scale (sklearn optional, pure-torch fallback per the proposal's "open risks" section).
- Un-skip `tests/test_init_multiframe.py`.
