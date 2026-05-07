---
id: MR-012
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-07
mr: <TBD>
thread: gsplat/losses.py (masked_ssim function)
status: addressed
labels: [losses, docs, tests, nit]
---

# MR-012 — masked_ssim convention dilutes the loss; lock the contract

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-07 [nit]:**

> The implementation multiplies both inputs by the mask and then takes an unmasked SSIM. In masked-out regions both `pred*mask` and `gt*mask` are exactly zero, so SSIM there is ~1 and contributes 0 to the (1 - SSIM) loss. The mean over the full image is therefore biased toward 0 in proportion to the masked-out fraction — a sparse mask makes any per-pixel loss in the active region look smaller than the masked-region-only mean would.
>
> This matches the G-SHARP v0.2 convention as the docstring says, so it isn't necessarily wrong — just worth being explicit about. Two small things would lock the contract in:
>
> 1. Add one sentence to the docstring noting the bias (e.g. "the mean is taken over the full image, so loss magnitude scales with mask coverage; this is intentional and matches the upstream convention").
> 2. Add a test with a partial mask (e.g. half the spatial area zeroed, identical inputs in the active half, mismatched in the masked-out half) and assert the loss is ~0 — that pins the "masked region is excluded" claim and would catch any future drift to the alternative "crop-then-SSIM" convention.
>
> Currently the only `masked_ssim` tests use an all-ones mask (identical-inputs and gradient-flow) plus a shape-mismatch case; there's no positive coverage of the actual masking semantic.

## Summary
`masked_ssim` works correctly per the G-SHARP convention but the bias-with-mask-coverage isn't documented and isn't test-pinned. Vincent asks for two small additions: docstring note + a partial-mask positive test. Code stays unchanged.

## Scope / affected files
- `gsplat/losses.py` — `masked_ssim` docstring (one new sentence).
- `tests/test_losses_depth.py` — add `test_masked_ssim_partial_mask_loss_near_zero`.

## Action plan
- [x] Add docstring sentence to `masked_ssim` calling out that the mask-coverage bias is intentional.
- [x] Add a partial-mask positive test: identical inputs in the masked-in half, mismatched in the masked-out half → loss ≈ 0.
- [x] Verify the new test passes alongside the existing 18.
- [ ] Confirm thread closure once committed.

## Resolution
Docstring + test landed (see `0006_round3_doc_fixes.md`). All 19 `tests/test_losses_depth.py` tests pass; new partial-mask test asserts `loss < 1e-3` matching the existing identical-inputs threshold.
