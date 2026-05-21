---
id: MR-023
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: gsplat/losses.py:240-248 (binocular_disparity_l1)
status: open
labels: [losses, depth, semantics]
---

# MR-023 — `binocular_disparity_l1` validity handling contradicts its docstring

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> `binocular_disparity_l1` zeroes `pred_inv` and `gt_inv` independently (lines 241, 246). When one side is invalid but the other is not, the loss is `|0 - 1/other|`, not zero — which contradicts the docstring at lines 226-228.
>
> Either:
> 1. Replace with a `pair_valid = valid_pred & valid_gt` mask (both must be valid for the term to contribute), and zero / drop the rest. Matches the docstring as written.
> 2. Or change the docstring to spell out the asymmetric behaviour.
>
> Either way, pin the chosen semantic with a test in `tests/test_losses_depth.py`.

## Summary
Numerical/semantic bug: `binocular_disparity_l1` independently zeroes invalid predictions and ground-truth values. When one is invalid and the other is valid, the loss spikes to `|0 - 1/other|` instead of being skipped. The docstring claims invalid pixels are excluded; need to either fix the impl or fix the docstring (and pin with a test either way).

## Scope / affected files
- `gsplat/losses.py` — `binocular_disparity_l1`, lines 226-248.
- `tests/test_losses_depth.py` — add a one-side-invalid positive test.

## Action plan
- [ ] Pick semantic (recommend #1: `pair_valid = valid_pred & valid_gt`; matches existing docstring intent).
- [ ] Implement: compute `pair_valid`, mask both sides, average over valid pair count.
- [ ] Add `test_binocular_disparity_l1_one_side_invalid_yields_zero` (or `_excludes_pair` if a partial mask is being tested).
- [ ] Re-run `pytest tests/test_losses_depth.py -v` to confirm.

## Resolution
<filled after fix lands>

## User-side reply draft
> You're right — implementation was asymmetric with the docstring. Switched to a `pair_valid = valid_pred & valid_gt` mask (so invalid-on-either-side excludes the pair entirely) and pinned with a new test `test_binocular_disparity_l1_one_side_invalid_excludes_pair`. Closes the gap.
