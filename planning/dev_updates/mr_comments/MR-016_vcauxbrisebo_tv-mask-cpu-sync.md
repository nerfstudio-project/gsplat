---
id: MR-016
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-14
mr: !169
thread: gsplat/regularizers.py:62 (compute_tv_loss_targeted)
status: open
labels: [regularizers, perf, contracts]
---

# MR-016 — compute_tv_loss_targeted mask validation forces a CPU sync per call

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-14 [should-fix — Owner]:**

> `compute_tv_loss_targeted` mask validation forces a CPU sync per call.
>
> At line 62: `if not (((mask == 0) | (mask == 1)).all()):`
>
> The `.all()` call produces a 0-dim bool tensor on the same device as `mask`. The `not` then triggers `__bool__`, which forces a host sync to read the value off the GPU. On a per-step training loop (TV loss is called every iteration) this is a measurable stall.
>
> `gsplat/losses.py` already gates similar contract checks behind an `ENFORCE_CONTRACTS` env var / flag. Either gate this check the same way, or drop it.

## Summary
Per-call host sync from a binary-mask validity check; remove or gate behind `ENFORCE_CONTRACTS` (matching the convention already established in `gsplat/losses.py`).

## Scope / affected files
- `gsplat/regularizers.py` — `compute_tv_loss_targeted`, line 62.

## Action plan
- [ ] Mirror the `ENFORCE_CONTRACTS` gate used in `gsplat/losses.py` and apply it to the `mask == 0 | == 1` check in `compute_tv_loss_targeted`.
- [ ] If a similar check exists elsewhere in `regularizers.py`, gate it too.
- [ ] Quick smoke run to confirm a measurable wall-clock win at training time (optional but worth noting in the resolution).

## Resolution
<filled after fix lands>

## User-side reply draft
> Good catch — gated the binary-mask check behind `ENFORCE_CONTRACTS` to match `gsplat/losses.py`. No more per-step host sync in the hot path.
