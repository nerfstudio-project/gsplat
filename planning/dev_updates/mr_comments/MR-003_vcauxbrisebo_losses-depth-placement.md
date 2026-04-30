---
id: MR-003
author: Vincent Caux-Brisebois (asked); Andre Maximo (decided)
date: 2026-04-25
mr: <TBD>
thread: gsplat/losses_depth.py (Andre's reply note id 51825241)
status: addressed
labels: [losses, regularizers, architecture]
related: [MR-002]
---

# MR-003 — Right home for losses_depth.py and regularizers.py

## Original comment
> **Vincent Caux-Brisebois (2026-04-25, on `gsplat/losses_depth.py`):**
>
> @amaximo can you comment on the right place for this? Should it move to `losses.py`. Same with regularizers — not sure if we currently distinguish between the two (we probably should).
>
> **Andre Maximo (2026-04-30) [self-resolve]:**
>
> Vincent — for `losses_depth.py`: I'd merge into `gsplat/losses.py` from the start. The existing precedent (`losses.py` next to `losses_fused.py`) supports splitting losses by implementation strategy (Python vs fused CUDA) rather than by use case (depth vs photometric). The proposal already says "to be merged into `gsplat/losses.py`" eventually — doing it now avoids a future API-surface rename for any external user who'd otherwise import from `gsplat.losses_depth import ...` and have to patch later. Four extra functions in `losses.py` is fine.
>
> For `regularizers.py`: keep it as a new top-level module. gsplat doesn't have an existing home for regularizers (no `regularizers.py`, no `regs/`), so creating one is the right move. Distinguishing losses-vs-regularizers is genuinely useful: losses operate on `(pred, gt)` pairs, regularizers operate on one tensor with no ground truth. That semantic split warrants two modules.
>
> Net: drop `gsplat/losses_depth.py` and put its four functions inside `gsplat/losses.py`; keep `gsplat/regularizers.py` as drafted.

## Summary
Two decisions:
1. **`losses_depth.py`** — drop; fold into `gsplat/losses.py`. gsplat splits files by implementation strategy (e.g. `losses_fused.py`), not by use case.
2. **`regularizers.py`** — keep as a new top-level module. gsplat has no existing home for regularizers; the losses-vs-regularizers distinction (pred-vs-gt vs single-tensor, no GT) is semantically meaningful.

## Scope / affected files
- See MR-002 action plan for the `losses_depth.py` drop
- `gsplat/regularizers.py` — no change required, decision is to keep as drafted

## Action plan
- [x] Same actions as MR-002 (linked) — landed 2026-04-30.
- [x] No action on regularizers — kept as drafted.

## Resolution
Code refactor landed 2026-04-30 alongside MR-002 (see `dev_updates/0002_post_review_changes.md`). `gsplat/regularizers.py` kept as drafted per @amaximo's split rationale (losses on `(pred, gt)` pairs; regularizers on a single tensor with no ground truth).
