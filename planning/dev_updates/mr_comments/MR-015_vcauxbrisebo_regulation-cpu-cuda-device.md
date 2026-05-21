---
id: MR-015
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-14
mr: !169
thread: gsplat/contrib/dynamic/regulation.py:38
status: open
labels: [contrib, dynamic, regulation, cuda, blocking]
---

# MR-015 — regulation.py initialises accumulator on CPU; crashes on CUDA planes

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-14 [blocking — Owner]:**

> regulation.py initialises `total` on CPU — crashes when planes are on CUDA.
>
> In `_second_difference_squared` and `time_l1`, `total: Tensor = torch.zeros((), dtype=torch.float32)` is created on the default device (CPU). When the HexPlane field lives on a CUDA device (which the trainer wires up), the subsequent `total = total + ...` triggers a cross-device op and PyTorch raises:
> ```
> RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
> ```
> `.to(dtype)` doesn't move device.
>
> Tests in `tests/test_contrib_regulation.py` only exercise CPU tensors, so this is invisible to CI but will fault on the first GPU training step.
>
> Fix: lazy-init inside the loop on `p.device` (`total = torch.zeros((), dtype=p.dtype, device=p.device)` on first iter, then accumulate), or compute on `p.device, p.dtype` from the start. Add a CUDA-gated regression test.

## Summary
Device-mismatch bug in two regulation helpers. CPU-only test suite hides it; first GPU step will crash. Need to init the accumulator on the plane's device + dtype, and add a CUDA-gated test so we don't regress.

## Scope / affected files
- `gsplat/contrib/dynamic/regulation.py` — `_second_difference_squared` (line 38), `time_l1` (line 102).
- `tests/test_contrib_regulation.py` — add a CUDA-gated test that runs the regulation fns on CUDA planes.

## Action plan
- [ ] Replace CPU-default `torch.zeros((), dtype=...)` with `torch.zeros((), dtype=p.dtype, device=p.device)` (or first-iteration lazy init).
- [ ] Verify the trainer's end-to-end CUDA path no longer faults on these losses.
- [ ] Add a CUDA-gated test in `tests/test_contrib_regulation.py` (skip if `not torch.cuda.is_available()`).
- [ ] Note: this didn't crash our recent long-run because the loss landed on CUDA via accumulator-on-rhs ops once a CUDA tensor was added — but order-of-ops dependence is fragile, fix it cleanly.

## Resolution
<filled after fix lands>

## User-side reply draft
> Confirmed — the CPU init was a latent bug, the long-run only worked because the first added tensor coerced device. Fixed by initialising on `p.device, p.dtype` and added a CUDA-gated regression test. Thanks for catching it.
