---
id: MR-021
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: tests/test_dynamic_surgical_trainer.py:127-128
status: open
labels: [tests, trainer, consistency]
---

# MR-021 — Test still asserts the `hexplane_params` / `deform_mlp_params` placeholders that were removed

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> `test_build_splats_from_parser_produces_expected_shapes` at line 127-128 still asserts the `hexplane_params` / `deform_mlp_params` placeholders, but `build_splats_from_parser` was intentionally trimmed to the five per-Gaussian keys (see commit `5f04a631`).
>
> Either drop the assertions or restore the placeholders. Also the docstring at `examples/dynamic_surgical_trainer.py:197-201` still mentions placeholders; sync that.

## Summary
After we removed the `hexplane_params` / `deform_mlp_params` placeholders from `build_splats_from_parser` (commit `5f04a631`, MR-013/0014 follow-up), the test in `tests/test_dynamic_surgical_trainer.py` and the docstring in `examples/dynamic_surgical_trainer.py:197-201` were left stale.

## Scope / affected files
- `tests/test_dynamic_surgical_trainer.py` — lines 127-128 of `test_build_splats_from_parser_produces_expected_shapes`.
- `examples/dynamic_surgical_trainer.py` — `build_splats_from_parser` docstring at lines 197-201.

## Action plan
- [ ] Update the test to assert only the five per-Gaussian keys (`means`, `quats`, `scales`, `opacities`, `colors`) and that `hexplane_params` / `deform_mlp_params` are **not** in `params`.
- [ ] Rewrite the docstring at `examples/dynamic_surgical_trainer.py:197-201` to mention only the five per-Gaussian keys and explain why HexPlane / DeformNet trainables sit in separate optimizers (already documented in `DynamicStrategy.__doc__`; cross-link).
- [ ] Re-run `pytest tests/test_dynamic_surgical_trainer.py -v` to confirm.

## Resolution
<filled after fix lands>

## User-side reply draft
> Good catch — the test + docstring weren't updated when I trimmed `build_splats_from_parser` to five per-Gaussian keys in `5f04a631`. Fixed both; `pytest tests/test_dynamic_surgical_trainer.py -v` passes.
