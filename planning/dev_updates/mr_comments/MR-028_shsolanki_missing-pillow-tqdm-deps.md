---
id: MR-028
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: setup.py:81-89 (install_requires)
status: open
labels: [setup, deps, packaging]
---

# MR-028 — `setup.py install_requires` is missing `Pillow` and `tqdm`

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> `setup.py`'s `install_requires` does not declare `Pillow` or `tqdm`, but the new tests/modules import them at module top:
> - `tests/test_dataset_endonerf.py:17` — `from PIL import Image`
> - `examples/datasets/endonerf.py:52` — `from PIL import Image`
> - `examples/dynamic_surgical_trainer.py:41` — `import tqdm`
>
> Add both to `install_requires` (or to the `extras_require['examples']` group if they're examples-only).
>
> Also: confirm CI installs whatever group these end up in.

## Summary
Hard imports of `Pillow` and `tqdm` from new modules without corresponding entries in `setup.py install_requires`. Need to either add them to the core deps or to an `extras_require` group, and confirm CI picks them up.

## Scope / affected files
- `setup.py` — `install_requires` (or `extras_require`).
- `.github/workflows/core_tests.yml` (if `extras_require` is the route, confirm CI installs that extra).
- `pyproject.toml` (if it declares deps separately).

## Action plan
- [ ] Decide split:
  - **`tqdm`** is used in the trainer; if we treat the trainer as part of `examples/`, put it in `extras_require['examples']`.
  - **`Pillow`** is needed by `EndoNeRFDataset` and its test — also examples-tier. Same `extras_require['examples']` group.
  - If we want CI to keep passing without an extra, add both to `install_requires` directly.
- [ ] Recommend: add `Pillow` + `tqdm` to `extras_require['examples']`, update CI to install `pip install -e .[examples]` for the test job that imports them.
- [ ] Confirm `pyproject.toml` doesn't shadow the change.

## Resolution
<filled after fix lands>

## User-side reply draft
> Added `Pillow` and `tqdm` to `extras_require['examples']` and updated `.github/workflows/core_tests.yml` to install `pip install -e .[examples]` for the test job that exercises `examples/datasets/endonerf.py`. Verified CI passes.
