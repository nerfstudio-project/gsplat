---
id: MR-029
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: 13 new source files (see Scope below)
status: open
labels: [licensing, headers, repo-hygiene]
---

# MR-029 — Missing SPDX copyright headers on 13 new source files

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> The 13 new source files added by this MR are missing the standard SPDX header block. Per repo convention (see `gsplat/rendering.py`), each file should open with:
>
> ```
> # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
> # SPDX-License-Identifier: Apache-2.0
> ```
>
> Ported files (HexPlane, DeformNet) should additionally include upstream attribution (Nerfstudio for HexPlane; G-SHARP for the deformation net).

## Summary
13 new source files don't have the SPDX copyright block this repo uses. Need to add the standard NVIDIA + Apache-2.0 block to each, plus upstream attribution comments on the ported files (HexPlane → Nerfstudio; DeformNet → G-SHARP).

## Scope / affected files
- `gsplat/contrib/__init__.py`
- `gsplat/contrib/dynamic/__init__.py`
- `gsplat/contrib/dynamic/deformation.py` *(add G-SHARP upstream attribution)*
- `gsplat/contrib/dynamic/hexplane.py` *(add Nerfstudio upstream attribution)*
- `gsplat/contrib/dynamic/regulation.py`
- `gsplat/contrib/dynamic/strategy.py`
- `gsplat/init_utils.py`
- `gsplat/regularizers.py`
- `gsplat/training/__init__.py`
- `gsplat/training/schedulers.py`
- `examples/datasets/endonerf.py`
- `examples/dynamic_surgical_trainer.py`
- (13th file — likely `gsplat/losses.py` if newly added; verify in current tree.)

## Action plan
- [ ] Read `gsplat/rendering.py` to confirm the exact SPDX block format.
- [ ] Add the block to each of the 13 files. Keep it as the *first* lines, above any existing `"""docstring"""`.
- [ ] On `hexplane.py` and `deformation.py`, add a short attribution comment block under the SPDX block referencing the upstream repos (Nerfstudio K-Planes/HexPlane; G-SHARP v0.2 endo_loader / gsplat_train).
- [ ] Confirm `ruff`/`black` doesn't strip the header.

## Resolution
<filled after fix lands>

## User-side reply draft
> Added the standard SPDX header block to all 13 new files (matching `gsplat/rendering.py`) and added upstream attribution comments to `hexplane.py` (Nerfstudio K-Planes) and `deformation.py` (G-SHARP v0.2). Verified `ruff format` doesn't strip the headers.
