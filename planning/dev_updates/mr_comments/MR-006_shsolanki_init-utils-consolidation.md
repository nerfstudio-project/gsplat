---
id: MR-006
author: Shikhar Solanki (request); Andre Maximo (pushback)
date: 2026-04-25
mr: <TBD>
thread: gsplat/init_utils.py
status: addressed
labels: [architecture, init]
---

# MR-006 — Move init_utils functions into the trainer file?

## Original comment
> **Shikhar Solanki (2026-04-25, edited, on `gsplat/init_utils.py`):**
>
> I think I would add these functions to the trainer file itself and get rid of the extra file
>
> **Andre Maximo (2026-04-30) [open — counter-proposal awaiting Shikhar]:**
>
> Pushing back here — `multi_frame_depth_unprojection` and `knn_scale_init` are general-purpose initializers that other trainers will want to reuse (any deformable-or-not pipeline that has masked depth + poses can call them). Burying them inside one trainer means the second trainer that needs them either re-implements or imports across `examples/` (which gsplat doesn't otherwise do). `gsplat/init_utils.py` next to other module-level utilities is the right home. Happy to revisit if there's a concrete reason the trainer-local placement scales better.

## Summary
Shikhar proposes folding the multi-frame depth init helpers into the trainer file. Andre disagrees: they're general-purpose initializers reusable across trainers, and gsplat doesn't otherwise import across `examples/`. Andre explicitly left this thread open for Shikhar's response.

## Scope / affected files
If the consolidation is adopted (decision flips):
- Delete `gsplat/init_utils.py`
- Inline functions into the trainer (path TBD per MR-005)
- Update `tests/test_init_multiframe.py` import targets
- Update `planning/gsharp_gsplat_plan.md` component map (both mirrored copies)

If kept as-is (Andre's position): no code change.

## Action plan
- [x] Decision (@vnath, 2026-04-30): keep `gsplat/init_utils.py` as a top-level module — siding with @amaximo. Reuse by other trainers (any pipeline with masked depth + poses, deformable or static) is realistic, not just hypothetical.
- [ ] Post reply on the GitLab thread stating the position so @shsolanki can ack.
- [ ] Await @shsolanki's response to fully resolve.

## Resolution
Sided with @amaximo: keep `gsplat/init_utils.py` separate. @vnath confirmed (as MR author) that other trainers are likely to reuse `multi_frame_depth_unprojection` and `knn_scale_init`, so a top-level module is the right home rather than burying them in a single trainer. Reply queued; awaiting @shsolanki's ack.
