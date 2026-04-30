---
id: MR-005
author: Shikhar Solanki (suggestion); Andre Maximo (agrees)
date: 2026-04-25
mr: <TBD>
thread: planning/gsharp_gsplat_plan.md (Examples / docs row)
status: addressed
labels: [naming, examples]
---

# MR-005 — dynamic_trainer.py rename

## Original comment
> **Shikhar Solanki (2026-04-25, edited):**
>
> instead of `dynamic_trainer`, maybe something more specific like `gsharp/endonerf_trainer.py`? cause we'll have dynamic objects for av scenes too
>
> **Andre Maximo (2026-04-30) [self-resolve]:**
>
> Agree on the spirit — "dynamic" is going to overload as soon as a non-medical dynamic-scene trainer lands. `examples/endonerf_trainer.py` (without the `gsharp/` prefix — gsharp is the origin, not the target domain) reads cleaner alongside `examples/simple_trainer.py`. Or `examples/dynamic_surgical_trainer.py` if Vishwesh wants to keep the "dynamic" hint. Either works; just pick one before downstream tutorials lock the import path.

## Summary
"dynamic_trainer" is too generic — will collide with future AV dynamic-scene trainers. Andre proposes `examples/endonerf_trainer.py` (preferred, matches `simple_trainer.py` style) or `examples/dynamic_surgical_trainer.py` (Vishwesh-friendly if he wants the "dynamic" hint). Decision needed before tutorials/imports lock.

## Scope / affected files
- `examples/dynamic_trainer.py` → rename
- `docs/source/examples/dynamic_surgical.rst` (path references)
- `docs/source/proposals/gsharp_v0_2_port.rst` (path references)
- `planning/gsharp_gsplat_plan.md` (component map + diagram, both mirrored copies)

## Action plan
- [x] Decision (@vnath, 2026-04-30): `examples/dynamic_surgical_trainer.py`.
- [x] Rename `examples/dynamic_trainer.py` → `examples/dynamic_surgical_trainer.py` (2026-04-30).
- [x] Update internal docstring + `NotImplementedError` message inside the renamed file (2026-04-30).
- [x] Update `docs/source/examples/dynamic_surgical.rst` references (2026-04-30).
- [x] Update `docs/source/proposals/gsharp_v0_2_port.rst` references (2026-04-30).
- [x] Update `planning/gsharp_gsplat_plan.md` (component map + diagram), both mirrored copies (2026-04-30).
- [ ] Post reply on the GitLab thread confirming the choice.

## Resolution
Rename + doc updates landed 2026-04-30 (see `dev_updates/0002_post_review_changes.md`). Reply still queued for the GitLab thread.
