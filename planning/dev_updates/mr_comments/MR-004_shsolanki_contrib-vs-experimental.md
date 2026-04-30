---
id: MR-004
author: Shikhar Solanki (suggestion); Andre Maximo (neutral)
date: 2026-04-25
mr: <TBD>
thread: planning/gsharp_gsplat_plan.md (Goals section)
status: addressed
labels: [naming, architecture, contrib]
---

# MR-004 — Naming: gsplat/contrib/ vs gsplat/experimental/

## Original comment
> **Shikhar Solanki (2026-04-25):**
>
> nit: I would maybe name it `gsplat/experimental/` instead
>
> **Andre Maximo (2026-04-30) [self-resolve]:**
>
> Both work. `contrib` is a common convention (TensorFlow, scikit-learn, jax all use it for "ships with the wheel, API may change"); `experimental` is also fine and is what scikit-learn settled on for `enable_*`-gated features. No strong preference from me. If we go with `experimental`, the contract should be the same: ships with the wheel, no API stability promise.

## Summary
Naming question for the experimental namespace. Andre is neutral. Decision needed before the namespace path locks in tutorials and external imports.

## Scope / affected files
If the rename is adopted, the touch list is:
- `gsplat/contrib/` directory tree (rename)
- `gsplat/contrib/__init__.py`, `gsplat/contrib/dynamic/__init__.py` (rename + module docstrings)
- All `gsplat/contrib/dynamic/*.py` (no change beyond import paths)
- `docs/source/apis/contrib.rst` (rename + content)
- `docs/source/proposals/gsharp_v0_2_port.rst` (any path references)
- `docs/source/examples/dynamic_surgical.rst` (path references)
- `planning/gsharp_gsplat_plan.md` (architecture diagram + tables)
- `examples/dynamic_trainer.py` and tests (imports)

## Action plan
- [x] Decision (@vnath, 2026-04-30): keep `gsplat/contrib/` — namespace already matches, no code change needed.
- [ ] Post reply on the GitLab thread confirming the choice.

## Resolution
Keeping `gsplat/contrib/` per the TF / scikit-learn / jax precedent @amaximo cited. No code change required since the scaffolded namespace already matches. Reply queued — exact text in the chat thread that recorded this decision.
