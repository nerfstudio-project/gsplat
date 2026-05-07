---
id: MR-013
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-07
mr: <TBD>
thread: gsplat/contrib/dynamic/deformation.py (file-level)
status: deferred
labels: [contrib, deformation, design-question]
---

# MR-013 — Question to @shsolanki on deformation integration

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-07:**

> @shsolanki could you advise on how to best get this integrated with the scene with transformation, context, and all?

## Summary
Vincent's question is directed at @shsolanki, asking for design input on how to integrate the deformation network with the existing scene transformation / context flow. Not a request to vnath; no code change required from this side. Track and wait.

## Scope / affected files
None directly. Decision will eventually inform the implementation of `gsplat/contrib/dynamic/deformation.py` (TDD step 6 in `0001_scaffolding_commit.md`).

## Action plan
- [ ] Wait for @shsolanki's design response on the GitLab thread.
- [ ] When the design lands, fold the implications into the deformation impl plan and/or the `0001` execution-order list.

## Resolution
_Deferred — directed at @shsolanki, awaiting his guidance. No @vnath action required at this point._
