---
id: MR-011
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-05-07
mr: <TBD>
thread: docs/source/proposals/gsharp_v0_2_port.rst (lines 132–133)
status: addressed
labels: [docs, rst, should-fix]
related: [MR-007, MR-009]
---

# MR-011 — Proposal RST still points at the deleted shareable HTML

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-07 [should-fix]:**

> Lines 132–133 of `docs/source/proposals/gsharp_v0_2_port.rst` say:
>
> > See the component × test-case matrix in `planning/gsharp_gsplat_plan.md` and the shareable HTML summary at `planning/gsharp_gsplat_plan.html`.
>
> MR-007/MR-009 dropped `planning/gsharp_gsplat_plan.html` from the repo and `.gitignored` `planning/*.html`, so this RST cross-reference now points at a path that does not (and cannot) exist in-tree. The matching round in `planning/gsharp_gsplat_plan.md` line 124 was rewritten ("Shareable HTML preview is now kept locally…"); the proposal RST was missed.
>
> Suggestion: replace the trailing clause with the same wording you used in the markdown (or just drop the HTML reference entirely), so the published proposal stops promising a file that is intentionally absent.
>
> Label: Verified by reading.

## Summary
The proposal RST at `docs/source/proposals/gsharp_v0_2_port.rst` was missed when MR-007 / MR-009 dropped the in-repo HTML. Lines 132–133 still cross-reference the now-deleted `planning/gsharp_gsplat_plan.html`. Same class as the missed-mirror cleanups.

## Scope / affected files
- `docs/source/proposals/gsharp_v0_2_port.rst` lines 132–133.

## Action plan
- [x] Drop the HTML reference; lines 132–133 now read just the markdown reference.
- [ ] Confirm thread closure once committed.

## Resolution
RST updated to drop the HTML clause, matching the cleanup in `gsharp_gsplat_plan.md` line 124. Landed in commit (see `0006_round3_doc_fixes.md`).
