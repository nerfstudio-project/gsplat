---
id: MR-009
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-04-30
mr: <TBD>
thread: planning/gsharp_gsplat_plan.html
status: addressed
labels: [planning, html, nit]
related: [MR-007]
---

# MR-009 — Consider not committing generated HTML

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-04-30 [nit, non-blocking]:**

> Consider not committing generated HTML
>
> Generated HTML in `planning/` is hard to keep in sync with the markdown source — this MR already demonstrates that, since it required a separate "Stale / to-flag" callout in `0002_post_review_changes.md`. If the HTML is just for ad-hoc sharing, generating on demand and `.gitignore`-ing it would remove a recurring drift hazard. Non-blocking; bring it up in the next round.

## Summary
Nit / non-blocking. Vincent points out that committing the rendered HTML alongside the markdown creates a recurring drift hazard (already demonstrated by MR-007). If we're not actively shipping it from-repo, dropping it and `.gitignore`-ing the path removes the class of bug. Decision is tied to MR-007.

## Scope / affected files
If we adopt this:
- Add `planning/*.html` (or a more specific entry) to `.gitignore` at the gsplat repo root.
- The actual file removal is handled by MR-007.

## Action plan
- [x] Folded into MR-007's Option A.
- [x] Added `planning/*.html` to `gsplat/.gitignore` so the file can't accidentally come back as a tracked artifact.

## Resolution
`.gitignore` updated. Combined with MR-007, future drift between the markdown and a checked-in HTML rendering is no longer possible — the HTML lives only outside the gsplat repo (workspace `planning/gsharp_gsplat_plan.html`) and any in-repo path matching `planning/*.html` is silently ignored.
