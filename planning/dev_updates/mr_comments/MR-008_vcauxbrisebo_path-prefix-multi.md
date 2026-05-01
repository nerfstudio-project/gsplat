---
id: MR-008
author: Vincent Caux-Brisebois (@vcauxbrisebo) — posted via review-mrs agent
date: 2026-04-30
mr: <TBD>
thread: planning/gsharp_gsplat_plan.md
status: addressed
labels: [planning, docs, path-fix, should-fix]
related: [MR-001]
---

# MR-008 — Lingering gsplat/ path-prefix bug at multiple sites

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-04-30 [should-fix]:**

> Lingering gsplat/ path-prefix bug (same class as MR-001, multiple sites)
>
> You flagged line 123 of this file as having the same `gsplat/docs/source/...` prefix bug that MR-001 fixed in `0000_init.md`. On re-read, the same bug also appears at lines 95, 96, 97, and 123:
>
> - line 95: `Tutorial | \`gsplat/docs/source/examples/dynamic_surgical.rst\``
> - line 96: `Contrib API docs | \`gsplat/docs/source/apis/contrib.rst\``
> - line 97: `Full proposal inside repo | \`gsplat/docs/source/proposals/gsharp_v0_2_port.rst\``
> - line 123: `Full proposal checked in at \`gsplat/docs/source/proposals/gsharp_v0_2_port.rst\`.`
>
> Repo root is already `gsplat/`, so these intra-repo paths should drop the `gsplat/` prefix to match `docs/source/...` elsewhere. Suggestion: fix all four occurrences in one pass since they're the same paste-error class CodeRabbit flagged before.
>
> Label: Verified by reading.

## Summary
Same `gsplat/docs/source/...` path-prefix bug as MR-001, now spotted at 4 more sites in `gsharp_gsplat_plan.md` (lines 95, 96, 97, 123). Inside the gsplat repo, the `gsplat/` prefix is redundant. Fix is mechanical: drop the prefix wherever it appears in front of `docs/source/...`.

## Scope / affected files
- `gsplat/planning/gsharp_gsplat_plan.md` (lines 95, 96, 97, 123) — three unique path strings.
- `planning/gsharp_gsplat_plan.md` (workspace mirror) — apply same fix to keep the two copies in sync.

Three unique path strings (each may appear more than once):
- `gsplat/docs/source/examples/dynamic_surgical.rst` → `docs/source/examples/dynamic_surgical.rst`
- `gsplat/docs/source/apis/contrib.rst` → `docs/source/apis/contrib.rst`
- `gsplat/docs/source/proposals/gsharp_v0_2_port.rst` → `docs/source/proposals/gsharp_v0_2_port.rst`

## Action plan
- [x] Applied the 3 path-string substitutions to both planning copies of `gsharp_gsplat_plan.md` (2026-04-30, `replace_all`). Also applied to the workspace HTML footer (line 413) before the in-repo HTML was dropped per MR-007.
- [x] Verified with grep — only remaining matches are in tracking files (verbatim quotes from MR-001 / MR-008 / mr_tracker) and the historical 0002 dev update note. All intentional.

## Resolution
4-site fix landed across both `gsharp_gsplat_plan.md` copies (lines 95, 96, 97, 123) and the workspace HTML footer. Same paste-error class as MR-001; this round catches the rest of the visible occurrences. The in-repo HTML's instance of the same bug is moot since that file is being dropped (MR-007).
