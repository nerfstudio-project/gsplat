---
id: MR-001
author: CodeRabbit (initial flag); Andre Maximo (correct diagnosis)
date: 2026-04-24
mr: <TBD>
thread: planning/dev_updates/0000_init.md
status: addressed
labels: [docs, planning]
---

# MR-001 — Wrong path prefix in 0000_init.md line 14

## Original comment
> **CodeRabbit (2026-04-24, on `planning/dev_updates/0000_init.md`):**
>
> ⚠️ Potential issue | 🟡 Minor — The proposal doc mentioned on line 14 doesn't exist. The update says the proposal was checked in at `gsplat/docs/source/proposals/gsharp_v0_2_port.rst`, but that file isn't in the repo. The planning docs (.md and .html) are there, though. Either add the missing proposal file or update this line to reflect what actually landed.
>
> **Andre Maximo (2026-04-30) [self-resolve]:**
>
> CodeRabbit's check is right but its diagnosis is slightly off. The proposal file does exist at `docs/source/proposals/gsharp_v0_2_port.rst` (it's in this same diff). What's wrong is the path in the dev update: it says `gsplat/docs/source/...` but the repo root is already the gsplat repo — the correct intra-repo path is `docs/source/proposals/gsharp_v0_2_port.rst` without the `gsplat/` prefix. One-character fix in `planning/dev_updates/0000_init.md` line 14.
>
> **CodeRabbit (2026-04-30):** acknowledged Andre's correction.

## Summary
The proposal file exists; only the path in the dev log is wrong. Line 14 of `planning/dev_updates/0000_init.md` references `gsplat/docs/source/proposals/...` — the repo root *is* the gsplat repo, so the correct intra-repo path drops the `gsplat/` prefix.

## Scope / affected files
- `planning/dev_updates/0000_init.md` line 14 — strip `gsplat/` prefix
- Mirror: `gsplat/planning/dev_updates/0000_init.md`

## Action plan
- [x] Edit both copies to remove the `gsplat/` prefix (landed 2026-04-30, see `dev_updates/0002_post_review_changes.md`).
- [ ] Confirm thread closure once committed and pushed.

## Resolution
Path prefix dropped in both `planning/dev_updates/0000_init.md` and `gsplat/planning/dev_updates/0000_init.md`. GitLab thread already self-resolved by @amaximo. Awaiting commit + push to fully close on the MR side.
