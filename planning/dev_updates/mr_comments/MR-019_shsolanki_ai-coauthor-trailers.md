---
id: MR-019
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: branch commit history c2652bca..5f04a631
status: open
labels: [policy, git, blocking-for-merge]
---

# MR-019 — Drop AI co-author trailers from branch commits before merge

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> 13 of the 14 branch commits carry a `Co-Authored-By: Claude … <noreply@anthropic.com>` trailer; our local commit rule disallows AI co-author trailers. The root commit `c2652bca` also has a `Made-with: Cursor` trailer.
>
> Please `git rebase -i nv/main`, drop the disallowed trailer(s), then force-push and re-trigger the pipeline.

## Summary
Repo policy disallows AI co-author trailers. We need to rebase the branch interactively, strip the `Co-Authored-By:` and `Made-with:` trailers from every commit, force-push, and re-trigger CI.

## Scope / affected files
- Branch commit metadata only — no source-file changes.
- 13 of 14 commits between `nv/main` and `vnath_gsharp` head (`5f04a631`).
- Root commit `c2652bca` has the `Made-with: Cursor` trailer too.

## Action plan
- [ ] Confirm with vnath that the force-push to `vnath_gsharp` is authorised (destructive op; per CLAUDE.md we always confirm).
- [ ] `git rebase -i nv/main` and remove the disallowed trailer lines from each commit message. Avoid `--no-edit` (incompatible with `git rebase`).
- [ ] Force-push `vnath_gsharp` to origin.
- [ ] Re-trigger pipeline; confirm CI starts a fresh run.
- [ ] Update `Co-Authored-By:` policy in `CLAUDE.md` memory so future commits skip the trailer automatically.

## Resolution
<filled after rebase + force-push>

## User-side reply draft
> Rebased onto `nv/main` and dropped the AI co-author trailers across all 14 commits; force-pushed to `vnath_gsharp`. Pipeline re-triggered. (Updated my local automation to skip the trailer going forward.)
