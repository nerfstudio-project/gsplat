---
id: MR-018
author: Vincent Caux-Brisebois (@vcauxbrisebo)
date: 2026-05-14
mr: !169
thread: planning/dev_updates/mr_comments/MR-001_coderabbit_dev-update-path-prefix.md (directory-level request)
status: open
labels: [planning, repo-hygiene, owner-question]
---

# MR-018 — Should `planning/dev_updates/mr_comments/` be removed before merge?

## Thread (verbatim)

**Vincent Caux-Brisebois, 2026-05-14:**
> @vnath could we remove this directory before we merge?

(Comment posted on `MR-001_coderabbit_dev-update-path-prefix.md` but applies to the whole `mr_comments/` directory.)

## Summary
Vincent is asking whether the in-repo `planning/dev_updates/mr_comments/` directory should be dropped from the MR before merge. The workspace-level mirror (`/home/vnath/Code/internal_gsplat_plus_gsharp_apr_2026/planning/dev_updates/mr_comments/`) stays as the working copy; the question is whether the tracked-in-repo copy serves a purpose post-merge.

Per established protocol (vnath, 2026-05-07 decision when MR-007/008/009 were discussed): we kept the in-repo copy then. This re-asks the question now that the MR is near merge.

## Scope / affected files
- `planning/dev_updates/mr_comments/` (the whole directory, ~14 files + this index)
- Possibly the `mr_tracker.md` references too, depending on the decision.

## Decision required from user (vnath)
- **Option A**: Drop the in-repo `mr_comments/` directory and `mr_tracker.md` before merge; keep only the workspace mirror. Audit trail lives in commit history + MR thread. *(matches Vincent's preference.)*
- **Option B**: Keep both copies as today. Pro: tracker lives next to the code it concerns; con: noisy MR.
- **Option C**: Keep the index `mr_tracker.md` but drop individual `mr_comments/*.md` files in-repo (workspace keeps everything).

**Decision (vnath, 2026-05-21):** **Option A** — drop the in-repo `mr_comments/` directory and `mr_tracker.md` immediately before the final push. The workspace mirror at `/home/vnath/Code/.../planning/dev_updates/` keeps the full audit trail.

## Action plan
- [ ] Get vnath's decision (A / B / C).
- [ ] If A: remove the directory and tracker, leave a single line in `0014_long_run_validation_and_render.md` (or successor) noting the audit trail moved to workspace + MR threads.
- [ ] If C: gut the per-file `mr_comments/*.md` from the in-repo copy, keep only `mr_tracker.md`.

## Resolution
<filled after vnath decides>

## User-side reply draft
- **If A**: "Done — dropped `planning/dev_updates/mr_comments/` and `mr_tracker.md` from the in-repo tree. Workspace mirror at `/home/vnath/Code/.../planning/dev_updates/` keeps the full audit trail."
- **If B**: "Keeping both copies — the in-repo tracker has been useful for cross-referencing reviewer threads with the code in subsequent commits, and the mr_tracker index files are small."
