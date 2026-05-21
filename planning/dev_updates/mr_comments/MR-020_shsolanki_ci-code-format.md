---
id: MR-020
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: GitLab pipeline 51354030; docs/source/index.rst:118
status: open
labels: [ci, formatting, blocking-for-merge]
---

# MR-020 — CI code-format job fails (14 files); fix and re-trigger

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> Pipeline 51354030 fails `code-format` (14 files). `git diff --check origin/nv/main...HEAD` also flags trailing whitespace at `docs/source/index.rst:118`.
>
> Run `pre-commit run -a` locally and re-push.

## Summary
The branch's `code-format` CI job is red. Need to run the repo's pre-commit hooks across all changed files and push the resulting formatter changes.

## Scope / affected files
- 14 files flagged by the CI code-format job (specific list lives in pipeline 51354030 logs).
- `docs/source/index.rst:118` (trailing whitespace, also flagged by `git diff --check`).
- Likely overlap with files touched in this MR.

## Action plan
- [ ] Pull pipeline log to enumerate the 14 files.
- [ ] Run `pre-commit run --all-files` locally (or the project's `ruff format` + `black` equivalent — confirm what `.pre-commit-config.yaml` declares).
- [ ] Strip trailing whitespace at `docs/source/index.rst:118`.
- [ ] Commit (separate "chore: ruff/black format" commit so the diff is reviewable).
- [ ] Force-push as part of the same rebase pass as MR-019, or push as a fresh commit if MR-019 isn't done yet.

## Resolution
<filled after pipeline is green>

## User-side reply draft
> Ran `pre-commit run --all-files`, fixed the trailing whitespace at `docs/source/index.rst:118`, and pushed. Pipeline 51354030 should pass on the next run.
