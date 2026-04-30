# MR Comment Tracker

Tracks reviewer comments pasted from the GitLab MR for the `vnath_gsharp` branch. Each row links to a detail file under `mr_comments/` with the verbatim comment, summary, scope, action plan, and resolution.

**Statuses:** `open` (just received) · `in-progress` (being worked on) · `addressed` (fix landed, awaiting reviewer ack) · `resolved` (reviewer marked thread resolved) · `wontfix` (declined, with rationale in detail file) · `deferred` (acknowledged, deferred to a follow-up).

## How a new comment lands

1. User pastes the comment (with author + date if visible).
2. Next `MR-NNN` ID is assigned (monotonic, zero-padded to 3 digits).
3. Detail file created at `mr_comments/MR-NNN_<author>_<slug>.md` using the template below.
4. Row appended to the table here.
5. Mirrored copy updated under workspace-level `planning/dev_updates/` to stay in sync.

## Detail file template

```markdown
---
id: MR-NNN
author: <reviewer>
date: YYYY-MM-DD
mr: <gitlab MR url or !number>
thread: <gitlab thread/discussion id if known>
status: open
labels: [<area tags, e.g. losses, contrib, docs>]
---

# MR-NNN — <short title>

## Original comment
> <verbatim quoted text from GitLab>

## Summary
<1–3 sentence restatement of what's being asked>

## Scope / affected files
- `path/to/file.py` (...)

## Action plan
- [ ] step 1
- [ ] step 2

## Resolution
<filled when addressed — commit SHA, dev_update link, or rationale if wontfix/deferred>
```

## Index

| ID | Author(s) | Date | Summary | Status | Detail |
|----|-----------|------|---------|--------|--------|
| MR-001 | CodeRabbit + Andre Maximo | 2026-04-24 | `0000_init.md` line 14 had wrong path prefix `gsplat/docs/...` — fixed in 0002 | addressed | [MR-001](mr_comments/MR-001_coderabbit_dev-update-path-prefix.md) |
| MR-002 | CodeRabbit + Andre Maximo | 2026-04-24 | losses_depth.py merge timing → fold into losses.py from the start | addressed | [MR-002](mr_comments/MR-002_coderabbit_losses-depth-merge-timing.md) |
| MR-003 | Vincent Caux-Brisebois + Andre Maximo | 2026-04-25 | Right home for losses_depth.py + regularizers.py → drop losses_depth.py into losses.py; keep regularizers.py | addressed | [MR-003](mr_comments/MR-003_vcauxbrisebo_losses-depth-placement.md) |
| MR-004 | Shikhar Solanki | 2026-04-25 | Naming nit `contrib/` vs `experimental/` → **keep `contrib/`** | addressed | [MR-004](mr_comments/MR-004_shsolanki_contrib-vs-experimental.md) |
| MR-005 | Shikhar Solanki | 2026-04-25 | Rename `dynamic_trainer.py` → **`dynamic_surgical_trainer.py`** | addressed | [MR-005](mr_comments/MR-005_shsolanki_trainer-naming.md) |
| MR-006 | Shikhar Solanki (Andre pushback) | 2026-04-25 | Move init_utils into trainer? → **keep separate** (vnath confirms reuse) | addressed | [MR-006](mr_comments/MR-006_shsolanki_init-utils-consolidation.md) |
