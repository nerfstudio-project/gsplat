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
| MR-001 | CodeRabbit + Andre Maximo | 2026-04-24 | `0000_init.md` line 14 wrong path prefix — fixed in `3c0dd4e` | resolved | [MR-001](mr_comments/MR-001_coderabbit_dev-update-path-prefix.md) |
| MR-002 | CodeRabbit + Andre Maximo | 2026-04-24 | `losses_depth.py` merge timing → folded into `losses.py` (`3c0dd4e`) | resolved | [MR-002](mr_comments/MR-002_coderabbit_losses-depth-merge-timing.md) |
| MR-003 | Vincent Caux-Brisebois + Andre Maximo | 2026-04-25 | Losses placement + regularizers split (`3c0dd4e`; impl in `7ed611e`) | resolved | [MR-003](mr_comments/MR-003_vcauxbrisebo_losses-depth-placement.md) |
| MR-004 | Shikhar Solanki | 2026-04-25 | Naming nit `contrib/` vs `experimental/` → kept `contrib/` (vnath replied) | resolved | [MR-004](mr_comments/MR-004_shsolanki_contrib-vs-experimental.md) |
| MR-005 | Shikhar Solanki | 2026-04-25 | Rename `dynamic_trainer.py` → `dynamic_surgical_trainer.py` (vnath replied) | resolved | [MR-005](mr_comments/MR-005_shsolanki_trainer-naming.md) |
| MR-006 | Shikhar Solanki (Andre pushback) | 2026-04-25 | Keep `init_utils.py` separate (vnath replied; awaiting @shsolanki ack) | addressed | [MR-006](mr_comments/MR-006_shsolanki_init-utils-consolidation.md) |
| MR-007 | Vincent Caux-Brisebois | 2026-04-30 | HTML plan stale → dropped in-repo (`dcb84ef`); resolved by Vincent | resolved | [MR-007](mr_comments/MR-007_vcauxbrisebo_html-stale.md) |
| MR-008 | Vincent Caux-Brisebois | 2026-04-30 | Multi-site path-prefix bug fixed (`dcb84ef`); resolved by Vincent | resolved | [MR-008](mr_comments/MR-008_vcauxbrisebo_path-prefix-multi.md) |
| MR-009 | Vincent Caux-Brisebois | 2026-04-30 | `.gitignore` for `planning/*.html` (`dcb84ef`); resolved by Vincent | resolved | [MR-009](mr_comments/MR-009_vcauxbrisebo_html-not-committed.md) |
| MR-010 | CodeRabbit | 2026-05-07 | `0004` line 59 function name + operation paste-error | addressed | [MR-010](mr_comments/MR-010_coderabbit_invisible-mask-name.md) |
| MR-011 | Vincent Caux-Brisebois | 2026-05-07 | Proposal RST still references the deleted HTML | addressed | [MR-011](mr_comments/MR-011_vcauxbrisebo_proposal-rst-html-ref.md) |
| MR-012 | Vincent Caux-Brisebois | 2026-05-07 | `masked_ssim` mask-bias docstring + partial-mask test | addressed | [MR-012](mr_comments/MR-012_vcauxbrisebo_masked-ssim-bias-doc.md) |
| MR-013 | Vincent Caux-Brisebois | 2026-05-07 | Question to @shsolanki on deformation integration (answered by MR-022) | deferred | [MR-013](mr_comments/MR-013_vcauxbrisebo_deformation-integration-question.md) |
| MR-014 | CodeRabbit | 2026-05-14 | Round-4 nits: regularizers Unicode, test format, 0006/MR-012 doc polish (7 sub-threads bundled) | open | [MR-014](mr_comments/MR-014_coderabbit_round4-doc-and-format-nits.md) |
| MR-015 | Vincent Caux-Brisebois | 2026-05-14 | `regulation.py` accumulator inits on CPU — crashes on CUDA planes (blocking) | open | [MR-015](mr_comments/MR-015_vcauxbrisebo_regulation-cpu-cuda-device.md) |
| MR-016 | Vincent Caux-Brisebois | 2026-05-14 | `compute_tv_loss_targeted` binary-mask check forces per-step host sync | open | [MR-016](mr_comments/MR-016_vcauxbrisebo_tv-mask-cpu-sync.md) |
| MR-017 | Vincent Caux-Brisebois | 2026-05-14 | `_grid_sample_wrapper` bare `.squeeze()` is shape-fragile (nit) | open | [MR-017](mr_comments/MR-017_vcauxbrisebo_hexplane-bare-squeeze.md) |
| MR-018 | Vincent Caux-Brisebois | 2026-05-14 | Remove `mr_comments/` directory before merge? (owner question) | open | [MR-018](mr_comments/MR-018_vcauxbrisebo_remove-mr-comments-dir.md) |
| MR-019 | Shikhar Solanki | 2026-05-20 | Drop AI co-author trailers from 13/14 branch commits (rebase + force-push) | open | [MR-019](mr_comments/MR-019_shsolanki_ai-coauthor-trailers.md) |
| MR-020 | Shikhar Solanki | 2026-05-20 | CI `code-format` job fails on 14 files; trailing whitespace in `docs/source/index.rst:118` | open | [MR-020](mr_comments/MR-020_shsolanki_ci-code-format.md) |
| MR-021 | Shikhar Solanki | 2026-05-20 | Test still asserts removed `hexplane_params` / `deform_mlp_params` placeholders | open | [MR-021](mr_comments/MR-021_shsolanki_test-asserts-removed-params.md) |
| MR-022 | Shikhar Solanki | 2026-05-20 | DeformationTable contract not honoured end-to-end (answers MR-013; design decision needed) | open | [MR-022](mr_comments/MR-022_shsolanki_deformation-table-contract.md) |
| MR-023 | Shikhar Solanki | 2026-05-20 | `binocular_disparity_l1` validity handling contradicts docstring | open | [MR-023](mr_comments/MR-023_shsolanki_binocular-disparity-pair-valid.md) |
| MR-024 | Shikhar Solanki | 2026-05-20 | `knn_scale_init` builds O(N²) `cdist` matrix; OOM on consumer GPUs at 50k pts | open | [MR-024](mr_comments/MR-024_shsolanki_knn-scale-init-memory.md) |
| MR-025 | Shikhar Solanki | 2026-05-20 | Tutorial quickstart shows CLI flags `--dataset_type` / `--strategy` that don't exist | open | [MR-025](mr_comments/MR-025_shsolanki_tutorial-bogus-cli-flags.md) |
| MR-026 | Shikhar Solanki | 2026-05-20 | `Config.hex_bounds=1.6` mismatch → HexPlane spatially constant (deformation effectively dead) **major** | open | [MR-026](mr_comments/MR-026_shsolanki_hex-bounds-mismatch-dead-hexplane.md) |
| MR-027 | Shikhar Solanki | 2026-05-20 | LLFF axis permutation in EndoNeRF parser (reviewer marked resolved; verify in tree) | addressed | [MR-027](mr_comments/MR-027_shsolanki_llff-axis-permutation.md) |
| MR-028 | Shikhar Solanki | 2026-05-20 | `setup.py install_requires` missing `Pillow` and `tqdm` | open | [MR-028](mr_comments/MR-028_shsolanki_missing-pillow-tqdm-deps.md) |
| MR-029 | Shikhar Solanki | 2026-05-20 | 13 new source files missing the standard SPDX copyright header block | open | [MR-029](mr_comments/MR-029_shsolanki_spdx-headers-missing.md) |
| MR-030 | Shikhar Solanki | 2026-05-20 | `gsplat/contrib/dynamic/__init__.py.__all__` exports internals (`DeformationTable` + plane-partitioned regs) | open | [MR-030](mr_comments/MR-030_shsolanki_contrib-dynamic-all-exposes-internals.md) |
| MR-031 | Shikhar Solanki | 2026-05-20 | EndoNeRF dataset has no documented acquisition path (no URL / no download script entry) | open | [MR-031](mr_comments/MR-031_shsolanki_endonerf-acquisition-path.md) |
