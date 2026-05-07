---
id: MR-010
author: CodeRabbit (bot)
date: 2026-05-07
mr: <TBD>
thread: planning/dev_updates/0004_post_review_round2.md (line 59)
status: addressed
labels: [planning, docs, nit, should-fix]
---

# MR-010 — Function name + operation mismatch in 0004 line 59

## Thread (verbatim)

**CodeRabbit, 2026-05-07 [should-fix nit]:**

> Function name and description mismatch.
>
> Line 59 lists `create_invisible_mask_from_paths` with "intersection of inverse masks", but the actual implementation is `create_invisible_mask` (no suffix) and performs a "union of per-frame instrument masks" according to its docstring and test name (`test_create_invisible_mask_union_of_inputs`). Since this is describing what's needed next, using the wrong name and operation could confuse anyone picking up this work.
>
> Update line 59 to match the actual implementation:
> - Correct name: `create_invisible_mask`
> - Correct operation: union (not intersection of inverse masks)

## Summary
Paste-error in `0004_post_review_round2.md` line 59: function name borrowed from holohub source (`create_invisible_mask_from_paths`) and operation described as "intersection of inverse masks" instead of "union". The actual function in our scaffold + impl is `create_invisible_mask` (union of per-frame masks).

## Scope / affected files
- `gsplat/planning/dev_updates/0004_post_review_round2.md` line 59
- Mirror: `planning/dev_updates/0004_post_review_round2.md` line 59

## Action plan
- [x] Update both copies of 0004 line 59: function name → `create_invisible_mask`, operation → "union of per-frame instrument masks".
- [ ] Confirm thread closure once committed.

## Resolution
Doc fix landed (see `0006_round3_doc_fixes.md`). The actual implementation in `gsplat/regularizers.py` (commit `7ed611e`) is correct and unchanged; only the dev-update narrative was wrong.
