---
id: MR-014
author: CodeRabbit
date: 2026-05-14
mr: <TBD>
thread: multiple — bundled review pass on commits 7ed611e6 and 58bf8d9d
status: open
labels: [nit, docs, formatting, ci]
---

# MR-014 — CodeRabbit round-4 nits (regularizers Unicode, test format, doc polish)

CodeRabbit posted seven sub-threads in one review pass on commits `7ed611e6` (regularizers impl) and `58bf8d9d` (round-3 doc fixes). All are minor / nitpick / formatting; bundled here per vnath direction (single audit trail rather than seven near-empty files).

## Sub-threads (verbatim, condensed)

### A1 — regularizers.py Unicode chars (commit `7ed611e6`)
> Ruff flagged a few spots where EN DASH (–) and multiplication sign (×) are used instead of ASCII equivalents.
> - Lines 9, 36, 135: – → -
> - Line 86: × → x
> Totally optional to fix, just mentioning it in case you want the linter green across the board.

### A2 — tests/test_regularizers_occlusion.py formatting (commit `7ed611e6`, CI failed)
> Pipeline failure: formatting check didn't pass. The CI flagged this file with "would reformat." Running the project's formatter (likely black or ruff format) should fix it. It might just be a missing trailing newline at the end of the file.

### A3 — 0006_round3_doc_fixes.md verification wording (lines 42-47)
> One small wording thing on line 46: the "→ no matches in active text" phrasing is slightly ambiguous since you then note it *is* found in the MR-010 file (intentionally). Maybe "→ found only in quoted context (intentional)" or "→ no matches in live code" would read a bit clearer.

### A4 — 0006_round3_doc_fixes.md missing commit SHA (lines 32-41)
> One tiny enhancement: you reference "this commit" a few times (e.g., line 39) but don't provide its hash, whereas earlier commits get their hashes (a380d40d, dcb84ef0, 3c0dd4e5). Adding the hash for this commit would make the audit trail even more complete — though it's totally fine to add that when you push!

### A5 — 0006_round3_doc_fixes.md "both copies" clarity (lines 11-13)
> Line 13 mentions fixing "both copies of 0004" but doesn't explain what the two copies are. … verify and correct the function/operation wording so `create_invisible_mask_from_paths` is replaced with `create_invisible_mask` and "intersection of inverse masks" is corrected to "union".

### A6 — MR-012 detail file: front matter mismatch (lines 5-7)
> The metadata shows status: addressed but mr: <TBD> is still a placeholder. Either fill in the MR number or change status to pending/in-progress.

### A7 — MR-012 detail file: checklist consistency (line 37)
> The checklist item "Confirm thread closure once committed" is still unchecked but the front matter declares status: addressed and the resolution claims changes landed.

## Summary
Pure documentation / formatting / linter cleanup. None of these affect runtime behaviour. A2 is the only CI-relevant one (would reformat); A1 is linter polish; A3–A7 are dev-update doc readability nits on past planning files.

## Scope / affected files
- `gsplat/regularizers.py` (A1)
- `tests/test_regularizers_occlusion.py` (A2)
- `planning/dev_updates/0006_round3_doc_fixes.md` (A3, A4, A5)
- `planning/dev_updates/mr_comments/MR-012_vcauxbrisebo_masked-ssim-bias-doc.md` (A6, A7)

## Action plan
- [ ] A1: ASCII-ify EN DASH and multiplication sign in `gsplat/regularizers.py` docstrings (lines 9, 36, 86, 135). Verify ruff goes green.
- [ ] A2: Run `ruff format` (or `black`) on `tests/test_regularizers_occlusion.py`; confirm trailing newline; ensure CI code-format passes (also see MR-020).
- [ ] A3: Reword "→ no matches in active text" in `0006_round3_doc_fixes.md` line 46 → "→ no matches in live code".
- [ ] A4: Backfill commit SHA for the round-3 fixes entry in `0006_round3_doc_fixes.md` (commit `58bf8d9d`).
- [ ] A5: Expand "both copies of 0004" to spell out which two copies (workspace mirror vs in-repo tracked); double-check `create_invisible_mask` / "union" wording.
- [ ] A6: Update `MR-012_vcauxbrisebo_masked-ssim-bias-doc.md` front matter to either drop `<TBD>` (replace with `!169`) or set `status: pending`.
- [ ] A7: In the same file, tick the "Confirm thread closure once committed" checklist item once thread is acked (or remove it if already acked).

## Resolution
<filled after fix lands>

## User-side reply draft (per sub-thread)

- **A1 & A2**: "Fixed in this round — ASCII-ified the unicode chars in `gsplat/regularizers.py` and ran `ruff format` on the test file. CI code-format should be green now."
- **A3, A4, A5**: "Polished the wording in `0006_round3_doc_fixes.md` and backfilled the `58bf8d9d` hash."
- **A6, A7**: "Updated `MR-012_*.md` front matter and ticked the closure checkbox now that thread is acked."
