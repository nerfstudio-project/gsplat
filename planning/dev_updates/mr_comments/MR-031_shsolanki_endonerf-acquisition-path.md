---
id: MR-031
author: Shikhar Solanki (@shsolanki) — posted via babysit-mrs agent
date: 2026-05-20
mr: !169
thread: examples/datasets/download_dataset.py; docs/source/examples/dynamic_surgical.rst:30-43; tests/test_dynamic_surgical_trainer.py:13
status: open
labels: [docs, datasets, usability]
---

# MR-031 — No acquisition path documented for EndoNeRF; SCARED mentioned but not implemented

## Thread (verbatim)

**Shikhar Solanki, 2026-05-20:**
> The tutorial at `dynamic_surgical.rst:30-43` names "EndoNeRF pulling" and SCARED but gives no acquisition path:
> - No URL in the tutorial.
> - No `examples/datasets/download_dataset.py` entry.
> - No manual instructions.
> - `tests/test_dynamic_surgical_trainer.py:13` references `ENDONERF_DATA_DIR` without saying where to obtain a valid directory.
>
> Add an `endonerf` entry to the download script's URL map (gate behind `--license-accept` if the dataset has a click-through). Add a "Getting the data" subsection with URL / layout / checksum. Drop or qualify the SCARED mention.

## Summary
The tutorial assumes the reader already has the EndoNeRF data on disk. No URL, no download helper, no layout doc. SCARED is mentioned but is a `NotImplementedError` stub. Need to add a download path and a "Getting the data" section.

The dataset is at https://drive.google.com/drive/folders/1zTcX80c1yrbntY9c6-EK2W2UVESVEug8?usp=sharing (pulling_soft_tissues subfolder) per holohub's `applications/surgical_scene_recon/README.md`. Upstream repo is https://github.com/med-air/EndoNeRF.

## Scope / affected files
- `examples/datasets/download_dataset.py` — add `endonerf` entry to the URL/checksum map.
- `docs/source/examples/dynamic_surgical.rst` — add "Getting the data" subsection at lines 30-43.
- `tests/test_dynamic_surgical_trainer.py` — line 13 docstring should point at the new tutorial section.
- (Optional) `README.md` at the repo root if there's a dataset section.

## Action plan
- [ ] Add `endonerf` to `examples/datasets/download_dataset.py`:
  - Primary source: med-air/EndoNeRF Google Drive folder (gdown can pull Drive folders).
  - Document expected directory layout (`poses_bounds.npy`, `images/`, `depth/`, `masks/`).
  - Add an MD5/SHA256 checksum if upstream is stable; otherwise skip and note "no upstream checksum".
- [ ] Add a "Getting the data" subsection to `docs/source/examples/dynamic_surgical.rst:30-43` with: download URL, expected layout, and the `mv pulling_soft_tissues data/EndoNeRF/pulling` step.
- [ ] Update the `ENDONERF_DATA_DIR` env-var docstring in `tests/test_dynamic_surgical_trainer.py:13` to cross-link the tutorial.
- [ ] Either delete the SCARED references from the tutorial, or wrap them with a `.. note:: SCARED loader is not yet implemented; see follow-up MR.`
- [ ] Verify the download command works end-to-end.

## Resolution
<filled after fix lands>

## User-side reply draft
> Added an `endonerf` entry to `examples/datasets/download_dataset.py` (pulls from the med-air Google Drive folder via gdown), wrote a "Getting the data" subsection in `dynamic_surgical.rst` with URL + expected layout, and updated the `ENDONERF_DATA_DIR` docstring to cross-link. Marked the SCARED references as "not yet implemented".
