# 0001 — Scaffolding commit ready to push

Date: 2026-04-23
Author: scaffolding agent
Branch: `vnath_gsharp` (off `nv/main`)
Commit: `c2652bc`

## What landed

- Mirrored `planning/` into `gsplat/planning/` so the HTML plan, markdown plan, and `dev_updates/` all ship with the PR. Workspace-level `planning/` is kept as a convenience copy.
- Single commit `c2652bc` on `vnath_gsharp` — 28 files, +1542 / -0, nothing modified except `docs/source/index.rst` (one-line append of a new "Proposals" toctree caption).

Commit contents (architecture-only, no implementation):

- **Proposal + tutorial + API stubs** (Sphinx): `docs/source/proposals/gsharp_v0_2_port.rst`, `docs/source/examples/dynamic_surgical.rst`, `docs/source/apis/contrib.rst`.
- **Shareable plan inside the repo**: `planning/gsharp_gsplat_plan.html`, `planning/gsharp_gsplat_plan.md`, `planning/dev_updates/0000_init.md`.
- **Core stubs** (stable): `gsplat/losses_depth.py`, `gsplat/regularizers.py`, `gsplat/init_utils.py`, `gsplat/training/{__init__,schedulers}.py`.
- **Experimental contrib stubs**: `gsplat/contrib/__init__.py`, `gsplat/contrib/dynamic/{__init__,hexplane,deformation,regulation,strategy}.py`.
- **Example stubs**: `examples/datasets/endonerf.py`, `examples/dynamic_trainer.py`.
- **Pytest stubs** (8 files, all `pytest.mark.skip` with reason strings; one real negative-case assertion for `TwoStageScheduler` arg validation).

## Verification

- `py_compile` on all 21 new Python files: clean.
- `ReadLints` on the full set: no warnings.
- `git status` clean after commit; working tree matches `HEAD`.
- Existing CUDA / rasterization code paths untouched; only additive changes + one toctree line.

## Blocked on

- **User push.** Network writes require credentials; user will run `git push -u origin vnath_gsharp` from `/home/vnath/Code/internal_gsplat_plus_gsharp_apr_2026/gsplat`.
- **Reviewer sign-off** on (a) the core-vs-contrib split, (b) the `gsplat.contrib.dynamic` namespace choice, (c) DA2 / MedSAM3 / VGGT staying out of the library, (d) the per-component test-matrix minimum bar.

## What unblocks next

Once the architecture review comes back green, implementation starts TDD per-component in the order from the pinned plan:

1. `binocular_disparity_l1` / `pearson_depth_loss` / `masked_l1` / `masked_ssim` → un-skip `tests/test_losses_depth.py`.
2. `compute_tv_loss_targeted` / `dilate_mask` / `create_invisible_mask` → un-skip `tests/test_regularizers_occlusion.py`.
3. `multi_frame_depth_unprojection` / `knn_scale_init` → un-skip `tests/test_init_multiframe.py`.
4. `TwoStageScheduler.step` → un-skip `tests/test_two_stage_scheduler.py`.
5. `HexPlaneField` → un-skip `tests/test_contrib_hexplane.py`.
6. `DeformNetwork` + `DeformationTable` → un-skip `tests/test_contrib_deformnet.py`.
7. `DynamicStrategy` → un-skip `tests/test_contrib_dynamic_strategy.py`.
8. `EndoNeRFParser` + `EndoNeRFDataset` → un-skip `tests/test_dataset_endonerf.py`.
9. Wire `examples/dynamic_trainer.py` end-to-end and run against a dataset the user provides.

Each item earns its own `NNNN_<slug>.md` dev log entry when it lands.

## Notes for the next agent picking this up

- Two mirrored copies of `planning/` exist (`gsplat/planning/` ← tracked; workspace `planning/` ← untracked). When appending a new dev update, write to **both** to keep them in sync, or drop the workspace-level one entirely and standardize on the in-repo copy.
- `gsplat/contrib/dynamic/strategy.py` intentionally drops the `@dataclass` decorator despite `DefaultStrategy` being a dataclass — this avoids dataclass-field ordering errors. Reintroduce `@dataclass` only if new fields are added with defaults.
- Top-level `import gsplat` triggers the CUDA JIT build, which needs write access to `~/.cache/torch_extensions/`. The sandbox in this agent environment forbids that; on a normal dev box the imports go through.
