# CityGS — block-based city-scale Gaussian Splatting on gsplat

A ground-up redesign of the [CityGaussian](https://github.com/Linketic/CityGaussian)
pipeline on top of gsplat, built around one principle: **use the right kind of
parallelism at each stage**.

| Stage | Parallelism | Why |
| --- | --- | --- |
| coarse global prior | gsplat `distributed=True` (optional) | breaks the single-GPU VRAM cap on the Gaussian budget; short stage, best ROI for all-to-all traffic |
| block finetune | one block per GPU, zero communication | the longest stage; PCIe stays idle, every GPU runs at full speed |

On PCIe-only boxes (e.g. 8x4090 without NVLink), measure distributed coarse
first — the built-in fallback is single-GPU coarse at lower resolution
(`--coarse.downsample`) with an MCMC-capped budget, which trades resolution
for Gaussians instead of paying the all-to-all.

## Pipeline

```
ingest ──> coarse ──> partition ──> finetune (queue) ──> merge ──> eval
  │          │            │               │                │
scene.json  coarse.pt   blocks +      block_XXXX.pt     merged.pt
points.npz  (MCMC)      camera         (one per cell)   merged.ply
                        assignment
```

All stages communicate through a single **`scene.json` manifest** (cameras,
normalization, contraction bounds, block definitions, camera assignments,
artifact paths). Every stage is idempotent: it skips itself when its outputs
exist, so a killed run resumes with the same command.

### Quick start

```bash
# everything, end to end (COLMAP data with undistorted pinhole cameras)
python -m citygs.pipeline \
    --data-dir data/my_city --result-dir results/my_city \
    --gpus 0 1 2 3 4 5 6 7 \
    --partition.grid 6 6 \
    --coarse.cap-max 20000000 --coarse-distributed

# resume after a crash: same command (finished blocks/stages are skipped)
# redo from a stage onwards:
python -m citygs.pipeline ... --force partition
```

Each stage is also a standalone CLI (`python -m citygs.scene.colmap`,
`citygs.train.coarse`, `citygs.partition.run`, `citygs.train.scheduler`,
`citygs.merge`, `citygs.eval`) — see `--help` on any of them; all options
live in `citygs/config.py`.

### Inspecting results

```bash
# interactive: per-block visibility, color-by-block, cell wireframes,
# per-block camera frustums (viser does the splat rendering)
python -m citygs.viz.viewer --manifest results/my_city/scene.json

# static top-down partition overview is written by the partition stage:
results/my_city/partition/partition.png

# merged.ply drops straight into SuperSplat / antimatter15 viewers
```

## How each stage works

**Ingest** (`citygs/scene/colmap.py`) — pycolmap → `scene.json` +
`points.npz` (SfM points, colors, and the per-camera point-visibility index
used for optional sparse-depth supervision). Normalizes the world (z-up,
recentered). Caches downsampled images to `images_<factor>/`. Only
undistorted pinhole cameras are accepted; run `colmap image_undistorter`
first (and for known poses, `colmap point_triangulator` beats full SfM).

**Coarse** (`citygs/train/coarse.py`) — global prior with the **MCMC
strategy**: `cap_max` fixes the *total* Gaussian budget, which both bounds
VRAM and keeps per-rank counts balanced in the distributed case. LR follows
the Grendel `sqrt(batch)` scaling rule. Trains at reduced resolution
(`downsample`, default 4) — a prior doesn't need full res.

**Partition** (`citygs/partition/`) — camera positions give a robust
(MAD-filtered) foreground box; a **per-axis contraction** maps all of space
into (-2,2)^3 (per-axis, not norm-coupled, so constant-altitude rigs don't
collapse the x/y layout). Blocks are grid cells of the contracted cube —
they tile space exactly, and the block count is decoupled from the GPU
count. Camera assignment uses a batched *coverage proxy* (opacity ×
projected area of the block's coarse Gaussians, no rendering) instead of
the original per-camera SSIM renders, with three guarantees: cameras inside
a cell always train it, every camera trains ≥1 block, every block gets
≥`min_cameras`.

**Finetune** (`citygs/train/finetune_block.py`, `scheduler.py`) — a work
queue runs one block per GPU, longest-first (LPT); block checkpoints double
as done markers, so reruns skip finished blocks. Each block initializes
from the coarse Gaussians in its cell + margin, trains with absgrad
densification (+ optional sparse-depth loss), gets an adaptive step budget
(`sqrt(cameras/median)`), and is cropped back to its exact cell at the end.

**Merge** (`citygs/merge.py`) — cells tile contracted space and every block
started from the same coarse prior, so merge is a plain concat (with an
idempotent re-crop as a guard). Exports `merged.pt` + `merged.ply`.

**Eval** (`citygs/eval.py`) — PSNR/SSIM (optional LPIPS) on the val split.

## Layout

```
citygs/
  config.py            all stage configs (tyro CLIs; JSON for subprocesses)
  pipeline.py          orchestrator with skip/resume/force
  ckpt.py              unified checkpoint schema (citygs/v1)
  scene/               manifest, COLMAP ingest, LRU-cached dataset
  partition/           contraction, grid blocks, camera assignment
  train/               shared trainer, coarse, per-block finetune, scheduler
  merge.py  eval.py
  viz/                 partition.png + viser block viewer
tests/citygs/          CPU-only unit tests (pytest tests/citygs)
```

Extra dependencies beyond gsplat: `tyro pycolmap imageio pillow`
(+ `viser` for the viewer, `matplotlib` for partition.png,
`tensorboard`/`scikit-learn` optional).
