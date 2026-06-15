# Inference Rendering Benchmarks

This directory contains benchmarks for the inference rendering path, which uses
packed quaternion, scale, and opacity (QSO) scene representation. The benchmark
measures GPU latency (via CUDA events) for the stateful `GaussianInferenceRenderer`
and optionally compares against the default gsplat `rasterization()` path for
speedup and image-quality (PSNR / LPIPS) metrics.

## Quick Start

Run from the repository root:

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py
```

By default this generates a synthetic scene (500k Gaussians), renders 1920×1080
frames on a circular camera orbit, and reports per-frame GPU timing statistics.

## Examples

**Synthetic scene with reference comparison:**

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py \
  --num-gaussians 1000000 \
  --num-frames 200 \
  --reference-tile-size 16
```

**Trained model from a PLY file:**

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py \
  --ply-path path/to/point_cloud.ply \
  --reference-tile-size 16
```

Pretrained scenes can be downloaded with `python examples/datasets/download_dataset.py`.

**Custom resolution and tile sizes:**

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py \
  --width 3840 --height 2160 \
  --inference-tile-size 8 \
  --reference-tile-size 16
```

**Save a rendered frame for visual inspection:**

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py \
  --ply-path path/to/point_cloud.ply \
  --save-image output/inference_preview.png \
  --reference-tile-size 16
```

When `--reference-tile-size` is set, a second image (`*_ref.png`) is also saved
from the reference `rasterization()` path.

## Options

Run with `--help` for defaults. Key flags:

### Scene

| Flag | Description |
|------|-------------|
| `--ply-path PATH` | Load a trained 3DGS PLY instead of generating a synthetic scene. When set, `--num-gaussians` is ignored and `--sh-degree` is inferred from the file. |
| `--num-gaussians N` | Number of Gaussians in the synthetic scene (default: 500,000). Ignored when `--ply-path` is set. |
| `--sh-degree {0,1,2,3,-1}` | Spherical-harmonics degree for the synthetic scene (default: 3). Use `-1` for pre-activated RGB colors instead of SH coefficients. Inferred automatically from a PLY. |

### Rendering

| Flag | Description |
|------|-------------|
| `--width W` | Output image width in pixels (default: 1920). |
| `--height H` | Output image height in pixels (default: 1080). |
| `--inference-tile-size {8,16}` | Tile size for the Inference rasterization kernels (default: 8). |
| `--reference-tile-size {4,16}` | Enable the reference `rasterization()` benchmark with the given tile size. When omitted, the reference path, speedup, and image-quality comparisons are skipped. |

### Timing

| Flag | Description |
|------|-------------|
| `--num-frames N` | Number of timed render frames per benchmark (default: 100). |
| `--warmup-frames N` | Untimed warmup frames before each benchmark (default: 10). |

### Image quality (requires `--reference-tile-size`)

| Flag | Description |
|------|-------------|
| `--quality-frames N` | Number of frames used for Inference-vs-reference PSNR/LPIPS comparison (default: 1). Set to `0` to skip image-quality metrics. |
| `--lpips-net {alex,vgg}` | LPIPS backbone for image-quality comparison (default: `alex`). Requires `torchmetrics`. |

### Output

| Flag | Description |
|------|-------------|
| `--save-image PATH` | Render one frame with the Inference path and save it as a PNG. Also saves a reference PNG when `--reference-tile-size` is set. |

### Device

| Flag | Description |
|------|-------------|
| `--device DEVICE` | CUDA device to use (default: `cuda:0`). |
