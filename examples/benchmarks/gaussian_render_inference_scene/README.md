# Inference Rendering Benchmarks

This directory contains benchmarks for the inference rendering path, which uses
packed quaternion, scale, and opacity (QSO) scene representation. The benchmark
measures GPU latency and optionally compares image quality against the default
gsplat `rasterization()` renderer.

## Python Benchmark

Run from the repository root:

```bash
python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py
```

By default, the benchmark uses a synthetic scene. To use a trained model, pass
`--ply-path <path_to_point_cloud.ply>`. Pretrained scenes can be downloaded
with `python examples/datasets/download_dataset.py`.

Pass `--reference-tile-size 16` to enable the reference `rasterization()` path
and print speedup and image-quality (PSNR / LPIPS) comparisons.

Run with `--help` for the full list of options.
