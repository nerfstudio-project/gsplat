<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Features: NHT (Neural Harmonic Textures)

NHT builds on the [3DGUT](3dgut.md) camera model (unscented transform + world-space evaluation) and requires `with_eval3d=True` and `with_ut=True`.

## Installing the optional dependency

NHT's deferred shader is built on top of [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) (`tcnn`). Because tcnn requires a working CUDA toolchain at install time, it is kept as an optional extra and is **not** installed by default. Install the extra with:

```bash
pip install "gsplat[nht]"
```

If `tinycudann` is missing at runtime, instantiating `gsplat.nht.DeferredShaderModule` / `DeferredShaderModuleAOV` raises a clear `ImportError` telling you to install the extra. The rest of `gsplat.nht` (the CUDA raster path, `HarmonicFeatures`, `NHTMCMCStrategy`, `NHTParams`, `export_splats_nht`) does not depend on `tcnn`.

## Build-time gate

The NHT CUDA kernels are gated by the `GSPLAT_BUILD_NHT` preprocessor flag (default on). Disable the extra translation units at build time by exporting `BUILD_NHT=0` before installing or rebuilding the extension.

---

## Key Differences from Plain gsplat (3DGS)

| Aspect | Standard gsplat (SH) | NHT |
|---|---|---|
| **Appearance model** | Per-Gaussian spherical harmonics (SH) coefficients | Per-Gaussian learned feature vectors decoded by a shared MLP |
| **Feature storage** | 48 SH coefficients (degree 3, 3 channels) | 4–80 feature values per primitive (divided among 4 tetrahedron vertices) |
| **Encoding** | None (direct SH evaluation) | Sine + cosine harmonic encoding of barycentrically interpolated features |
| **Decoding** | Direct color per-primitive, blended along ray | Deferred: features are blended along ray, then a shallow MLP decodes final RGB in image space |
| **Rasterization kernel** | Standard 3DGS or 3DGUT | Custom CUDA kernels for tetrahedral interpolation + harmonic encoding (forward and backward) |
| **MLP backend** | None | tiny-cuda-nn JIT-compiled cooperative-vector MLPs (FP16 inference) |
| **View dependence** | SH basis evaluated per-primitive | SH-encoded ray direction appended to blended features, decoded jointly by MLP |
---

## How to Use

### For users running `examples` in gsplat

#### Training

NHT training uses `simple_trainer_nht.py` with the MCMC densification strategy:

```bash
# Standard NHT training (48-feature, 1M primitives, MipNeRF 360)
cd examples
CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py mcmc \
    --disable_viewer --data_factor 4 \
    --strategy.cap-max 1000000 \
    --render_traj_path ellipse \
    --data_dir data/360_v2/garden/ \
    --result_dir results/nht_garden/
```

> **Tip:** Cross-platform convenience wrappers (`scripts/train.{sh,ps1}`, `scripts/view.{sh,ps1}`, `scripts/eval.{sh,ps1}`) and dataset download helpers are maintained in the [neural-harmonic-textures superrepo](https://github.com/nv-tlabs/neural-harmonic-textures), which vendors this gsplat fork as a submodule.

##### Dataset backends

`simple_trainer_nht.py` supports the same dataset backends as the upstream `simple_trainer.py`, selected via `--data_type`:

| `--data_type` | Source | `--data_dir` points to |
|---|---|---|
| `colmap` (default) | COLMAP / Mip-NeRF 360 layout | Scene directory containing `sparse/0/` and images |
| `ncore` | NCore v4 sequence | Meta-JSON file (e.g. `path/to/sequence.json`) |

For NCore sequences, the relevant flags are the same as in `simple_trainer.py` (camera/point-cloud source IDs, component group names, etc.). Lidar (or radar / native point cloud) sources are used only to seed the initial Gaussian point cloud via the NCore parser; there is no per-frame lidar supervision.

```bash
# NCore example
CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py mcmc \
    --disable_viewer \
    --data_type ncore \
    --data_dir path/to/sequence.json \
    --ncore_camera_ids cam_front \
    --ncore_lidar_ids lidar_top \
    --result_dir results/nht_ncore/
```

#### Key NHT-Specific Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--deferred_opt_feature_dim` | `48` | Total feature dimensionality per primitive (divided among 4 tetrahedron vertices, i.e. 12 per vertex) |
| `--deferred_features_lr` | `0.015` | Learning rate for the per-primitive features |
| `--deferred_mlp_lr` | `0.00072` | Learning rate for the deferred MLP |
| `--deferred_mlp_hidden_dim` | `128` | Width of each hidden layer in the deferred MLP |
| `--deferred_mlp_num_layers` | `3` | Number of hidden layers in the deferred MLP |
| `--deferred_mlp_ema` | `True` | Enable Exponential Moving Average on MLP weights (decay=0.95) |
| `--deferred_opt_center_ray_encoding` | `False` | Use per-tile center ray instead of per-pixel ray for view encoding. **Recommended for high parallax and densely captured scenes** |
| `--deferred_opt_view_encoding_type` | `"sh"` | View encoding type: `"sh"` (spherical harmonics) or `"fourier"` |
| `--deferred_opt_sh_degree` | `3` | SH degree for view direction encoding |
| `--deferred_opt_sh_scale` | `3.0` | Scale applied to normalized directions before SH evaluation |
| `--deferred_lr_scheduler` | `"cosine"` | LR schedule for features and MLP: `"cosine"` or `"exponential"` |
| `--deferred_features_lr_decay_final` | `0.1` | Final multiplier for feature LR decay (e.g. 0.1 = ends at 10% of initial) |
| `--deferred_mlp_lr_decay_final` | `0.1` | Final multiplier for MLP LR decay |
| `--color_refine_steps` | `3000` | Number of steps at end of training where geometry is frozen and only features + MLP are optimized |
| `--opacity_reg` | `0.02` | Opacity regularization weight |
| `--scale_reg` | `0.01` | Scale regularization weight |
| `--ssim_lambda` | `0.2` | D-SSIM weight in the loss (vs. L1) |
| `--tile_size` | `16` | Rasterization tile size. Lower to 8 if running out of shared memory with large `feature_dim` |

#### Rendering / Viewing

Once trained, view the result interactively with the NHT viewer:

```bash
CUDA_VISIBLE_DEVICES=0 python simple_viewer_nht.py \
    --ckpt results/nht_garden/ckpts/ckpt_29999_rank0.pt
```

The viewer starts a [viser](https://viser.studio/) server. Open `http://localhost:8080` in your browser to interact with the model. PowerShell wrappers (`scripts/view.{sh,ps1}`) are available in the [neural-harmonic-textures superrepo](https://github.com/nv-tlabs/neural-harmonic-textures).

**Viewer render modes** (selectable in the UI dropdown):

| Mode | Description |
|---|---|
| `rgb` | Final decoded RGB color (features → MLP → color) |
| `depth(accumulated)` | Accumulated z-depth (alpha-weighted sum of depths) |
| `depth(expected)` | Expected depth (accumulated depth normalized by alpha) |
| `normal` | Rendered surface normals (when the model was trained with normal supervision) |
| `alpha` | Accumulated opacity / transmittance map |

Additional viewer controls: near/far plane, radius clip, 2D epsilon, background color, anti-aliasing mode (`classic` / `antialiased`), camera model (`pinhole` / `ortho` / `fisheye`), and depth colormap.

#### Evaluation (Quality Metrics + Runtime Benchmark)

Evaluate a trained checkpoint with quality metrics (PSNR, SSIM, LPIPS) and optionally render a trajectory video:

```bash
CUDA_VISIBLE_DEVICES=0 python simple_trainer_nht.py mcmc \
    --disable_viewer \
    --data_dir data/360_v2/garden/ \
    --data_factor 4 \
    --result_dir results/nht_garden/ \
    --ckpt results/nht_garden/ckpts/ckpt_29999_rank0.pt
```

Results are saved to `<result_dir>/stats/val_step<N>.json` with PSNR, SSIM, LPIPS, and number of Gaussians.

For a **standalone runtime benchmark** that measures rasterization, deferred MLP, and total time separately (`benchmark_nht.py`), and for `scripts/eval.{sh,ps1}` wrappers that combine quality eval with runtime timing, see the [neural-harmonic-textures superrepo](https://github.com/nv-tlabs/neural-harmonic-textures).

---

## Rendering Modes (API level)

The `rasterization()` function supports several rendering modes via its arguments:

| `render_mode` | Output |
|---|---|
| `"RGB"` | Rendered colors only (features in NHT mode, SH colors otherwise) |
| `"D"` | Accumulated depth only |
| `"ED"` | Expected depth only |
| `"RGB+D"` | Colors + accumulated depth (concatenated on last dim) |
| `"RGB+ED"` | Colors + expected depth |

| `rasterize_mode` | Description |
|---|---|
| `"classic"` | Standard rasterization |
| `"antialiased"` | Applies a view-dependent compensation factor for anti-aliasing |

| `camera_model` | Description |
|---|---|
| `"pinhole"` | Standard pinhole camera (default) |
| `"ortho"` | Orthographic projection |
| `"fisheye"` | Fisheye lens model |
| `"ftheta"` | F-theta fisheye model |

---

## Reproducing Paper Results

The paper evaluates on three standard benchmarks: **MipNeRF 360**, **Tanks & Temples**, and **Deep Blending**. The scripts shipped inside this gsplat tree cover the **Mip-NeRF 360** portion of each paper table; for full reproduction across all three benchmarks (plus PowerShell wrappers, AOV mode, and the standalone runtime benchmark) see the [neural-harmonic-textures superrepo](https://github.com/nv-tlabs/neural-harmonic-textures), which vendors this fork as a submodule.

| Inside gsplat (Mip-NeRF 360 only) | Superrepo (Mip-NeRF 360 + T&T + Deep Blending) |
|---|---|
| `examples/benchmarks/nht/mcmc.sh` | `benchmarks/nht/benchmark_nht.sh` / `.ps1` |
| `examples/benchmarks/nht/mcmc_split.sh` | `benchmarks/nht/benchmark_nht_split.sh` / `.ps1` |
| `examples/benchmarks/nht/mcmc_high.sh` | `benchmarks/nht/benchmark_nht_high.sh` / `.ps1` |

All Mip-NeRF 360 commands below assume you have downloaded the dataset to `data/360_v2/` and run them from inside `gsplat/examples/`.

### Table 2 — Controlled Comparison (1M primitives, 30k steps)

This table compares our method against 3DGS+SH, 3DGUT+SH, 3DGUT+Spher.Voronoi, and 3DGUT+NHT under identical conditions: 1M primitives, 48 appearance parameters per primitive, 30k training steps, same hyperparameters for all scenes.

```bash
# Mip-NeRF 360 portion (from gsplat/examples/)
bash benchmarks/nht/mcmc.sh

# Run specific scenes only — edit SCENE_LIST at the top of the script,
# or override the environment variable when invoked from the superrepo.
```

**Measured results (RTX A6000 Ada):**

| Method (w/ MCMC) | M360 PSNR | M360 SSIM | M360 LPIPS | T&T PSNR | T&T SSIM | T&T LPIPS | DB PSNR | DB SSIM | DB LPIPS |
|---|---|---|---|---|---|---|---|---|---|
| 3DGS + SH | 27.94 | 0.829 | 0.246 | 24.25 | 0.861 | 0.188 | 29.98 | 0.912 | 0.317 |
| 3DGUT + SH | 27.93 | 0.828 | 0.247 | 23.99 | 0.859 | 0.192 | 30.21 | 0.913 | 0.318 |
| 3DGUT + NHT (Ours) | **28.46** | **0.830** | **0.232** | **24.79** | **0.875** | **0.169** | **30.88** | **0.918** | **0.311** |

### Table 1 — Split-Strategy Benchmark (Best Quality, Per-Dataset Config)

This is the highest-quality configuration from the paper. Different primitive counts, training lengths, and ray-encoding settings are used per dataset group:

| Dataset Group | Primitives | Steps | Ray Encoding | Data Factor |
|---|---|---|---|---|
| M360 Outdoor | 4.5M | 20k | per-pixel ray | 4 |
| M360 Indoor | 2M | 45k | center ray | 2 |
| Tanks & Temples | 2.5M | 40k | center ray | 1 |
| Deep Blending | 2M | 30k | center ray | 1 |

```bash
# Mip-NeRF 360 portion (from gsplat/examples/)
bash benchmarks/nht/mcmc_split.sh
```

**Measured results:**

| Dataset | PSNR | SSIM | LPIPS |
|---|---|---|---|
| M360 Outdoor Avg | 25.34 | 0.755 | 0.236 |
| M360 Indoor Avg | 33.59 | 0.947 | 0.183 |
| **M360 Total** | **29.01** | **0.840** | **0.212** |
| **T&T Avg** | **25.68** | **0.882** | **0.141** |
| **DB Avg** | **30.94** | **0.919** | **0.302** |

Per-scene breakdown:

| Scene | PSNR | SSIM | LPIPS |
|---|---|---|---|
| garden | 28.33 | 0.881 | 0.106 |
| bicycle | 25.88 | 0.788 | 0.216 |
| stump | 27.06 | 0.795 | 0.223 |
| treehill | 23.36 | 0.670 | 0.306 |
| flowers | 22.08 | 0.640 | 0.330 |
| bonsai | 35.73 | 0.964 | 0.192 |
| counter | 31.00 | 0.934 | 0.193 |
| kitchen | 33.59 | 0.945 | 0.123 |
| room | 34.03 | 0.947 | 0.221 |
| truck | 26.91 | 0.900 | 0.112 |
| train | 24.45 | 0.865 | 0.169 |
| drjohnson | 30.43 | 0.918 | 0.309 |
| playroom | 31.45 | 0.921 | 0.296 |

### Table 7 — High Primitive Count (Per-Scene 3DGS Caps)

Matches the per-scene primitive counts from the original 3DGS-MCMC (e.g. garden=5.2M, bicycle=5.9M, bonsai=1.3M, etc.), using 64 features, 128x3 MLP, and center-ray encoding:

```bash
# Mip-NeRF 360 portion (from gsplat/examples/)
bash benchmarks/nht/mcmc_high.sh
```

### Optional extensions in the superrepository

The [neural-harmonic-textures superrepo](https://github.com/nv-tlabs/neural-harmonic-textures) vendors this gsplat fork as a submodule and adds workflows that are out of scope for this document:

- Convenience wrappers (`scripts/train.{sh,ps1}`, `scripts/view.{sh,ps1}`, `scripts/eval.{sh,ps1}`)
- Full multi-dataset paper-reproduction scripts (M360 + Tanks & Temples + Deep Blending) under `benchmarks/nht/`
- Standalone runtime timing tool (`benchmarks/benchmark_nht.py`)
- AOV extensions for arbitrary output variables (LSEG / DINOv3 / RGB2X) under `aov/`

---

## Paper Setup

### Hardware

All paper results were measured on an **NVIDIA RTX A6000 Ada** (48 GB, Ada Lovelace architecture, compute capability 8.9).

### Software Stack

| Component | Version / Detail |
|---|---|
| **Framework** | gsplat (this repository) |
| **PyTorch** | >= 2.0 with CUDA support |
| **CUDA** | >= 12.0 (required for tiny-cuda-nn cooperative vectors) |
| **tiny-cuda-nn** | JIT-compiled cooperative-vector MLPs, FP16 inference |
| **Python** | >= 3.8 |
| **GPU requirement** | Ada Lovelace or newer recommended (RTX 4090, A6000 Ada, L40, etc.); Ampere GPUs (A100, RTX 3090) also work but cooperative-vector MLPs may fall back to standard GEMM |

### WARNING ON LPIPS METRIC
- **LPIPS metric**: Uses VGG backbone with inputs normalized to [-1, 1]. This is different to INRIA 3DGS, which does not normalize the inputs (incorrect).

---

## For users using gsplat's API

The relevant arguments in `rasterization()` are:

- `nht=True` — enables the NHT rasterization path (tetrahedral interpolation + frequency encoding)
- `with_eval3d=True`, `with_ut=True` — required (NHT builds on the 3DGUT pipeline)
- `center_ray_mode=True/False` — whether to use per-pixel or center-of-tile ray directions
- `ray_dir_scale` — scaling factor for the appended ray direction channels

The deferred shading module lives in `gsplat.nht`:

```python
from gsplat.nht import DeferredShaderModule, HarmonicFeatures
from gsplat.rendering import rasterization

# Rasterize features + ray directions
renders, alphas, meta = rasterization(
    means, quats, scales, opacities, features,
    viewmats, Ks, width, height,
    nht=True, with_eval3d=True, with_ut=True,
    sh_degree=None,
)
# renders[..., :-3] = encoded features, renders[..., -3:] = ray dirs

# Decode to RGB with the deferred shader
rgb = deferred_shader(renders)
```

### Full Example: Loading a Checkpoint and Rendering

```python
import torch
from gsplat.rendering import rasterization
from gsplat.nht.deferred_shader import DeferredShaderModule

device = torch.device("cuda:0")
ckpt = torch.load("results/garden/ckpts/ckpt_29999_rank0.pt", map_location=device)
splats = {k: v.to(device) for k, v in ckpt["splats"].items()}

# Restore deferred module
dm_state = ckpt["deferred_module"]
dm = DeferredShaderModule(**dm_state["config"]).to(device)
dm.load_state_dict(dm_state["state_dict"])
if "ema" in dm_state:
    for n, p in dm.named_parameters():
        if n in dm_state["ema"]:
            p.data.copy_(dm_state["ema"][n])
dm.eval()

# Prepare splats
means = splats["means"]
quats = torch.nn.functional.normalize(splats["quats"], p=2, dim=-1)
scales = torch.exp(splats["scales"])
opacities = torch.sigmoid(splats["opacities"])
features = splats["features"].half()

# Rasterize
with torch.no_grad():
    render_colors, render_alphas, info = rasterization(
        means=means, quats=quats, scales=scales,
        opacities=opacities, colors=features,
        viewmats=viewmat[None], Ks=K[None],
        width=W, height=H,
        nht=True, with_eval3d=True, with_ut=True,
        sh_degree=None,
        center_ray_mode=dm.center_ray_encoding,
        ray_dir_scale=dm.ray_dir_scale,
    )
    rgb, extras = dm(render_colors)
    rgb = rgb[0].clamp(0, 1)
```
