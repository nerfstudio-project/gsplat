# AV Trainer: Multi-Camera Autonomous Driving with gsplat

Train 3D Gaussians on surround-view driving scenes using gsplat's rasterizer.
Supports pinhole cameras (PandaSet) and FTheta cameras with 3DGUT rendering
(NCore v4), with optional LiDAR rendering for geometric supervision.

NCore v4 dataset loading uses the shared `datasets/ncore.py` module
(`NCoreParser` + `NCoreDataset`). The av_trainer adds LiDAR rendering
via gsplat's native LiDAR renderer (`camera_model="lidar"`) with
`lidar_distance_loss` on top.

## Data Formats

### PandaSet (NPZ)

Pre-processed data with pinhole cameras. `prepare_pandaset.py` (in
`examples/`) either downloads a scene from the gated HuggingFace mirror
(requires `HF_TOKEN`) or converts a local PandaSet checkout with
`--pandaset-dir`:

```bash
HF_TOKEN=hf_... python examples/prepare_pandaset.py --download --scene 019
python examples/av_trainer.py --scene pandaset_019.npz --max-steps 15000
```

### NCore v4

Native FTheta camera data loaded via
[NCore SDK](https://github.com/NVIDIA/ncore):

```bash
git clone https://github.com/NVIDIA/ncore.git
pip install universal-pathlib "zarr<3" cbor2 simplejpeg dataclasses-json opencv-python-headless
export PYTHONPATH=/path/to/ncore:$PYTHONPATH
```

Download a clip from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --revision ncore_test --include "004c2001*" \
    --local-dir ncore_data
```

Train (the `...` stands for the full clip UUID; e.g.
`ncore_data/004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6/pai_004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6.json`):

```bash
python examples/av_trainer.py \
    --scene ncore_data/004c2001-.../pai_004c2001-...json \
    --cameras camera_front_wide_120fov \
    --duration 4.0 --downscale 4 --max-steps 10000
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `assets/test_pandaset.npz` | NPZ or NCore v4 JSON manifest |
| `--cameras` | all | Comma-separated camera IDs (NCore) |
| `--duration` | full | Clip duration in seconds (NCore) |
| `--downscale` | 1 | Image downscale factor (NCore) |
| `--max-lidar` | 150000 | Max LiDAR init points |
| `--max-steps` | 15000 | Training iterations |
| `--lr` | 0.005 | Base learning rate |
| `--mcmc` | off | Enable MCMC densification |
| `--cap-max` | 300000 | Max Gaussians (MCMC) |
| `--sh-degree` | 0 | SH degree (0=flat, 3=full) |
| `--sh-degree-interval` | 1000 | Steps between SH degree bumps (applies when `--sh-degree > 0`) |
| `--lidar-render` | off | Enable gsplat LiDAR renderer |
| `--lidar-render-subsample` | 112 | Subsample LiDAR rays |
| `--lidar-render-weight` | 0.0003 | LiDAR distance loss weight |

## Benchmark: NCore v4

Results on clip `004c2001`, `camera_front_wide_120fov`, 480x270 (ds4),
4s clip, 10K steps, 150K init / 300K cap. Timings on NVIDIA A40 (48 GB):

| # | Config                      | PSNR      | Gaussians | Time     |
|---|-----------------------------|-----------|-----------|----------|
| 1 | Baseline 3DGUT (ftheta)     | 26.46 dB  | 150K      | 6.1 min  |
| 2 | + MCMC                      | 27.97 dB  | 300K      | 22.8 min |
| 3 | + SH3                       | 29.36 dB  | 300K      | 26.8 min |
| 4 | + LiDAR rendering           | 29.54 dB  | 300K      | 28.8 min |

Incremental gains: MCMC +1.5 dB, SH3 +1.4 dB, LiDAR +0.2 dB (total +3.1 dB over baseline).

Note: MCMC uses random relocation; results may vary +/-0.5 dB between runs.
Reported numbers are from a single run on the commit shown above.

### Reproduce

```bash
SCENE=ncore_data/004c2001-.../pai_004c2001-...json
COMMON="--scene $SCENE --cameras camera_front_wide_120fov \
    --duration 4.0 --downscale 4 --max-steps 10000"

# 1. Baseline 3DGUT (ftheta)
python examples/av_trainer.py $COMMON --result-dir results/1_ftheta

# 2. + MCMC
python examples/av_trainer.py $COMMON --mcmc --result-dir results/2_mcmc

# 3. + SH3
python examples/av_trainer.py $COMMON --mcmc --sh-degree 3 --result-dir results/3_sh3

# 4. + LiDAR rendering
python examples/av_trainer.py $COMMON --mcmc --sh-degree 3 --lidar-render --result-dir results/4_lidar
```
