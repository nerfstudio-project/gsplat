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

#### Offline `dynamic_flag` preparation

`prepare_ncore_dynamics.py` materializes a derived NCore v4 clip with per-point
`dynamic_flag` arrays on the LiDAR store, using annotation-time LiDAR-spin
association (same semantics as rigid-dynamic static exclusion):

```bash
python examples/prepare_ncore_dynamics.py \
    --input ncore_data/004c2001-.../pai_004c2001-...json \
    --output-dir ncore_data/004c2001-...-dynamic \
    --class-ids automobile,person,heavy_truck
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | (required) | NCore v4 JSON manifest |
| `--output-dir` | (required) | New directory for the derived manifest and component stores |
| `--class-ids` | default actor classes | Comma-separated cuboid class IDs |
| `--lidar-id` | unset | LiDAR sensor ID; required only when the manifest has multiple LiDARs |

Constraints:

- `--output-dir` must not already exist (`FileExistsError` on rerun).
- `--lidar-id` is required when the manifest has multiple LiDARs; otherwise the
  sole LiDAR is chosen automatically.
- The selected LiDAR must own its own component store (no mixed or multi-LiDAR
  stores).

**Producer / consumer semantics.** The prepare tool writes an offline export
artifact (`dynamic_flag` per LiDAR return). `av_trainer.py` / `NCoreParser`
intentionally does **not** read baked `dynamic_flag`; it recomputes static
exclusion live from cuboid annotations when `--rigid-dynamic-track-class-ids`
is set. A plain `av_trainer.py --scene X.json` (no rigid classes) does **not**
drop baked-dynamic returns — exclusion only happens when rigid classes are
requested. A runtime warning may be emitted when a flag-carrying manifest is
loaded without rigid classes.

**Disk layout.** The tool writes a full derived clip directory under
`--output-dir`. Unchanged component stores (e.g. camera `.itar` files) are
hard-linked into the output (falling back to symlink, then copy); only the
rewritten LiDAR `dynamic_flag` store is newly written, so the derived clip is
not a full duplicate of the source. Across filesystems, unchanged stores may
still be fully copied — plan disk accordingly (~2 GB per clip on typical
multi-camera manifests).

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | `assets/test_pandaset.npz` | NPZ or NCore v4 JSON manifest |
| `--cameras` | all | Comma-separated camera IDs (NCore) |
| `--duration` | full | Clip duration in seconds (NCore) |
| `--downscale` | 1 | Image downscale factor (NCore) |
| `--rigid-dynamic-track-class-ids` | unset | Comma-separated NCore cuboid class IDs to load as rigid dynamic tracks in a dynamic scene |
| `--rigid-dynamic-static-baseline` | off | Keep selected moving-object returns in the static scene instead of splitting them into rigid components |
| `--max-lidar` | 150000 | Max initial LiDAR points; with rigid dynamics this is the total across static and dynamic scenes |
| `--max-steps` | 15000 | Training iterations |
| `--lr` | 0.005 | Base learning rate |
| `--mcmc` | off | Enable MCMC densification |
| `--cap-max` | 300000 | Max Gaussians (MCMC) |
| `--sh-degree` | 0 | SH degree (0=flat, 3=full) |
| `--sh-degree-interval` | 1000 | Steps between SH degree bumps (applies when `--sh-degree > 0`) |
| `--lidar-render` | off | Enable gsplat LiDAR renderer |
| `--lidar-render-subsample` | 112 | Subsample LiDAR rays |
| `--lidar-render-weight` | 0.0003 | LiDAR distance loss weight |

## Outputs

Unless `--no-save-model` is passed, training writes `model.pt` with the same
schema as the training checkpoints — `{"scene_id", "splats"}` — so
`sample_inference.py` can load it directly. Rigid dynamic runs additionally
store `"render_scene_id"` and `"scenes"` (one `GaussianScene.state_dict()` per
member scene, including pose context and the transform graph). All tensors in
`model.pt` are CPU-native, so the export loads on machines without CUDA.

`sample_inference.py` rebuilds the static + dynamic collection from the saved
scenes and renders each NCore camera at its frame-midpoint timestamp. The
checkpoint records the training-time camera set, clip duration, and downscale
(the scene frame's origin depends on them), and `sample_inference.py` uses the
recorded values by default; `--cameras`/`--duration`/`--downscale` override.

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

Periodic evaluations run after completed training steps `k * eval_every` and at
`max_steps`. Artifact names and checkpoint metadata use the same completed-step
number, so a step boundary produces matching `stats/stepNNNNN.json` and
`ckpts/ckpt_NNNNN.pt` artifacts.

Each stats file records the evaluation `mean_psnr`, and `summary.json` includes
that value in its `checkpoints` entries. Final evaluation renders are written
under `renders/`.
