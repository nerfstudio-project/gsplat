# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-camera autonomous driving trainer with gsplat.

Trains 3D Gaussians on surround-view driving scenes using NCore v4 data
(FTheta cameras + 3DGUT) or PandaSet NPZ (pinhole cameras). Follows the
same patterns as simple_trainer.py for NCore dataset loading, with the
addition of gsplat's native LiDAR rendering for geometric supervision.

See examples/AV_TRAINER.md for full documentation, arguments, and benchmarks.

Additions over simple_trainer.py:
  - PandaSet NPZ loading (load_scene_npz)
  - LiDAR rendering via gsplat (camera_model="lidar") with lidar_distance_loss
  - Sky masking and sky penalty in loss
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import gsplat
from gsplat_scene import GaussianScene
from gsplat_stage import Stage
from gsplat.strategy import MCMCStrategy

# ---------------------------------------------------------------------------
# PandaSet NPZ loading
# ---------------------------------------------------------------------------


def load_scene_npz(path: str, device: str = "cuda") -> SimpleNamespace:
    """Load a scene from npz file and prepare per-camera tensors."""
    data = dict(np.load(path, allow_pickle=True))

    scene = SimpleNamespace()
    scene.images = torch.from_numpy(data["images"]).float().to(device) / 255.0
    cam_intrinsics = torch.from_numpy(data["cam_intrinsics"]).float().to(device)
    cam_to_worlds = torch.from_numpy(data["cam_to_worlds"]).float().to(device)
    scene.lidar_points = torch.from_numpy(data["lidar_points"]).float().to(device)
    scene.lidar_frame_indices = (
        torch.from_numpy(data["lidar_frame_indices"]).long().to(device)
    )
    scene.is_test = data["is_test"]
    scene.n_frames, scene.n_cams, scene.H, scene.W, _ = scene.images.shape
    scene.camera_names = list(
        data.get("camera_names", [f"cam{i}" for i in range(scene.n_cams)])
    )

    # Per-camera K matrices [N_cams, 1, 3, 3]
    scene.Ks = torch.zeros(scene.n_cams, 1, 3, 3, device=device)
    scene.Ks[:, 0, 0, 0] = cam_intrinsics[:, 0]  # horizontal focal length
    scene.Ks[:, 0, 1, 1] = cam_intrinsics[:, 1]  # vertical focal length
    scene.Ks[:, 0, 0, 2] = cam_intrinsics[:, 2]  # x principal point
    scene.Ks[:, 0, 1, 2] = cam_intrinsics[:, 3]  # y principal point
    scene.Ks[:, 0, 2, 2] = 1.0

    scene.viewmats = torch.linalg.inv(cam_to_worlds)

    return scene


def load_scene(path: str, device: str = "cuda") -> SimpleNamespace:
    """Compatibility wrapper for PandaSet-style NPZ scenes."""
    return load_scene_npz(path, device=device)


# ---------------------------------------------------------------------------
# LiDAR rendering helpers (addition over simple_trainer.py)
#
# gsplat supports camera_model="lidar" for rasterizing Gaussians along
# actual LiDAR rays. This section builds the structured LiDAR model
# parameters and GT distance maps from NCore data.
# ---------------------------------------------------------------------------


def _build_lidar_renderer(ncore_lidar_model, device: str = "cuda"):
    """Build gsplat LiDAR renderer params from NCore LiDAR model."""
    from gsplat.cuda._lidar import (
        RowOffsetStructuredSpinningLidarModelParameters as GsplatLidarParams,
        SpinningDirection,
        compute_angles_to_columns_map,
        compute_tiling,
    )
    from gsplat.cuda._wrapper import (
        RowOffsetStructuredSpinningLidarModelParametersExt,
    )

    spin_dir = (
        SpinningDirection.CLOCKWISE
        if ncore_lidar_model.spinning_direction in ("cw", "CLOCKWISE")
        else SpinningDirection.COUNTER_CLOCKWISE
    )
    params = GsplatLidarParams(
        row_elevations_rad=torch.from_numpy(ncore_lidar_model.row_elevations_rad)
        .float()
        .to(device),
        column_azimuths_rad=torch.from_numpy(ncore_lidar_model.column_azimuths_rad)
        .float()
        .to(device),
        row_azimuth_offsets_rad=torch.from_numpy(
            ncore_lidar_model.row_azimuth_offsets_rad
        )
        .float()
        .to(device),
        spinning_frequency_hz=float(ncore_lidar_model.spinning_frequency_hz),
        spinning_direction=spin_dir,
    )
    a2c_map = compute_angles_to_columns_map(params)
    tiling = compute_tiling(params)
    return RowOffsetStructuredSpinningLidarModelParametersExt(
        params,
        a2c_map.to(device),
        tiling,
    )


def _load_ncore_lidar_gt(
    lidar_sensor,
    parser,
    n_lidar: int,
    n_rows: int,
    n_columns: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build structured GT distance maps and viewmats for LiDAR rendering loss.

    Uses parser._ncore_world_to_scene_poses() for coordinate frame consistency
    with camera poses (world -> world_global -> scene).
    """
    import ncore.data

    gt_dist = torch.full((n_lidar, n_rows, n_columns, 1), float("nan"), device=device)
    valid = torch.zeros(
        (n_lidar, n_rows, n_columns, 1), dtype=torch.bool, device=device
    )
    viewmats = torch.zeros((n_lidar, 4, 4), device=device)
    viewmats_rs = torch.zeros(
        (n_lidar, 4, 4), device=device
    )  # end-of-frame for rolling shutter

    for lfi in range(n_lidar):
        me = lidar_sensor.get_frame_ray_bundle_model_element(lfi)
        if me is None:
            continue
        dist = lidar_sensor.get_frame_ray_bundle_return_distance_m(lfi, return_index=0)
        rows, cols = me[:, 0].astype(np.int64), me[:, 1].astype(np.int64)
        gt_dist[lfi, rows, cols, 0] = torch.from_numpy(dist).float().to(device)
        valid[lfi, rows, cols, 0] = True

        # Start pose (mid-frame viewmat for rasterization)
        T_start = lidar_sensor.get_frames_T_sensor_target(
            "world",
            lfi,
            frame_timepoint=ncore.data.FrameTimepoint.START,
        ).astype(np.float32)
        T_start_scene = parser._ncore_world_to_scene_poses(
            T_start.reshape(1, 4, 4)
        ).reshape(4, 4)
        viewmats[lfi] = torch.linalg.inv(
            torch.from_numpy(T_start_scene).float().to(device)
        )

        # End pose (rolling shutter end-of-frame)
        T_end = lidar_sensor.get_frames_T_sensor_target(
            "world",
            lfi,
            frame_timepoint=ncore.data.FrameTimepoint.END,
        ).astype(np.float32)
        T_end_scene = parser._ncore_world_to_scene_poses(
            T_end.reshape(1, 4, 4)
        ).reshape(4, 4)
        viewmats_rs[lfi] = torch.linalg.inv(
            torch.from_numpy(T_end_scene).float().to(device)
        )

    return gt_dist, valid, viewmats, viewmats_rs


def setup_lidar_renderer(
    meta_json_path: str,
    parser,
    device: str,
    duration_sec: float | None,
    subsample: int,
    weight: float,
) -> SimpleNamespace | None:
    """Build LiDAR renderer from NCore data for distance loss."""
    from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
    from ncore.data import (
        RowOffsetStructuredSpinningLidarModelParameters as NCoreLidarParams,
    )

    if not parser.lidar_ids:
        return None

    loader = SequenceLoaderV4(
        SequenceComponentGroupsReader(
            [Path(meta_json_path)],
            open_consolidated=True,
        )
    )
    lidar_sensor = loader.get_lidar_sensor(parser.lidar_ids[0])
    lidar_model = lidar_sensor.model_parameters
    if not isinstance(lidar_model, NCoreLidarParams):
        return None

    n_lidar = lidar_sensor.frames_count
    if duration_sec is not None:
        cam0 = loader.get_camera_sensor(parser.camera_ids[0])
        cam_ts = cam0.frames_timestamps_us
        start_us = cam_ts[0, 0]
        stop_us = min(start_us + int(duration_sec * 1e6), cam_ts[-1, 1])
        lidar_ts = lidar_sensor.frames_timestamps_us[:, 1]
        n_lidar = min(n_lidar, int(np.searchsorted(lidar_ts, stop_us)) + 1)

    lidar_coeffs = _build_lidar_renderer(lidar_model, device=device)
    gt_dist, valid_mask, lidar_viewmats, lidar_viewmats_rs = _load_ncore_lidar_gt(
        lidar_sensor,
        parser,
        n_lidar,
        lidar_model.n_rows,
        lidar_model.n_columns,
        device,
    )

    n_valid = int(valid_mask[0].sum())
    n_kept = n_valid // subsample
    print(
        f"  LiDAR renderer: {lidar_model.n_rows}x{lidar_model.n_columns}, "
        f"{n_lidar} frames, {n_kept}/{n_valid} rays/frame (sub={subsample}), "
        f"weight={weight}, rolling_shutter=True, far_plane=200m"
    )
    return SimpleNamespace(
        coeffs=lidar_coeffs,
        gt_distances=gt_dist,
        valid_masks=valid_mask,
        viewmats=lidar_viewmats,
        viewmats_rs=lidar_viewmats_rs,
        n_frames=n_lidar,
        n_rows=lidar_model.n_rows,
        n_columns=lidar_model.n_columns,
        subsample=subsample,
        loss_weight=weight,
    )


# ---------------------------------------------------------------------------
# Gaussian initialization
# ---------------------------------------------------------------------------

SH_C0 = 0.28209479177387814


def init_gaussians(
    points: torch.Tensor,
    colors: torch.Tensor,
    device: str = "cuda",
    sh_degree: int = 0,
) -> torch.nn.ParameterDict:
    """Initialize Gaussians from point cloud (LiDAR or SfM)."""
    N = points.shape[0]
    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points.clone()),
            "scales": torch.nn.Parameter(torch.full((N, 3), -5.0, device=device)),
            "quats": torch.nn.Parameter(
                F.normalize(torch.randn(N, 4, device=device), dim=-1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((N,), 0.1, device=device))
            ),
        }
    )
    if sh_degree > 0:
        K = (sh_degree + 1) ** 2
        sh0 = ((colors.clamp(0, 1) - 0.5) / SH_C0).unsqueeze(1)
        shN = torch.zeros(N, K - 1, 3, device=device)
        params["sh0"] = torch.nn.Parameter(sh0)
        params["shN"] = torch.nn.Parameter(shN)
    else:
        params["colors"] = torch.nn.Parameter(colors.clamp(0, 1).clone())
    return params


def init_gaussians_from_lidar(
    scene: SimpleNamespace,
    device: str = "cuda",
    sh_degree: int = 0,
) -> torch.nn.ParameterDict:
    """Initialize one Gaussian per train-frame PandaSet LiDAR point."""
    train_mask = ~scene.is_test
    train_frame_ids = torch.where(torch.from_numpy(train_mask))[0]
    pts_mask = torch.zeros(len(scene.lidar_points), dtype=torch.bool, device=device)
    for fid in train_frame_ids:
        pts_mask |= scene.lidar_frame_indices == fid.item()
    points = scene.lidar_points[pts_mask, :3]
    intensities = scene.lidar_points[pts_mask, 3:4]
    colors = intensities.expand(-1, 3)
    return init_gaussians(points, colors, device, sh_degree)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def estimate_sky_mask(image, height_fraction=0.4, brightness_threshold=0.85):
    """Estimate sky mask from image brightness (addition over simple_trainer)."""
    H = image.shape[0]
    upper = int(H * height_fraction)
    gray = image.mean(dim=-1)
    mask = torch.zeros_like(gray, dtype=torch.bool)
    mask[:upper] = gray[:upper] > brightness_threshold
    return mask


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt).item()
    return float("inf") if mse == 0 else -10.0 * np.log10(mse)


def get_render_params(ncore_camera_data, camera_idx, device):
    """Extract per-camera rendering params (same pattern as simple_trainer.rasterize_splats)."""
    cam = ncore_camera_data[camera_idx]
    kwargs = {
        "camera_model": cam.camera_model,
        "ftheta_coeffs": cam.ftheta_coeffs,
    }
    if cam.radial_coeffs is not None:
        kwargs["radial_coeffs"] = (
            torch.from_numpy(cam.radial_coeffs).to(device).unsqueeze(0)
        )
    if cam.tangential_coeffs is not None:
        kwargs["tangential_coeffs"] = (
            torch.from_numpy(cam.tangential_coeffs).to(device).unsqueeze(0)
        )
    if cam.thin_prism_coeffs is not None:
        kwargs["thin_prism_coeffs"] = (
            torch.from_numpy(cam.thin_prism_coeffs).to(device).unsqueeze(0)
        )
    return kwargs


def render_gaussians(
    viewmat,
    K,
    W,
    H,
    render_mode="RGB",
    sh_degree_to_use=None,
    camera_model="pinhole",
    ftheta_coeffs=None,
    radial_coeffs=None,
    tangential_coeffs=None,
    thin_prism_coeffs=None,
    *,
    splats: torch.nn.ParameterDict,
):
    """Rasterize Gaussians (matches simple_trainer.rasterize_splats pattern).

    Returns (renders, alphas, info, activated_scales, activated_opacities).
    Activations are computed once here to avoid redundant computation in loss.

    ``splats`` is a required keyword-only argument; Stage supplies it via
    ``splats=scene.splats`` on every dispatch.
    """

    if "sh0" in splats:
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
    else:
        colors = splats["colors"]

    # The unscented transform path is also required for pinhole cameras that
    # carry non-zero distortion coefficients (OpenCV radial/tangential/thin
    # prism); gsplat.rasterization otherwise asserts.
    use_ut = camera_model in ("ftheta", "fisheye") or any(
        c is not None
        for c in (ftheta_coeffs, radial_coeffs, tangential_coeffs, thin_prism_coeffs)
    )
    packed = not use_ut

    activated_scales = torch.exp(splats["scales"])
    activated_opacities = torch.sigmoid(splats["opacities"])

    renders, alphas, info = gsplat.rasterization(
        splats["means"],
        F.normalize(splats["quats"], dim=-1),
        activated_scales,
        activated_opacities,
        colors,
        viewmat,
        K,
        W,
        H,
        camera_model=camera_model,
        sh_degree=sh_degree_to_use,
        packed=packed,
        render_mode=render_mode,
        with_ut=use_ut,
        with_eval3d=use_ut,
        ftheta_coeffs=ftheta_coeffs,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
    )
    return renders, alphas, info, activated_scales, activated_opacities


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def create_optimizers(params, lr):
    lr_map = {
        "means": lr * 0.032,
        "scales": lr,
        "quats": lr * 0.2,
        "opacities": lr * 10,
    }
    if "sh0" in params:
        lr_map["sh0"] = lr * 0.5
        lr_map["shN"] = lr * 0.5 / 20
    else:
        lr_map["colors"] = lr * 0.5
    return {
        name: torch.optim.Adam(
            [{"params": params[name], "lr": lr_map[name], "name": name}],
            eps=1e-15,
        )
        for name in lr_map
    }


def log_training_step(
    step: int, max_steps: int, total_loss: torch.Tensor, start_time: float
) -> None:
    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    print(
        f"Step {step}/{max_steps} | loss={total_loss.item():.4f} | "
        f"{max(1, step) / elapsed:.0f} it/s | {mem_peak:.0f} MB"
    )


def prepare_output_dirs(
    result_dir: str, include_checkpoints: bool = False
) -> tuple[str, str, str | None]:
    os.makedirs(result_dir, exist_ok=True)
    renders_dir = os.path.join(result_dir, "renders")
    stats_dir = os.path.join(result_dir, "stats")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    ckpt_dir = None
    if include_checkpoints:
        ckpt_dir = os.path.join(result_dir, "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)

    return renders_dir, stats_dir, ckpt_dir


def checkpoint_name(step: int) -> str:
    return f"ckpt_{step:05d}.pt"


def gaussian_scene_from_checkpoint(checkpoint: dict) -> GaussianScene:
    splats_state = checkpoint.get("splats")
    if splats_state is None or (
        "colors" not in splats_state and "sh0" not in splats_state
    ):
        raise ValueError(
            "Checkpoint does not contain AV-compatible splats: "
            "expected 'splats' with 'colors' or 'sh0' entries."
        )
    if "scene_id" not in checkpoint:
        raise ValueError(
            "Checkpoint missing 'scene_id'. Re-train with the current av_trainer "
            "to produce a checkpoint that records scene_id."
        )

    splats = torch.nn.ParameterDict(
        {key: torch.nn.Parameter(value.clone()) for key, value in splats_state.items()}
    )
    return GaussianScene.from_splats(splats, id=checkpoint["scene_id"])


def save_checkpoint(
    gaussian_scene: GaussianScene,
    scene_path: str,
    ckpt_dir: str,
    step: int,
) -> str:
    ckpt_path = os.path.join(ckpt_dir, checkpoint_name(step))
    torch.save(
        {
            "step": step,
            "scene_path": scene_path,
            "scene_id": gaussian_scene.id,
            "splats": gaussian_scene.splats.state_dict(),
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    ckpt_path: str,
    device: str | torch.device = "cpu",
) -> tuple[int, GaussianScene, str | None]:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "step" not in checkpoint:
        raise ValueError("Checkpoint is missing required 'step' field.")
    return (
        int(checkpoint["step"]),
        gaussian_scene_from_checkpoint(checkpoint),
        checkpoint.get("scene_path"),
    )


def evaluate(
    stage: Stage,
    gaussian_scene: GaussianScene,
    scene: SimpleNamespace,
    test_frame_ids: list[int],
    W: int,
    H: int,
    renders_dir: str | None = None,
    sh_degree: int | None = None,
) -> float:
    # Route through `stage.render` rather than calling `render_gaussians`
    # directly so checkpoint eval shares the same forward path as `train()`
    # (line ~860). Any future Stage-side change (e.g. an extra preprocess
    # hook, a wrapped renderer) then applies to checkpoint eval automatically
    # instead of silently diverging.
    psnrs = []
    with torch.no_grad():
        for tfid in test_frame_ids:
            for tci in range(scene.n_cams):
                gt = scene.images[tfid, tci].unsqueeze(0)
                vm = scene.viewmats[tfid, tci].unsqueeze(0)
                K = scene.Ks[tci]
                rd, _, _, _, _ = stage.render(
                    scene_id=gaussian_scene.id,
                    viewmat=vm,
                    K=K,
                    W=W,
                    H=H,
                    sh_degree_to_use=sh_degree,
                )
                psnrs.append(compute_psnr(rd[0].clamp(0, 1), gt[0]))

                if renders_dir is not None:
                    gt_np = (gt[0].cpu().numpy() * 255).astype(np.uint8)
                    rd_np = (rd[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                    canvas = np.concatenate([gt_np, rd_np], axis=1)
                    imageio.imwrite(
                        os.path.join(
                            renders_dir, f"eval_f{tfid}_{scene.camera_names[tci]}.png"
                        ),
                        canvas,
                    )
    return float(np.mean(psnrs)) if psnrs else 0.0


def evaluate_checkpoint(
    ckpt_path: str,
    scene_path: str | None,
    result_dir: str,
) -> dict:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    renders_dir, stats_dir, _ = prepare_output_dirs(result_dir)

    step, gaussian_scene, checkpoint_scene_path = load_checkpoint(ckpt_path, device)
    if scene_path is None:
        scene_path = checkpoint_scene_path
    if scene_path is None:
        raise ValueError("A scene path is required when the checkpoint has none.")
    if (
        scene_path.endswith(".json")
        or scene_path.endswith(".zarr.itar")
        or Path(scene_path).is_dir()
    ):
        raise ValueError("--ckpt evaluation currently supports PandaSet NPZ scenes.")

    scene = load_scene(scene_path, device=device)
    test_frame_ids = [i for i in range(scene.n_frames) if scene.is_test[i]]
    start_time = time.time()
    result = {
        "step": step,
        "scene_path": scene_path,
        "source_checkpoint": ckpt_path,
        "num_gaussians": gaussian_scene.splats["means"].shape[0],
    }
    # Build a Stage exactly as `train()` does (line ~787) so the eval forward
    # path is identical to the training forward path.
    stage = Stage()
    stage.add_scene(gaussian_scene, render_gaussians)
    if test_frame_ids:
        result["mean_psnr"] = evaluate(
            stage,
            gaussian_scene,
            scene,
            test_frame_ids,
            scene.W,
            scene.H,
            renders_dir=renders_dir,
        )
        print(
            f"  Eval PSNR: {result['mean_psnr']:.2f} dB "
            f"({len(test_frame_ids) * scene.n_cams} views)"
        )

    result["elapsed_s"] = time.time() - start_time
    result["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6 if use_cuda else 0.0
    with open(os.path.join(stats_dir, f"step{step:05d}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "loaded_checkpoint": ckpt_path,
                "scene_path": scene_path,
                "num_gaussians": result["num_gaussians"],
                "checkpoints": [result],
            },
            f,
            indent=2,
        )
    print(f"Results saved to {result_dir}/")
    return result


def train(
    scene_path: str,
    max_steps: int,
    lr: float,
    log_every: int,
    eval_every: int,
    result_dir: str,
    use_mcmc: bool = False,
    cap_max: int = 300_000,
    sh_degree: int = 0,
    sh_degree_interval: int = 1000,
    save_model: bool = True,
    # NCore-specific
    cameras: list[str] | None = None,
    duration: float | None = None,
    max_lidar: int = 150_000,
    downscale: int = 1,
    # LiDAR rendering (addition over simple_trainer)
    lidar_render: bool = False,
    lidar_render_subsample: int = 112,
    lidar_render_weight: float = 0.0003,
):
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    renders_dir, stats_dir, ckpt_dir = prepare_output_dirs(
        result_dir, include_checkpoints=True
    )

    # --- Detect data format and load ---
    # NCore path follows simple_trainer.py: NCoreParser + NCoreDataset
    is_ncore = (
        scene_path.endswith(".json")
        or scene_path.endswith(".zarr.itar")
        or Path(scene_path).is_dir()
    )

    ncore_camera_data = None
    lidar_r = None

    if is_ncore:
        from datasets.ncore import NCoreParser, NCoreDataset

        parser = NCoreParser(
            meta_json_path=scene_path,
            factor=1.0 / downscale if downscale > 1 else 1.0,
            test_every=8,
            camera_ids=cameras or None,
            duration_sec=duration,
            max_lidar_points=max_lidar,
        )
        # Pre-stacking below requires every selected camera to share (W, H).
        # Mixed-resolution surround-view rigs (e.g. wide 120fov + tele 30fov)
        # would raise at torch.stack time; fail early with a clear message.
        cam_sizes = {cid: parser.imsize_dict[cid] for cid in parser.camera_ids}
        if len(set(cam_sizes.values())) > 1:
            raise ValueError(
                f"av_trainer requires uniform (W, H) across selected cameras; "
                f"got {cam_sizes}. Pass a single camera via --cameras, or "
                f"train per-camera."
            )

        trainset = NCoreDataset(parser, split="train")
        valset = NCoreDataset(parser, split="val")

        # Pre-load all frames into contiguous GPU tensors for fast training.
        # simple_trainer.py uses DataLoader with lazy loading; we pre-load
        # since av_trainer scenes are small enough to fit in GPU memory.
        print(f"  Pre-loading {len(trainset)} train + {len(valset)} val frames...")
        t0 = time.time()
        train_raw = [trainset[i] for i in range(len(trainset))]
        val_raw = [valset[i] for i in range(len(valset))]

        def _stack_to_gpu(samples, dev):
            images = torch.stack([d["image"] for d in samples]).float().to(dev) / 255.0
            viewmats = torch.linalg.inv(
                torch.stack([d["camtoworld"] for d in samples]).float().to(dev)
            )
            Ks = torch.stack([d["K"] for d in samples]).to(dev)
            cam_idx = [d["camera_idx"] for d in samples]
            masks = None
            if "mask" in samples[0]:
                masks = torch.stack([d["mask"] for d in samples]).to(dev)
            return images, viewmats, Ks, cam_idx, masks

        (
            train_images,
            train_viewmats,
            train_Ks,
            train_cam_idx,
            train_masks,
        ) = _stack_to_gpu(train_raw, device)
        val_images, val_viewmats, val_Ks, val_cam_idx, _val_masks = _stack_to_gpu(
            val_raw, device
        )
        # Pre-compute sky masks (deterministic function of GT images)
        train_sky_masks = torch.stack(
            [estimate_sky_mask(train_images[i]) for i in range(len(train_raw))]
        ).to(device)
        print(f"  Pre-loaded in {time.time() - t0:.1f}s")

        # Per-camera render data (same as simple_trainer.ncore_camera_data)
        ncore_camera_data = [
            parser.camera_render_data[cam_id] for cam_id in parser.camera_ids
        ]

        # Gaussian init from LiDAR (same as simple_trainer init_type="lidar")
        points = torch.from_numpy(parser.points).float().to(device)
        colors = torch.from_numpy(parser.points_rgb).float().to(device) / 255.0
        params = init_gaussians(points, colors, device, sh_degree)

        first_cam = parser.camera_ids[0]
        W, H = parser.imsize_dict[first_cam]
        n_train = len(train_raw)
        n_val = len(val_raw)

        # LiDAR renderer (addition over simple_trainer)
        if lidar_render:
            lidar_r = setup_lidar_renderer(
                scene_path,
                parser,
                device,
                duration,
                lidar_render_subsample,
                lidar_render_weight,
            )

        print(
            f"Device: {device} | NCore {parser.sequence_id} | "
            f"{n_train} train, {n_val} val views, {W}x{H} | "
            f"{len(parser.points)} init pts | {len(params['means'])} Gaussians"
        )
    else:
        # PandaSet NPZ path (not in simple_trainer)
        scene = load_scene(scene_path, device=device)
        train_mask = ~scene.is_test
        params = init_gaussians_from_lidar(scene, device=device, sh_degree=sh_degree)
        H, W = scene.H, scene.W
        n_train = (
            sum(1 for fi in range(scene.n_frames) if train_mask[fi]) * scene.n_cams
        )
        n_val = (
            sum(1 for fi in range(scene.n_frames) if scene.is_test[fi]) * scene.n_cams
        )
        print(
            f"Device: {device} | PandaSet | "
            f"{n_train} train, {n_val} val views, {W}x{H} | "
            f"{len(scene.lidar_points)} init pts | {len(params['means'])} Gaussians"
        )

    gaussian_scene = GaussianScene.from_splats(params, id="av_scene")
    stage = Stage()
    stage.add_scene(gaussian_scene, render_gaussians)
    splats = gaussian_scene.splats

    N = len(splats["means"])
    optimizers = create_optimizers(splats, lr)

    # MCMC (same as simple_trainer)
    strategy = None
    strategy_state = None
    if use_mcmc:
        strategy = MCMCStrategy(
            cap_max=cap_max,
            noise_lr=5e5,
            refine_start_iter=500,
            refine_stop_iter=int(max_steps * 0.8),
            refine_every=100,
            verbose=True,
        )
        strategy.check_sanity(splats, optimizers)
        strategy_state = strategy.initialize_state()
        print(f"MCMC: cap_max={cap_max}")

    # LR schedule (same as simple_trainer)
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        ),
    ] + [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=max_steps)
        for name in optimizers
        if name != "means"
    ]

    losses_history = []
    checkpoints = []
    start_time = time.time()

    # --- Training loop ---
    for step in range(max_steps):
        if is_ncore:
            # Data access pattern: pre-loaded contiguous GPU tensors
            idx = step % n_train
            pixels = train_images[idx].unsqueeze(0)
            viewmat = train_viewmats[idx].unsqueeze(0)
            K = train_Ks[idx].unsqueeze(0)
            mask = train_masks[idx] if train_masks is not None else None
            camera_idx = train_cam_idx[idx]

            # Per-camera distortion (same as simple_trainer.rasterize_splats)
            rp = get_render_params(ncore_camera_data, camera_idx, device)
            cam_model = rp.pop("camera_model")
        else:
            # PandaSet: direct tensor indexing
            fi, ci = divmod(step % n_train, scene.n_cams)
            train_frames = [i for i in range(scene.n_frames) if not scene.is_test[i]]
            fi = train_frames[fi % len(train_frames)]
            pixels = scene.images[fi, ci].unsqueeze(0)
            viewmat = scene.viewmats[fi, ci].unsqueeze(0)
            K = scene.Ks[ci]
            mask = None
            cam_model = "pinhole"
            rp = {}

        height, width = pixels.shape[1], pixels.shape[2]
        sh_degree_to_use = (
            min(step // sh_degree_interval, sh_degree) if sh_degree > 0 else None
        )

        # Forward (matches simple_trainer pattern). PandaSet gets an extra depth
        # channel ("RGB+ED") so sparse-LiDAR depth supervision below can sample it.
        render_mode = "RGB" if is_ncore else "RGB+ED"
        renders, alphas, info, act_scales, act_opacities = stage.render(
            scene_id=gaussian_scene.id,
            viewmat=viewmat,
            K=K,
            W=width,
            H=height,
            render_mode=render_mode,
            sh_degree_to_use=sh_degree_to_use,
            camera_model=cam_model,
            **rp,
        )
        colors_render = renders[..., :3]
        depths_render = None if is_ncore else renders[..., 3:4]

        # Loss (similar to simple_trainer: L1 + SSIM)
        # Addition: sky masking and ego mask support
        sky_mask = train_sky_masks[idx] if is_ncore else estimate_sky_mask(pixels[0])
        non_sky = (~sky_mask).to(torch.int32).unsqueeze(0)
        if mask is not None:
            non_sky = non_sky * mask.to(torch.int32).unsqueeze(0)

        per_pixel_l1 = gsplat.losses.l1_loss(colors_render, pixels).mean(dim=-1)
        l1loss = gsplat.losses.reduce_mean(per_pixel_l1, mask=non_sky)

        ssimloss = gsplat.losses.ssim_loss(
            colors_render.clamp(0, 1).permute(0, 3, 1, 2),
            pixels.permute(0, 3, 1, 2),
        )
        loss = 0.8 * l1loss + 0.2 * ssimloss

        # Regularization (same as simple_trainer, using pre-computed activations)
        loss += 0.005 * gsplat.losses.gaussian_scale_reg(act_scales).mean()
        loss += 0.005 * gsplat.losses.gaussian_density_reg(act_opacities).mean()

        # Sky penalty (addition over simple_trainer)
        coverage = alphas[0, :, :, 0]
        loss += 0.001 * (coverage * sky_mask.float()).mean()

        # LiDAR rendering loss (addition over simple_trainer)
        if lidar_r is not None:
            lfi = step % lidar_r.n_frames
            lidar_vm = lidar_r.viewmats[lfi].unsqueeze(0)
            lidar_vm_rs = lidar_r.viewmats_rs[lfi].unsqueeze(0)
            lidar_K = torch.eye(3, device=device).unsqueeze(0)
            N_gs = splats["means"].shape[0]
            dummy_colors = torch.zeros(N_gs, 3, device=device)

            lidar_renders, _, _ = gsplat.rasterization(
                splats["means"],
                F.normalize(splats["quats"], dim=-1),
                torch.exp(splats["scales"]),
                torch.sigmoid(splats["opacities"]),
                dummy_colors,
                lidar_vm,
                lidar_K,
                lidar_r.n_columns,
                lidar_r.n_rows,
                camera_model="lidar",
                lidar_coeffs=lidar_r.coeffs,
                with_ut=True,
                with_eval3d=True,
                packed=False,
                render_mode="RGB-Ed",  # expected hit distance (along-ray), not z-depth
                global_z_order=False,  # spinning sensor: no single camera-Z axis
                far_plane=200.0,
                rolling_shutter=gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
                viewmats_rs=lidar_vm_rs,
            )
            pred_dist = lidar_renders[0, :, :, 3:4]
            gt_dist = lidar_r.gt_distances[lfi]
            lmask = lidar_r.valid_masks[lfi]
            if lidar_r.subsample > 1:
                valid_idx = lmask.reshape(-1).nonzero(as_tuple=True)[0]
                keep = valid_idx[:: lidar_r.subsample]
                sub_mask = torch.zeros_like(lmask.reshape(-1), dtype=torch.bool)
                sub_mask[keep] = True
                lmask = sub_mask.reshape(lmask.shape)
            loss = loss + lidar_r.loss_weight * gsplat.losses.lidar_distance_loss(
                pred_dist, gt_dist, valid_mask=lmask
            )

        # PandaSet: sparse LiDAR depth supervision via simple_trainer's pattern
        # (project LiDAR points to the camera, sample rendered depth at those
        # pixels, L1 in disparity). NCore uses native LiDAR rendering above.
        if not is_ncore:
            pts_mask = scene.lidar_frame_indices == fi
            if pts_mask.any():
                lidar_world = scene.lidar_points[pts_mask, :3]
                ones = torch.ones_like(lidar_world[:, :1])
                homog = torch.cat([lidar_world, ones], dim=-1)
                pts_cam = (viewmat[0] @ homog.T).T[:, :3]
                uv_hom = (K[0] @ pts_cam.T).T
                uv = uv_hom[:, :2] / uv_hom[:, 2:3]
                depth_gt = pts_cam[:, 2]
                keep = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < width)
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < height)
                    & (depth_gt > 0)
                )
                uv = uv[keep]
                depth_gt = depth_gt[keep]
                if len(depth_gt) > 0:
                    grid = torch.stack(
                        [
                            uv[:, 0] / (width - 1) * 2 - 1,
                            uv[:, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    ).reshape(1, -1, 1, 2)
                    depth_pred = F.grid_sample(
                        depths_render.permute(0, 3, 1, 2),
                        grid,
                        align_corners=True,
                    ).reshape(-1)
                    loss = loss + 0.01 * gsplat.losses.depth_l1_loss(
                        depth_pred, depth_gt
                    )

        # Backward (same as simple_trainer)
        for opt in optimizers.values():
            opt.zero_grad()
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        for sched in schedulers:
            sched.step()
        losses_history.append(loss.item())

        # MCMC (same as simple_trainer)
        if strategy is not None:
            strategy.step_post_backward(
                splats,
                optimizers,
                strategy_state,
                step,
                info={},
                lr=schedulers[0].get_last_lr()[0],
                scene=gaussian_scene,
            )
        N = len(splats["means"])

        # Logging
        if step % log_every == 0:
            elapsed = time.time() - start_time
            mem_peak = torch.cuda.max_memory_allocated() / 1e6
            print(
                f"Step {step}/{max_steps} | loss={loss.item():.4f} | "
                f"{max(1, step) / elapsed:.0f} it/s | {mem_peak:.0f} MB"
            )

        # Evaluation
        if (
            eval_every > 0 and step > 0 and step % eval_every == 0
        ) or step == max_steps - 1:
            is_last = step == max_steps - 1
            psnrs = []
            with torch.no_grad():
                if is_ncore:
                    for vi in range(n_val):
                        gt = val_images[vi].unsqueeze(0)
                        vm = val_viewmats[vi].unsqueeze(0)
                        Kv = val_Ks[vi].unsqueeze(0)
                        ci = val_cam_idx[vi]
                        erp = get_render_params(ncore_camera_data, ci, device)
                        ecm = erp.pop("camera_model")
                        val_h, val_w = gt.shape[1], gt.shape[2]
                        rd, _, _, _, _ = stage.render(
                            scene_id=gaussian_scene.id,
                            viewmat=vm,
                            K=Kv,
                            W=val_w,
                            H=val_h,
                            sh_degree_to_use=sh_degree_to_use,
                            camera_model=ecm,
                            **erp,
                        )
                        psnrs.append(compute_psnr(rd[0].clamp(0, 1), gt[0]))
                        if is_last and renders_dir:
                            gt_np = (gt[0].cpu().numpy() * 255).astype(np.uint8)
                            rd_np = (rd[0].clamp(0, 1).cpu().numpy() * 255).astype(
                                np.uint8
                            )
                            canvas = np.concatenate([gt_np, rd_np], axis=1)
                            cam_id, frame_idx = parser.frame_list[valset.indices[vi]]
                            imageio.imwrite(
                                os.path.join(
                                    renders_dir, f"eval_f{frame_idx}_{cam_id}.png"
                                ),
                                canvas,
                            )
                else:
                    test_ids = [i for i in range(scene.n_frames) if scene.is_test[i]]
                    for tfid in test_ids:
                        for tci in range(scene.n_cams):
                            gt = scene.images[tfid, tci].unsqueeze(0)
                            vm = scene.viewmats[tfid, tci].unsqueeze(0)
                            Kv = scene.Ks[tci]
                            rd, _, _, _, _ = stage.render(
                                scene_id=gaussian_scene.id,
                                viewmat=vm,
                                K=Kv,
                                W=W,
                                H=H,
                                sh_degree_to_use=sh_degree_to_use,
                            )
                            psnrs.append(compute_psnr(rd[0].clamp(0, 1), gt[0]))
                            if is_last and renders_dir:
                                gt_np = (gt[0].cpu().numpy() * 255).astype(np.uint8)
                                rd_np = (rd[0].clamp(0, 1).cpu().numpy() * 255).astype(
                                    np.uint8
                                )
                                canvas = np.concatenate([gt_np, rd_np], axis=1)
                                imageio.imwrite(
                                    os.path.join(
                                        renders_dir,
                                        f"eval_f{tfid}_{scene.camera_names[tci]}.png",
                                    ),
                                    canvas,
                                )

            elapsed = time.time() - start_time
            mem_peak = torch.cuda.max_memory_allocated() / 1e6
            mean_psnr = float(np.mean(psnrs)) if psnrs else 0.0
            ckpt_path = save_checkpoint(gaussian_scene, scene_path, ckpt_dir, step + 1)
            ckpt = {
                "step": step + 1,
                "loss": loss.item(),
                "elapsed_s": elapsed,
                "gpu_peak_mb": mem_peak,
                "num_gaussians": N,
                "mean_psnr": mean_psnr,
                "checkpoint_path": ckpt_path,
            }
            print(f"  Eval PSNR: {mean_psnr:.2f} dB ({len(psnrs)} views)")
            with open(os.path.join(stats_dir, f"step{step + 1:05d}.json"), "w") as f:
                json.dump(ckpt, f, indent=2)
            checkpoints.append(ckpt)

    # Summary
    elapsed = time.time() - start_time
    summary = {
        "max_steps": max_steps,
        "total_time_s": elapsed,
        "initial_loss": losses_history[0] if losses_history else None,
        "final_loss": losses_history[-1] if losses_history else None,
        "num_gaussians": N,
        "checkpoints": checkpoints,
    }
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if save_model:
        model_path = os.path.join(result_dir, "model.pt")
        torch.save({k: v.detach().cpu() for k, v in splats.items()}, model_path)
        print(f"Model saved to {model_path} ({N} Gaussians)")

    print(f"\nDone: {max_steps} steps in {elapsed:.1f}s")
    print(f"Results saved to {result_dir}/")
    return losses_history, checkpoints


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    default_scene_path = os.path.join(
        os.path.dirname(__file__), "../assets/test_pandaset.npz"
    )
    p = argparse.ArgumentParser(description="AV trainer with gsplat")
    p.add_argument(
        "--scene",
        type=str,
        default=None,
        help="NPZ file or NCore v4 JSON manifest",
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to an AV checkpoint for eval/render only",
    )
    p.add_argument(
        "--cameras", type=str, default=None, help="comma-separated camera IDs (NCore)"
    )
    p.add_argument(
        "--duration", type=float, default=None, help="clip duration in seconds (NCore)"
    )
    p.add_argument("--max-lidar", type=int, default=150_000)
    p.add_argument("--downscale", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--log-every", type=int, default=3000)
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="evaluate every N steps (0 = only at end)",
    )
    p.add_argument("--result-dir", type=str, default="results/av")
    p.add_argument("--sh-degree", type=int, default=0)
    p.add_argument("--sh-degree-interval", type=int, default=1000)
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--mcmc", action="store_true")
    p.add_argument("--cap-max", type=int, default=300_000)
    # LiDAR rendering (addition over simple_trainer)
    p.add_argument(
        "--lidar-render",
        action="store_true",
        help="enable gsplat LiDAR rendering with distance loss",
    )
    p.add_argument("--lidar-render-subsample", type=int, default=112)
    p.add_argument("--lidar-render-weight", type=float, default=0.0003)
    args = p.parse_args()

    if args.ckpt is not None:
        evaluate_checkpoint(
            ckpt_path=args.ckpt,
            scene_path=args.scene,
            result_dir=args.result_dir,
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for gsplat rasterization.")

    cameras_list = args.cameras.split(",") if args.cameras else None
    train(
        scene_path=args.scene or default_scene_path,
        max_steps=args.max_steps,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        result_dir=args.result_dir,
        use_mcmc=args.mcmc,
        cap_max=args.cap_max,
        sh_degree=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        save_model=not args.no_save_model,
        cameras=cameras_list,
        duration=args.duration,
        max_lidar=args.max_lidar,
        downscale=args.downscale,
        lidar_render=args.lidar_render,
        lidar_render_subsample=args.lidar_render_subsample,
        lidar_render_weight=args.lidar_render_weight,
    )


if __name__ == "__main__":
    main()
