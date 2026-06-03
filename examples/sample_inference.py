# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Render a trained AV scene from various viewpoints using Stage + GaussianScene.

Loads a checkpoint and dispatches on ``--scene`` to one of three formats
(PandaSet ``.npz``, NCore ``.json``, or COLMAP directory).

Usage:
    python examples/sample_inference.py \\
        --ckpt results/showcase/ckpts/ckpt_00500.pt \\
        --scene assets/test_pandaset.npz \\
        --output-dir results/showcase
"""

from __future__ import annotations

import argparse
import math
import os

import imageio
import numpy as np
import torch

from examples.av_trainer import (
    get_render_params,
    load_scene_npz,
    render_gaussians,
)
from datasets.colmap import Parser as ColmapParser
from datasets.ncore import NCoreParser
from gsplat_scene import GaussianScene
from gsplat_stage import Stage


def load_stage(ckpt_path: str, device: torch.device) -> tuple[Stage, str, int | None]:
    """Load a gsplat checkpoint into a Stage."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "scene_id" not in checkpoint:
        raise ValueError(f"Checkpoint {ckpt_path} missing 'scene_id'.")
    splats_state = checkpoint["splats"]
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v, requires_grad=False) for k, v in splats_state.items()}
    )
    scene = GaussianScene.from_splats(splats, id=checkpoint["scene_id"])
    stage = Stage()
    stage.add_scene(scene, render_gaussians)

    sh_degree = None
    if "shN" in splats_state:
        # shN stores (deg+1)^2 - 1 rows; sh0 is the missing row.
        sh_degree = int(math.isqrt(splats_state["shN"].shape[1] + 1)) - 1
    print(
        f"Loaded checkpoint (step {checkpoint.get('step', '?')}): "
        f"{scene.num_gaussians()} Gaussians, sh_degree={sh_degree}"
    )
    return stage, scene.id, sh_degree


# ---------------------------------------------------------------------------
# Camera extraction (PandaSet npz, NCore json, or COLMAP directory)
# ---------------------------------------------------------------------------


def _cam_entry(
    c2w: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    label: str,
    render_kwargs: dict | None = None,
) -> dict:
    return {
        "viewmat": torch.linalg.inv(c2w).unsqueeze(0),
        "K": K if K.ndim == 3 else K.unsqueeze(0),
        "W": W,
        "H": H,
        "label": label,
        "render_kwargs": render_kwargs or {},
    }


def _extract_cameras_colmap(
    data_dir: str,
    device: torch.device,
    frame_filter: list[int] | None,
) -> dict[str, dict]:
    parser = ColmapParser(data_dir=data_dir, factor=4, normalize=True)
    n = len(parser.camtoworlds)
    indices = frame_filter if frame_filter is not None else [0, n // 2, n - 1]
    cameras: dict[str, dict] = {}
    for fi in indices:
        if not (0 <= fi < n):
            continue
        cam_id = parser.camera_ids[fi]
        c2w = torch.from_numpy(parser.camtoworlds[fi]).float().to(device)
        K = torch.from_numpy(parser.Ks_dict[cam_id]).float().to(device)
        W, H = parser.imsize_dict[cam_id]
        cameras[f"colmap_f{fi}"] = _cam_entry(
            c2w, K, W, H, f"colmap frame {fi} ({parser.image_names[fi]})"
        )
    return cameras


def _extract_cameras_ncore(
    json_path: str,
    device: torch.device,
    camera_filter: list[str] | None,
    frame_filter: list[int] | None,
) -> dict[str, dict]:
    """Pull poses + FTheta params straight from NCoreParser (no image decode)."""
    parser = NCoreParser(
        meta_json_path=json_path,
        factor=0.25,
        test_every=8,
        camera_ids=camera_filter or None,
        duration_sec=4.0,
    )
    n = len(parser.frame_list)
    indices = frame_filter if frame_filter is not None else [0, n // 2, n - 1]
    cam_render_data = [parser.camera_render_data[c] for c in parser.camera_ids]
    cameras: dict[str, dict] = {}
    for fi in indices:
        if not (0 <= fi < n):
            continue
        cam_id, _frame_idx = parser.frame_list[fi]
        cam_idx = parser.camera_idx_per_frame[fi]
        c2w = torch.from_numpy(parser.camtoworlds[fi]).float().to(device)
        K = torch.from_numpy(parser.Ks_dict[cam_id]).float().to(device)
        W, H = parser.imsize_dict[cam_id]
        cameras[f"{cam_id}_f{fi}"] = _cam_entry(
            c2w,
            K,
            W,
            H,
            f"{cam_id} (frame {fi})",
            render_kwargs=get_render_params(cam_render_data, cam_idx, device),
        )
    return cameras


def _extract_cameras_pandaset(
    npz_path: str,
    device: torch.device,
    camera_filter: list[str] | None,
    frame_filter: list[int] | None,
    include_synthetic: bool,
) -> dict[str, dict]:
    scene = load_scene_npz(npz_path, device=str(device))
    frames = frame_filter if frame_filter is not None else [0]
    cameras: dict[str, dict] = {}

    # Real cameras: selected cameras x selected frames.
    for fi in frames:
        if not (0 <= fi < scene.n_frames):
            print(f"  Warning: frame {fi} out of range [0, {scene.n_frames}), skipping")
            continue
        for ci in range(scene.n_cams):
            name = scene.camera_names[ci]
            if camera_filter and name not in camera_filter:
                continue
            cameras[f"{name}_f{fi}"] = _cam_entry(
                torch.linalg.inv(scene.viewmats[fi, ci]),
                scene.Ks[ci, 0],
                scene.W,
                scene.H,
                f"{name} (frame {fi})",
            )

    if not include_synthetic or not cameras:
        return cameras

    # Synthetic: chase cam offset 5m back / 3m up from the front camera.
    ref = frames[0] if frames[0] < scene.n_frames else 0
    front_c2w = torch.linalg.inv(scene.viewmats[ref, 0]).clone()
    forward = front_c2w[:3, 2]  # OpenGL camera looks down +z
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    chase_pos = front_c2w[:3, 3] - forward * 5.0 + up * 3.0
    look_dir = front_c2w[:3, 3] - chase_pos
    look_dir = look_dir / look_dir.norm()
    right = torch.linalg.cross(look_dir, up)
    # Degenerate when the front camera points along world ``up`` (look_dir ‖ up).
    # Practically never true for ground-driving AV scenes; assert so synthetic
    # rigs that violate it fail loudly instead of producing NaN renders.
    assert (
        right.norm() > 1e-6
    ), "chase-cam basis is degenerate: front camera is aligned with world up"
    right = right / right.norm()
    cam_up = torch.linalg.cross(right, look_dir)
    chase_c2w = torch.eye(4, device=device)
    chase_c2w[:3, 0] = right
    chase_c2w[:3, 1] = cam_up
    chase_c2w[:3, 2] = look_dir
    chase_c2w[:3, 3] = chase_pos
    K_chase = scene.Ks[1, 0].clone()
    K_chase[0, 2] = scene.W / 2.0
    K_chase[1, 2] = scene.H / 2.0
    cameras["chase_cam"] = _cam_entry(
        chase_c2w,
        K_chase,
        scene.W,
        scene.H,
        "Chase cam (synthetic, 5m back + 3m up)",
    )

    # Synthetic: zoomed front camera (2x focal).
    K_zoom = scene.Ks[0, 0].clone()
    K_zoom[0, 0] *= 2
    K_zoom[1, 1] *= 2
    cameras["zoomed_front"] = _cam_entry(
        torch.linalg.inv(scene.viewmats[ref, 0]),
        K_zoom,
        scene.W,
        scene.H,
        "Zoomed front (2x focal length)",
    )
    return cameras


def extract_cameras(
    scene_path: str,
    device: torch.device,
    camera_filter: list[str] | None = None,
    frame_filter: list[int] | None = None,
    include_synthetic: bool = True,
) -> dict[str, dict]:
    if scene_path.endswith(".json"):
        return _extract_cameras_ncore(scene_path, device, camera_filter, frame_filter)
    if os.path.isdir(scene_path):
        return _extract_cameras_colmap(scene_path, device, frame_filter)
    return _extract_cameras_pandaset(
        scene_path, device, camera_filter, frame_filter, include_synthetic
    )


def render_gallery(
    stage: Stage,
    scene_id: str,
    cameras: dict[str, dict],
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nRendering {len(cameras)} viewpoints to {output_dir}/")
    for name, cam in cameras.items():
        renders, alphas, *_ = stage.render(
            scene_id,
            viewmat=cam["viewmat"],
            K=cam["K"],
            W=cam["W"],
            H=cam["H"],
            **cam["render_kwargs"],
        )
        rgb_np = (renders[0, :, :, :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        alpha_np = (alphas[0, :, :, 0].cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(output_dir, f"{name}.png"), rgb_np)
        imageio.imwrite(os.path.join(output_dir, f"{name}_alpha.png"), alpha_np)
        print(f"  {name:40s} → {cam['label']}")
    print(f"\n{len(cameras)} renders saved to {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a trained AV scene from various viewpoints",
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to AV checkpoint (.pt)"
    )
    parser.add_argument(
        "--scene",
        type=str,
        required=True,
        help="path to scene (npz, json, or COLMAP dir)",
    )
    parser.add_argument("--output-dir", type=str, default="results/showcase")
    parser.add_argument("--cameras", type=str, nargs="+", default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=None)
    parser.add_argument("--no-synthetic", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        stage, scene_id, sh_degree = load_stage(args.ckpt, device)
        cameras = extract_cameras(
            args.scene,
            device,
            camera_filter=args.cameras,
            frame_filter=args.frames,
            include_synthetic=not args.no_synthetic,
        )
        if sh_degree is not None:
            for cam in cameras.values():
                cam["render_kwargs"]["sh_degree_to_use"] = sh_degree
        render_gallery(stage, scene_id, cameras, args.output_dir)


if __name__ == "__main__":
    main()
