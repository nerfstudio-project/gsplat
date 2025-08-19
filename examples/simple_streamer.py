'''
A fork from simple_viewer.py to stream the scene.
'''

import argparse
import json
import math
import os
import sys
import time
import shutil

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from examples.gsplat_viewer import GsplatViewer, GsplatRenderTabState
from examples.config import Config, load_config_from_toml, merge_config
from examples.utils import get_color
from stream import frustum_culling, distance_culling

RENDER_MODE_MAP = {
                    "rgb": "RGB",
                    "depth(accumulated)": "D",
                    "depth(expected)": "ED",
                    "alpha": "RGB",
                }

def save_viewer_poses(poses_dir, saved_poses, only_highest_res=True):

    # Only save the highest resolution poses
    max_width = max(pose["width"] for pose in saved_poses)
    max_height = max(pose["height"] for pose in saved_poses)
    print(f"Max height: {max_height}, Max width: {max_width}")

    # Filter for highest resolution if requested
    if only_highest_res:
        saved_poses = [pose for pose in saved_poses if pose["width"] == max_width and pose["height"] == max_height]

    # Add sequential pose_id to each pose
    for i, pose in enumerate(saved_poses):
        pose["pose_id"] = i

    # Save all poses to a single JSON file
    poses_file = os.path.join(poses_dir, "viewer_poses.json")
    with open(poses_file, 'w') as f:
        json.dump(saved_poses, f, indent=2)
    print(f"Saved {len(saved_poses)} poses to {poses_file}")

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if cfg.ckpt is None:
        raise ValueError("ckpt is not provided")
    
    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    ckpt = torch.load(cfg.ckpt, map_location=device)["splats"]
    print("Checkpoint loaded from: ", cfg.ckpt)
    print("Checkpoint keys: ", ckpt.keys())
    means.append(ckpt["means"])
    quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
    scales.append(torch.exp(ckpt["scales"]))
    opacities.append(torch.sigmoid(ckpt["opacities"]))
    sh0.append(ckpt["sh0"])
    shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0) # [N, 3]
    quats = torch.cat(quats, dim=0) # [N, 4]
    scales = torch.cat(scales, dim=0) # [N, 3]
    opacities = torch.cat(opacities, dim=0) # [N]
    sh0 = torch.cat(sh0, dim=0) # [N, 1, 3]
    shN = torch.cat(shN, dim=0) # [N, S, 3], e.g. S = 15
    colors = torch.cat([sh0, shN], dim=-2) # [N, K, 3], e.g. K = 1 + S = 16
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", means.shape[0])

    # Create distance culling directory
    distance_culling_dir = os.path.join(cfg.result_dir, "distance_culling")
    if os.path.exists(distance_culling_dir):
        shutil.rmtree(distance_culling_dir)
    os.makedirs(distance_culling_dir)
    
    # Create directory and storage for viewer poses (if enabled)
    if cfg.save_viewer_poses:
        poses_dir = os.path.join(cfg.result_dir, "viewer_poses")
        if os.path.exists(poses_dir):
            shutil.rmtree(poses_dir)
        os.makedirs(poses_dir)
        
        # Storage for all poses
        saved_poses = []
    else:
        poses_dir = None
        saved_poses = []

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height)) # [3, 3]
        c2w = torch.from_numpy(c2w).float().to(device) # [4, 4]
        K = torch.from_numpy(K).float().to(device) # [3, 3]
        viewmat = c2w.inverse() # [4, 4]
        print(f"Rendering Image - Height: {height}, Width: {width}, Near Plane: {render_tab_state.near_plane}, Far Plane: {render_tab_state.far_plane}")

        # Calculate bounding box of all gaussians
        bbox_min = torch.min(means, dim=0)[0]  # [3]
        bbox_max = torch.max(means, dim=0)[0]  # [3]
        bbox_center = (bbox_min + bbox_max) / 2.0  # [3]
        
        # Calculate distance from camera to bounding box center
        distance_to_bbox_center = torch.norm(c2w[:3, 3] - bbox_center).item()
        print(f"Distance to bbox center: {distance_to_bbox_center:.3f} m")

        if cfg.save_viewer_poses:
            # Save current viewer pose for later evaluation
            pose_data = {
                "timestamp": time.time(),
                "frame_id": len(saved_poses),
                "c2w_matrix": c2w.cpu().numpy().tolist(),  # Camera-to-world transformation
                "K_matrix": K.cpu().numpy().tolist(),      # Intrinsic matrix
                "width": width,
                "height": height,
                "near_plane": render_tab_state.near_plane,
                "far_plane": render_tab_state.far_plane,
                "render_mode": render_tab_state.render_mode,
                "max_sh_degree": render_tab_state.max_sh_degree,
                "bbox_center": bbox_center.cpu().numpy().tolist(),
                "distance_to_bbox_center": distance_to_bbox_center,
            }
            saved_poses.append(pose_data)

        # Create local copies for rendering (and potential frustum culling)
        render_means = means
        render_quats = quats
        render_scales = scales
        render_opacities = opacities
        render_colors = colors
        num_original = render_means.shape[0]
        # Perform frustum culling
        if cfg.frustum_culling:
            start_time = time.time()
            culling_mask = frustum_culling(
                render_means, render_scales, viewmat, K, width, height, 
                near=render_tab_state.near_plane, far=render_tab_state.far_plane
            )
            
            # Filter Gaussians based on culling mask
            render_means = render_means[culling_mask]
            render_quats = render_quats[culling_mask]
            render_scales = render_scales[culling_mask]
            render_opacities = render_opacities[culling_mask]
            render_colors = render_colors[culling_mask]
            end_time = time.time()
            print(f"Frustum culling time: {(end_time - start_time) * 1000:.2f}ms")
            
            num_after_cull = render_means.shape[0]
            culling_ratio = (num_original - num_after_cull) / num_original * 100.0 # avoid division by zero
            
            print(f"#Gaussians: {num_original}, #Frustum Culled Gaussians: {num_original - num_after_cull}, #Rendered Gaussians: {num_after_cull}, #Frustum Culling Ratio: {culling_ratio:.1f}%")

        if cfg.distance_culling:
            start_time = time.time()

            culling_mask = distance_culling(
                render_means, render_quats, render_scales, render_opacities, viewmat, K, width, height, dump_file=None
            )
            render_means = render_means[culling_mask]
            render_quats = render_quats[culling_mask]
            render_scales = render_scales[culling_mask]
            render_opacities = render_opacities[culling_mask]
            render_colors = render_colors[culling_mask]
            end_time = time.time()
            print(f"Distance culling time: {(end_time - start_time) * 1000:.2f}ms")

            num_after_cull = render_means.shape[0]
            culling_ratio = (num_original - num_after_cull) / num_original * 100.0 # avoid division by zero
            print(f"#Gaussians: {num_original}, #Distance Culled Gaussians: {num_original - num_after_cull}, #Rendered Gaussians: {num_after_cull}, #Distance Culling Ratio: {culling_ratio:.1f}%")
        
        print(f"#Gaussians: {num_original}, #Culled Gaussians: {num_original - render_means.shape[0]}, #Rendered Gaussians: {render_means.shape[0]}, #Total Culling Ratio: {(num_original - render_means.shape[0]) / num_original * 100:.1f}%\n")

        if render_means.shape[0] == 0: # Otherwise, the rendering will crash
            print("No Gaussians left after frustum culling. Returning empty image.")
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        render_colors, render_alphas, info = rasterization(
            render_means,  # [N, 3]
            render_quats,  # [N, 4]
            render_scales,  # [N, 3]
            render_opacities,  # [N]
            render_colors,  # [N, K, 3]
            viewmat[None],  # [4, 4]
            K[None],  # [3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=cfg.with_ut,
            with_eval3d=cfg.with_eval3d,
        )
        render_tab_state.total_gs_count = len(render_means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=cfg.port, verbose=False)
    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(cfg.result_dir),
        mode="rendering",
    )
    # Set initial background color from config
    viewer.render_tab_state.backgrounds = get_color(cfg.viewer_bkgd_color)
    viewer.render_tab_state.near_plane = cfg.near_plane
    viewer.render_tab_state.far_plane = cfg.far_plane
    print("Viewer running... Ctrl+C to exit.")
    
    try:
        time.sleep(100000)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, saving poses...")
    finally:
        # Final cleanup and save all poses
        if cfg.save_viewer_poses and saved_poses:
            save_viewer_poses(poses_dir, saved_poses, only_highest_res=True)
            
            # Save summary file with statistics
            summary_file = os.path.join(poses_dir, "poses_summary.json")
            summary = {
                "total_poses": len(saved_poses),
                "first_timestamp": saved_poses[0]["timestamp"] if saved_poses else None,
                "last_timestamp": saved_poses[-1]["timestamp"] if saved_poses else None,
                "duration_seconds": (saved_poses[-1]["timestamp"] - saved_poses[0]["timestamp"]) if len(saved_poses) > 1 else 0,
                "unique_resolutions": list(set(f"{pose['width']}x{pose['height']}" for pose in saved_poses)),
                "render_modes_used": list(set(pose["render_mode"] for pose in saved_poses)),
                "poses_file": "viewer_poses.json"
            }
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Pose summary saved to {summary_file}")
        print("Cleanup completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    args = parser.parse_args()

    print(args)
    # build default config
    default_cfg = Config()
    
    # read the template of yaml from file
    template_path = "../configs/actorshq_stream.toml"
    cfg = load_config_from_toml(template_path)
    cfg = merge_config(default_cfg, cfg)

    cfg.result_dir = args.output_dir
    cfg.ckpt = args.ckpt
    cfg.port = args.port
    cfg.with_ut = args.with_ut
    cfg.with_eval3d = args.with_eval3d

    cli(main, cfg, verbose=True)

'''
CUDA_VISIBLE_DEVICES=0 python simple_streamer.py \
    --ckpt /main/rajrup/Dropbox/Project/GsplatStream/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ckpts/ckpt_29999_rank0.pt \
    --output_dir /main/rajrup/Dropbox/Project/GsplatStream/gsplat/results/actorshq_l1_0.5_ssim_0.5_alpha_1.0/Actor01/Sequence1/resolution_4/0/ \
    --port 8082
'''
