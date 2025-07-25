'''
A fork from simple_viewer.py to stream the scene.
'''

import argparse
import math
import os
import sys
import time

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
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    shN = torch.cat(shN, dim=0)
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", means.shape[0])

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
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
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
        render_tab_state.total_gs_count = len(means)
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
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


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
