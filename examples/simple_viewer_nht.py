# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import argparse
import os
import time

import torch
import torch.nn.functional as F
import viser
from pathlib import Path

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.nht.deferred_shader import DeferredShaderModule

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer_nht import GsplatNHTViewer, GsplatNHTRenderTabState


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    assert args.ckpt is not None, "NHT viewer requires --ckpt checkpoint(s)"

    # Load checkpoint(s)
    ckpt_means, ckpt_quats, ckpt_scales, ckpt_opacities, ckpt_features = (
        [],
        [],
        [],
        [],
        [],
    )
    deferred_state_dict = None

    ema_shadow = None
    for i, ckpt_path in enumerate(args.ckpt):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        splats = ckpt["splats"]
        ckpt_means.append(splats["means"])
        ckpt_quats.append(F.normalize(splats["quats"], p=2, dim=-1))
        ckpt_scales.append(torch.exp(splats["scales"]))
        ckpt_opacities.append(torch.sigmoid(splats["opacities"]))
        ckpt_features.append(splats["features"])
        if i == 0:
            deferred_state_dict = ckpt["deferred_module"]
            if "deferred_ema" in ckpt:
                ema_shadow = ckpt["deferred_ema"]

    means = torch.cat(ckpt_means, dim=0)
    quats = torch.cat(ckpt_quats, dim=0)
    scales = torch.cat(ckpt_scales, dim=0)
    opacities = torch.cat(ckpt_opacities, dim=0)
    features = torch.cat(ckpt_features, dim=0).half()

    feature_dim = features.shape[-1]
    print(f"Number of Gaussians: {len(means)}, Feature dim: {feature_dim}")

    # Instantiate and load the deferred shading module
    deferred_module = DeferredShaderModule(
        feature_dim=feature_dim,
        enable_view_encoding=args.enable_view_encoding,
        view_encoding_type=args.view_encoding_type,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_num_layers=args.mlp_num_layers,
        sh_degree=args.deferred_sh_degree,
        sh_scale=args.deferred_sh_scale,
        fourier_num_freqs=args.fourier_num_freqs,
        center_ray_encoding=args.center_ray_encoding,
    ).to(device)
    deferred_module.load_state_dict(deferred_state_dict)
    if ema_shadow is not None:
        for n, p in deferred_module.named_parameters():
            if n in ema_shadow:
                p.data.copy_(ema_shadow[n])
        print("Applied EMA weights to deferred module")
    deferred_module.eval()



    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatNHTRenderTabState)
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

        need_depth = render_tab_state.render_mode in [
            "depth(accumulated)",
            "depth(expected)",
        ]
        if need_depth:
            depth_suffix = (
                "D"
                if render_tab_state.render_mode == "depth(accumulated)"
                else "ED"
            )
            rast_render_mode = f"RGB+{depth_suffix}"
        else:
            rast_render_mode = "RGB"

        render_colors, render_alphas, info = rasterization(
            means,
            quats,
            scales,
            opacities,
            features,
            viewmat[None],
            K[None],
            width,
            height,
            sh_degree=None,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            render_mode=rast_render_mode,
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            tile_size=args.tile_size,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
            nht=True,
            center_ray_mode=args.center_ray_encoding,
            ray_dir_scale=deferred_module.ray_dir_scale,
        )

        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        rgb, extras = deferred_module(render_colors)

        if render_tab_state.render_mode == "rgb":
            rgb = rgb[0, ..., 0:3].clamp(0, 1)
            bkgd = (
                torch.tensor(render_tab_state.backgrounds, device=device).float()
                / 255.0
            )
            rgb = rgb + bkgd * (1.0 - render_alphas[0])
            return rgb.clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in [
            "depth(accumulated)",
            "depth(expected)",
        ]:
            depth = extras[0, ..., 0:1]
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
            return (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            return (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatNHTViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # View an NHT checkpoint trained with simple_trainer_nht.py
    CUDA_VISIBLE_DEVICES=0 python simple_viewer_nht.py \
        --ckpt results/garden/ckpts/ckpt_29999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="path to NHT .pt checkpoint file(s)",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut",
        action="store_true",
        default=True,
        help="use uncentered transform (default: True)",
    )
    parser.add_argument(
        "--with_eval3d",
        action="store_true",
        default=True,
        help="use eval 3D (default: True)",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=16,
        help="tile size for rasterization",
    )
    # Deferred shading module configuration
    # These defaults match simple_trainer_nht.py Config defaults
    parser.add_argument(
        "--no_view_encoding",
        dest="enable_view_encoding",
        action="store_false",
        help="disable view-dependent encoding",
    )
    parser.set_defaults(enable_view_encoding=True)
    parser.add_argument(
        "--view_encoding_type",
        type=str,
        default="sh",
        choices=["sh", "fourier"],
        help="view encoding type",
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=128,
        help="deferred MLP hidden dimension",
    )
    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=3,
        help="deferred MLP number of hidden layers",
    )
    parser.add_argument(
        "--deferred_sh_degree",
        type=int,
        default=3,
        help="SH degree for view encoding",
    )
    parser.add_argument(
        "--deferred_sh_scale",
        type=float,
        default=3.0,
        help="scale applied to dirs before SH evaluation",
    )
    parser.add_argument(
        "--fourier_num_freqs",
        type=int,
        default=4,
        help="number of Fourier frequency levels",
    )
    parser.add_argument(
        "--no_center_ray_encoding",
        dest="center_ray_encoding",
        action="store_false",
        help="disable center ray encoding (use per-pixel rays)",
    )
    parser.set_defaults(center_ray_encoding=False)

    args = parser.parse_args()
    cli(main, args, verbose=True)
