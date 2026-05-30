# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser
from pathlib import Path
from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.exporter import load_ply_to_splats
from gsplat.rendering import rasterization
from libs.scene import GaussianScene
import experimental

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ply is not None:
        splats_dict = load_ply_to_splats(args.ply)
        means = splats_dict["means"].to(device)
        quats = F.normalize(splats_dict["quats"].to(device), p=2, dim=-1)
        scales_raw = splats_dict["scales"].to(device)
        opacities_raw = splats_dict["opacities"].to(device)
        sh0_t = splats_dict["sh0"].to(device)
        shN_t = splats_dict["shN"].to(device)
        colors = torch.cat([sh0_t, shN_t], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means, requires_grad=False),
                "quats": torch.nn.Parameter(quats, requires_grad=False),
                "scales": torch.nn.Parameter(scales_raw, requires_grad=False),
                "opacities": torch.nn.Parameter(opacities_raw, requires_grad=False),
                "colors": torch.nn.Parameter(colors, requires_grad=False),
            }
        )
        gaussian_scene = GaussianScene.from_splats(splats, id="viewer_scene")
    elif args.ckpt is None:
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
        ) = load_test_data(device=device, scene_grid=args.scene_grid)

        assert world_size <= 2
        means = means[world_rank::world_size].contiguous()
        quats = quats[world_rank::world_size].contiguous()
        scales = scales[world_rank::world_size].contiguous()
        opacities = opacities[world_rank::world_size].contiguous()
        colors = colors[world_rank::world_size].contiguous()

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render (skipped for inference render: no backward, no RGB+D)
        for _ in tqdm.trange(0 if args.use_gaussian_render_inference_scene else 1):
            render_colors, render_alphas, meta = rasterization(
                means,  # [N, 3]
                quats,  # [N, 4]
                scales,  # [N, 3]
                opacities,  # [N]
                colors,  # [N, S, 3]
                viewmats,  # [C, 4, 4]
                Ks,  # [C, 3, 3]
                width,
                height,
                render_mode="RGB+D",
                packed=False,
                distributed=world_size > 1,
            )
        if not args.use_gaussian_render_inference_scene:
            C = render_colors.shape[0]
            assert render_colors.shape == (C, height, width, 4)
            assert render_alphas.shape == (C, height, width, 1)

            render_rgbs = render_colors[..., 0:3]
            render_depths = render_colors[..., 3:4]
            render_depths = render_depths / render_depths.max()

            # dump batch images
            os.makedirs(args.output_dir, exist_ok=True)
            canvas = (
                torch.cat(
                    [
                        render_rgbs.reshape(C * height, width, 3),
                        render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                        render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
                    ],
                    dim=1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            imageio.imsave(
                f"{args.output_dir}/render_rank{world_rank}.png",
                (canvas * 255).astype(np.uint8),
            )

        # Build GaussianScene for viewer callback (needs raw/log-space tensors)
        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means.detach(), requires_grad=False),
                "quats": torch.nn.Parameter(quats.detach(), requires_grad=False),
                "scales": torch.nn.Parameter(
                    scales.detach().log(), requires_grad=False
                ),
                "opacities": torch.nn.Parameter(
                    torch.logit(opacities.detach().clamp(1e-6, 1 - 1e-6)),
                    requires_grad=False,
                ),
                "colors": torch.nn.Parameter(colors.detach(), requires_grad=False),
            }
        )
        gaussian_scene = GaussianScene.from_splats(splats, id="viewer_scene")
    else:
        means, quats_list, scales_raw, opacities_raw, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
            means.append(ckpt["means"])
            quats_list.append(F.normalize(ckpt["quats"], p=2, dim=-1))
            scales_raw.append(ckpt["scales"])  # raw log-space
            opacities_raw.append(ckpt["opacities"])  # raw logit-space
            sh0.append(ckpt["sh0"])
            shN.append(ckpt["shN"])
        means = torch.cat(means, dim=0)
        quats = torch.cat(quats_list, dim=0)
        scales_raw = torch.cat(scales_raw, dim=0)
        opacities_raw = torch.cat(opacities_raw, dim=0)
        sh0 = torch.cat(sh0, dim=0)
        shN = torch.cat(shN, dim=0)
        colors = torch.cat([sh0, shN], dim=-2)
        sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
        print("Number of Gaussians:", len(means))

        splats = torch.nn.ParameterDict(
            {
                "means": torch.nn.Parameter(means, requires_grad=False),
                "quats": torch.nn.Parameter(quats, requires_grad=False),
                "scales": torch.nn.Parameter(scales_raw, requires_grad=False),
                "opacities": torch.nn.Parameter(opacities_raw, requires_grad=False),
                "colors": torch.nn.Parameter(colors, requires_grad=False),
            }
        )
        gaussian_scene = GaussianScene.from_splats(splats, id="viewer_scene")

    # Build Inference scene if requested (after both loading paths)
    inference_scene = None
    inference_renderer = None
    if args.use_gaussian_render_inference_scene:
        inference_scene = experimental.GaussianInferenceScene.from_gaussian_scene(
            gaussian_scene, id="viewer_scene", sh_compression="32b"
        )
        inference_renderer = experimental.GaussianInferenceRenderer(inference_scene)

    # register and open viewer
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
        viewmat = c2w.inverse().contiguous()

        if args.use_gaussian_render_inference_scene:
            bg_raw = render_tab_state.backgrounds
            bg = torch.tensor(bg_raw, device=device) / 255.0
            render_kwargs = dict(
                viewmat=viewmat,
                K=K,
                width=width,
                height=height,
                near_plane=render_tab_state.near_plane,
                far_plane=render_tab_state.far_plane,
                radius_clip=render_tab_state.radius_clip,
                eps2d=render_tab_state.eps2d,
                background=bg,
            )
        else:
            RENDER_MODE_MAP = {
                "rgb": "RGB",
                "depth(accumulated)": "D",
                "depth(expected)": "ED",
                "alpha": "RGB",
            }
            splats = gaussian_scene.splats
            render_kwargs = dict(
                viewmats=viewmat[None],
                Ks=K[None],
                width=width,
                height=height,
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
                with_ut=args.with_ut,
                with_eval3d=args.with_eval3d,
            )

        with torch.inference_mode():
            if args.use_gaussian_render_inference_scene:
                ret = inference_renderer.render(**render_kwargs)
            else:
                render_colors_out, render_alphas_out, render_info = rasterization(
                    splats["means"],
                    splats["quats"],
                    splats["scales"].exp(),
                    splats["opacities"].sigmoid(),
                    splats.get("colors", splats.get("sh0")),
                    **render_kwargs,
                )

        if args.use_gaussian_render_inference_scene:
            render_tab_state.total_gs_count = inference_scene.num_gaussians
            render_tab_state.rendered_gs_count = 0
            # Stateful Inference returns native half4 RGBT; the viewer displays RGB.
            renders = ret.frame.squeeze(0)[..., :3].float().clamp(0, 1).cpu().numpy()
            return renders

        # Vanilla branch post-processing (using direct rasterization outputs)
        render_tab_state.total_gs_count = gaussian_scene.splats["means"].shape[0]
        radii = render_info.get("radii")
        if radii is not None:
            render_tab_state.rendered_gs_count = (radii > 0).all(-1).sum().item()
        else:
            render_tab_state.rendered_gs_count = 0

        if render_tab_state.render_mode == "rgb":
            render_colors = render_colors_out[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors_out[0, ..., 0:1]
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
            alpha = render_alphas_out[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    viewer_kwargs = dict(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    if args.use_gaussian_render_inference_scene:
        viewer_kwargs["render_modes"] = ("rgb",)
    _ = GsplatViewer(**viewer_kwargs)
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --ckpt results/garden/ckpts/ckpt_6999_rank0.pt \
        --output_dir results/garden/ \
        --port 8082

    CUDA_VISIBLE_DEVICES=9 python -m simple_viewer \
        --output_dir results/garden/ \
        --port 8082
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument("--ply", type=str, default=None, help="path to a 3DGS PLY file")
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    parser.add_argument(
        "--use_gaussian_render_inference_scene",
        action="store_true",
        help="use the Inference (non-differentiable) rasterization path",
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
