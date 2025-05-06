import argparse
import math
import os
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
from gsplat_viewer import GsplatViewer, GsplatRenderTabState


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    if args.ckpt is None:
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
        means.requires_grad = True
        quats = quats[world_rank::world_size].contiguous()
        quats.requires_grad = True
        scales = scales[world_rank::world_size].contiguous()
        scales.requires_grad = True
        opacities = opacities[world_rank::world_size].contiguous()
        opacities.requires_grad = True
        colors = colors[world_rank::world_size].contiguous()
        colors.requires_grad = True

        viewmats = viewmats[world_rank::world_size][:1].contiguous()
        Ks = Ks[world_rank::world_size][:1].contiguous()

        sh_degree = None
        C = len(viewmats)
        N = len(means)
        print("rank", world_rank, "Number of Gaussians:", N, "Number of Cameras:", C)

        # batched render
        for _ in tqdm.trange(1):
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
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)
        render_colors.sum().backward()

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
    else:
        means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=device)["splats"]
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

        # # crop
        # aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
        # edges = aabb[3:] - aabb[:3]
        # sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
        # sel = torch.where(sel)[0]
        # means, quats, scales, colors, opacities = (
        #     means[sel],
        #     quats[sel],
        #     scales[sel],
        #     colors[sel],
        #     opacities[sel],
        # )

        # # repeat the scene into a grid (to mimic a large-scale setting)
        # repeats = args.scene_grid
        # gridx, gridy = torch.meshgrid(
        #     [
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        #     ],
        #     indexing="ij",
        # )
        # grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
        #     -1, 3
        # )
        # means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
        # means = means.reshape(-1, 3)
        # quats = quats.repeat(repeats**2, 1)
        # scales = scales.repeat(repeats**2, 1)
        # colors = colors.repeat(repeats**2, 1, 1)
        # opacities = opacities.repeat(repeats**2)
        print("Number of Gaussians:", len(means))

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
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
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
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
