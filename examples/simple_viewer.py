"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import viser
from torch import Tensor

from gsplat._helper import load_test_data
from gsplat.rendering import _rasterization, rasterization

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="results/", help="where to dump outputs"
)
parser.add_argument(
    "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
)
parser.add_argument("--ckpt", type=str, default=None, help="path to the .pt file")
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
parser.add_argument(
    "--backend", type=str, default="gsplat", help="gsplat, gsplat_legacy, inria"
)
args = parser.parse_args()
assert args.scene_grid % 2 == 1, "scene_grid must be odd"

torch.manual_seed(42)
device = "cuda"


def getProjectionMatrix(znear, zfar, fovX, fovY, device="cuda"):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def _depths_to_points(depthmap, world_view_transform, full_proj_transform):
    c2w = (world_view_transform.T).inverse()
    H, W = depthmap.shape[:2]
    ndc2pix = (
        torch.tensor([[W / 2, 0, 0, (W) / 2], [0, H / 2, 0, (H) / 2], [0, 0, 0, 1]])
        .float()
        .cuda()
        .T
    )
    projection_matrix = c2w.T @ full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device="cuda").float(),
        torch.arange(H, device="cuda").float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(
        -1, 3
    )
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def _depth_to_normal(depth, world_view_transform, full_proj_transform):
    points = _depths_to_points(
        depth, world_view_transform, full_proj_transform
    ).reshape(*depth.shape[:2], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output


def depth_to_normal(
    depths: Tensor,  # [C, H, W, 1]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> Tensor:
    height, width = depths.shape[1:3]

    normals = []
    for cid, depth in enumerate(depths):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = getProjectionMatrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=depths.device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        normal = _depth_to_normal(depth, world_view_transform, full_proj_transform)
        normals.append(normal)
    normals = torch.stack(normals, dim=0)
    return normals


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
    sh_degree = None
    C = len(viewmats)
    N = len(means)
    print("Number of Gaussians:", N)

    ckpt = torch.load("results/garden/ckpts/ckpt_6999.pt", map_location=device)[
        "splats"
    ]
    means = ckpt["means3d"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # batched render
    render_colors, render_alphas, meta = _rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmats,  # [C, 4, 4]
        Ks,  # [C, 3, 3]
        width,
        height,
        render_mode="RGB+ED",
        sh_degree=sh_degree,
        accurate_depth=False,
    )
    assert render_colors.shape == (C, height, width, 4)
    assert render_alphas.shape == (C, height, width, 1)

    render_rgbs = render_colors[..., 0:3]
    render_depths = render_colors[..., 3:4]
    render_normals = depth_to_normal(render_depths, viewmats, Ks)
    render_normals = render_normals * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    render_depths = render_depths / render_depths.max()

    # dump batch images
    os.makedirs(args.output_dir, exist_ok=True)
    canvas = (
        torch.cat(
            [
                render_rgbs.reshape(C * height, width, 3),
                render_depths.reshape(C * height, width, 1).expand(-1, -1, 3),
                render_normals.reshape(C * height, width, 3),
                render_alphas.reshape(C * height, width, 1).expand(-1, -1, 3),
            ],
            dim=1,
        )
        .cpu()
        .numpy()
    )
    imageio.imsave(f"{args.output_dir}/render.png", (canvas * 255).astype(np.uint8))
else:
    ckpt = torch.load(args.ckpt, map_location=device)["splats"]
    means = ckpt["means3d"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # crop
    aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
    edges = aabb[3:] - aabb[:3]
    sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    sel = torch.where(sel)[0]
    means, quats, scales, colors, opacities = (
        means[sel],
        quats[sel],
        scales[sel],
        colors[sel],
        opacities[sel],
    )

    # repeat the scene into a grid (to mimic a large-scale setting)
    repeats = args.scene_grid
    gridx, gridy = torch.meshgrid(
        [
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
            torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
        ],
        indexing="ij",
    )
    grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(-1, 3)
    means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    means = means.reshape(-1, 3)
    quats = quats.repeat(repeats**2, 1)
    scales = scales.repeat(repeats**2, 1)
    colors = colors.repeat(repeats**2, 1, 1)
    opacities = opacities.repeat(repeats**2)
    print("Number of Gaussians:", len(means))


# register and open viewer
@torch.no_grad()
def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    width, height = img_wh
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    c2w = torch.from_numpy(c2w).float().to(device)
    K = torch.from_numpy(K).float().to(device)
    viewmat = c2w.inverse()

    if args.backend == "gsplat":
        rasterization_fn = rasterization
    elif args.backend == "gsplat_legacy":
        from gsplat import rasterization_legacy_wrapper

        rasterization_fn = rasterization_legacy_wrapper
    elif args.backend == "inria":
        from gsplat import rasterization_inria_wrapper

        rasterization_fn = rasterization_inria_wrapper
    else:
        raise ValueError

    render_colors, render_alphas, meta = rasterization_fn(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        sh_degree=sh_degree,
        render_mode="RGB",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    return render_rgbs


server = viser.ViserServer(port=args.port, verbose=False)
_ = nerfview.Viewer(
    server=server,
    render_fn=viewer_render_fn,
    mode="rendering",
)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
