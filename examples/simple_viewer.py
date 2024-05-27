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
import numpy as np
import torch
import torch.nn.functional as F

from gsplat._helper import load_test_data
from gsplat.rendering import rasterization

try:
    from nerfview import CameraState, ViewerServer
except ImportError:
    print("Please install nerfview: pip install git+https://github.com/hangg7/nerfview")

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

if args.ckpt is None:
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = (
        load_test_data(device=device, scene_grid=args.scene_grid)
    )
    sh_degree = None
    C = len(viewmats)
    N = len(means)
    print("Number of Gaussians:", N)

    # batched render
    render_colors, render_alphas, meta = rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmats,  # [C, 4, 4]
        Ks,  # [C, 3, 3]
        width,
        height,
        render_mode="RGB+D",
    )
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
def viewer_render_fn(camera_state: CameraState, img_wh: Tuple[int, int]):
    fov = camera_state.fov
    c2w = camera_state.c2w
    width, height = img_wh

    focal_length = height / 2.0 / np.tan(fov / 2.0)
    K = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
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


server = ViewerServer(port=args.port, render_fn=viewer_render_fn)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
