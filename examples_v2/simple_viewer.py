"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import os
import time
from typing import Tuple

import imageio
import numpy as np
import torch

from gsplat._helper_v2 import load_test_data
from gsplat.rendering_v2 import rasterization

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
parser.add_argument("--port", type=int, default=8080, help="port for the viewer server")
args = parser.parse_args()
assert args.scene_grid % 2 == 1, "scene_grid must be odd"

torch.manual_seed(42)
device = "cuda"

means, quats, scales, opacities, colors, viewmats, Ks, width, height = load_test_data(
    device=device, scene_grid=args.scene_grid
)
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

    # Switch between different backends (read their docstrings before using them!)
    # from gsplat2._helper import rasterization_inria_wrapper as rasterization
    # from gsplat2._helper import rasterization_legacy_wrapper as rasterization
    render_colors, render_alphas, meta = rasterization(
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, 3]
        viewmat[None],  # [1, 4, 4]
        K[None],  # [1, 3, 3]
        width,
        height,
        render_mode="RGB",
        # this is to speedup large-scale rendering by skipping far-away Gaussians.
        radius_clip=3,
    )
    render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
    return render_rgbs


server = ViewerServer(port=args.port, render_fn=viewer_render_fn)
print("Viewer running... Ctrl+C to exit.")
time.sleep(100000)
