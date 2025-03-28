import math
import os

import imageio
import numpy as np
import torch

from gsplat import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
)
from gsplat._helper import load_test_data
from gsplat.cuda._backend import _C
from gsplat.cuda._wrapper import (
    fully_fused_projection_with_ut,
    rasterize_to_pixels_eval3d,
)
from gsplat.utils import so3_matrix_to_quat

torch.manual_seed(42)

device = torch.device("cuda:0")


def test_data():
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
    ) = load_test_data(
        device=device,
        data_path=os.path.join(os.path.dirname(__file__), "../assets/test_garden.npz"),
    )
    colors = colors[None].repeat(len(viewmats), 1, 1)
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": width,
        "height": height,
    }


data = test_data()
Ks = data["Ks"][:1].contiguous()
viewmats = data["viewmats"][:1].contiguous()
height = data["height"]
width = data["width"]
quats = data["quats"].contiguous()
scales = data["scales"].contiguous()
means = data["means"].contiguous()
opacities = data["opacities"].contiguous()
C = len(Ks)
colors = data["colors"][:1].contiguous()


def rasterizer_and_save(radii, means2d, depths, conics, file_name="render.png"):
    # Identify intersecting tiles
    tile_size = 16
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # # forward
    # render_colors, render_alphas = rasterize_to_pixels(
    #     means2d,
    #     conics,
    #     colors,
    #     opacities.repeat(C, 1),
    #     width,
    #     height,
    #     tile_size,
    #     isect_offsets,
    #     flatten_ids,
    # )

    render_colors, render_alphas = rasterize_to_pixels_eval3d(
        means,
        quats,
        scales,
        colors,
        opacities.repeat(C, 1),
        viewmats,
        Ks,
        width,
        height,
        "pinhole",
        tile_size,
        isect_offsets,
        flatten_ids,
    )

    imageio.imsave(file_name, (render_colors[0].cpu().numpy() * 255).astype(np.uint8))


radii, means2d, depths, conics, _ = fully_fused_projection_with_ut(
    means, quats, scales, None, viewmats, None, Ks, width, height, 0.3, 0.01, 1e10, 0.0
)
rasterizer_and_save(radii, means2d, depths, conics, "results/ut_eval3d.png")

radii, means2d, depths, conics, _ = fully_fused_projection(
    means, None, quats, scales, viewmats, Ks, width, height, 0.3, 0.01, 1e10, 0.0
)
rasterizer_and_save(radii, means2d, depths, conics, "results/ewa_eval3d.png")
