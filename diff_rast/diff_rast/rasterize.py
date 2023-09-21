"""Python bindings for custom Cuda functions"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

import cuda_lib  # make sure to import torch before diff_rast


class RasterizeGaussians(Function):
    """Main gaussian-splatting rendering function

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): colors associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        background (Tensor): background color
    """

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        conics: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        background: Optional[Float[Tensor, "channels"]] = None,
    ):
        if colors.dtype == torch.uint8:
            # make sure colors are float [0,1]
            colors = colors.float() / 255

        if background is not None:
            assert (
                background.shape[0] == colors.shape[-1]
            ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
        else:
            background = torch.ones(3, dtype=torch.float32)
        (
            out_img,
            final_Ts,
            final_idx,
            tile_bins,
            gaussian_ids_sorted,
            gaussian_ids_unsorted,
            isect_ids_sorted,
            isect_ids_unsorted,
        ) = cuda_lib.rasterize_forward(
            xys.contiguous().cuda(),
            depths.contiguous().cuda(),
            radii.contiguous().cuda(),
            conics.contiguous().cuda(),
            num_tiles_hit.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacity.contiguous().cuda(),
            img_height,
            img_width,
            background.contiguous().cuda(),
        )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            final_Ts,
            final_idx,
        )

        return out_img

    @staticmethod
    def backward(ctx, v_out_img):
        img_height = ctx.img_height
        img_width = ctx.img_width

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors

        v_xy, v_conic, v_colors, v_opacity = cuda_lib.rasterize_backward(
            img_height,
            img_width,
            gaussian_ids_sorted.contiguous().cuda(),
            tile_bins,
            xys.contiguous().cuda(),
            conics.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacity.contiguous().cuda(),
            final_Ts.contiguous().cuda(),
            final_idx.contiguous().cuda(),
            v_out_img.contiguous().cuda(),
        )

        v_opacity = v_opacity.squeeze(-1)

        return (
            v_xy,  # xys
            None,  # depths
            None,  # radii
            v_conic,  # conics
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # background
        )


# helper to save image
def vis_image(image, image_path):
    from PIL import Image

    """Generic image saver"""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
        image = image[..., [2, 1, 0]].copy()
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir()

    im = Image.fromarray(image)
    print("saving to: ", image_path)
    im.save(image_path)


if __name__ == "__main__":
    """Test bindings"""
    # rasterizer testing
    import math
    import random

    device = torch.device("cuda:0")

    BLOCK_X = 16
    BLOCK_Y = 16
    num_points = 2048
    fov_x = math.pi / 2.0
    W = 256
    H = 256
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x)
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1
    img_size = torch.tensor([W, H, 1], device=device)
    block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=device)

    means = torch.empty((num_points, 3), device=device)
    scales = torch.empty((num_points, 3), device=device)
    quats = torch.empty((num_points, 4), device=device)
    rgbs = torch.empty((num_points, 3), device=device)
    opacities = torch.empty(num_points, device=device)
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        device=device,
    )
    bd = 2
    random.seed()
    for i in range(num_points):
        means[i] = torch.tensor(
            [
                bd * (random.random() - 0.5),
                bd * (random.random() - 0.5),
                bd * (random.random() - 0.5),
            ],
            device=device,
        )
        scales[i] = torch.tensor([random.random(), random.random(), random.random()], device=device)
        rgbs[i] = torch.tensor([random.random(), random.random(), random.random()], device=device)
        u = random.random()
        v = random.random()
        w = random.random()
        quats[i] = torch.tensor(
            [
                math.sqrt(1.0 - u) * math.sin(2.0 * math.pi * v),
                math.sqrt(1.0 - u) * math.cos(2.0 * math.pi * v),
                math.sqrt(u) * math.sin(2.0 * math.pi * w),
                math.sqrt(u) * math.cos(2.0 * math.pi * w),
            ],
            device=device,
        )
        opacities[i] = 0.90

    means.requires_grad = True
    scales.requires_grad = True
    quats.requires_grad = True
    rgbs.requires_grad = True
    opacities.requires_grad = True
    viewmat.requires_grad = True

    means = means.to(device)
    scales = scales.to(device)
    quats = quats.to(device)
    rgbs = rgbs.to(device)
    viewmat = viewmat.to(device)

    # Test new rasterizer
    # 1. Project gaussians
    from diff_rast.project_gaussians import ProjectGaussians
    xys, depths, radii, conics, num_tiles_hit = ProjectGaussians(
        means, scales, 1, quats, viewmat, viewmat, focal, focal, H, W, tile_bounds
    )

    # 2. Rasterize gaussians
    out_img = RasterizeGaussians.apply(xys, depths, radii, conics, num_tiles_hit, rgbs, opacities, H, W)

    vis_image(out_img, os.getcwd() + "/python_forward.png")
    gt_rgb = torch.ones((W, H, 3), dtype=out_img.dtype).to("cuda:0")
    mse = torch.nn.MSELoss()
    out_img = out_img
    gt_rgb = gt_rgb
    loss = mse(out_img, gt_rgb)
    loss.backward()

    import pdb

    pdb.set_trace()
