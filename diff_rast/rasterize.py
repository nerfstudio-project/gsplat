"""Python bindings for custom Cuda functions"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.autograd import Function

from diff_rast import cuda_lib  # make sure to import torch before diff_rast


class rasterize(Function):
    """Main gaussian-splatting rendering function

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       rotations_quat (Tensor): rotations in quaternion [w,x,y,z] format.
       colors (Tensor): colors associated with the gaussians.
       opacity (Tensor): opacity/transparency of the gaussians.
       view_matrix (Tensor): view matrix for rendering.
       proj_matrix (Tensor): projection matrix for rendering.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
    """

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        rotations_quat: Float[Tensor, "*batch 4"],
        colors: Float[Tensor, "*batch 3"],
        opacity: Float[Tensor, "*batch 1"],
        view_matrix: Float[Tensor, "4 4"],
        proj_matrix: Float[Tensor, "4 4"],
        img_height: int,
        img_width: int,
        fx: int,
        fy: int,
    ):
        for name, input in {
            "means3d": means3d,
            "scales": scales,
            "rotations.quat": rotations_quat,
            "colors": colors,
            "opacity": opacity,
        }.items():
            assert (
                getattr(input, "shape")[0] == means3d.shape[0]
            ), f"Incorrect shape of input {name}. Batch size should be {means.shape[0]}, but got {input.shape[0]}."
            assert (
                getattr(input, "dim") != 2
            ), f"Incorrect number of dimensions for input {name}. Num of dimensions should be 2, got {input.dim()}"

        if proj_matrix.shape == (3, 4):
            proj_matrix = torch.cat(
                [
                    proj_matrix,
                    torch.Tensor([0, 0, 0, 1], device=proj_matrix.device).unsqueeze(0),
                ],
                dim=0,
            )
        if view_matrix.shape == (3, 4):
            view_matrix = torch.cat(
                [
                    view_matrix,
                    torch.Tensor([0, 0, 0, 1], device=proj_matrix.device).unsqueeze(0),
                ],
                dim=0,
            )
        assert proj_matrix.shape == (
            4,
            4,
        ), f"Incorrect shape for projection matrix, got{proj_matrix.shape}, should be (4,4)."
        assert view_matrix.shape == (
            4,
            4,
        ), f"Incorrect shape for view matrix, got{view_matrix.shape}, should be (4,4)."

        # move tensors to cuda and call forward
        (
            num_rendered,
            out_img,
            out_radii,
            final_Ts,
            final_idx,
            gaussian_ids_sorted,
            tile_bins,
            xy,
            conics,
        ) = cuda_lib.rasterize_forward(
            means3d.contiguous().cuda(),
            scales.contiguous().cuda(),
            glob_scale,
            rotations_quat.contiguous().cuda(),
            colors.contiguous().cuda(),
            opacity.contiguous().cuda(),
            view_matrix.contiguous().cuda(),
            proj_matrix.contiguous().cuda(),
            img_height,
            img_width,
            fx,
            fy,
        )

        ctx.num_rendered = num_rendered
        ctx.glob_scale = glob_scale
        ctx.view_matrix = view_matrix
        ctx.proj_matrix = proj_matrix
        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.fx = fx
        ctx.fy = fy
        ctx.save_for_backward(
            colors,
            means3d,
            scales,
            rotations_quat,
            out_radii,
            final_Ts,
            final_idx,
            gaussian_ids_sorted,
            tile_bins,
            xy,
            conics,
        )

        return num_rendered, out_img, out_radii

    @staticmethod
    def backward(ctx, _, grad_out_img, grad_out_radii):
        num_rendered = ctx.num_rendered
        glob_scale = ctx.glob_scale
        view_matrix = ctx.view_matrix
        proj_matrix = ctx.proj_matrix
        img_height = ctx.img_height
        img_width = ctx.img_width
        fx = ctx.fx
        fy = ctx.fy

        (
            colors,
            means3d,
            scales,
            rotations_quat,
            out_radii,
            final_Ts,
            final_idx,
            gaussian_ids_sorted,
            tile_bins,
            xy,
            conics,
        ) = ctx.saved_tensors

        num_rendered, out_img, out_radii, final_Ts, final_idx = cuda_lib.rasterize_backward(
            means3d.contiguous().cuda(),
            out_radii.contiguous().cuda(),
            colors.contiguous().cuda(),
            scales.contiguous().cuda(),
            rotations_quat.contiguous().cuda(),
            glob_scale,
            view_matrix.contiguous().cuda(),
            proj_matrix.contiguous().cuda(),
            grad_out_img.contiguous().cuda(),  # v_output also called dL_dout_color
            int(img_height),
            int(img_width),
            float(fx),
            float(fy),
            gaussian_ids_sorted,
            tile_bins,
            conics,
            colors,
            opacities,
            final_Ts,
            final_idx,
        )

        return (
            None,  # means3d
            None,  # scales
            None,  # glob_scale
            None,  # rotations_quat
            None,  # colors
            None,  # opacity
            None,  # view_matrix
            None,  # proj_matrix
            None,  # img_height
            None,  # img_width
            None,  # fx
            None,  # fy
        )


# helper to save image
def vis_image(image, image_path):
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

    BLOCK_X = 16
    BLOCK_Y = 16
    num_points = 2048
    fov_x = math.pi / 2.0
    W = 256
    H = 256
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x)
    tile_bounds = torch.tensor([(W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1])
    img_size = torch.tensor([W, H, 1])
    block = torch.tensor([BLOCK_X, BLOCK_Y, 1])

    means = torch.empty((num_points, 3))
    scales = torch.empty((num_points, 3))
    quats = torch.empty((num_points, 4))
    rgbs = torch.empty((num_points, 3))
    opacities = torch.empty(num_points)
    viewmat = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    bd = 2.0
    random.seed()
    for i in range(num_points):
        means[i] = torch.tensor(
            [
                bd * (random.random() - 0.5),
                bd * (random.random() - 0.5),
                bd * (random.random() - 0.5),
            ]
        )
        scales[i] = torch.tensor([random.random(), random.random(), random.random()])
        rgbs[i] = torch.tensor([random.random(), random.random(), random.random()])
        u = random.random()
        v = random.random()
        w = random.random()
        quats[i] = torch.tensor(
            [
                math.sqrt(1.0 - u) * math.sin(2.0 * math.pi * v),
                math.sqrt(1.0 - u) * math.cos(2.0 * math.pi * v),
                math.sqrt(u) * math.sin(2.0 * math.pi * w),
                math.sqrt(u) * math.cos(2.0 * math.pi * w),
            ]
        )
        opacities[i] = 0.9

    means.requires_grad = True
    scales.requires_grad = True
    quats.requires_grad = True
    rgbs.requires_grad = True
    opacities.requires_grad = True
    viewmat.requires_grad = True

    # currently proj_mat = view_mat
    num_rendered, out_img, out_radii = rasterize.apply(
        means, scales, 1, quats, rgbs, opacities, viewmat, viewmat, H, W, focal, focal
    )
    # vis_image(out_img, os.getcwd() + "/python_forward.png")

    gt_rgb = torch.ones((W, H, 3), dtype=out_img.dtype).cuda()
    mse = torch.nn.MSELoss()
    out_img = out_img.cuda()
    loss = mse(out_img.cpu(), gt_rgb.cpu())  # BUG: why cpu required here
    loss.backward()
