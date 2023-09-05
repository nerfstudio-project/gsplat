"""Python bindings for custom Cuda functions"""

from typing import Tuple

import torch
from diff_rast import cuda_lib  # make sure to import torch before diff_rast
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from pathlib import Path
import os
from PIL import Image
import numpy as np


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
                [proj_matrix, torch.Tensor([0, 0, 0, 1], device=proj_matrix.device).unsqueeze(0)], dim=0
            )
        if view_matrix.shape == (3, 4):
            view_matrix = torch.cat(
                [view_matrix, torch.Tensor([0, 0, 0, 1], device=proj_matrix.device).unsqueeze(0)], dim=0
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
        outputs = cuda_lib.rasterize_forward(
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
        return outputs

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


class compute_cov2d_bounds(Function):
    """Computes bounds of 2D covariance matrix

    expects input cov2d to be size (batch, 3) of upper triangular values

    Returns: tuple of conics (batch, 3) and radii (batch, 1)
    """

    @staticmethod
    def forward(
        ctx, cov2d: Float[Tensor, "batch 3"]
    ) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
        num_pts = cov2d.shape[0]
        assert num_pts > 0

        output = cuda_lib.compute_cov2d_bounds_forward(num_pts, cov2d)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError


# helper to save image
def vis_image(image, image_path):
    """Generic image saver"""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
        image = image[..., [2, 1, 0]].copy()
        # image = image[..., ::-1].copy()
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
    viewmat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 8.0], [0.0, 0.0, 0.0, 1.0]])
    bd = 2.0
    random.seed()
    for i in range(num_points):
        means[i] = torch.tensor(
            [bd * (random.random() - 0.5), bd * (random.random() - 0.5), bd * (random.random() - 0.5)]
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

    # currently proj_mat = view_mat
    num_rendered, out_img, out_radii = rasterize.apply(
        means, scales, 1, quats, rgbs, opacities, viewmat, viewmat, H, W, focal, focal
    )
    print(out_img.shape)
    vis_image(out_img.permute(2, 1, 0), os.getcwd() + "/python_forward.png")

    # cov2d bounds
    f = compute_cov2d_bounds()
    A = torch.rand(4, 3, device="cuda:0").requires_grad_()
    conics, radii = f.apply(A)
