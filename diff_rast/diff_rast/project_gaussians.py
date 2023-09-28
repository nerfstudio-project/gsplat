"""Python bindings for custom Cuda functions"""

from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import cuda_lib  # make sure to import torch before diff_rast


class ProjectGaussians(Function):
    """Project 3D Gaussians to 2D.

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
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        projmat: Float[Tensor, "4 4"],
        fx: int,
        fy: int,
        img_height: int,
        img_width: int,
        tile_bounds: Tuple[int, int, int],
    ):
        num_points = means3d.shape[-2]

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = cuda_lib.project_gaussians_forward(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            projmat,
            fx,
            fy,
            img_height,
            img_width,
            tile_bounds,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            projmat,
            cov3d,
            radii,
            conics,
        )

        return (xys, depths, radii, conics, num_tiles_hit,cov3d)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit,v_cov3d):
        (
            means3d,
            scales,
            quats,
            viewmat,
            projmat,
            cov3d,
            radii,
            conics,
        ) = ctx.saved_tensors

        (
            v_cov2d,
            v_cov3d,
            v_mean3d,
            v_scale,
            v_quat,
        ) = cuda_lib.project_gaussians_backward(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            projmat,
            ctx.fx,
            ctx.fy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            v_xys,
            v_conics,
        )

        # Return a gradient for each input.
        return (
            # means3d: Float[Tensor, "*batch 3"],
            v_mean3d,
            # scales: Float[Tensor, "*batch 3"],
            v_scale,
            # glob_scale: float,
            None,
            # quats: Float[Tensor, "*batch 4"],
            v_quat,
            # viewmat: Float[Tensor, "4 4"],
            None,
            # projmat: Float[Tensor, "4 4"],
            None,
            # fx: int,
            None,
            # fy: int,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # tile_bounds: Tuple[int, int, int],
            None,
        )


if __name__ == "__main__":
    device = torch.device("cuda:0")
    num_points = 256
    means3d = torch.randn((num_points, 3), device=device, requires_grad=True)
    scales = torch.randn((num_points, 3), device=device)
    glob_scale = 0.3
    quats = torch.randn((num_points, 4), device=device)
    quats /= torch.linalg.norm(quats, dim=-1, keepdim=True)
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 2
    # projmat = torch.eye(4, device=device)
    projmat = viewmat
    fx = 3.0
    fy = 3.0
    img_height = 256
    img_width = 256

    W = 256
    H = 256
    BLOCK_X = 16
    BLOCK_Y = 16
    tile_bounds = (W + BLOCK_X - 1) // BLOCK_X, (H + BLOCK_Y - 1) // BLOCK_Y, 1

    xys, conics = ProjectGaussians.apply(
        means3d,
        scales,
        glob_scale,
        quats,
        viewmat,
        projmat,
        fx,
        fy,
        img_height,
        img_width,
        tile_bounds,
    )
    print(f"{xys.shape=}")
    print(f"{conics.shape=}")

    mse = torch.nn.MSELoss()
    loss = mse(xys, torch.ones(size=xys.shape, device=xys.device))
    # loss = xys.sum() + conics.sum()
    loss.backward()
    # print(f"{means3d.grad.shape=}")
    # print(f"{means3d.grad.sum()}")
