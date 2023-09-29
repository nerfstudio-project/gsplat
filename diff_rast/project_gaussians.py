"""Python bindings for 3D gaussian projection"""

from typing import Tuple

from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function

import diff_rast.cuda as _C


class ProjectGaussians(Function):
    """Project 3D Gaussians to 2D.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       projmat (Tensor): projection matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).
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
        clip_thresh:float=0.01
    ):
        num_points = means3d.shape[-2]

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
        ) = _C.project_gaussians_forward(
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
            clip_thresh,
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

        return (xys, depths, radii, conics, num_tiles_hit, cov3d)

    @staticmethod
    def backward(ctx, v_xys, v_depths, v_radii, v_conics, v_num_tiles_hit, v_cov3d):
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
        ) = _C.project_gaussians_backward(
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
