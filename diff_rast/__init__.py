from jaxtyping import Float
from torch import Tensor

from . import rasterize


def rasterize(
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
    """Alias for diff_rast.cuda_lib.rasterize."""
    return rasterize.rasterize.apply(
        means3d,
        scales,
        glob_scale,
        rotations_quat,
        colors,
        opacity,
        view_matrix,
        proj_matrix,
        img_height,
        img_width,
        fx,
        fy,
    )
