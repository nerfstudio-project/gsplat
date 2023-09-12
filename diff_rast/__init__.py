from jaxtyping import Float, Int
from typing import Tuple
from torch import Tensor

from .rasterize import rasterize as rast
from .project_gaussians import project_gaussians as proj_gauss


def rasterize(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    conics: Float[Tensor, "*batch 3"],  # upper triangular?
    num_tiles_hit: Int[Tensor, "*batch 1"],  # (num points, 1)
    colors: Float[Tensor, "*batch 3"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
):
    """Alias for diff_rast.cuda_lib.rasterize."""
    return rast.apply(
        xys,
        depths,
        radii,
        conics,
        num_tiles_hit,
        colors,
        opacity,
        img_height,
        img_width,
    )


def project_gaussians(
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
    """Alias for diff_rast.cuda_lib.project_gaussians."""
    return proj_gauss.apply(
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
