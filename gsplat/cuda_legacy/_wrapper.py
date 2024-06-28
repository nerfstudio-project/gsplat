"""Python bindings for binning and sorting gaussians"""

from typing import Callable, Literal, Optional, Tuple

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.autograd import Function

def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_cuda

def map_gaussian_to_intersects(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[Float[Tensor, "cum_tiles_hit 1"], Float[Tensor, "cum_tiles_hit 1"]]:
    """Map each gaussian intersection to a unique tile ID and depth value for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): total number of tile intersections.
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor}:

        - **isect_ids** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit.
    """
    isect_ids, gaussian_ids = _make_lazy_cuda_func("map_gaussian_to_intersects")(
        num_points,
        num_intersects,
        xys.contiguous(),
        depths.contiguous(),
        radii.contiguous(),
        cum_tiles_hit.contiguous(),
        tile_bounds,
        block_size,
    )
    return (isect_ids, gaussian_ids)

def get_tile_bin_edges(
    num_intersects: int,
    isect_ids_sorted: Int[Tensor, "num_intersects 1"],
    tile_bounds: Tuple[int, int, int],
) -> Int[Tensor, "num_intersects 2"]:
    """Map sorted intersection IDs to tile bins which give the range of unique gaussian IDs belonging to each tile.

    Expects that intersection IDs are sorted by increasing tile ID.

    Indexing into tile_bins[tile_idx] returns the range (lower,upper) of gaussian IDs that hit tile_idx.

    Note:
        This function is not differentiable to any input.

    Args:
        num_intersects (int): total number of gaussian intersects.
        isect_ids_sorted (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A Tensor:

        - **tile_bins** (Tensor): range of gaussians IDs hit per tile.
    """
    return _make_lazy_cuda_func("get_tile_bin_edges")(
        num_intersects, isect_ids_sorted.contiguous(), tile_bounds
    )
    
def compute_cov2d_bounds(
    cov2d: Float[Tensor, "batch 3"]
) -> Tuple[Float[Tensor, "batch_conics 3"], Float[Tensor, "batch_radii 1"]]:
    """Computes bounds of 2D covariance matrix

    Args:
        cov2d (Tensor): input cov2d of size  (batch, 3) of upper triangular 2D covariance values

    Returns:
        A tuple of {Tensor, Tensor}:

        - **conic** (Tensor): conic parameters for 2D gaussian.
        - **radii** (Tensor): radii of 2D gaussian projections.
    """
    assert (
        cov2d.shape[-1] == 3
    ), f"Expected input cov2d to be of shape (*batch, 3) (upper triangular values), but got {tuple(cov2d.shape)}"
    num_pts = cov2d.shape[0]
    assert num_pts > 0
    return _make_lazy_cuda_func("compute_cov2d_bounds")(num_pts, cov2d.contiguous())

def compute_cumulative_intersects(
    num_tiles_hit: Float[Tensor, "batch 1"]
) -> Tuple[int, Float[Tensor, "batch 1"]]:
    """Computes cumulative intersections of gaussians. This is useful for creating unique gaussian IDs and for sorting.

    Note:
        This function is not differentiable to any input.

    Args:
        num_tiles_hit (Tensor): number of intersected tiles per gaussian.

    Returns:
        A tuple of {int, Tensor}:

        - **num_intersects** (int): total number of tile intersections.
        - **cum_tiles_hit** (Tensor): a tensor of cumulated intersections (used for sorting).
    """
    cum_tiles_hit = torch.cumsum(num_tiles_hit, dim=0, dtype=torch.int32)
    num_intersects = cum_tiles_hit[-1].item()
    return num_intersects, cum_tiles_hit


def bin_and_sort_gaussians(
    num_points: int,
    num_intersects: int,
    xys: Float[Tensor, "batch 2"],
    depths: Float[Tensor, "batch 1"],
    radii: Float[Tensor, "batch 1"],
    cum_tiles_hit: Float[Tensor, "batch 1"],
    tile_bounds: Tuple[int, int, int],
    block_size: int,
) -> Tuple[
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 1"],
    Float[Tensor, "num_intersects 2"],
]:
    """Mapping gaussians to sorted unique intersection IDs and tile bins used for fast rasterization.

    We return both sorted and unsorted versions of intersect IDs and gaussian IDs for testing purposes.

    Note:
        This function is not differentiable to any input.

    Args:
        num_points (int): number of gaussians.
        num_intersects (int): cumulative number of total gaussian intersections
        xys (Tensor): x,y locations of 2D gaussian projections.
        depths (Tensor): z depth of gaussians.
        radii (Tensor): radii of 2D gaussian projections.
        cum_tiles_hit (Tensor): list of cumulative tiles hit.
        tile_bounds (Tuple): tile dimensions as a len 3 tuple (tiles.x , tiles.y, 1).

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor}:

        - **isect_ids_unsorted** (Tensor): unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_unsorted** (Tensor): Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **isect_ids_sorted** (Tensor): sorted unique IDs for each gaussian in the form (tile | depth id).
        - **gaussian_ids_sorted** (Tensor): sorted Tensor that maps isect_ids back to cum_tiles_hit. Useful for identifying gaussians.
        - **tile_bins** (Tensor): range of gaussians hit per tile.
    """
    isect_ids, gaussian_ids = map_gaussian_to_intersects(
        num_points,
        num_intersects,
        xys,
        depths,
        radii,
        cum_tiles_hit,
        tile_bounds,
        block_size,
    )
    isect_ids_sorted, sorted_indices = torch.sort(isect_ids)
    gaussian_ids_sorted = torch.gather(gaussian_ids, 0, sorted_indices)
    tile_bins = get_tile_bin_edges(num_intersects, isect_ids_sorted, tile_bounds)
    return isect_ids, gaussian_ids, isect_ids_sorted, gaussian_ids_sorted, tile_bins


def spherical_harmonics(
    degrees_to_use: int,
    viewdirs: Float[Tensor, "*batch 3"],
    coeffs: Float[Tensor, "*batch D C"],
    method: Literal["poly", "fast"] = "fast",
) -> Float[Tensor, "*batch C"]:
    """Compute spherical harmonics

    Note:
        This function is only differentiable to the input coeffs.

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.

    Returns:
        The spherical harmonics.
    """
    assert coeffs.shape[-2] >= num_sh_bases(degrees_to_use)
    assert method in ["poly", "fast"]
    return _SphericalHarmonics.apply(
        method, degrees_to_use, viewdirs.contiguous(), coeffs.contiguous()
    )


def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25


def deg_from_sh(num_bases: int):
    if num_bases == 1:
        return 0
    if num_bases == 4:
        return 1
    if num_bases == 9:
        return 2
    if num_bases == 16:
        return 3
    if num_bases == 25:
        return 4
    assert False, "Invalid number of SH bases"


class _SphericalHarmonics(Function):
    """Compute spherical harmonics

    Args:
        degrees_to_use (int): degree of SHs to use (<= total number available).
        viewdirs (Tensor): viewing directions.
        coeffs (Tensor): harmonic coefficients.
    """

    @staticmethod
    def forward(
        ctx,
        method: Literal["poly", "fast"],
        degrees_to_use: int,
        viewdirs: Float[Tensor, "*batch 3"],
        coeffs: Float[Tensor, "*batch D C"],
    ):
        num_points = coeffs.shape[0]
        ctx.degrees_to_use = degrees_to_use
        degree = deg_from_sh(coeffs.shape[-2])
        ctx.degree = degree
        ctx.method = method
        ctx.save_for_backward(viewdirs)
        return _make_lazy_cuda_func("compute_sh_forward")(
            method, num_points, degree, degrees_to_use, viewdirs, coeffs
        )

    @staticmethod
    def backward(ctx, v_colors: Float[Tensor, "*batch 3"]):
        degrees_to_use = ctx.degrees_to_use
        degree = ctx.degree
        method = ctx.method
        viewdirs = ctx.saved_tensors[0]
        num_points = v_colors.shape[0]
        return (
            None,
            None,
            None,
            _make_lazy_cuda_func("compute_sh_backward")(
                method, num_points, degree, degrees_to_use, viewdirs, v_colors
            ),
        )

def rasterize_gaussians(
    xys: Float[Tensor, "*batch 2"],
    depths: Float[Tensor, "*batch 1"],
    radii: Float[Tensor, "*batch 1"],
    num_tiles_hit: Int[Tensor, "*batch 1"],
    colors: Float[Tensor, "*batch channels"],
    opacity: Float[Tensor, "*batch 1"],
    img_height: int,
    img_width: int,
    block_width: int,
    background: Optional[Float[Tensor, "channels"]] = None,
    return_alpha: Optional[bool] = False,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
    conics: Float[Tensor, "*batch 3"] = None,
    ray_transformations: Float[Tensor, "batch 3 3"] = None,
    normals: Float[Tensor, "*batc 3"] = None,
) -> Tensor:
    """Rasterizes 2D gaussians by sorting and binning gaussian intersections for each tile and returns an N-dimensional output using alpha-compositing.

    Note:
        This function is differentiable w.r.t the xys, conics, colors, and opacity inputs.

    Args:
        xys (Tensor): xy coords of 2D gaussians.
        depths (Tensor): depths of 2D gaussians.
        radii (Tensor): radii of 2D gaussians
        conics (Tensor): conics (inverse of covariance) of 2D gaussians in upper triangular format
        num_tiles_hit (Tensor): number of tiles hit per gaussian
        colors (Tensor): N-dimensional features associated with the gaussians.
        opacity (Tensor): opacity associated with the gaussians.
        img_height (int): height of the rendered image.
        img_width (int): width of the rendered image.
        block_width (int): MUST match whatever block width was used in the project_gaussians call. integer number of pixels between 2 and 16 inclusive
        background (Tensor): background color
        return_alpha (bool): whether to return alpha channel
        model_type (str): which splat model to use; choose from [3dgs, 2dgs]
        ray_transformations: transformations that convert rays in image coordinate into camera coordinate for 2D Gaussian query.
        normals: normals of 2D gaussians.
    Returns:
        A Tensor:

        - **out_img** (Tensor): N-dimensional rendered output image.
        - **out_alpha** (Optional[Tensor]): Alpha channel of the rendered output image.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    if colors.dtype == torch.uint8:
        # make sure colors are float [0,1]
        colors = colors.float() / 255

    if background is not None:
        assert (
            background.shape[0] == colors.shape[-1]
        ), f"incorrect shape of background color tensor, expected shape {colors.shape[-1]}"
    else:
        background = torch.ones(
            colors.shape[-1], dtype=torch.float32, device=colors.device
        )

    if xys.ndimension() != 2 or xys.size(1) != 2:
        raise ValueError("xys must have dimensions (N, 2)")

    if colors.ndimension() != 2:
        raise ValueError("colors must have dimensions (N, D)")

    if model_type == "3dgs":
        return _RasterizeGaussians_3DGS.apply(
            xys.contiguous(),
            depths.contiguous(),
            radii.contiguous(),
            conics.contiguous(),
            num_tiles_hit.contiguous(),
            colors.contiguous(),
            opacity.contiguous(),
            img_height,
            img_width,
            block_width,
            background.contiguous(),
            return_alpha,
        )
    elif model_type == "2dgs":
        return _RasterizeGaussians_2DGS.apply(
            xys.contiguous(),
            depths.contiguous(),
            radii.contiguous(),
            ray_transformations.contiguous(),
            normals.contiguous(),
            num_tiles_hit.contiguous(),
            colors.contiguous(),
            opacity.contiguous(),
            img_height,
            img_width,
            block_width,
            background.contiguous(),
            return_alpha,
        )
    else:
        raise NotImplementedError()

class _RasterizeGaussians_3DGS(Function):
    """Rasterizes 2D gaussians"""

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
        block_width: int,
        background: Float[Tensor, "channels"],
        return_alpha: Optional[bool] = False,
    ) -> Tensor:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )
            if colors.shape[-1] == 3:
                rasterize_fn = _make_lazy_cuda_func("rasterize_forward")
            else:
                rasterize_fn = _make_lazy_cuda_func("nd_rasterize_forward")

            out_img, final_Ts, final_idx = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                opacity,
                background,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        )

        if return_alpha:
            out_alpha = 1 - final_Ts
            return out_img, out_alpha
        else:
            return out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):
        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            conics,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_xy_abs = torch.zeros_like(xys)
            v_conic = torch.zeros_like(conics)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)

        else:
            if colors.shape[-1] == 3:
                rasterize_fn = _make_lazy_cuda_func("rasterize_backward")
            else:
                rasterize_fn = _make_lazy_cuda_func("nd_rasterize_backward")
            v_xy, v_xy_abs, v_conic, v_colors, v_opacity = rasterize_fn(
                img_height,
                img_width,
                ctx.block_width,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                conics,
                colors,
                opacity,
                background,
                final_Ts,
                final_idx,
                v_out_img,
                v_out_alpha,
            )
        v_background = None
        if background.requires_grad:
            v_background = torch.matmul(
                v_out_img.float().view(-1, 3).t(), final_Ts.float().view(-1, 1)
            ).squeeze()

        # Abs grad for gaussian splitting criterion. See
        # - "AbsGS: Recovering Fine Details for 3D Gaussian Splatting"
        # - "EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene Representation"
        xys.absgrad = v_xy_abs

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
            None,  # block_width
            v_background,  # background
            None,  # return_alpha
        )

class _RasterizeGaussians_2DGS(Function):
    """Rasterizes 2D gaussians"""

    @staticmethod
    def forward(
        ctx,
        xys: Float[Tensor, "*batch 2"],
        depths: Float[Tensor, "*batch 1"],
        radii: Float[Tensor, "*batch 1"],
        ray_transformations: Float[Tensor, "*batch 3 3"],
        normals: Float[Tensor, "*batch 3"],
        num_tiles_hit: Int[Tensor, "*batch 1"],
        colors: Float[Tensor, "*batch channels"],
        opacity: Float[Tensor, "*batch 1"],
        img_height: int,
        img_width: int,
        block_width: int,
        background: Optional[Float[Tensor, "channels"]] = None,
        return_alpha: Optional[bool] = False,
    ) -> Tensor:
        num_points = xys.size(0)
        tile_bounds = (
            (img_width + block_width - 1) // block_width,
            (img_height + block_width - 1) // block_width,
            1,
        )
        block = (block_width, block_width, 1)
        img_size = (img_width, img_height, 1)

        num_intersects, cum_tiles_hit = compute_cumulative_intersects(num_tiles_hit)

        # pdb.set_trace()

        if num_intersects < 1:
            out_img = (
                torch.ones(img_height, img_width, colors.shape[-1], device=xys.device)
                * background
            )
            gaussian_ids_sorted = torch.zeros(0, 1, device=xys.device)
            tile_bins = torch.zeros(0, 2, device=xys.device)
            final_Ts = torch.zeros(img_height, img_width, device=xys.device)
            final_idx = torch.zeros(img_height, img_width, device=xys.device)
        else:
            (
                isect_ids_unsorted,
                gaussian_ids_unsorted,
                isect_ids_sorted,
                gaussian_ids_sorted,
                tile_bins,
            ) = bin_and_sort_gaussians(
                num_points,
                num_intersects,
                xys,
                depths,
                radii,
                cum_tiles_hit,
                tile_bounds,
                block_width,
            )

            # pdb.set_trace()
            
            if colors.shape[-1] == 3:
                # rasterize_fn = _C.rasterize_forward_2dgs
                rasterize_fn = _make_lazy_cuda_func("rasterize_forward_2dgs")
            else:
                #TODO (weijia): currently n-d feature rasterization is not supported.
                # rasterize_fn = _C.nd_rasterize_forward
                # rasterize_fn = _C.rasterize_forward_2dgs
                rasterize_fn = _make_lazy_cuda_func("rasterize_forward_2dgs")

            (
                out_img, 
                final_Ts, 
                final_idx,
                out_normal,
                out_distortion,
                out_depth
             ) = rasterize_fn(
                tile_bounds,
                block,
                img_size,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                ray_transformations,
                colors,
                opacity,
                background,
                normals,
            )

        ctx.img_width = img_width
        ctx.img_height = img_height
        ctx.num_intersects = num_intersects
        ctx.block_width = block_width
        ctx.save_for_backward(
            gaussian_ids_sorted,
            tile_bins,
            xys,
            ray_transformations,
            normals,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
            out_normal,
            out_distortion,
            out_depth
        )

        if return_alpha:
            out_alpha = 1 - final_Ts
            return out_img, out_alpha
        else:
            return out_img

    @staticmethod
    def backward(ctx, v_out_img, v_out_alpha=None):

        # pdb.set_trace()

        img_height = ctx.img_height
        img_width = ctx.img_width
        num_intersects = ctx.num_intersects

        if v_out_alpha is None:
            v_out_alpha = torch.zeros_like(v_out_img[..., 0])

        (
            gaussian_ids_sorted,
            tile_bins,
            xys,
            ray_transformations,
            normals,
            colors,
            opacity,
            background,
            final_Ts,
            final_idx,
            out_normal,
            out_distortion,
            out_depth,
        ) = ctx.saved_tensors

        if num_intersects < 1:
            v_xy = torch.zeros_like(xys)
            v_xy_abs = torch.zeros_like(xys)
            # v_conic = torch.zeros_like(conics)
            v_ray_transformations = torch.zeros_like(ray_transformations)
            v_colors = torch.zeros_like(colors)
            v_opacity = torch.zeros_like(opacity)

        else:
            if colors.shape[-1] == 3:
                # rasterize_fn = _C.rasterize_backward_2dgs
                raseterize_fn = _make_lazy_cuda_func("rasterize_backward_2dgs")
            else:
                # rasterize_fn = _C.nd_rasterize_backward
                #TODO (weijia): currently n-d feature rasterization is not supported.
                rasterize_fn = _make_lazy_cuda_func("rasterize_backward_2dgs")

            v_xy, v_xy_abs, v_ray_transformations, v_colors, v_opacity, v_normal = rasterize_fn(
                img_height,
                img_width,
                ctx.block_width,
                gaussian_ids_sorted,
                tile_bins,
                xys,
                ray_transformations,
                colors,
                opacity,
                background,
                final_Ts,
                final_idx,
                v_out_img,
                v_out_alpha,
            )
            # pdb.set_trace()


        # Abs grad for gaussian splitting criterion. See
        # - "AbsGS: Recovering Fine Details for 3D Gaussian Splatting"
        # - "EfficientGS: Streamlining Gaussian Splatting for Large-Scale High-Resolution Scene Representation"
        xys.absgrad = v_xy_abs

        # print("Here")
        # print(v_xy)
        # print(v_ray_transformations)
        # print(v_colors)
        # print(v_opacity)

        return (
            v_xy,  # xys
            None,  # depths
            None,  # radii
            # v_conic,  # conics
            v_ray_transformations, # ray_transformations
            None,  # num_tiles_hit
            v_colors,  # colors
            v_opacity,  # opacity
            None,  # img_height
            None,  # img_width
            None,  # block_width
            None,  # background
            None,  # return_alpha
        )


def project_gaussians(
    means3d: Float[Tensor, "*batch 3"],
    scales: Float[Tensor, "*batch 3"],
    glob_scale: float,
    quats: Float[Tensor, "*batch 4"],
    viewmat: Float[Tensor, "4 4"],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_height: int,
    img_width: int,
    block_width: int,
    clip_thresh: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs"
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """This function projects 3D gaussians to 2D using the EWA splatting method for gaussian splatting.

    Note:
        This function is differentiable w.r.t the means3d, scales and quats inputs.

    Args:
       means3d (Tensor): xyzs of gaussians.
       scales (Tensor): scales of the gaussians.
       glob_scale (float): A global scaling factor applied to the scene.
       quats (Tensor): rotations in normalized quaternion [w,x,y,z] format.
       viewmat (Tensor): view matrix for rendering.
       fx (float): focal length x.
       fy (float): focal length y.
       cx (float): principal point x.
       cy (float): principal point y.
       img_height (int): height of the rendered image.
       img_width (int): width of the rendered image.
       block_width (int): side length of tiles inside projection/rasterization in pixels (always square). 16 is a good default value, must be between 2 and 16 inclusive.
       clip_thresh (float): minimum z depth threshold.
       model_type (str): which splat model to use; choose from [3dgs, 2dgs]

    Returns:
        A tuple of {Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor}:

        ====== for 3DGS ======
        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **conics** (Tensor): conic parameters for 2D gaussian.
        - **compensation** (Tensor): the density compensation for blurring 2D kernel
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
        ====== for 2DGS ======
        - **xys** (Tensor): x,y locations of 2D gaussian projections.
        - **depths** (Tensor): z depth of gaussians.
        - **radii** (Tensor): radii of 2D gaussian projections.
        - **num_tiles_hit** (Tensor): number of tiles hit per gaussian.
        - **cov3d** (Tensor): 3D covariances.
        - **ray_transformations** (Tensor): transformations that convert rays in image coordinate into camera coordinate for 2D Gaussian query.
        - **normals** (Tensor): normals of 2DGS.
    """
    assert block_width > 1 and block_width <= 16, "block_width must be between 2 and 16"
    assert (quats.norm(dim=-1) - 1 < 1e-6).all(), "quats must be normalized"
    
    if model_type == "3dgs":
        return _ProjectGaussians_3DGS.apply(
            means3d.contiguous(),
            scales.contiguous(),
            glob_scale,
            quats.contiguous(),
            viewmat.contiguous(),
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )
    elif model_type == "2dgs":
        return _ProjectGaussians_2DGS.apply(
            means3d.contiguous(),
            scales.contiguous(),
            glob_scale,
            quats.contiguous(),
            viewmat.contiguous(),
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )
    else:
        raise NotImplementedError()

class _ProjectGaussians_3DGS(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            radii,
            conics,
            compensation,
            num_tiles_hit,
        ) = _make_lazy_cuda_func("project_gaussians_forward")(
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        )

        return (xys, depths, radii, conics, compensation, num_tiles_hit, cov3d)

    @staticmethod
    def backward(
        ctx,
        v_xys,
        v_depths,
        v_radii,
        v_conics,
        v_compensation,
        v_num_tiles_hit,
        v_cov3d,
    ):
        (
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            conics,
            compensation,
        ) = ctx.saved_tensors

        (v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat) = _make_lazy_cuda_func(
            "project_gaussians_backward"
        )(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            conics,
            compensation,
            v_xys,
            v_depths,
            v_conics,
            v_compensation,
        )

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = g(R @ mean3d + t,
            #           R @ cov3d_world(q, s) @ R^T ))
            #
            # Then, the Jacobian w.r.t., t is:
            #
            #   d f / d t = df / d mean3d @ R^T
            #
            # and, in the context of fine tuning camera poses, it is reasonable
            # to assume that
            #
            #   d f / d R_ij =~ \sum_l d f / d t_l * d (R @ mean3d)_l / d R_ij
            #                = d f / d_t_i * mean3d[j]
            #
            # Gradients for R and t can then be obtained by summing over
            # all the Gaussians.
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(
                        v_mean3d_cam[..., j], means3d[..., l]
                    )
        else:
            v_viewmat = None

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
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            # clip_thresh,
            None,
        )
        
class _ProjectGaussians_2DGS(Function):
    """Project 3D gaussians to 2D."""

    @staticmethod
    def forward(
        ctx,
        means3d: Float[Tensor, "*batch 3"],
        scales: Float[Tensor, "*batch 3"],
        glob_scale: float,
        quats: Float[Tensor, "*batch 4"],
        viewmat: Float[Tensor, "4 4"],
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        img_height: int,
        img_width: int,
        block_width: int,
        clip_thresh: float = 0.01,
    ):
        num_points = means3d.shape[-2]
        if num_points < 1 or means3d.shape[-1] != 3:
            raise ValueError(f"Invalid shape for means3d: {means3d.shape}")

        (
            cov3d,
            xys,
            depths,
            radii,
            num_tiles_hit,
            ray_transformations,
            normals,
        ) = _make_lazy_cuda_func(
            "project_gaussians_forward_2dgs") 
        (
            num_points,
            means3d,
            scales,
            glob_scale,
            quats,
            viewmat,
            fx,
            fy,
            cx,
            cy,
            img_height,
            img_width,
            block_width,
            clip_thresh,
        )

        # Save non-tensors.
        ctx.img_height = img_height
        ctx.img_width = img_width
        ctx.num_points = num_points
        ctx.glob_scale = glob_scale
        ctx.fx = fx
        ctx.fy = fy
        ctx.cx = cx
        ctx.cy = cy

        # Save tensors.
        ctx.save_for_backward(
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            ray_transformations,
            normals,
        )
        # pdb.set_trace()
        return (xys, depths, radii, num_tiles_hit, cov3d, ray_transformations, normals)

    @staticmethod
    def backward(
        ctx,
        dL_dxys,
        dL_ddepths,
        dL_dradii,
        dL_dnum_tile_hit,
        dL_dcov3d,
        dL_dray_transformations
        # v_xys,
        # v_depths,
        # v_radii,
        # v_conics,
        # v_compensation,
        # v_num_tiles_hit,
        # v_cov3d,
    ):
        # pdb.set_trace()
        (
            means3d,
            scales,
            quats,
            viewmat,
            cov3d,
            radii,
            ray_transformations,
            normals,
        ) = ctx.saved_tensors

        # pdb.set_trace()
        (v_mean3d, v_scale, v_quat, v_normal2d) = _C.project_gaussians_backward_2dgs(
            ctx.num_points,
            means3d,
            scales,
            ctx.glob_scale,
            quats,
            viewmat,
            ray_transformations,
            ctx.fx,
            ctx.fy,
            ctx.cx,
            ctx.cy,
            ctx.img_height,
            ctx.img_width,
            cov3d,
            radii,
            dL_dray_transformations,
            # dL_dnormal3Ds
            # conics,
            # compensation,
            # v_xys,
            # v_depths,
            # v_conics,
            # v_compensation,
        )

        # pdb.set_trace()

        if viewmat.requires_grad:
            v_viewmat = torch.zeros_like(viewmat)
            R = viewmat[..., :3, :3]

            # Denote ProjectGaussians for a single Gaussian (mean3d, q, s)
            # viemwat = [R, t] as:
            #
            #   f(mean3d, q, s, R, t, intrinsics)
            #       = g(R @ mean3d + t,
            #           R @ cov3d_world(q, s) @ R^T ))
            #
            # Then, the Jacobian w.r.t., t is:
            #
            #   d f / d t = df / d mean3d @ R^T
            #
            # and, in the context of fine tuning camera poses, it is reasonable
            # to assume that
            #
            #   d f / d R_ij =~ \sum_l d f / d t_l * d (R @ mean3d)_l / d R_ij
            #                = d f / d_t_i * mean3d[j]
            #
            # Gradients for R and t can then be obtained by summing over
            # all the Gaussians.
            v_mean3d_cam = torch.matmul(v_mean3d, R.transpose(-1, -2))

            # gradient w.r.t. view matrix translation
            v_viewmat[..., :3, 3] = v_mean3d_cam.sum(-2)

            # gradent w.r.t. view matrix rotation
            for j in range(3):
                for l in range(3):
                    v_viewmat[..., j, l] = torch.dot(
                        v_mean3d_cam[..., j], means3d[..., l]
                    )
        else:
            v_viewmat = None

        v_scale = torch.concatenate((v_scale, torch.zeros((v_scale.shape[0], 1), device="cuda:0")), dim=-1)
        # pdb.set_trace()
        # Return a gradient for each input.
        # print(v_mean3d)
        # print(v_scale)
        # print(v_quat)
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
            v_viewmat,
            # fx: float,
            None,
            # fy: float,
            None,
            # cx: float,
            None,
            # cy: float,
            None,
            # img_height: int,
            None,
            # img_width: int,
            None,
            # block_width: int,
            None,
            # clip_thresh,
            None,
        )