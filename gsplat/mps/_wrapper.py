import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal


def _make_lazy_mps_func(name: str) -> Callable:
    def call_mps(*args, **kwargs):
        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    return call_mps


def _make_lazy_mps_obj(name: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    obj = _C
    for name_split in name.split("."):
        obj = getattr(_C, name_split)
    return obj


class RollingShutterType(Enum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4

    def to_cpp(self) -> Any:
        return _make_lazy_mps_obj(f"ShutterType.{self.name}")


@dataclass
class UnscentedTransformParameters:
    alpha: float = 0.1
    beta: float = 2.0
    kappa: float = 0.0
    in_image_margin_factor: float = 0.1
    require_all_sigma_points_valid: bool = True

    def to_cpp(self) -> Any:
        p = _make_lazy_mps_obj("UnscentedTransformParameters")()
        p.alpha = self.alpha
        p.beta = self.beta
        p.kappa = self.kappa
        p.in_image_margin_factor = self.in_image_margin_factor
        p.require_all_sigma_points_valid = self.require_all_sigma_points_valid
        return p


@dataclass
class FThetaPolynomialType(Enum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1

    def to_cpp(self) -> Any:
        return _make_lazy_mps_obj(f"FThetaPolynomialType.{self.name}")


@dataclass
class FThetaCameraDistortionParameters:
    reference_poly: FThetaPolynomialType
    pixeldist_to_angle_poly: Tuple[float, float, float, float, float, float]
    angle_to_pixeldist_poly: Tuple[float, float, float, float, float, float]
    max_angle: float
    linear_cde: Tuple[float, float, float]

    def to_cpp(self) -> Any:
        p = _make_lazy_mps_obj("FThetaCameraDistortionParameters")()
        p.reference_poly = self.reference_poly.to_cpp()
        p.pixeldist_to_angle_poly = self.pixeldist_to_angle_poly
        p.angle_to_pixeldist_poly = self.angle_to_pixeldist_poly
        p.max_angle = self.max_angle
        p.linear_cde = self.linear_cde
        return p

    @classmethod
    def to_cpp_default(cls) -> Any:
        p = _make_lazy_mps_obj("FThetaCameraDistortionParameters")()
        return p


def world_to_cam(
    means: Tensor,
    covars: Tensor,
    viewmats: Tensor,
) -> Tuple[Tensor, Tensor]:
    from ..cuda._torch_impl import _world_to_cam

    warnings.warn(
        "world_to_cam() is removed from the backend as it's relatively easy to "
        "implement in PyTorch. Currently use the PyTorch implementation instead. "
        "This function will be completely removed in a future release.",
        DeprecationWarning,
    )
    return _world_to_cam(means, covars, viewmats)


def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,
    coeffs: Tensor,
    masks: Optional[Tensor] = None,
) -> Tensor:
    return _make_lazy_mps_func("compute_sh_forward")(
        degrees_to_use, dirs, coeffs, masks
    )


def quat_scale_to_covar_preci(
    quats: Tensor,
    scales: Tensor,
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    raise NotImplementedError("quat_scale_to_covar_preci not implemented for MPS yet")


def proj(
    means: Tensor,
    covars: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
) -> Tuple[Tensor, Tensor]:
    return _make_lazy_mps_func("project_gaussians_forward")(
        means, covars, Ks, width, height, camera_model
    )


def fully_fused_projection(
    means: Tensor,
    covars: Optional[Tensor],
    quats: Optional[Tensor],
    scales: Optional[Tensor],
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye", "ftheta"] = "pinhole",
    opacities: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    raise NotImplementedError("fully_fused_projection not implemented for MPS yet")


def isect_tiles(
    means2d: Tensor,
    radii: Tensor,
    depths: Tensor,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError("isect_tiles not implemented for MPS yet")


def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    raise NotImplementedError("isect_offset_encode not implemented for MPS yet")


def rasterize_to_pixels(
    means2d: Tensor,
    conics: Tensor,
    colors: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    return _make_lazy_mps_func("rasterize_forward")(
        means2d,
        conics,
        colors,
        opacities,
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets,
        flatten_ids,
    )


def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,
    means2d: Tensor,
    conics: Tensor,
    opacities: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    raise NotImplementedError(
        "rasterize_to_indices_in_range not implemented for MPS yet"
    )


def fully_fused_projection_2dgs(*args, **kwargs):
    raise NotImplementedError("fully_fused_projection_2dgs not implemented for MPS yet")


def rasterize_to_pixels_2dgs(*args, **kwargs):
    raise NotImplementedError("rasterize_to_pixels_2dgs not implemented for MPS yet")


def rasterize_to_indices_in_range_2dgs(*args, **kwargs):
    raise NotImplementedError(
        "rasterize_to_indices_in_range_2dgs not implemented for MPS yet"
    )


def accumulate(*args, **kwargs):
    raise NotImplementedError("accumulate not implemented for MPS yet")


def accumulate_2dgs(*args, **kwargs):
    raise NotImplementedError("accumulate_2dgs not implemented for MPS yet")


def fully_fused_projection_with_ut(*args, **kwargs):
    raise NotImplementedError(
        "fully_fused_projection_with_ut not implemented for MPS yet"
    )


def rasterize_to_pixels_eval3d(*args, **kwargs):
    raise NotImplementedError("rasterize_to_pixels_eval3d not implemented for MPS yet")
