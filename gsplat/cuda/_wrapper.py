# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import types
import warnings
from dataclasses import dataclass
from enum import IntEnum
from abc import ABC
from typing import Any, Callable, Mapping, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Literal
from gsplat.trace import trace_function
from gsplat.cuda._lidar import (
    SpinningDirection,
    LidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParameters,
    RowOffsetStructuredSpinningLidarModelParametersExt as RowOffsetStructuredSpinningLidarModelParametersExtBase,
    FOV as FOVBase,
)

ExternalDistortionModelMeta = Literal["bivariate-windshield"]
CameraModel = Literal["pinhole", "ortho", "fisheye", "ftheta", "lidar"]

# Autograd for the migrated ops is attached in Python (torch.library.register_autograd)
# rather than C++. The C++ module exports only each op's `<op>_fwd` and `<op>_bwd`; the
# backward is wired to the forward op when this module is imported (the call at the end
# of the file), so direct torch.ops.gsplat.* callers get autograd too, not only the
# Python wrappers.
_AUTOGRAD_REGISTRATIONS_DONE = False


def _make_lazy_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        # The following import statement is required to ensure that C++ module
        # gsplat/csrc.so is loaded (and JIT-compiled if necessary). Upon module
        # load, the gsplat PyTorch operators are imported into the
        # torch.ops.gsplat submodule.

        # pylint: disable=import-outside-toplevel
        from ._backend import _C

        if _C is not None:
            _ensure_autograd_registrations()
        return getattr(torch.ops.gsplat, name)(*args, **kwargs)

    return call_cuda


def _has_schema(op_name: str) -> bool:
    """Whether `gsplat::<op_name>` is registered (it may be compiled out by build flags)."""
    try:
        torch._C._dispatch_find_schema_or_throw(f"gsplat::{op_name}", "")
    except RuntimeError:
        return False
    return True


def _register_autograd(register: type) -> None:
    """Attach a Python autograd backward to the C++ forward op ``gsplat::<base>``.

    ``register`` is a per-op class grouping the op's hooks: its ``base`` op name and
    static ``setup_context`` / ``backward`` methods. Registering autograd on the
    forward op makes the op itself differentiable through the dispatcher, so its
    backward is recorded whenever it runs under autograd; ``backward`` invokes
    ``torch.ops.gsplat.<base>_bwd``. No-op if either op was compiled out.
    """
    base = register.base
    if not (_has_schema(base) and _has_schema(f"{base}_bwd")):
        return
    torch.library.register_autograd(
        f"gsplat::{base}", register.backward, setup_context=register.setup_context
    )


def _ensure_autograd_registrations() -> None:
    """Install the Python autograd backends for the migrated ops, exactly once.

    Populated one op at a time as each C++ autograd Function is ported to Python.
    """
    global _AUTOGRAD_REGISTRATIONS_DONE
    if _AUTOGRAD_REGISTRATIONS_DONE:
        return
    _register_autograd(RegisterSphericalHarmonics)
    _register_autograd(RegisterSphericalHarmonicsL0)
    _register_autograd(RegisterSphericalHarmonicsL1Plus)
    _register_autograd(RegisterQuatScaleToCovarPreci)
    _register_autograd(RegisterProjectionEWASimple)
    _register_autograd(RegisterProjectionEWA3DGSFused)
    _register_autograd(RegisterProjectionEWA3DGSPacked)
    _register_autograd(RegisterProjection2DGSFused)
    _register_autograd(RegisterProjection2DGSPacked)
    _register_autograd(RegisterRasterizeToPixels3DGS)
    _register_autograd(RegisterRasterizeToPixels2DGS)
    _register_autograd(RegisterRasterizeToPixelsSparse)
    _AUTOGRAD_REGISTRATIONS_DONE = True


def _make_lazy_cuda_cls(name: str) -> Any:
    # The following import statement is required to ensure that C++ module
    # gsplat/csrc.so is loaded (and JIT-compiled if necessary). Upon module
    # load, the gsplat PyTorch custom classes are imported into the
    # torch.classes.gsplat submodule.

    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    if _C is None:
        return _unavailable_cuda_cls(name)

    try:
        return getattr(torch.classes.gsplat, name)
    except RuntimeError as e:
        # Class not registered (e.g. extension built without it or partial load).
        if "does not exist" in str(e) or "torch::class_" in str(e):
            return _unavailable_cuda_cls(name)
        raise


def _unavailable_cuda_cls(name: str) -> Any:
    """Placeholder class when the CUDA extension is not available."""

    class _UnavailableCudaCls:
        __name__ = name

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "gsplat CUDA extension is not available (not built or failed to load). "
                f"Cannot instantiate '{name}'."
            )

    return _UnavailableCudaCls


def _make_lazy_cuda_obj(name: str) -> Any:
    # pylint: disable=import-outside-toplevel
    from ._backend import _C

    if _C is None:
        raise RuntimeError(
            "gsplat CUDA extension is not available (not built or failed to load). "
            f"Cannot access '{name}'."
        )
    obj = _C
    for name_split in name.split("."):
        obj = getattr(obj, name_split)
    return obj


def renderer_config_mixed_batch() -> Any:
    """Return the CUDA enum value for the MixedBatch renderer config."""
    return _make_lazy_cuda_obj("RendererConfig.MIXED_BATCH")


def renderer_config_parallel_batch() -> Any:
    """Return the CUDA enum value for the ParallelBatch renderer config."""
    return _make_lazy_cuda_obj("RendererConfig.PARALLEL_BATCH")


def _renderer_config_to_cuda(renderer_config: Any) -> Any:
    if renderer_config is None:
        return renderer_config_mixed_batch()

    # RendererConfig lives in gsplat.rendering, which imports this module.
    # Import lazily here so the public low-level wrapper accepts the same
    # config objects as gsplat.rasterization without creating an import cycle.
    from gsplat.rendering import (  # pylint: disable=import-outside-toplevel
        RendererConfig,
        _renderer_config_type,
    )

    if isinstance(renderer_config, RendererConfig):
        # Delegate the config -> CUDA enum mapping to the single source of
        # truth in gsplat.rendering (which raises for unknown subtypes).
        return _renderer_config_type(renderer_config)
    return renderer_config


class RollingShutterType(IntEnum):
    ROLLING_TOP_TO_BOTTOM = 0
    ROLLING_LEFT_TO_RIGHT = 1
    ROLLING_BOTTOM_TO_TOP = 2
    ROLLING_RIGHT_TO_LEFT = 3
    GLOBAL = 4


class FThetaPolynomialType(IntEnum):
    PIXELDIST_TO_ANGLE = 0
    ANGLE_TO_PIXELDIST = 1


UnscentedTransformParameters = _make_lazy_cuda_cls("UnscentedTransformParameters")
FThetaCameraDistortionParameters = _make_lazy_cuda_cls(
    "FThetaCameraDistortionParameters"
)


class ExternalDistortionModelParameters(ABC):
    """Base class for external distortion model parameters.

    All concrete external distortion models (e.g. BivariateWindshieldModelParameters)
    should inherit from this class so that the rendering API can accept any
    distortion model through a single type-erased parameter.
    """


class ExternalDistortionReferencePolynomial(IntEnum):
    FORWARD = 1
    BACKWARD = 2


class BivariateWindshieldModelParameters(ExternalDistortionModelParameters):
    """Thin wrapper around the CUDA BivariateWindshieldModelParameters class.

    torch::Library bindings does not allow standalone constants. This
    wrapper fetches MAX_ORDER and MAX_COEFFS from the C++ static getters
    and exposes them as class-level attributes, preserving the existing
    attribute-access calling convention.
    """

    _cuda_cls = None
    MAX_ORDER: int = 5  # default, overriden by C++ value
    MAX_COEFFS: int = 21  # default, overriden by C++ value

    @classmethod
    def _ensure_cuda_cls(cls):
        if cls._cuda_cls is None:
            cls._cuda_cls = _make_lazy_cuda_cls("BivariateWindshieldModelParameters")
            cls.MAX_ORDER = cls._cuda_cls.get_max_order()
            cls.MAX_COEFFS = cls._cuda_cls.get_max_coeffs()

    def __new__(cls):
        cls._ensure_cuda_cls()
        return cls._cuda_cls()


@functools.lru_cache(maxsize=1)
def _build_config() -> Mapping[str, bool]:
    try:
        from ._backend import _C

        return (
            types.MappingProxyType(_C.build_config())
            if _C is not None
            else types.MappingProxyType({})
        )
    except (ImportError, AttributeError):
        return types.MappingProxyType({})


def _has_build_feature(name: str) -> bool:
    return _build_config().get(name, False)


def has_camera_wrappers():
    return _has_build_feature("camera_wrappers")


def has_2dgs():
    return _has_build_feature("2dgs")


def has_3dgs():
    return _has_build_feature("3dgs")


def has_3dgut():
    return _has_build_feature("3dgut")


def has_adam():
    return _has_build_feature("adam")


def has_reloc():
    return _has_build_feature("reloc")


def has_losses():
    return _has_build_feature("losses")


def create_camera_model(
    camera_model: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    principal_points: Optional[Tensor] = None,
    focal_lengths: Optional[Tensor] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    rs_type: RollingShutterType = RollingShutterType.GLOBAL,
    lidar_coeffs: Optional["RowOffsetStructuredSpinningLidarModelParametersExt"] = None,
):
    if camera_model == "lidar":
        assert (
            lidar_coeffs is not None
        ), "lidar_coeffs is required for lidar camera model"
        RowOffsetStructuredSpinningLidarModelCUDA = _make_lazy_cuda_cls(
            "RowOffsetStructuredSpinningLidarModel"
        )
        return RowOffsetStructuredSpinningLidarModelCUDA(lidar_coeffs.to_cpp())
    else:
        assert width is not None, "width is required for non-lidar camera models"
        assert height is not None, "height is required for non-lidar camera models"
        assert (
            principal_points is not None
        ), "principal_points is required for non-lidar camera models"
        BaseCameraModelCUDA = _make_lazy_cuda_cls("BaseCameraModel")
        return BaseCameraModelCUDA.create(
            width,
            height,
            camera_model,
            principal_points,
            focal_lengths,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            external_distortion_coeffs,
            rs_type,
        )


class FOV(FOVBase):
    @classmethod
    def from_base(cls, base: FOVBase) -> "FOV":
        return cls(start=base.start, span=base.span, direction=base.direction)

    def to_cpp(self):
        FOVCUDA = _make_lazy_cuda_cls("FOV")
        return FOVCUDA(start=self.start, span=self.span)


class RowOffsetStructuredSpinningLidarModelParametersExt(
    RowOffsetStructuredSpinningLidarModelParametersExtBase
):
    """Lidar camera parameters extended with acceleration structures"""

    def to_cpp(self) -> Any:
        """Convert to C++ custom class instance."""
        LidarParamsCUDA = _make_lazy_cuda_cls(
            "RowOffsetStructuredSpinningLidarModelParametersExt"
        )
        return LidarParamsCUDA(
            row_elevations_rad=self.row_elevations_rad.contiguous(),
            column_azimuths_rad=self.column_azimuths_rad.contiguous(),
            row_azimuth_offsets_rad=self.row_azimuth_offsets_rad.contiguous(),
            spinning_direction=self.spinning_direction.value,
            spinning_frequency_hz=self.spinning_frequency_hz,
            fov_vert_rad=FOV.from_base(self.fov_vert_rad).to_cpp(),
            fov_horiz_rad=FOV.from_base(self.fov_horiz_rad).to_cpp(),
            fov_eps_rad=self.fov_eps_rad,
            angles_to_columns_map=self.angles_to_columns_map,
            n_bins_azimuth=self.tiling.n_bins_azimuth,
            n_bins_elevation=self.tiling.n_bins_elevation,
            cdf_elevation=self.tiling.cdf_elevation.contiguous(),
            cdf_dense_ray_mask=self.tiling.cdf_dense_ray_mask.contiguous(),
            tiles_to_elements_map=self.tiling.tiles_to_elements_map.contiguous(),
            tiles_pack_info=self.tiling.tiles_pack_info.contiguous(),
        )


def world_to_cam(
    means: Tensor,  # [..., N, 3]
    covars: Tensor,  # [..., N, 3, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
) -> Tuple[Tensor, Tensor]:
    """Transforms Gaussians from world to camera coordinate system.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances. [..., N, 3, 3]
        viewmats: World-to-camera transformation matrices. [..., C, 4, 4]

    Returns:
        A tuple:

        - **Gaussian means in camera coordinate system**. [..., C, N, 3]
        - **Gaussian covariances in camera coordinate system**. [..., C, N, 3, 3]
    """
    from ._torch_impl import _world_to_cam

    warnings.warn(
        "world_to_cam() is removed from the CUDA backend as it's relatively easy to "
        "implement in PyTorch. Currently use the PyTorch implementation instead. "
        "This function will be completely removed in a future release.",
        DeprecationWarning,
    )
    batch_dims = means.shape[:-2]
    N = means.shape[-2]
    C = viewmats.shape[-3]
    assert means.shape == batch_dims + (N, 3), means.shape
    assert covars.shape == batch_dims + (N, 3, 3), covars.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    means = means.contiguous()
    covars = covars.contiguous()
    viewmats = viewmats.contiguous()
    return _world_to_cam(means, covars, viewmats)


def adam(
    param: Tensor,
    param_grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    valid: Tensor,
    lr: float,
    b1: float,
    b2: float,
    eps: float,
) -> None:
    _make_lazy_cuda_func("adam")(
        param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps
    )


@trace_function("sh-fwd")
def spherical_harmonics(
    degrees_to_use: int,
    dirs: Tensor,  # [..., N, 3]
    coeffs: Tensor,  # [N, K, D]
    masks: Optional[Tensor] = None,  # [..., N]
) -> Tensor:
    """Computes spherical harmonics.

    The output channel count ``D`` is taken from the last dim of ``coeffs`` and
    can be any positive integer (e.g. 3 for RGB, 1 for scalar features).

    In packed mode, callers pre-gather coeffs by ``gaussian_ids`` so ``N`` is
    ``nnz`` and ``dirs`` has no leading dims.

    Args:
        degrees_to_use: SH degree to evaluate.
        dirs: View directions. ``[..., N, 3]``; any leading shape, rank ≥ 2.
        coeffs: SH coefficients. ``[N, K, D]``, with ``N`` matching ``dirs.shape[-2]``.
        masks: Optional boolean masks. ``[..., N]`` matching ``dirs.shape[:-1]``.

    Returns:
        Spherical harmonics. ``[..., N, D]``.
    """
    if masks is not None:
        masks = masks.contiguous()
    return _make_lazy_cuda_func("spherical_harmonics")(
        degrees_to_use, dirs.contiguous(), coeffs.contiguous(), masks
    )


@trace_function("sh-l0-fwd")
def spherical_harmonics_l0(
    sh0: Tensor,  # [N, 1, D]
) -> Tensor:
    """Computes the l=0 component of spherical harmonics.

    Args:
        sh0: SH coefficients for l=0. ``[N, 1, D]``.

    Returns:
        l=0 features. ``[N, D]``.
    """
    return _make_lazy_cuda_func("spherical_harmonics_l0")(sh0.contiguous())


@trace_function("sh-l1-plus-fwd")
def spherical_harmonics_l1_plus(
    degrees_to_use: int,
    dirs: Tensor,  # [..., N, 3]
    shN: Tensor,  # [N, K - 1, D]
    masks: Optional[Tensor] = None,  # [..., N]
) -> Tensor:
    """Computes the l>=1 components of spherical harmonics.

    Unlike :func:`spherical_harmonics`, ``shN`` starts at the degree-one SH
    basis; the degree-zero coefficient is intentionally omitted.

    Args:
        degrees_to_use: SH degree to evaluate.
        dirs: View directions. ``[..., N, 3]``; any leading shape, rank >= 2.
        shN: SH coefficients for l>=1. ``[N, K - 1, D]``.
        masks: Optional boolean masks. ``[..., N]`` matching
            ``dirs.shape[:-1]``.

    Returns:
        l>=1 features. ``[..., N, D]``.
    """
    if masks is not None:
        masks = masks.contiguous()
    return _make_lazy_cuda_func("spherical_harmonics_l1_plus")(
        degrees_to_use, dirs.contiguous(), shN.contiguous(), masks
    )


class RegisterSphericalHarmonics:
    """Python autograd hooks for the gsplat::spherical_harmonics op."""

    base = "spherical_harmonics"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        degrees_to_use, dirs, coeffs, masks = inputs
        ctx.degrees_to_use = degrees_to_use
        # save_for_backward round-trips None as None, so the optional masks save directly.
        ctx.save_for_backward(dirs, coeffs, masks)

    @classmethod
    def backward(cls, ctx, v_colors: Tensor):
        dirs, coeffs, masks = ctx.saved_tensors
        # dirs is forward input index 1; its gradient is wanted only when requested.
        compute_v_dirs = ctx.needs_input_grad[1]
        v_coeffs, v_dirs = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            ctx.degrees_to_use,
            dirs,
            coeffs,
            masks,
            v_colors,
            compute_v_dirs,
        )
        return (
            None,  # degrees_to_use
            v_dirs,
            v_coeffs,
            None,  # masks
        )


class RegisterSphericalHarmonicsL0:
    """Python autograd hooks for the l=0 SH op."""

    base = "spherical_harmonics_l0"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (sh0,) = inputs
        ctx.save_for_backward(sh0)

    @classmethod
    def backward(cls, ctx, v_colors: Tensor):
        (sh0,) = ctx.saved_tensors
        return (_make_lazy_cuda_func(f"{cls.base}_bwd")(sh0, v_colors),)


class RegisterSphericalHarmonicsL1Plus(RegisterSphericalHarmonics):
    """Python autograd hooks for the l>=1 SH op."""

    base = "spherical_harmonics_l1_plus"


def quat_scale_to_covar_preci(
    quats: Tensor,  # [..., 4],
    scales: Tensor,  # [..., 3],
    compute_covar: bool = True,
    compute_preci: bool = True,
    triu: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """Converts quaternions and scales to covariance and precision matrices.

    Args:
        quats: Quaternions (No need to be normalized). [..., 4]
        scales: Scales. [..., 3]
        compute_covar: Whether to compute covariance matrices. Default: True. If False,
            the returned covariance matrices will be None.
        compute_preci: Whether to compute precision matrices. Default: True. If False,
            the returned precision matrices will be None.
        triu: If True, the return matrices will be upper triangular. Default: False.

    Returns:
        A tuple:

        - **Covariance matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
        - **Precision matrices**. If `triu` is True the returned shape is [..., 6], otherwise [..., 3, 3].
    """
    quats = quats.contiguous()
    scales = scales.contiguous()
    return _make_lazy_cuda_func("quat_scale_to_covar_preci")(
        quats, scales, compute_covar, compute_preci, triu
    )


class RegisterQuatScaleToCovarPreci:
    """Python autograd hooks for the gsplat::quat_scale_to_covar_preci op."""

    base = "quat_scale_to_covar_preci"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        quats, scales, _compute_covar, _compute_preci, triu = inputs
        ctx.triu = triu
        ctx.save_for_backward(quats, scales)

    @classmethod
    def backward(cls, ctx, v_covars, v_precis):
        quats, scales = ctx.saved_tensors
        # A disabled output has a None grad, which passes through unchanged.
        v_quats, v_scales = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            quats,
            scales,
            ctx.triu,
            v_covars,
            v_precis,
        )
        return (
            v_quats,
            v_scales,
            None,  # compute_covar
            None,  # compute_preci
            None,  # triu
        )


def persp_proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
) -> Tuple[Tensor, Tensor]:
    """Perspective projection on Gaussians.
    DEPRECATED: please use `proj` with `ortho=False` instead.

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    warnings.warn(
        "persp_proj is deprecated and will be removed in a future release. "
        "Use proj with ortho=False instead.",
        DeprecationWarning,
    )
    return proj(means, covars, Ks, width, height, ortho=False)


def proj(
    means: Tensor,  # [..., C, N, 3]
    covars: Tensor,  # [..., C, N, 3, 3]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    camera_model: CameraModel = "pinhole",
) -> Tuple[Tensor, Tensor]:
    """Projection of Gaussians (perspective or orthographic).

    Args:
        means: Gaussian means. [..., C, N, 3]
        covars: Gaussian covariances. [..., C, N, 3, 3]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.

    Returns:
        A tuple:

        - **Projected means**. [..., C, N, 2]
        - **Projected covariances**. [..., C, N, 2, 2]
    """
    means = means.contiguous()
    covars = covars.contiguous()
    Ks = Ks.contiguous()
    camera_model_type = _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")
    return _make_lazy_cuda_func("projection_ewa_simple")(
        means, covars, Ks, width, height, camera_model_type
    )


class RegisterProjectionEWASimple:
    """Python autograd hooks for the gsplat::projection_ewa_simple op."""

    base = "projection_ewa_simple"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        means, covars, Ks, width, height, camera_model = inputs
        ctx.width = width
        ctx.height = height
        ctx.camera_model = camera_model
        ctx.save_for_backward(means, covars, Ks)

    @classmethod
    def backward(cls, ctx, v_means2d, v_covars2d):
        means, covars, Ks = ctx.saved_tensors
        v_means, v_covars = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            means,
            covars,
            Ks,
            ctx.width,
            ctx.height,
            ctx.camera_model,
            v_means2d,
            v_covars2d,
        )
        return (
            v_means,
            v_covars,
            None,  # Ks
            None,  # width
            None,  # height
            None,  # camera_model
        )


@trace_function("project-fwd")
def fully_fused_projection(
    means: Tensor,  # [..., N, 3]
    covars: Optional[Tensor],  # [..., N, 6] or None
    quats: Optional[Tensor],  # [..., N, 4] or None
    scales: Optional[Tensor],  # [..., N, 3] or None
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    opacities: Optional[Tensor] = None,  # [..., N] or None
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D.

    This function fuse the process of computing covariances
    (:func:`quat_scale_to_covar_preci()`), transforming to camera space (:func:`world_to_cam()`),
    and projection (:func:`proj()`).

    .. note::

        During projection, we ignore the Gaussians that are outside of the camera frustum.
        So not all the elements in the output tensors are valid. The output `radii` could serve as
        an indicator, in which zero radii means the corresponding elements are invalid in
        the output tensors and will be ignored in the next rasterization process. If `packed=True`,
        the output tensors will be packed into a flattened tensor, in which all elements are valid.
        In this case, a `batch_ids` tensor and `camera_ids` tensor will be returned to indicate the
        batch, camera and gaussian indices of the packed flattened tensor, which is essentially following the
        COO sparse tensor format.

    .. note::

        This functions supports projecting Gaussians with either covariances or {quaternions, scales},
        which will be converted to covariances internally in a fused CUDA kernel. Either `covars` or
        {`quats`, `scales`} should be provided.

    Args:
        means: Gaussian means. [..., N, 3]
        covars: Gaussian covariances (flattened upper triangle). [..., N, 6] Optional.
        quats: Quaternions (No need to be normalized). [..., N, 4] Optional.
        scales: Scales. [..., N, 3] Optional.
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        eps2d: A epsilon added to the 2D covariance for numerical stability. Default: 0.3.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 1e10.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad: This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.
        calc_compensations: If True, a view-dependent opacity compensation factor will be computed, which
          is useful for anti-aliasing. Default: False.
        opacities: Gaussian opacities in range [0, 1]. If provided, will use it to compute a tighter bounds.
            [..., N] or None. Default: None.

    Returns:
        A tuple:

        If `packed` is True:

        - **batch_ids**. The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **camera_ids**. The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **indptr**. CSR-style index pointer into gaussian_ids for batch-camera pairs. Int32 tensor of shape [B*C+1].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [nnz, 3]
        - **compensations**. The view-dependent opacity compensation factor. [nnz]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **conics**. Inverse of the projected covariances. Return the flattend upper triangle with [..., C, N, 3]
        - **compensations**. The view-dependent opacity compensation factor. [..., C, N]
    """
    means = means.contiguous()
    if covars is not None:
        covars = covars.contiguous()
    else:
        if quats is not None:
            quats = quats.contiguous()
        if scales is not None:
            scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    if opacities is not None:
        opacities = opacities.contiguous()

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )
        return _make_lazy_cuda_func("projection_ewa_3dgs_packed")(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
            calc_compensations,
            camera_model_type,
        )
    else:
        camera_model_type = _make_lazy_cuda_obj(
            f"CameraModelType.{camera_model.upper()}"
        )
        return _make_lazy_cuda_func("projection_ewa_3dgs_fused")(
            means,
            covars,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
            calc_compensations,
            camera_model_type,
        )


class RegisterProjectionEWA3DGSFused:
    """Python autograd hooks for the gsplat::projection_ewa_3dgs_fused op."""

    base = "projection_ewa_3dgs_fused"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means,
            covars,
            quats,
            scales,
            _opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            _near_plane,
            _far_plane,
            _radius_clip,
            _calc_compensations,
            camera_model,
        ) = inputs
        radii, _means2d, _depths, conics, compensations = output
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model = camera_model
        # covars / quats / scales / compensations may be None; save_for_backward
        # round-trips None as None.
        ctx.save_for_backward(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        )

    @classmethod
    def backward(cls, ctx, v_radii, v_means2d, v_depths, v_conics, v_compensations):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            conics,
            compensations,
        ) = ctx.saved_tensors
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            f"{cls.base}_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            ctx.camera_model,
            radii,
            conics,
            compensations,
            v_means2d,
            v_depths,
            v_conics,
            v_compensations,
            ctx.needs_input_grad[
                5
            ],  # viewmats_requires_grad (viewmats is input index 5)
        )
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,  # opacities
            v_viewmats,
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # calc_compensations
            None,  # camera_model
        )


class RegisterProjectionEWA3DGSPacked:
    """Python autograd hooks for the gsplat::projection_ewa_3dgs_packed op."""

    base = "projection_ewa_3dgs_packed"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means,
            covars,
            quats,
            scales,
            _opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            _near_plane,
            _far_plane,
            _radius_clip,
            sparse_grad,
            _calc_compensations,
            camera_model,
        ) = inputs
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            _indptr,
            _radii,
            _means2d,
            _depths,
            conics,
            compensations,
        ) = output
        ctx.width = width
        ctx.height = height
        ctx.eps2d = eps2d
        ctx.camera_model = camera_model
        ctx.sparse_grad = sparse_grad
        ctx.save_for_backward(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
        )

    @classmethod
    def backward(
        cls,
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_indptr,
        v_radii,
        v_means2d,
        v_depths,
        v_conics,
        v_compensations,
    ):
        (
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
        ) = ctx.saved_tensors
        v_means, v_covars, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            f"{cls.base}_bwd"
        )(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.eps2d,
            ctx.camera_model,
            ctx.sparse_grad,
            batch_ids,
            camera_ids,
            gaussian_ids,
            conics,
            compensations,
            v_means2d,
            v_depths,
            v_conics,
            v_compensations,
            ctx.needs_input_grad[
                5
            ],  # viewmats_requires_grad (viewmats is input index 5)
        )
        return (
            v_means,
            v_covars,
            v_quats,
            v_scales,
            None,  # opacities
            v_viewmats,
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # sparse_grad
            None,  # calc_compensations
            None,  # camera_model
        )


@torch.no_grad()
@trace_function("isect-camera")
def isect_tiles(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    tile_size: int,
    tile_width: int,
    tile_height: int,
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
    conics: Optional[
        Tensor
    ] = None,  # [..., N, 3] or [nnz, 3], enables AccuTile when provided
    opacities: Optional[
        Tensor
    ] = None,  # [..., N] or [nnz], enables AccuTile when provided
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    When `conics` and `opacities` are provided the kernel uses conservative ellipse intersection (AccuTile/SNUGBOX),
    skipping tiles that the opacity-thresholded ellipse does not touch. When either is `None` the kernel falls back to the original axis-aligned bounding box.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [..., N] if packed is False, [nnz] if packed is True.
        tile_size: Tile size.
        tile_width: Tile width.
        tile_height: Tile height.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        segmented: If True, segmented radix sort will be used to sort the intersections. Default: False.
        packed: If True, the input tensors are packed. Default: False.
        n_images: Number of images. Required if packed is True.
        image_ids: The image indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.
        conics: Inverse of projected covariances (upper triangle). [..., N, 3] if packed is False, [nnz, 3] if packed is True. Enables AccuTile when provided together with opacities.
        opacities: Gaussian opacities. [..., N] if packed is False, [nnz] if packed is True. Enables AccuTile when provided together with conics.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [..., N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the image and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [I * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func("intersect_tile")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        conics.contiguous() if conics is not None else None,
        opacities.contiguous() if opacities is not None else None,
        image_ids,
        gaussian_ids,
        n_images,
        tile_size,
        tile_width,
        tile_height,
        sort,
        segmented,
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
@trace_function("isect-lidar")
def isect_tiles_lidar(
    lidar: RowOffsetStructuredSpinningLidarModelParametersExt,
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    radii: Tensor,  # [..., N, 2] or [nnz, 2]
    depths: Tensor,  # [..., N] or [nnz]
    sort: bool = True,
    segmented: bool = False,
    packed: bool = False,
    n_images: Optional[int] = None,
    image_ids: Optional[Tensor] = None,
    gaussian_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Maps projected Gaussians to intersecting tiles.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        radii: Maximum radii of the projected Gaussians. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        depths: Z-depth of the projected Gaussians. [..., N] if packed is False, [nnz] if packed is True.
        sort: If True, the returned intersections will be sorted by the intersection ids. Default: True.
        segmented: If True, segmented radix sort will be used to sort the intersections. Default: False.
        packed: If True, the input tensors are packed. Default: False.
        n_images: Number of images. Required if packed is True.
        image_ids: The image indices of the projected Gaussians. Required if packed is True.
        gaussian_ids: The column indices of the projected Gaussians. Required if packed is True.

    Returns:
        A tuple:

        - **Tiles per Gaussian**. The number of tiles intersected by each Gaussian.
          Int32 [..., N] if packed is False, Int32 [nnz] if packed is True.
        - **Intersection ids**. Each id is an 64-bit integer with the following
          information: image_id (Xc bits) | tile_id (Xt bits) | depth (32 bits).
          Xc and Xt are the maximum number of bits required to represent the image and
          tile ids, respectively. Int64 [n_isects]
        - **Flatten ids**. The global flatten indices in [I * N] or [nnz] (packed). [n_isects]
    """
    if packed:
        image_ids = image_ids.contiguous()
        gaussian_ids = gaussian_ids.contiguous()
    tiles_per_gauss, isect_ids, flatten_ids = _make_lazy_cuda_func(
        "intersect_tile_lidar"
    )(
        lidar.to_cpp(),
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids,
        gaussian_ids,
        n_images,
        sort,
        segmented,
    )
    return tiles_per_gauss, isect_ids, flatten_ids


@torch.no_grad()
@trace_function("offsets")
def isect_offset_encode(
    isect_ids: Tensor,
    n_images: int,
    tile_width: int,
    tile_height: int,
) -> Tensor:
    """Encodes intersection ids to offsets.

    Args:
        isect_ids: Intersection ids. [n_isects]
        n_images: Number of images.
        tile_width: Tile width.
        tile_height: Tile height.

    Returns:
        Offsets. [I, tile_height, tile_width]
    """
    return _make_lazy_cuda_func("intersect_offset")(
        isect_ids.contiguous(), n_images, tile_width, tile_height
    )


@torch.no_grad()
@trace_function("isect-sparse")
def isect_tiles_sparse(
    means2d: Tensor,  # [I, N, 2] or [nnz, 2]
    radii: Tensor,  # [I, N, 2] or [nnz, 2]
    depths: Tensor,  # [I, N] or [nnz]
    tile_mask: Tensor,  # [I, tile_height, tile_width] bool
    active_tiles: Tensor,  # [num_active_tiles] int32
    n_images: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    image_ids: Optional[Tensor] = None,  # [nnz] int32, required in packed mode
) -> Tuple[Tensor, Tensor]:
    """Maps projected Gaussians to a caller-supplied set of *active* tiles.

    The sparse counterpart of :func:`isect_tiles` + :func:`isect_offset_encode`:
    it enumerates only the intersections that fall in tiles flagged active by
    ``tile_mask`` and returns a compacted per-active-tile offset table.

    Args:
        means2d: Projected Gaussian means. [I, N, 2] (dense) or [nnz, 2] (packed).
        radii: Per-axis pixel radii. [I, N, 2] (dense) or [nnz, 2] (packed).
        depths: Z-depth of the projected Gaussians. [I, N] or [nnz].
        tile_mask: Bool tile-activity mask. [n_images, tile_height, tile_width].
        active_tiles: Ascending dense tile ids (``image_id * tile_height *
            tile_width + y * tile_width + x``) of the active tiles. [num_active_tiles],
            int32. Conventionally ``nonzero(tile_mask.flatten())``.
        n_images: Number of images.
        tile_size: Tile size in pixels.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.
        image_ids: The image index of each Gaussian. [nnz], int32. Required (and
            only used) in packed mode.

    Returns:
        A tuple:

        - **Tile offsets**. Int32 [num_active_tiles + 1]. ``tile_offsets[i]`` is
          the exclusive prefix-sum start of the flatten-id range for
          ``active_tiles[i]``; the trailing sentinel ``tile_offsets[-1] ==
          n_isects``.
        - **Flatten ids**. Int32 [n_isects]. The global flatten indices in
          [I * N] (dense) or [nnz] (packed), sorted by ``(image_id, tile_id,
          depth)`` near-to-far.
    """
    packed = means2d.dim() == 2
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert radii.shape == (nnz, 2), radii.shape
        assert depths.shape == (nnz,), depths.shape
        assert image_ids is not None, "image_ids is required when packed ([nnz, 2])"
        assert image_ids.shape == (nnz,), image_ids.shape
    else:
        I, N = means2d.shape[0], means2d.shape[1]
        assert means2d.shape == (I, N, 2), means2d.shape
        assert radii.shape == (I, N, 2), radii.shape
        assert depths.shape == (I, N), depths.shape
        assert I == n_images, (I, n_images)

    assert tile_mask.shape == (n_images, tile_height, tile_width), tile_mask.shape
    assert tile_mask.dtype == torch.bool, tile_mask.dtype
    assert active_tiles.dim() == 1 and active_tiles.dtype == torch.int32, (
        active_tiles.shape,
        active_tiles.dtype,
    )

    tile_offsets, flatten_ids = _make_lazy_cuda_func("intersect_tile_sparse")(
        means2d.contiguous(),
        radii.contiguous(),
        depths.contiguous(),
        image_ids.contiguous() if image_ids is not None else None,
        tile_mask.contiguous(),
        active_tiles.contiguous(),
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )
    return tile_offsets, flatten_ids


def build_sparse_tile_layout(
    pixels: Tensor,  # [P, 2] int, (row, col)
    image_ids: Tensor,  # [P] int, image/camera index of each pixel
    n_images: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Builds the per-active-tile layout consumed by sparse rasterization.

    Given the pixels to render, it computes the tile/pixel bookkeeping that lets
    sparse rasterization touch only active tiles and write only active pixels.
    Pixels are passed in *packed* form (a flat ``[P, 2]`` tensor plus ``[P]``
    ``image_ids``, where ``image_ids`` is the per-element batch index).

    Coordinate convention: ``pixels[p] = (row, col)`` with ``row`` in
    ``[0, height)`` and ``col`` in ``[0, width)``. A pixel's dense tile id is
    ``image_id * (tile_height * tile_width) + (row // tile_size) * tile_width +
    (col // tile_size)``.

    Args:
        pixels: Pixel coordinates to render, ``(row, col)``. [P, 2], int.
        image_ids: Image/camera index of each pixel. [P], int.
        n_images: Number of images (sizes ``active_tile_mask``); may exceed the
            largest id present in ``image_ids``.
        tile_size: Tile size in pixels.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.

    Returns:
        A tuple:

        - **active_tiles**. Int32 [AT]. Ascending dense tile ids of tiles holding
          at least one active pixel. Equals ``nonzero(active_tile_mask.flatten())``.
        - **active_tile_mask**. Bool [n_images, tile_height, tile_width].
        - **tile_pixel_mask**. UInt64 [AT, words_per_tile], ``words_per_tile =
          ceil(tile_size**2 / 64)``. Raster-order bitmask of active pixels in each
          active tile.
        - **tile_pixel_cumsum**. Int64 [AT]. *Inclusive* prefix sum of the
          active-pixel count per active tile; consumers read ``cumsum[t-1]`` as
          the start of active tile ``t``.
        - **pixel_map**. Int64 [P]. Argsort taking pixels into (tile_id, in-tile)
          sorted order -- the write order of pixels within tiles.

    Note:
        Callers must deduplicate: ``pixels`` must not contain a repeated
        ``(image, row, col)``. The empty case (``P == 0``) returns a length-1
        ``tile_pixel_cumsum`` of zero.
    """
    assert pixels.dim() == 2 and pixels.shape[1] == 2, pixels.shape
    P = pixels.shape[0]
    assert image_ids.shape == (P,), (image_ids.shape, P)

    return _make_lazy_cuda_func("build_sparse_tile_layout")(
        pixels.contiguous(),
        image_ids.contiguous(),
        n_images,
        tile_size,
        tile_width,
        tile_height,
    )


@trace_function("render2D-fwd")
def rasterize_to_pixels(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
            ``colors.shape[-1]`` must be one of the channel counts compiled into ``GSPLAT_NUM_CHANNELS``
            (see ``gsplat/cuda/csrc/Config.h``); otherwise the CUDA kernel raises ``ValueError``.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [..., image_height, image_width, channels]
        - **Rendered alphas**. [..., image_height, image_width, 1]
    """
    if backgrounds is not None:
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        masks = masks.contiguous()

    render_colors, render_alphas, means2d_absgrad, _last_ids = _make_lazy_cuda_func(
        "rasterize_to_pixels_3dgs"
    )(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        packed,
        absgrad,
    )
    if absgrad:
        means2d.absgrad = means2d_absgrad

    return render_colors, render_alphas


@trace_function("render2D-sparse-fwd")
def rasterize_to_pixels_sparse(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    colors: Tensor,  # [..., N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., N] or [nnz]
    image_ids: Tensor,  # [P]
    active_tiles: Tensor,  # [AT]
    tile_offsets: Tensor,  # [AT + 1]
    flatten_ids: Tensor,  # [n_isects]
    tile_pixel_mask: Tensor,  # [AT, words]
    tile_pixel_cumsum: Tensor,  # [AT]
    pixel_map: Tensor,  # [P]
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    backgrounds: Optional[Tensor] = None,  # [n_images, channels]
    masks: Optional[Tensor] = None,  # [n_images, tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to a packed set of pixels (sparse rasterization).

    Renders only the pixels described by the sparse tile layout, touching only
    active tiles. Outputs are packed in the original pixel order -- the same
    order as the ``pixels`` / ``image_ids`` passed to
    :func:`build_sparse_tile_layout` -- as ``[P, ...]`` rather than dense images.

    The layout tensors (``active_tiles`` / ``tile_pixel_mask`` /
    ``tile_pixel_cumsum`` / ``pixel_map``) come from
    :func:`build_sparse_tile_layout`; ``tile_offsets`` / ``flatten_ids`` come
    from :func:`intersect_tile_sparse`. ``image_ids`` is the same ``[P]`` tensor
    given to :func:`build_sparse_tile_layout` (needed for the background
    gradient).

    Args:
        means2d: Projected Gaussian means. [..., N, 2] or [nnz, 2] if packed.
        conics: Inverse projected covariances (upper triangle). [..., N, 3] or [nnz, 3].
        colors: Gaussian colors or ND features. [..., N, channels] or [nnz, channels].
            ``colors.shape[-1]`` must be a channel count compiled into ``GSPLAT_NUM_CHANNELS``.
        opacities: Gaussian opacities. [..., N] or [nnz].
        image_ids: Image index of each requested pixel. [P].
        active_tiles: Ascending dense ids of active tiles. [AT].
        tile_offsets: Per-active-tile intersection offsets. [AT + 1].
        flatten_ids: Flattened Gaussian indices. [n_isects].
        tile_pixel_mask: Per-active-tile raster-order active-pixel bitmask. [AT, words].
        tile_pixel_cumsum: Inclusive per-active-tile active-pixel count. [AT].
        pixel_map: Argsort taking pixels into (tile, in-tile) order. [P].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.
        backgrounds: Background colors. [n_images, channels]. Default: None.
        masks: Tile mask to skip masked tiles. [n_images, tile_height, tile_width]. Default: None.
        packed: If True, inputs are packed with shape [nnz, ...]. Default: False.
        absgrad: If True, backward computes a ``.absgrad`` attribute for ``means2d``. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**. [P, channels]
        - **Rendered alphas**. [P, 1]
    """
    if backgrounds is not None:
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        masks = masks.contiguous()

    render_colors, render_alphas, means2d_absgrad, _last_ids = _make_lazy_cuda_func(
        "rasterize_to_pixels_sparse"
    )(
        means2d.contiguous(),
        conics.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds,
        masks,
        image_ids.contiguous(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tiles.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        tile_pixel_mask.contiguous(),
        tile_pixel_cumsum.contiguous(),
        pixel_map.contiguous(),
        packed,
        absgrad,
    )
    if absgrad:
        means2d.absgrad = means2d_absgrad

    return render_colors, render_alphas


@torch.no_grad()
@trace_function("render2D-count")
def rasterize_num_contributing_gaussians(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    tile_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    image_width: int,
    image_height: int,
    tile_size: int,
) -> Tuple[Tensor, Tensor]:
    """Counts contributing Gaussians per pixel and returns accumulated alpha.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] for dense inputs, or [nnz, 2] for packed inputs.
        conics: Inverse projected covariances. [..., N, 3] for dense inputs, or [nnz, 3] for packed inputs.
        opacities: Gaussian opacities. [..., N] for dense inputs, or [nnz] for packed inputs.
        tile_offsets: Intersection offsets from :func:`isect_offset_encode`. [..., tile_height, tile_width].
        flatten_ids: Flattened Gaussian indices from :func:`isect_tiles`. [n_isects].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.

    Returns:
        A tuple:

        - **Number of contributing Gaussians**. [..., image_height, image_width]
        - **Rendered alphas**. [..., image_height, image_width]
    """
    return _make_lazy_cuda_func("rasterize_num_contributing_gaussians")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        image_width,
        image_height,
        tile_size,
    )


@torch.no_grad()
@trace_function("render2D-count-sparse")
def rasterize_num_contributing_gaussians_sparse(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    active_tiles: Tensor,  # [AT]
    tile_offsets: Tensor,  # [AT + 1]
    flatten_ids: Tensor,  # [n_isects]
    tile_pixel_mask: Tensor,  # [AT, words]
    tile_pixel_cumsum: Tensor,  # [AT]
    pixel_map: Tensor,  # [P]
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor]:
    """Sparse counterpart of :func:`rasterize_num_contributing_gaussians`.

    Counts contributing Gaussians (and accumulated alpha) for only the requested
    pixels, packed in original-pixel order (the order of the ``pixels`` /
    ``image_ids`` given to :func:`build_sparse_tile_layout`). Consumes the
    layout tensors from :func:`build_sparse_tile_layout` and the intersections
    from :func:`intersect_tile_sparse`.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] or [nnz, 2] if packed.
        conics: Inverse projected covariances. [..., N, 3] or [nnz, 3].
        opacities: Gaussian opacities. [..., N] or [nnz].
        active_tiles: Ascending dense ids of active tiles. [AT].
        tile_offsets: Per-active-tile intersection offsets. [AT + 1].
        flatten_ids: Flattened Gaussian indices. [n_isects].
        tile_pixel_mask: Per-active-tile raster-order active-pixel bitmask. [AT, words].
        tile_pixel_cumsum: Inclusive per-active-tile active-pixel count. [AT].
        pixel_map: Argsort taking pixels into (tile, in-tile) order. [P].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.

    Returns:
        A tuple:

        - **Number of contributing Gaussians**. Int32 [P].
        - **Rendered alphas**. [P].
    """
    return _make_lazy_cuda_func("rasterize_num_contributing_gaussians_sparse")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tiles.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        tile_pixel_mask.contiguous(),
        tile_pixel_cumsum.contiguous(),
        pixel_map.contiguous(),
    )


@torch.no_grad()
@trace_function("render2D-contributors")
def rasterize_contributing_gaussian_ids(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    tile_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    image_width: int,
    image_height: int,
    tile_size: int,
    num_contributing_gaussians: Tensor,  # [..., image_height, image_width]
) -> Tuple[Tensor, Tensor]:
    """Returns all contributing Gaussian IDs and weights per pixel.

    The output is padded to ``num_contributing_gaussians.max()`` samples per
    pixel. Valid entries are in front-to-back order, IDs are padded with ``-1``,
    and weights are padded with ``0``.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] for dense inputs, or [nnz, 2] for packed inputs.
        conics: Inverse projected covariances. [..., N, 3] for dense inputs, or [nnz, 3] for packed inputs.
        opacities: Gaussian opacities. [..., N] for dense inputs, or [nnz] for packed inputs.
        tile_offsets: Intersection offsets from :func:`isect_offset_encode`. [..., tile_height, tile_width].
        flatten_ids: Flattened Gaussian indices from :func:`isect_tiles`. [n_isects].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        num_contributing_gaussians: Number of valid contributors per pixel. [..., image_height, image_width].

    Returns:
        A tuple:

        - **Gaussian IDs**. [..., image_height, image_width, max_num_contributing]
        - **Radiance weights**. [..., image_height, image_width, max_num_contributing]
    """
    return _make_lazy_cuda_func("rasterize_contributing_gaussian_ids")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        image_width,
        image_height,
        tile_size,
        num_contributing_gaussians.contiguous(),
    )


@torch.no_grad()
@trace_function("render2D-contributors-sparse")
def rasterize_contributing_gaussian_ids_sparse(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    active_tiles: Tensor,  # [AT]
    tile_offsets: Tensor,  # [AT + 1]
    flatten_ids: Tensor,  # [n_isects]
    tile_pixel_mask: Tensor,  # [AT, words]
    tile_pixel_cumsum: Tensor,  # [AT]
    pixel_map: Tensor,  # [P]
    num_contributing_gaussians: Tensor,  # [P] int32
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
) -> Tuple[Tensor, Tensor]:
    """Sparse counterpart of :func:`rasterize_contributing_gaussian_ids`.

    Records the contributing Gaussian ids and weights for only the requested
    pixels, packed in original-pixel order (``[P, K]`` where ``K`` is the max
    per-pixel contributor count). Consumes the layout from
    :func:`build_sparse_tile_layout`, the intersections from
    :func:`intersect_tile_sparse`, and the per-pixel counts from
    :func:`rasterize_num_contributing_gaussians_sparse`.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] or [nnz, 2] if packed.
        conics: Inverse projected covariances. [..., N, 3] or [nnz, 3].
        opacities: Gaussian opacities. [..., N] or [nnz].
        active_tiles: Ascending dense ids of active tiles. [AT].
        tile_offsets: Per-active-tile intersection offsets. [AT + 1].
        flatten_ids: Flattened Gaussian indices. [n_isects].
        tile_pixel_mask: Per-active-tile raster-order active-pixel bitmask. [AT, words].
        tile_pixel_cumsum: Inclusive per-active-tile active-pixel count. [AT].
        pixel_map: Argsort taking pixels into (tile, in-tile) order. [P].
        num_contributing_gaussians: Per-pixel contributor counts. Int32 [P].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.

    Returns:
        A tuple:

        - **Contributing Gaussian ids**. Int32 [P, K] (-1 padded).
        - **Contributing weights**. [P, K].
    """
    return _make_lazy_cuda_func("rasterize_contributing_gaussian_ids_sparse")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        active_tiles.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        tile_pixel_mask.contiguous(),
        tile_pixel_cumsum.contiguous(),
        pixel_map.contiguous(),
        num_contributing_gaussians.contiguous(),
    )


@torch.no_grad()
@trace_function("render2D-top-contributors")
def rasterize_top_contributing_gaussian_ids(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    tile_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    image_width: int,
    image_height: int,
    tile_size: int,
    num_depth_samples: int,
) -> Tuple[Tensor, Tensor]:
    """Returns the top radiance-weight Gaussian IDs and weights per pixel.

    The selected samples are the strongest contributors by ``alpha * T`` during
    front-to-back rasterization, then sorted back into front-to-back order.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] for dense inputs, or [nnz, 2] for packed inputs.
        conics: Inverse projected covariances. [..., N, 3] for dense inputs, or [nnz, 3] for packed inputs.
        opacities: Gaussian opacities. [..., N] for dense inputs, or [nnz] for packed inputs.
        tile_offsets: Intersection offsets from :func:`isect_offset_encode`. [..., tile_height, tile_width].
        flatten_ids: Flattened Gaussian indices from :func:`isect_tiles`. [n_isects].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        num_depth_samples: Number of contributors to return per pixel.

    Returns:
        A tuple:

        - **Gaussian IDs**. [..., image_height, image_width, num_depth_samples]
        - **Radiance weights**. [..., image_height, image_width, num_depth_samples]
    """
    return _make_lazy_cuda_func("rasterize_top_contributing_gaussian_ids")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        image_width,
        image_height,
        tile_size,
        num_depth_samples,
    )


@torch.no_grad()
@trace_function("render2D-top-contributors-sparse")
def rasterize_top_contributing_gaussian_ids_sparse(
    means2d: Tensor,  # [..., N, 2] or [nnz, 2]
    conics: Tensor,  # [..., N, 3] or [nnz, 3]
    opacities: Tensor,  # [..., N] or [nnz]
    active_tiles: Tensor,  # [AT]
    tile_offsets: Tensor,  # [AT + 1]
    flatten_ids: Tensor,  # [n_isects]
    tile_pixel_mask: Tensor,  # [AT, words]
    tile_pixel_cumsum: Tensor,  # [AT]
    pixel_map: Tensor,  # [P]
    image_width: int,
    image_height: int,
    tile_size: int,
    tile_width: int,
    tile_height: int,
    num_depth_samples: int,
) -> Tuple[Tensor, Tensor]:
    """Sparse counterpart of :func:`rasterize_top_contributing_gaussian_ids`.

    Keeps the top-``num_depth_samples`` contributors (by weight, in depth order)
    for only the requested pixels, packed in original-pixel order
    (``[P, num_depth_samples]``). Consumes the layout from
    :func:`build_sparse_tile_layout` and the intersections from
    :func:`intersect_tile_sparse`.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] or [nnz, 2] if packed.
        conics: Inverse projected covariances. [..., N, 3] or [nnz, 3].
        opacities: Gaussian opacities. [..., N] or [nnz].
        active_tiles: Ascending dense ids of active tiles. [AT].
        tile_offsets: Per-active-tile intersection offsets. [AT + 1].
        flatten_ids: Flattened Gaussian indices. [n_isects].
        tile_pixel_mask: Per-active-tile raster-order active-pixel bitmask. [AT, words].
        tile_pixel_cumsum: Inclusive per-active-tile active-pixel count. [AT].
        pixel_map: Argsort taking pixels into (tile, in-tile) order. [P].
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        tile_width: Number of tiles along the image width.
        tile_height: Number of tiles along the image height.
        num_depth_samples: Number of top contributors to keep per pixel.

    Returns:
        A tuple:

        - **Top Gaussian ids**. Int32 [P, num_depth_samples] (-1 padded).
        - **Top weights**. [P, num_depth_samples].
    """
    return _make_lazy_cuda_func("rasterize_top_contributing_gaussian_ids_sparse")(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        tile_width,
        tile_height,
        num_depth_samples,
        active_tiles.contiguous(),
        tile_offsets.contiguous(),
        flatten_ids.contiguous(),
        tile_pixel_mask.contiguous(),
        tile_pixel_cumsum.contiguous(),
        pixel_map.contiguous(),
    )


class RegisterRasterizeToPixels3DGS:
    """Python autograd hooks for the gsplat::rasterize_to_pixels_3dgs op."""

    base = "rasterize_to_pixels_3dgs"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
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
            _packed,
            absgrad,
        ) = inputs
        _render_colors, render_alphas, means2d_absgrad, last_ids = output
        # last_ids and the absgrad holder are forward-internal; the backward fills the
        # holder in place (it must not be tracked by autograd).
        ctx.mark_non_differentiable(last_ids, means2d_absgrad)
        ctx.width = image_width
        ctx.height = image_height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            means2d_absgrad,
        )

    @classmethod
    def backward(
        cls, ctx, v_render_colors, v_render_alphas, v_means2d_absgrad, v_last_ids
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            means2d_absgrad,
        ) = ctx.saved_tensors
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
        ) = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            ctx.absgrad,
            v_render_colors,
            v_render_alphas,
            ctx.needs_input_grad[
                4
            ],  # compute_v_backgrounds (backgrounds is input index 4)
        )
        # The abs gradient is not a returned input grad; surface it by filling the
        # saved means2d.absgrad holder in place.
        if ctx.absgrad and v_means2d_abs is not None:
            means2d_absgrad.copy_(v_means2d_abs)
        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,  # masks
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # packed
            None,  # absgrad
        )


class RegisterRasterizeToPixelsSparse:
    """Python autograd hooks for the gsplat::rasterize_to_pixels_sparse op."""

    base = "rasterize_to_pixels_sparse"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_ids,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            active_tiles,
            tile_offsets,
            flatten_ids,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            _packed,
            absgrad,
        ) = inputs
        _render_colors, render_alphas, means2d_absgrad, last_ids = output
        # last_ids and the absgrad holder are forward-internal; the backward fills
        # the holder in place (it must not be tracked by autograd).
        ctx.mark_non_differentiable(last_ids, means2d_absgrad)
        ctx.width = image_width
        ctx.height = image_height
        ctx.tile_size = tile_size
        ctx.tile_width = tile_width
        ctx.tile_height = tile_height
        ctx.absgrad = absgrad
        ctx.save_for_backward(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_ids,
            active_tiles,
            tile_offsets,
            flatten_ids,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            render_alphas,
            last_ids,
            means2d_absgrad,
        )

    @classmethod
    def backward(
        cls, ctx, v_render_colors, v_render_alphas, v_means2d_absgrad, v_last_ids
    ):
        (
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_ids,
            active_tiles,
            tile_offsets,
            flatten_ids,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            render_alphas,
            last_ids,
            means2d_absgrad,
        ) = ctx.saved_tensors
        (
            v_means2d_abs,
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
        ) = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_ids,
            active_tiles,
            tile_offsets,
            flatten_ids,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            render_alphas,
            last_ids,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            ctx.tile_width,
            ctx.tile_height,
            ctx.absgrad,
            v_render_colors,
            v_render_alphas,
            ctx.needs_input_grad[
                4
            ],  # compute_v_backgrounds (backgrounds is input index 4)
        )
        # The abs gradient is not a returned input grad; surface it by filling the
        # saved means2d.absgrad holder in place.
        if ctx.absgrad and v_means2d_abs is not None:
            means2d_absgrad.copy_(v_means2d_abs)
        return (
            v_means2d,
            v_conics,
            v_colors,
            v_opacities,
            v_backgrounds,
            None,  # masks
            None,  # image_ids
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # tile_width
            None,  # tile_height
            None,  # active_tiles
            None,  # tile_offsets
            None,  # flatten_ids
            None,  # tile_pixel_mask
            None,  # tile_pixel_cumsum
            None,  # pixel_map
            None,  # packed
            None,  # absgrad
        )


def rasterize_to_pixels_eval3d(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, H, W, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    use_hit_distance: bool = False,
    return_normals: bool = False,
    renderer_config: Any = None,
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels.

    Similar to `rasterize_to_pixels()`, but compute the Gaussian responses in the
    3D world space instead of the 2D image space. Supports rolling shutter and
    camera distortion.

    ``colors.shape[-1]`` must be one of the channel counts compiled into
    ``GSPLAT_NUM_CHANNELS`` (see ``gsplat/cuda/csrc/Config.h``); otherwise the CUDA
    kernel raises ``ValueError``.

    Args:
        renderer_config: Eval3d renderer selector. ``None`` uses the default
            ``RendererConfig_MixedBatch`` policy. Pass public
            ``RendererConfig_MixedBatch`` / ``RendererConfig_ParallelBatch``
            instances, or the already-translated low-level CUDA config value.

    Returns:
        A tuple:

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()

    colors, alphas, *_ = rasterize_to_pixels_eval3d_extra(
        means=means,
        quats=quats,
        scales=scales,
        colors=colors,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
        rays=rays,
        image_width=image_width,
        image_height=image_height,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
        backgrounds=backgrounds,
        masks=masks,
        camera_model=camera_model,
        ut_params=ut_params,
        radial_coeffs=radial_coeffs,
        tangential_coeffs=tangential_coeffs,
        thin_prism_coeffs=thin_prism_coeffs,
        ftheta_coeffs=ftheta_coeffs,
        lidar_coeffs=lidar_coeffs,
        external_distortion_coeffs=external_distortion_coeffs,
        rolling_shutter=rolling_shutter,
        viewmats_rs=viewmats_rs,
        return_last_ids=False,
        return_sample_counts=False,
        use_hit_distance=use_hit_distance,
        return_normals=return_normals,
        renderer_config=renderer_config,
    )
    return colors, alphas


@trace_function("raster3D-fwd")
def rasterize_to_pixels_eval3d_extra(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    colors: Tensor,  # [..., C, N, channels] or [nnz, channels]
    opacities: Tensor,  # [..., C, N] or [nnz]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., C, channels]
    masks: Optional[Tensor] = None,  # [..., C, tile_height, tile_width]
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    rays: Optional[Tensor] = None,  # [..., C, P, 6]
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
    renderer_config: Any = None,
    return_last_ids: bool = True,
    unsafe_masked_tile_outputs: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Rasterizes Gaussians to pixels, returning extra information for debugging.

    Similar to `rasterize_to_pixels_eval3d()`, but can return the last gaussian id
    accumulated in a pixel and optionally the number of accumulated samples per pixel.

    ``colors.shape[-1]`` must be one of the channel counts compiled into
    ``GSPLAT_NUM_CHANNELS`` (see ``gsplat/cuda/csrc/Config.h``); otherwise the CUDA
    kernel raises ``ValueError``.

    Args:
        return_last_ids: If True, also return last flatten_idx per pixel. Default: True.
        return_sample_counts: If True, also return number of accumulated samples per pixel. Default: False.
        return_normals: If True, compute and return accumulated normals per pixel.
            Normals are computed from Gaussian quaternions (canonical normal = (0,0,1)
            transformed by rotation, flipped if facing away from ray). Default: False.
        renderer_config: Eval3d renderer selector. ``None`` uses the default
            ``RendererConfig_MixedBatch`` policy. Pass public
            ``RendererConfig_MixedBatch`` / ``RendererConfig_ParallelBatch``
            instances, or the already-translated low-level CUDA config value.
        unsafe_masked_tile_outputs: If True, outputs for masked tiles are left undefined
            and must not be read by the caller. Default False writes per-pixel safe
            values for masked tiles: render_colors = backgrounds (or 0.0 when no
            backgrounds are provided), render_alphas = 0.0, render_normals = 0.0,
            last_ids = -1, sample_counts = 0.

    Returns:
        A tuple (contents depend on return flags):

        - **Rendered colors**. [..., C, image_height, image_width, channels]
        - **Rendered alphas**. [..., C, image_height, image_width, 1]
        - **Last flatten_idx** (optional). [..., C, image_height, image_width]. If return_last_ids=True.
        - **Sample counts** (optional). [..., C, image_height, image_width]. If return_sample_counts=True.
        - **Rendered normals** (optional). [..., C, image_height, image_width, 3]. If return_normals=True.
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()
    renderer_config = _renderer_config_to_cuda(renderer_config)

    # Packed colors (colors.ndim == means.ndim) are not supported yet.
    if colors.ndim == means.ndim:
        raise NotImplementedError("packed mode is not supported yet")

    camera_model_type = _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")
    ftheta_coeffs = (
        ftheta_coeffs
        if ftheta_coeffs is not None
        else FThetaCameraDistortionParameters()
    )
    lidar_coeffs = lidar_coeffs.to_cpp() if lidar_coeffs is not None else None

    (
        render_colors,
        render_alphas,
        last_ids,
        sample_counts,
        render_normals,
    ) = _make_lazy_cuda_func("rasterize_to_pixels_from_world_3dgs")(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds.contiguous() if backgrounds is not None else None,
        masks.contiguous() if masks is not None else None,
        image_width,
        image_height,
        tile_size,
        viewmats.contiguous(),
        viewmats_rs.contiguous() if viewmats_rs is not None else None,
        Ks.contiguous(),
        camera_model_type,
        ut_params,
        rolling_shutter,
        rays.contiguous() if rays is not None else None,
        # distortion
        radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_coeffs,
        lidar_coeffs,
        external_distortion_coeffs,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        return_sample_counts,  # Pass flag to forward
        use_hit_distance,
        return_normals,  # Pass return_normals flag to forward
        renderer_config,
        return_last_ids,
        unsafe_masked_tile_outputs,
    )

    return render_colors, render_alphas, last_ids, sample_counts, render_normals


@torch.no_grad()
def rasterize_to_indices_in_range(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    conics: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [..., image_height, image_width]
        means2d: Projected Gaussian means. [..., N, 2]
        conics: Inverse of the projected covariances with only upper triangle values. [..., N, 3]
        opacities: Gaussian opacities that support per-view values. [..., N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Image ids**. image indices. A flattened list of shape [M].
    """

    return _make_lazy_cuda_func("rasterize_to_indices_3dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )


@trace_function("projectUT-fwd")
def fully_fused_projection_with_ut(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Optional[Tensor],  # [..., N]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    calc_compensations: bool = False,
    camera_model: CameraModel = "pinhole",
    ut_params: Optional[UnscentedTransformParameters] = None,
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[BivariateWindshieldModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    global_z_order: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Projects Gaussians to 2D using Unscented Transform (UT).

    similar to `fully_fused_projection()`, but supports camera distortion and
    rolling shutter.

    .. warning::
        This function is not differentiable to any input.

    Args:
        global_z_order: Defines how Gaussians are sorted for depth ordering. If True (default),
            Gaussians are sorted by their z-coordinate in camera space. If False, they are sorted
            by their Euclidean distance from the camera origin. The z-coordinate sorting is typically
            faster and sufficient for most cases, while Euclidean distance can be useful for scenes
            with wide field-of-view or non-standard camera models. Default: True.
    """
    if lidar_coeffs is not None:
        assert isinstance(
            lidar_coeffs, RowOffsetStructuredSpinningLidarModelParametersExt
        )

    camera_model_type = _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")

    radii, means2d, depths, conics, compensations = _make_lazy_cuda_func(
        "projection_ut_3dgs_fused"
    )(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        opacities.contiguous() if opacities is not None else None,
        viewmats.contiguous(),
        viewmats_rs.contiguous() if viewmats_rs is not None else None,
        Ks.contiguous(),
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        calc_compensations,
        camera_model_type,
        global_z_order,
        ut_params,
        rolling_shutter,
        radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_coeffs,
        lidar_coeffs.to_cpp() if lidar_coeffs is not None else None,
        external_distortion_coeffs,
    )
    return radii, means2d, depths, conics, compensations


###### 2DGS ######
def fully_fused_projection_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    eps2d: float = 0.3,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    packed: bool = False,
    sparse_grad: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Prepare Gaussians for rasterization

    This function prepares ray-splat intersection matrices, computes
    per splat bounding box and 2D means in image space.

    Args:
        means: Gaussian means. [..., N, 3]
        quats: Quaternions (No need to be normalized). [..., N, 4].
        scales: Scales. [..., N, 3].
        viewmats: World-to-camera matrices. [..., C, 4, 4]
        Ks: Camera intrinsics. [..., C, 3, 3]
        width: Image width.
        height: Image height.
        near_plane: Near plane distance. Default: 0.01.
        far_plane: Far plane distance. Default: 200.
        radius_clip: Gaussians with projected radii smaller than this value will be ignored. Default: 0.0.
        packed: If True, the output tensors will be packed into a flattened tensor. Default: False.
        sparse_grad (Experimental): This is only effective when `packed` is True. If True, during backward the gradients
          of {`means`, `covars`, `quats`, `scales`} will be a sparse Tensor in COO layout. Default: False.

    Returns:
        A tuple:

        If `packed` is True:

        - **batch_ids**. The batch indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **camera_ids**. The camera indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **gaussian_ids**. The column indices of the projected Gaussians. Int32 tensor of shape [nnz].
        - **indptr**. CSR-style index pointer into gaussian_ids for batch-camera pairs. Int32 tensor of shape [B*C+1].
        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [nnz, 2].
        - **means**. Projected Gaussian means in 2D. [nnz, 2]
        - **depths**. The z-depth of the projected Gaussians. [nnz]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates (WH)^T in equation (9) in paper [nnz, 3, 3]
        - **normals**. The normals in camera spaces. [nnz, 3]

        If `packed` is False:

        - **radii**. The maximum radius of the projected Gaussians in pixel unit. Int32 tensor of shape [..., C, N, 2].
        - **means**. Projected Gaussian means in 2D. [..., C, N, 2]
        - **depths**. The z-depth of the projected Gaussians. [..., C, N]
        - **ray_transforms**. transformation matrices that transforms xy-planes in pixel spaces into splat coordinates [..., C, N, 3, 3]
        - **normals**. The normals in camera spaces. [..., C, N, 3]

    """
    means = means.contiguous()
    quats = quats.contiguous()
    scales = scales.contiguous()
    if sparse_grad:
        assert packed, "sparse_grad is only supported when packed is True"

    viewmats = viewmats.contiguous()
    Ks = Ks.contiguous()
    if packed:
        return _make_lazy_cuda_func("projection_2dgs_packed")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            near_plane,
            far_plane,
            radius_clip,
            sparse_grad,
        )
    else:
        return _make_lazy_cuda_func("projection_2dgs_fused")(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d,
            near_plane,
            far_plane,
            radius_clip,
        )


class RegisterProjection2DGSFused:
    """Python autograd hooks for the gsplat::projection_2dgs_fused op."""

    base = "projection_2dgs_fused"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            _eps2d,
            _near_plane,
            _far_plane,
            _radius_clip,
        ) = inputs
        radii, _means2d, _depths, ray_transforms, _normals = output
        ctx.width = width
        ctx.height = height
        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
        )

    @classmethod
    def backward(cls, ctx, v_radii, v_means2d, v_depths, v_ray_transforms, v_normals):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            radii,
            ray_transforms,
        ) = ctx.saved_tensors
        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            f"{cls.base}_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            radii,
            ray_transforms,
            v_means2d,
            v_depths,
            v_ray_transforms,
            v_normals,
            ctx.needs_input_grad[
                3
            ],  # viewmats_requires_grad (viewmats is input index 3)
        )
        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # eps2d
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
        )


class RegisterProjection2DGSPacked:
    """Python autograd hooks for the gsplat::projection_2dgs_packed op."""

    base = "projection_2dgs_packed"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            _near_plane,
            _far_plane,
            _radius_clip,
            sparse_grad,
        ) = inputs
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            _indptr,
            _radii,
            _means2d,
            _depths,
            ray_transforms,
            _normals,
        ) = output
        ctx.width = width
        ctx.height = height
        ctx.sparse_grad = sparse_grad
        ctx.save_for_backward(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
        )

    @classmethod
    def backward(
        cls,
        ctx,
        v_batch_ids,
        v_camera_ids,
        v_gaussian_ids,
        v_indptr,
        v_radii,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
    ):
        (
            means,
            quats,
            scales,
            viewmats,
            Ks,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
        ) = ctx.saved_tensors
        v_means, v_quats, v_scales, v_viewmats = _make_lazy_cuda_func(
            f"{cls.base}_bwd"
        )(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            ctx.width,
            ctx.height,
            ctx.sparse_grad,
            batch_ids,
            camera_ids,
            gaussian_ids,
            ray_transforms,
            v_means2d,
            v_depths,
            v_ray_transforms,
            v_normals,
            ctx.needs_input_grad[
                3
            ],  # viewmats_requires_grad (viewmats is input index 3)
        )
        return (
            v_means,
            v_quats,
            v_scales,
            v_viewmats,
            None,  # Ks
            None,  # image_width
            None,  # image_height
            None,  # near_plane
            None,  # far_plane
            None,  # radius_clip
            None,  # sparse_grad
        )


def rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    colors: Tensor,  # [..., N, channels]
    opacities: Tensor,  # [..., N]
    normals: Tensor,  # [..., N, 3]
    densify: Tensor,  # [..., N, 2]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [..., tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [..., channels]
    masks: Optional[Tensor] = None,  # [..., tile_height, tile_width]
    packed: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.

    Args:
        means2d: Projected Gaussian means. [..., N, 2] if packed is False, [nnz, 2] if packed is True.
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [..., N, 3, 3] if packed is False, [nnz, channels] if packed is True.
        colors: Gaussian colors or ND features. [..., N, channels] if packed is False, [nnz, channels] if packed is True.
            ``colors.shape[-1]`` must be one of the channel counts compiled into ``GSPLAT_NUM_CHANNELS``
            (see ``gsplat/cuda/csrc/Config.h``); otherwise the CUDA kernel raises ``ValueError``.
        opacities: Gaussian opacities that support per-view values. [..., N] if packed is False, [nnz] if packed is True.
        normals: The normals in camera space. [..., N, 3] if packed is False, [nnz, 3] if packed is True.
        densify: Dummy variable to keep track of gradient for densification. [..., N, 2] if packed, [nnz, 3] if packed is True.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] or [nnz] from  `isect_tiles()`. [n_isects]
        backgrounds: Background colors. [..., channels]. Default: None.
        masks: Optional tile mask to skip rendering GS to masked tiles. [..., tile_height, tile_width]. Default: None.
        packed: If True, the input tensors are expected to be packed with shape [nnz, ...]. Default: False.
        absgrad: If True, the backward pass will compute a `.absgrad` attribute for `means2d`. Default: False.

    Returns:
        A tuple:

        - **Rendered colors**.      [..., image_height, image_width, channels]
        - **Rendered alphas**.      [..., image_height, image_width, 1]
        - **Rendered normals**.     [..., image_height, image_width, 3]
        - **Rendered distortion**.  [..., image_height, image_width, 1]
        - **Rendered median depth**.[..., image_height, image_width, 1]


    """
    if backgrounds is not None:
        backgrounds = backgrounds.contiguous()
    if masks is not None:
        masks = masks.contiguous()

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
        means2d_absgrad,
        _last_ids,
        _median_ids,
    ) = _make_lazy_cuda_func("rasterize_to_pixels_2dgs")(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        normals.contiguous(),
        densify.contiguous(),
        backgrounds,
        masks,
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        packed,
        absgrad,
        distloss,
    )
    if absgrad:
        means2d.absgrad = means2d_absgrad

    return render_colors, render_alphas, render_normals, render_distort, render_median


class RegisterRasterizeToPixels2DGS:
    """Python autograd hooks for the gsplat::rasterize_to_pixels_2dgs op."""

    base = "rasterize_to_pixels_2dgs"

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            _packed,
            absgrad,
            _distloss,
        ) = inputs
        (
            render_colors,
            render_alphas,
            _render_normals,
            _render_distort,
            _render_median,
            means2d_absgrad,
            last_ids,
            median_ids,
        ) = output
        # last_ids / median_ids and the absgrad holder are forward-internal; the
        # backward fills the holder in place (it must not be tracked by autograd).
        ctx.mark_non_differentiable(last_ids, median_ids, means2d_absgrad)
        ctx.width = image_width
        ctx.height = image_height
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.save_for_backward(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            tile_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            means2d_absgrad,
        )

    @classmethod
    def backward(
        cls,
        ctx,
        v_render_colors,
        v_render_alphas,
        v_render_normals,
        v_render_distort,
        v_render_median,
        v_means2d_absgrad,
        v_last_ids,
        v_median_ids,
    ):
        (
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            tile_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            means2d_absgrad,
        ) = ctx.saved_tensors
        (
            v_means2d_abs,
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
        ) = _make_lazy_cuda_func(f"{cls.base}_bwd")(
            means2d,
            ray_transforms,
            colors,
            opacities,
            normals,
            densify,
            backgrounds,
            masks,
            tile_offsets,
            flatten_ids,
            render_colors,
            render_alphas,
            last_ids,
            median_ids,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            ctx.absgrad,
            v_render_colors,
            v_render_alphas,
            v_render_normals,
            v_render_distort,
            v_render_median,
            ctx.needs_input_grad[
                6
            ],  # compute_v_backgrounds (backgrounds is input index 6)
        )
        if ctx.absgrad and v_means2d_abs is not None:
            means2d_absgrad.copy_(v_means2d_abs)
        return (
            v_means2d,
            v_ray_transforms,
            v_colors,
            v_opacities,
            v_normals,
            v_densify,
            v_backgrounds,
            None,  # masks
            None,  # image_width
            None,  # image_height
            None,  # tile_size
            None,  # tile_offsets
            None,  # flatten_ids
            None,  # packed
            None,  # absgrad
            None,  # distloss
        )


@torch.no_grad()
def rasterize_to_indices_in_range_2dgs(
    range_start: int,
    range_end: int,
    transmittances: Tensor,  # [..., image_height, image_width]
    means2d: Tensor,  # [..., N, 2]
    ray_transforms: Tensor,  # [..., N, 3, 3]
    opacities: Tensor,  # [..., N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Rasterizes a batch of Gaussians to images but only returns the indices.

    .. note::

        This function supports iterative rasterization, in which each call of this function
        will rasterize a batch of Gaussians from near to far, defined by `[range_start, range_end)`.
        If a one-step full rasterization is desired, set `range_start` to 0 and `range_end` to a really
        large number, e.g, 1e10.

    Args:
        range_start: The start batch of Gaussians to be rasterized (inclusive).
        range_end: The end batch of Gaussians to be rasterized (exclusive).
        transmittances: Currently transmittances. [..., image_height, image_width]
        means2d: Projected Gaussian means. [..., N, 2]
        ray_transforms: transformation matrices that transforms xy-planes in pixel spaces into splat coordinates. [..., N, 3, 3]
        opacities: Gaussian opacities that support per-view values. [..., N]
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [..., tile_height, tile_width]
        flatten_ids: The global flatten indices in [I * N] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Gaussian ids**. Gaussian ids for the pixel intersection. A flattened list of shape [M].
        - **Pixel ids**. pixel indices (row-major). A flattened list of shape [M].
        - **Camera ids**. Camera indices. A flattened list of shape [M].
        - **Batch ids**. Batch indices. A flattened list of shape [M].
    """

    return _make_lazy_cuda_func("rasterize_to_indices_2dgs")(
        range_start,
        range_end,
        transmittances.contiguous(),
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )


# Wire the Python autograd backends now, at import time. The module-level handles above
# already load the C extension, so by here the op schemas exist; _register_autograd is a
# no-op for any op whose schema is absent, and the whole pass is idempotent.
_ensure_autograd_registrations()
