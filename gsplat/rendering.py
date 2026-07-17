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

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal
from .trace import trace_function
from .profile import capture_inputs

from .cuda._wrapper import (
    RollingShutterType,
    CameraModel,
    FThetaCameraDistortionParameters,
    FThetaPolynomialType,
    RowOffsetStructuredSpinningLidarModelParametersExt,
    UnscentedTransformParameters,
    ExternalDistortionModelMeta,
    ExternalDistortionModelParameters,
    ExternalDistortionReferencePolynomial,
    BivariateWindshieldModelParameters,
    fully_fused_projection,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    isect_tiles_lidar,
    _make_lazy_cuda_func,
    _make_lazy_cuda_obj,
    rasterize_to_pixels,
    rasterize_to_pixels_eval3d,
    rasterize_to_pixels_eval3d_extra,
    renderer_config_mixed_batch,
    renderer_config_parallel_batch,
    spherical_harmonics,
)
from .utils import depth_to_normal, get_projection_matrix


# Fused post-transform codes for the proj_features assembler (SHPostOp in CUDA):
# none -> identity, shift -> +0.5, shift_relu -> max(x + 0.5, 0).
_POST_CODE = {"none": 0, "shift": 1, "shift_relu": 2}


# Gaussian depth modes (D/ED): use projection depth (controlled by global_z_order)
# Hit distance modes (d/Ed): compute along-ray distance in rasterization
RenderMode = Literal["RGB", "d", "Ed", "D", "ED", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"]
RasterizeMode = Literal["classic", "antialiased"]


class RendererConfig:
    """Base class for public rasterizer selection configs.

    Instantiate one of the concrete renderer configs instead of this base
    class. Unsupported subclasses remain possible so future policies can fail
    with an explicit "unsupported" error at the validation boundary.
    """

    def __new__(cls, *args, **kwargs):
        if cls is RendererConfig:
            raise TypeError(
                "RendererConfig is a base class; instantiate "
                "RendererConfig_MixedBatch or RendererConfig_ParallelBatch."
            )
        return super().__new__(cls)


@dataclass
class RendererConfig_MixedBatch(RendererConfig):
    """Eval3d rasterizer: serial-batch forward, batch-parallel backward.

    "Mixed" = the two passes batch differently. The forward composites each
    tile's depth-sorted Gaussian batches serially (one CTA per tile, front to
    back); the backward is batch-parallel (one CTA per batch).
    """


@dataclass
class RendererConfig_ParallelBatch(RendererConfig):
    """Eval3d rasterizer: batch-parallel forward and backward.

    Both passes are batch-parallel (one CTA per batch); the forward adds a
    partials/scan/replay pipeline so independent batches composite concurrently,
    at the cost of per-batch forward state persisted for the backward. The
    backward kernel is shared with MixedBatch.
    """


def _validate_renderer_config(renderer_config: RendererConfig) -> None:
    if renderer_config is None:
        raise TypeError("renderer_config must be a RendererConfig instance, got None.")
    if not isinstance(renderer_config, RendererConfig):
        raise TypeError(
            "renderer_config must be a RendererConfig instance, "
            f"got {type(renderer_config).__name__}."
        )
    if isinstance(renderer_config, RendererConfig_MixedBatch):
        return
    if isinstance(renderer_config, RendererConfig_ParallelBatch):
        return
    raise NotImplementedError(
        f"Unsupported renderer_config type: {type(renderer_config).__name__}."
    )


def _renderer_config_type(renderer_config: RendererConfig) -> Any:
    _validate_renderer_config(renderer_config)
    if isinstance(renderer_config, RendererConfig_MixedBatch):
        return renderer_config_mixed_batch()
    if isinstance(renderer_config, RendererConfig_ParallelBatch):
        return renderer_config_parallel_batch()
    raise AssertionError(
        f"unreachable renderer_config: {type(renderer_config).__name__}"
    )


# TODO: RenderMode should be an enum so that we can add these query methods to it.
# The problem is that it'd break backward compatibllity due to some symbols used, e.g. RGB+D or RGB-d.
def render_mode_has_color(mode: RenderMode) -> bool:
    return mode in {"RGB", "RGB-d", "RGB-Ed", "RGB+D", "RGB+ED"}


def render_mode_has_hit_distance(mode: RenderMode) -> bool:
    return mode in {"d", "Ed", "RGB-d", "RGB-Ed"}


def render_mode_has_depth(mode: RenderMode) -> bool:
    return mode in {"D", "ED", "RGB+D", "RGB+ED"}


def render_mode_has_expected_depth(mode: RenderMode) -> bool:
    return mode in {"Ed", "ED", "RGB-Ed", "RGB+ED"}


def render_mode_has_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth(mode) or render_mode_has_hit_distance(mode)


def render_mode_has_only_depth_channel(mode: RenderMode) -> bool:
    return render_mode_has_depth_channel(mode) and not render_mode_has_color(mode)


def render_mode_has_only_color(mode: RenderMode) -> bool:
    return not render_mode_has_depth_channel(mode) and render_mode_has_color(mode)


def _validate_3dgut_rasterize_mode(
    rasterize_mode: RasterizeMode,
    *,
    with_ut: bool,
    with_eval3d: bool,
) -> None:
    if rasterize_mode != "classic" and (with_ut or with_eval3d):
        raise ValueError(
            "3DGUT rendering only supports rasterize_mode='classic'. "
            f"Got rasterize_mode='{rasterize_mode}' with "
            f"with_ut={with_ut} and with_eval3d={with_eval3d}."
        )


def _get_default_nccl_process_group_name() -> str:
    """Return the default NCCL process-group name used by c10d functional ops."""

    if not torch.distributed.is_available():
        raise ValueError("distributed=True requires torch.distributed to be available.")
    if not torch.distributed.is_initialized():
        raise ValueError(
            "distributed=True requires an initialized default torch.distributed "
            "process group."
        )

    import torch.distributed.distributed_c10d as distributed_c10d

    process_group = distributed_c10d._get_default_group()
    backend = torch.distributed.get_backend(process_group)
    if str(backend).lower() != "nccl":
        raise ValueError(
            "distributed=True currently supports only the default NCCL process "
            f"group; got backend '{backend}'."
        )
    return distributed_c10d._get_process_group_name(process_group)


def _resolve_tile_size(
    tile_size: Optional[int], with_eval3d: bool, width: int, height: int
) -> int:
    """Resolve a None-valued tile_size to the path-correct default.

    The 3DGS kernel is compiled with TILE_SIZE=16 only; the 3DGUT kernel
    dispatches at compile time on tile_size in {8, 16} and the optimum is
    workload-dependent:
      - tile=8 (CTA=32, PPT=2) is the compact-CTA path. Wins below 1080p
        where smaller per-tile shmem lets many CTAs co-reside per SM and
        intersect+sort cost is small.
      - tile=16 (CTA=256, PPT=1) is one thread per pixel. Wins at 1080p+
        where intersect+sort dominates and fewer/larger tiles shrink it.

    Spinning lidar grids are wide but shallow (e.g. pandar128 = 128 rows x
    3600 cols, at128 = 128 x 1200); min gates on the row count keeps lidar at
    tile=8 alongside sub-1080p cameras. Cameras at 1080p+ have min(W,H) >= 1080
    and pick tile=16.

    Callers must pass `width`/`height` AFTER any lidar dim override so the
    gate sees the right dims for lidar.

    Explicit non-None tile_size is returned unchanged (caller override wins).
    """
    if tile_size is not None:
        return tile_size
    if with_eval3d:
        return 16 if min(width, height) >= 1080 else 8
    return 16


@trace_function("render")
@capture_inputs(envvar="GSPLAT_INPUT_CAPTURE_RASTERIZATION")
def rasterization(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Optional[
        Tensor
    ],  # [..., (C,) N, D] for post-activation colors, or [N, K, D] for SH coefficients; None for depth-only render_modes
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: RasterizeMode = "classic",
    distributed: bool = False,
    camera_model: CameraModel = "pinhole",
    segmented: bool = False,
    covars: Optional[Tensor] = None,
    with_ut: bool = False,
    with_eval3d: bool = False,
    macro_tile: bool = False,
    return_normals: bool = False,
    global_z_order: bool = True,
    rays: Optional[
        Tensor
    ] = None,  # [..., C, H, W, 6] -> ox, oy, oz, dx*spread, dy*spread, dz*spread
    # distortion
    radial_coeffs: Optional[Tensor] = None,  # [..., C, 6] or [..., C, 4]
    tangential_coeffs: Optional[Tensor] = None,  # [..., C, 2]
    thin_prism_coeffs: Optional[Tensor] = None,  # [..., C, 4]
    ftheta_coeffs: Optional[FThetaCameraDistortionParameters] = None,
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    external_distortion_coeffs: Optional[ExternalDistortionModelParameters] = None,
    # rolling shutter
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,  # [..., C, 4, 4]
    # unscented transform (for 3DGUT)
    ut_params: Optional[UnscentedTransformParameters] = None,
    # extra signal channels (order in output: RGB, depth, extra)
    extra_signals: Optional[
        Tensor
    ] = None,  # [..., (C,) N, E], or [N, K, E] when extra_signals_sh_degree set
    extra_signals_sh_degree: Optional[
        int
    ] = None,  # Currently only None or 3 is accepted.
    renderer_config: Optional[RendererConfig] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [..., N, D] or [..., C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, D], shared across all batch and camera dims (i.e. no leading `...` or `C` dims),
        where K is the number of SH bases and D is the number of feature channels. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.

        **Gaussian Depth Modes** (use projection depth, controlled by `global_z_order`):
        - "D": Accumulated Gaussian depth :math:`\\sum_i w_i z_i`
        - "ED": Expected Gaussian depth :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`
        - "RGB+D": RGB + accumulated Gaussian depth
        - "RGB+ED": RGB + expected Gaussian depth

        **Hit Distance Modes** (compute along-ray distance in rasterization):
        - "d": Accumulated hit distance :math:`\\sum_i w_i d_i`
        - "Ed": Expected hit distance :math:`\\frac{\\sum_i w_i d_i}{\\sum_i w_i}`
        - "RGB-d": RGB + accumulated hit distance
        - "RGB-Ed": RGB + expected hit distance

        "RGB" renders only the colored image. For combined modes, depth is the last channel.
        When extra_signals are present, render_colors is RGB + depth only (4 channels);
        extra channels are returned in ``meta["render_extra_signals"]``.

    .. note::
        **Extra signals**: Optional `extra_signals` are rendered and returned in ``meta["render_extra_signals"]``
        (shape [..., C, height, width, E]). If `extra_signals_sh_degree` is set, extra_signals are
        SH coefficients of shape [N, K, E] (shared across batch/camera dims), evaluated per view.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [..., C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.
        On the non-Eval3D path, this option raises ``RuntimeError`` when the total
        rendered feature width requires multiple rasterization passes.

    .. note::
        **Camera Distortion and Rolling Shutter**: The function supports rendering with opencv
        distortion formula for pinhole and fisheye cameras (`radial_coeffs`, `tangential_coeffs`, `thin_prism_coeffs`).
        It also supports rolling shutter rendering with the `rolling_shutter` argument. We take
        reference from the paper `3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting
        <https://arxiv.org/abs/2412.12507>`_.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] for post-activation colors, or [N, K, D] for SH coefficients (shared across batch/camera dims).
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
          For lidar sensors, this is ignored. The width is taken from lidar_coeffs.n_columns.
        height: The height of the image.
          For lidar sensors, this is ignored. The height is taken from lidar_coeffs.n_rows.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [N, K, D] SH coefficients (shared
            across batch/camera dims), else the `colors` should be [..., (C,) N, D]
            post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [..., C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "d", "Ed", "D", "ED",
            "RGB-d", "RGB-Ed", "RGB+D", and "RGB+ED". "RGB" renders the colored image.
            Gaussian depth modes (D, ED, RGB+D, RGB+ED) use projection depth. Hit distance
            modes (d, Ed, RGB-d, RGB-Ed) compute along-ray distance. Expected modes (Ed, ED)
            are normalized by opacity. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. On the non-Eval3D path, a total rendered
            feature width that requires multiple rasterization passes raises
            ``RuntimeError``. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        camera_model: The camera model to use. Supported models are "pinhole", "ortho",
            "fisheye", and "ftheta". Default is "pinhole".
        segmented: Whether to use segmented radix sort. Default is False.
            Segmented radix sort performs sorting in segments, which is more efficient for the sorting operation itself.
            However, since it requires offset indices as input, additional global memory access is needed, which results
            in slower overall performance in most use cases.
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [..., N, 3, 3], Default is None.
        with_ut: Whether to use Unscented Transform (UT) for projection. Default is False.
        with_eval3d: Whether to calculate Gaussian response in 3D world space, instead
            of 2D image space. Default is False.
        macro_tile: Whether to use the macro-tile (MT) two-level hierarchical
            intersection path. Currently supports the unpacked classic 3DGS
            EWA path only. Default is False.
        return_normals: Whether to compute and return accumulated normals per pixel.
            Normals are computed from Gaussian quaternions (canonical normal = (0,0,1)
            transformed by rotation, flipped if facing away from ray). Requires
            with_eval3d=True. Default is False.
        global_z_order: Whether to use z-depth (True) or Euclidean distance (False) for
            sorting Gaussians during rasterization. When True, Gaussians are sorted by their
            z-coordinate in camera space. When False, they are sorted by their Euclidean
            distance from the camera origin. Default is True.
        radial_coeffs: Opencv pinhole/fisheye radial distortion coefficients. Default is None.
            For pinhole camera, the shape should be [..., C, 6]. For fisheye camera, the shape
            should be [..., C, 4].
        tangential_coeffs: Opencv pinhole tangential distortion coefficients. Default is None.
            The shape should be [..., C, 2] if provided.
        thin_prism_coeffs: Opencv pinhole thin prism distortion coefficients. Default is None.
            The shape should be [..., C, 4] if provided.
        ftheta_coeffs: F-Theta camera distortion coefficients shared for all cameras.
            Default is None. See `FThetaCameraDistortionParameters` for details.
        rolling_shutter: The rolling shutter type. Default `RollingShutterType.GLOBAL` means
            global shutter.
        viewmats_rs: The second viewmat when rolling shutter is used. Default is None.
        renderer_config: The eval3d rasterizer implementation selector. Default is
            :class:`RendererConfig_MixedBatch`, which uses the existing mixed-batch
            rasterizer implementation. Non-default configs require
            ``with_eval3d=True``.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    has_color = render_mode_has_color(render_mode)

    external_distortion_coeffs = cast(
        Optional[BivariateWindshieldModelParameters], external_distortion_coeffs
    )

    if lidar_coeffs is not None:
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows

    tile_size = _resolve_tile_size(tile_size, with_eval3d, width, height)

    if covars is not None:
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]

    # Resolve the renderer configuration (MixedBatch / ParallelBatch). The
    # custom-class instances live only in Python; flatten to the CUDA enum
    # before crossing into the C++ orchestration op. ParallelBatch is only
    # valid on the eval3d (from-world) path.
    if renderer_config is None:
        renderer_config = RendererConfig_MixedBatch()
    _validate_renderer_config(renderer_config)
    if not with_eval3d and not isinstance(renderer_config, RendererConfig_MixedBatch):
        raise ValueError(
            f"{type(renderer_config).__name__} requires with_eval3d=True; "
            "the non-eval3d path only supports RendererConfig_MixedBatch."
        )
    renderer_config_impl = _renderer_config_type(renderer_config)

    # Both paths run the single C++ orchestrator. `distributed` is honored as
    # passed (no world_size-based downgrade): at one rank the gathers/scatters are
    # identities, so distributed=True stays numerically identical to the
    # single-GPU path and remains exercisable on a single GPU. Python derives the
    # NCCL group name (which also validates an initialized NCCL group) and the
    # world size; the C++ op owns all distributed validation.
    if distributed:
        process_group_name = _get_default_nccl_process_group_name()
        world_size = torch.distributed.get_world_size()
    else:
        process_group_name = None
        world_size = 1

    camera_model_type = _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")
    sh_degree_value = sh_degree if sh_degree is not None else -1
    extra_signals_sh_degree_value = (
        extra_signals_sh_degree if extra_signals_sh_degree is not None else -1
    )
    ftheta_params = (
        ftheta_coeffs
        if ftheta_coeffs is not None
        else FThetaCameraDistortionParameters()
    )
    (
        render_colors,
        render_alphas,
        render_extra_signals,
        render_normals,
        means2d_absgrad,
        batch_ids,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        projected_opacities,
        tiles_per_gauss,
        isect_ids,
        flatten_ids,
        isect_offsets,
        tile_width,
        tile_height,
    ) = _make_lazy_cuda_func("rasterization_3dgs")(
        means.contiguous(),
        covars.contiguous() if covars is not None else None,
        quats.contiguous() if quats is not None else None,
        scales.contiguous() if scales is not None else None,
        opacities.contiguous(),
        colors.contiguous() if (has_color and colors is not None) else None,
        viewmats.contiguous(),
        Ks.contiguous(),
        width,
        height,
        tile_size,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        backgrounds.contiguous() if backgrounds is not None else None,
        packed,
        sparse_grad,
        absgrad,
        rasterize_mode == "antialiased",
        rasterize_mode == "classic",
        camera_model_type,
        segmented,
        has_color,
        sh_degree_value,
        extra_signals.contiguous() if extra_signals is not None else None,
        extra_signals_sh_degree_value,
        render_mode_has_depth_channel(render_mode),
        render_mode_has_expected_depth(render_mode),
        with_eval3d,
        with_ut,
        rays.contiguous() if rays is not None else None,
        viewmats_rs.contiguous() if viewmats_rs is not None else None,
        ut_params if ut_params is not None else UnscentedTransformParameters(),
        rolling_shutter,
        radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_params,
        lidar_coeffs.to_cpp() if lidar_coeffs is not None else None,
        external_distortion_coeffs,
        global_z_order,
        render_mode_has_hit_distance(render_mode),
        return_normals,
        renderer_config_impl,
        process_group_name,
        world_size,
        macro_tile,
    )

    if absgrad and not with_eval3d:
        means2d.absgrad = means2d_absgrad

    if not packed:
        batch_ids = None
        camera_ids = None
        gaussian_ids = None

    batch_dims = means.shape[:-2]
    B = math.prod(batch_dims)
    C = viewmats.shape[-3]
    meta = {
        "batch_ids": batch_ids,
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": projected_opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_batches": B,
        "n_cameras": C,
    }

    if extra_signals is not None:
        meta["render_extra_signals"] = render_extra_signals
    if return_normals:
        meta["normals"] = render_normals

    return render_colors, render_alphas, meta


def _maybe_evaluate_sh(
    sh_degree, features, means, radii, viewmats, batch_dims, C, N, clamp
):
    num_batch_dims = len(batch_dims)

    # Turn features into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if features.dim() == num_batch_dims + 2:
            # Turn [..., N, D] into [..., C, N, D]
            features = torch.broadcast_to(
                features[..., None, :, :], batch_dims + (C, N, -1)
            )
        else:
            # features is already [..., C, N, D]
            pass
    else:
        camtoworlds = torch.inverse(viewmats)  # [..., C, 4, 4]
        dirs = means[..., None, :, :] - camtoworlds[..., None, :3, 3]  # [..., C, N, 3]
        masks = (radii > 0).all(dim=-1)  # [..., C, N]
        features = spherical_harmonics(
            sh_degree, dirs, features, masks=masks
        )  # [..., C, N, D]
        if clamp:
            # make it apple-to-apple with Inria's CUDA Backend.
            features = torch.clamp_min(features + 0.5, 0.0)
        else:
            features = features + 0.5
    return features


def _rasterization(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] for post-activation colors, or [N, K, D] for SH coefficients
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: Optional[int] = None,
    rays: Optional[
        Tensor
    ] = None,  # [..., C, H, W, 6] -> ox, oy, oz, dx*spread, dy*spread, dz*spread
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    rasterize_mode: RasterizeMode = "classic",
    _max_channels_per_launch: int = 32,
    batch_per_iter: int = 100,
    with_eval3d: bool = False,
    with_ut: bool = False,
    camera_model: CameraModel = "pinhole",
    lidar_coeffs: Optional[RowOffsetStructuredSpinningLidarModelParametersExt] = None,
    extra_signals: Optional[
        Tensor
    ] = None,  # [..., (C,) N, E], or [N, K, E] when extra_signals_sh_degree set
    extra_signals_sh_degree: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """
    from gsplat.cuda._torch_impl import _fully_fused_projection
    from gsplat.cuda._torch_impl import (
        _rasterize_to_pixels as _torch_rasterize_to_pixels,
    )
    from gsplat.cuda._torch_impl_eval3d import _rasterize_to_pixels_eval3d
    from gsplat.cuda._torch_impl_ut import _fully_fused_projection_with_ut
    from gsplat.cuda._math import _quat_scale_to_covar_preci

    if lidar_coeffs is not None:
        width = lidar_coeffs.n_columns
        height = lidar_coeffs.n_rows

    tile_size = _resolve_tile_size(tile_size, with_eval3d, width, height)

    has_color = render_mode_has_color(render_mode)

    def _rasterize_to_pixels(
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
        batch_per_iter: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        return _torch_rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            batch_per_iter=batch_per_iter,
        )

    _validate_3dgut_rasterize_mode(
        rasterize_mode, with_ut=with_ut, with_eval3d=with_eval3d
    )

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    D = (
        colors.shape[-1] if has_color else 0
    )  # number of primary color channels; 0 for depth-only
    I = B * C
    H = height
    W = width
    device = means.device
    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert rays is None or rays.shape == batch_dims + (C, H, W, 6), rays.shape

    if has_color:
        if sh_degree is None:
            # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
            assert (
                colors.dim() == num_batch_dims + 2
                and colors.shape[:-1] == batch_dims + (N,)
            ) or (
                colors.dim() == num_batch_dims + 3
                and colors.shape[:-1] == batch_dims + (C, N)
            ), colors.shape
        else:
            # treat colors as SH coefficients, must be in shape [N, K, D].
            # Allowing for activating partial SH bands.
            assert colors.dim() == 3 and colors.shape[0] == N, colors.shape
            assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    if with_ut:
        radii, means2d, depths, conics, compensations = _fully_fused_projection_with_ut(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            lidar_coeffs=lidar_coeffs,
        )
    else:
        if rays is not None:
            raise ValueError("Rays input is only supported with with_eval3d=True")
        assert camera_model == "pinhole", camera_model

        # Project Gaussians to 2D.
        # The results are with shape [..., C, N, ...]. Only the elements with radii > 0 are valid.
        covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
        radii, means2d, depths, conics, compensations = _fully_fused_projection(
            means,
            covars,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            calc_compensations=(rasterize_mode == "antialiased"),
        )
    opacities = torch.broadcast_to(
        opacities[..., None, :], batch_dims + (C, N)
    )  # [..., C, N]
    batch_ids, camera_ids, gaussian_ids = None, None, None
    image_ids = None

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    if lidar_coeffs is not None:
        tile_width = lidar_coeffs.tiling.n_bins_azimuth
        tile_height = lidar_coeffs.tiling.n_bins_elevation
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles_lidar(
            lidar_coeffs,
            means2d,
            radii,
            depths,
            packed=False,
            n_images=I,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
        )
    else:
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_images=I,
            image_ids=image_ids,
            gaussian_ids=gaussian_ids,
            conics=None if with_ut else conics,
            opacities=None if with_ut else opacities,
        )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # Turn colors into [..., C, N, D] or [..., nnz, D] to pass into rasterize_to_pixels()
    # Make sure they're clamped if evaluating SH.
    if has_color:
        colors = _maybe_evaluate_sh(
            sh_degree, colors, means, radii, viewmats, batch_dims, C, N, True
        )

    # Now do the same to the extra signals.
    if extra_signals is not None:
        # Do not clamp it.
        extra_signals = _maybe_evaluate_sh(
            extra_signals_sh_degree,
            extra_signals,
            means,
            radii,
            viewmats,
            batch_dims,
            C,
            N,
            False,
        )
        if has_color:
            # Concatenate colors and extra_signals for joint rasterization.
            assert colors.shape[:-1] == extra_signals.shape[:-1], (
                colors.shape,
                extra_signals.shape,
            )
            colors = torch.cat([colors, extra_signals], dim=-1)

    # Rasterize to pixels
    if render_mode_has_depth_channel(render_mode) and render_mode_has_color(
        render_mode
    ):
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode_has_only_depth_channel(render_mode):
        # In depth-only mode, extra_signals were not concatenated into colors
        # above. Place them before depth so depth stays last.
        if extra_signals is not None and not has_color:
            colors = torch.cat([extra_signals, depths[..., None]], dim=-1)
        else:
            colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(
                batch_dims + (C, colors.shape[-1]), device=backgrounds.device
            )
    else:  # RGB
        pass

    # Chunking logic for both eval3d and standard paths
    chunk_width = _max_channels_per_launch
    if colors.shape[-1] > chunk_width:
        # slice into chunks
        n_chunks = (colors.shape[-1] + chunk_width - 1) // chunk_width
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * chunk_width : (i + 1) * chunk_width]
            backgrounds_chunk = (
                backgrounds[..., i * chunk_width : (i + 1) * chunk_width]
                if backgrounds is not None
                else None
            )
            if with_eval3d:
                # Using CUDA code due to its speed. This function is already
                # being thoroughtly tested in test_basic.py
                render_colors_, render_alphas_ = rasterize_to_pixels_eval3d(
                    means=means,
                    quats=quats,
                    scales=scales,
                    colors=colors_chunk,
                    opacities=opacities,
                    viewmats=viewmats,
                    camera_model=camera_model,
                    Ks=Ks,
                    image_width=width,
                    image_height=height,
                    rays=rays,
                    lidar_coeffs=lidar_coeffs,
                    tile_size=tile_size,
                    isect_offsets=isect_offsets,
                    flatten_ids=flatten_ids,
                    backgrounds=backgrounds_chunk,
                )
            else:
                if rays is not None:
                    raise ValueError(
                        "Rays input is only supported with with_eval3d=True"
                    )
                assert camera_model == "pinhole", camera_model
                render_colors_, render_alphas_ = _rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    batch_per_iter=batch_per_iter,
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        # No chunking needed
        if with_eval3d:
            # Using CUDA code due to its speed. This function is already
            # being thoroughtly tested in test_basic.py
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means=means,
                quats=quats,
                scales=scales,
                colors=colors,
                opacities=opacities,
                viewmats=viewmats,
                Ks=Ks,
                image_width=width,
                image_height=height,
                rays=rays,
                camera_model=camera_model,
                lidar_coeffs=lidar_coeffs,
                tile_size=tile_size,
                isect_offsets=isect_offsets,
                flatten_ids=flatten_ids,
                backgrounds=backgrounds,
            )
        else:
            if rays is not None:
                raise ValueError("Rays input is only supported with with_eval3d=True")
            assert camera_model == "pinhole", camera_model
            render_colors, render_alphas = _rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                batch_per_iter=batch_per_iter,
            )

    if extra_signals is not None:
        # Extract the extra signals (per ray) from render_colors
        E = extra_signals.shape[-1]
        render_extra_signals = render_colors[..., D : D + E]
        # Leave only colors (and possibly depth)
        if render_mode_has_depth_channel(render_mode):
            render_depth = render_colors[..., -1:]

            # Normalize depth for expected modes (Ed, ED, RGB-Ed, RGB+ED)
            if render_mode_has_expected_depth(render_mode):
                render_depth = render_depth / render_alphas.clamp(min=1e-10)

            render_colors = torch.cat([render_colors[..., 0:D], render_depth], dim=-1)
        else:
            render_colors = render_colors[..., 0:D]
    else:
        render_extra_signals = None
        # Normalize depth for expected modes (Ed, ED, RGB-Ed, RGB+ED)
        if render_mode_has_expected_depth(render_mode):
            # normalize the accumulated depth to get the expected depth
            render_depth = render_colors[..., -1:] / render_alphas.clamp(min=1e-10)
            render_colors = torch.cat([render_colors[..., :D], render_depth], dim=-1)

    meta = {
        "batch_ids": batch_ids,
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_batches": B,
        "n_cameras": C,
    }

    if render_extra_signals is not None:
        meta["render_extra_signals"] = render_extra_signals

    return render_colors, render_alphas, meta


# def rasterization_legacy_wrapper(
#     means: Tensor,  # [N, 3]
#     quats: Tensor,  # [N, 4]
#     scales: Tensor,  # [N, 3]
#     opacities: Tensor,  # [N]
#     colors: Tensor,  # [N, D] or [N, K, D]
#     viewmats: Tensor,  # [C, 4, 4]
#     Ks: Tensor,  # [C, 3, 3]
#     width: int,
#     height: int,
#     near_plane: float = 0.01,
#     eps2d: float = 0.3,
#     sh_degree: Optional[int] = None,
#     tile_size: int = 16,
#     backgrounds: Optional[Tensor] = None,
#     **kwargs,
# ) -> Tuple[Tensor, Tensor, Dict]:
#     """Wrapper for old version gsplat.

#     .. warning::
#         This function exists for comparison purpose only. So we skip collecting
#         the intermidiate variables, and only return an empty dict.

#     """
#     from gsplat.cuda_legacy._wrapper import (
#         project_gaussians,
#         rasterize_gaussians,
#         spherical_harmonics,
#     )

#     assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
#     C = len(viewmats)

#     render_colors, render_alphas = [], []
#     for cid in range(C):
#         fx, fy = Ks[cid, 0, 0], Ks[cid, 1, 1]
#         cx, cy = Ks[cid, 0, 2], Ks[cid, 1, 2]
#         viewmat = viewmats[cid]

#         means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
#             means3d=means,
#             scales=scales,
#             glob_scale=1.0,
#             quats=quats,
#             viewmat=viewmat,
#             fx=fx,
#             fy=fy,
#             cx=cx,
#             cy=cy,
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             clip_thresh=near_plane,
#         )

#         if colors.dim() == 3:
#             c2w = viewmat.inverse()
#             viewdirs = means - c2w[:3, 3]
#             # viewdirs = F.normalize(viewdirs, dim=-1).detach()
#             if sh_degree is None:
#                 sh_degree = int(math.sqrt(colors.shape[1]) - 1)
#             colors = spherical_harmonics(sh_degree, viewdirs, colors)  # [N, 3]

#         background = (
#             backgrounds[cid]
#             if backgrounds is not None
#             else torch.zeros(colors.shape[-1], device=means.device)
#         )

#         render_colors_, render_alphas_ = rasterize_gaussians(
#             xys=means2d,
#             depths=depths,
#             radii=radii,
#             conics=conics,
#             num_tiles_hit=num_tiles_hit,
#             colors=colors,
#             opacity=opacities[..., None],
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             background=background,
#             return_alpha=True,
#         )
#         render_colors.append(render_colors_)
#         render_alphas.append(render_alphas_[..., None])
#     render_colors = torch.stack(render_colors, dim=0)
#     render_alphas = torch.stack(render_alphas, dim=0)
#     return render_colors, render_alphas, {}


def rasterization_inria_wrapper(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., N, D] for post-activation colors, or [N, K, 3] for SH coefficients
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for Inria's rasterization backend.

    .. warning::
        This function exists for comparison purpose only. Only rendered image is
        returned.

    .. warning::
        Inria's CUDA backend has its own LICENSE, so this function should be used with
        the respect to the original LICENSE at:
        https://github.com/graphdeco-inria/diff-gaussian-rasterization

    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    N = means.shape[-2]
    B = math.prod(batch_dims)
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (
            colors.dim() == num_batch_dims + 2
            and colors.shape[:-1] == batch_dims + (N,)
        ) or (
            colors.dim() == num_batch_dims + 3
            and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, must be in shape [N, K, 3].
        # Allowing for activating partial SH bands.
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # flatten all batch dimensions
    means = means.reshape(B, N, 3)
    quats = quats.reshape(B, N, 4)
    scales = scales.reshape(B, N, 3)
    opacities = opacities.reshape(B, N)
    viewmats = viewmats.reshape(B, C, 4, 4)
    Ks = Ks.reshape(B, C, 3, 3)
    if sh_degree is not None:
        # SH coefficients are deduplicated as (N, K, 3), shared across batch dims.
        colors = colors.unsqueeze(0).expand(B, -1, -1, -1)
    elif colors.dim() == num_batch_dims + 2:
        colors = colors.reshape(B, N, -1)
    elif colors.dim() == num_batch_dims + 3:
        colors = colors.reshape(B, C, N, -1)

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]

    render_colors = []
    for bid in range(B):
        for cid in range(C):
            FoVx = 2 * math.atan(width / (2 * Ks[bid, cid, 0, 0].item()))
            FoVy = 2 * math.atan(height / (2 * Ks[bid, cid, 1, 1].item()))
            tanfovx = math.tan(FoVx * 0.5)
            tanfovy = math.tan(FoVy * 0.5)

            world_view_transform = viewmats[bid, cid].transpose(0, 1)
            projection_matrix = get_projection_matrix(
                znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
            ).transpose(0, 1)
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            background = (
                backgrounds[bid, cid]
                if backgrounds is not None
                else torch.zeros(3, device=device)
            )

            raster_settings = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=background,
                scale_modifier=1.0,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0 if sh_degree is None else sh_degree,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )

            rasterizer = GaussianRasterizer(raster_settings=raster_settings)

            means2D = torch.zeros_like(means, requires_grad=True, device=device)

            render_colors_ = []
            for i in range(0, channels, 3):
                _colors = colors[bid, ..., i : i + 3]
                if _colors.shape[-1] < 3:
                    pad = torch.zeros(
                        _colors.shape[:-1], 3 - _colors.shape[-1], device=device
                    )
                    _colors = torch.cat([_colors, pad], dim=-1)
                _render_colors_, radii = rasterizer(
                    means3D=means[bid],
                    means2D=means2D[bid],
                    shs=_colors if colors.dim() == 4 else None,
                    colors_precomp=_colors if colors.dim() == 3 else None,
                    opacities=opacities[..., None],
                    scales=scales[bid],
                    rotations=quats[bid],
                    cov3D_precomp=None,
                )
                if _colors.shape[-1] < 3:
                    _render_colors_ = _render_colors_[..., : _colors.shape[-1]]
                render_colors_.append(_render_colors_)
            render_colors_ = torch.cat(render_colors_, dim=-1)

            render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
            render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)
    render_colors = render_colors.reshape(batch_dims + (height, width, channels))
    return render_colors, None, {}


###### 2DGS ######
def rasterization_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] for post-activation colors, or [N, K, D] for SH coefficients
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] for post-activation colors, or [N, K, D] for SH coefficients (shared across batch/camera dims).
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [N, K, D] SH coefficients (shared
            across batch/camera dims), else the `colors` should [(C,) N, D]
            post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "Ed", "D", "ED",
            "RGB+D", and "RGB+ED". "RGB" renders the colored image.
            Gaussian depth modes (D, ED, RGB+D, RGB+ED) use projection depth.
            Expected modes (Ed, ED) are normalized by opacity. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. This option raises ``RuntimeError`` when
            the total rendered feature width requires multiple rasterization
            passes. Default is False.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.
    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **render_normals**: The rendered normals. [..., C, height, width, 3].

        **surf_normals**: surface normal from depth. [..., C, height, width, 3]

        **render_distort**: The rendered distortions. [..., C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        means2d_absgrad,
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        ray_transforms,
        opacities,
        normals,
        tiles_per_gauss,
        isect_ids,
        flatten_ids,
        isect_offsets,
        densify,
        tile_width,
        tile_height,
        n_cameras,
    ) = _make_lazy_cuda_func("rasterization_2dgs")(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        opacities.contiguous(),
        colors.contiguous(),
        viewmats.contiguous(),
        Ks.contiguous(),
        width,
        height,
        tile_size,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        backgrounds.contiguous() if backgrounds is not None else None,
        packed,
        sparse_grad,
        absgrad,
        distloss,
        sh_degree,
        render_mode,
        depth_mode,
    )

    if absgrad:
        means2d.absgrad = means2d_absgrad

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": n_cameras,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


def rasterization_2dgs_inria_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    depth_ratio: int = 0,
    **kwargs,
) -> Tuple[Tuple, Dict]:
    """Wrapper for 2DGS's rasterization backend which is based on Inria's backend.

    Install the 2DGS rasterization backend from
        https://github.com/hbb1/diff-surfel-rasterization

    Credit to Jeffrey Hu https://github.com/jefequien

    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]
    scales = scales[:, :2]  # [N, 2]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)

        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            _render_colors_, radii, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
        render_colors_ = torch.cat(render_colors_, dim=-1)

        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)

    # additional maps
    allmap = allmap.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    render_depth_expected = allmap[..., 0:1]
    render_alphas = allmap[..., 1:2]
    render_normal = allmap[..., 2:5]
    render_depth_median = allmap[..., 5:6]
    render_dist = allmap[..., 6:7]

    render_normal = render_normal @ (world_view_transform[:3, :3].T)
    render_depth_expected = render_depth_expected / render_alphas
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # render_depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    render_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    normals_surf = depth_to_normal(
        render_depth, torch.linalg.inv_ex(viewmats).inverse, Ks
    )
    normals_surf = normals_surf * (render_alphas).detach()

    render_colors = torch.cat([render_colors, render_depth], dim=-1)

    meta = {
        "normals_rend": render_normal,
        "normals_surf": normals_surf,
        "render_distloss": render_dist,
        "means2d": means2D,
        "width": width,
        "height": height,
        "radii": radii.unsqueeze(0),
        "n_cameras": C,
        "gaussian_ids": None,
    }
    return (render_colors, render_alphas), meta
