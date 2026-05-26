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
"""Python wrappers around the NHT CUDA ops.

Kept here (rather than in ``gsplat.cuda._wrapper``) so that the core wrapper
module has no NHT-specific surface. ``rasterization(..., nht_params=...)`` from
:mod:`gsplat.rendering` delegates here via :mod:`gsplat.nht._rendering`.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..cuda._wrapper import (
    FThetaCameraDistortionParameters,
    RollingShutterType,
    UnscentedTransformParameters,
    _make_lazy_cuda_func,
    _make_lazy_cuda_obj,
)


# ---------------------------------------------------------------------------
# NHT compile-time constants exposed to Python via the C++ extension.
# ---------------------------------------------------------------------------


def get_encoding_expansion_factor() -> int:
    """Output expansion factor per base feature after harmonic encoding.

    With sin+cos encoding at ``N`` frequencies, each base feature produces
    ``2*N`` output channels. Returns the compile-time ``ENCF`` constant.
    """
    return _make_lazy_cuda_obj("encoding_expansion_factor")


def get_num_encoding_frequencies() -> int:
    """Number of encoding frequencies used internally (``ENCF / 2``)."""
    return _make_lazy_cuda_obj("num_encoding_frequencies")


def get_feature_divisor() -> int:
    """``VERTEX_PER_PRIM``: number of vertices per primitive (4 for tetrahedral).

    Input feature channels are divided evenly among this many vertices.
    """
    return _make_lazy_cuda_obj("feature_divisor")


# ---------------------------------------------------------------------------
# Channels supported by the NHT CUDA kernels' templated switch.
# ---------------------------------------------------------------------------
NHT_SUPPORTED_CHANNELS: Tuple[int, ...] = (
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    64,
    80,
    96,
    128,
    256,
)


def _find_next_supported(channels: int, supported: Tuple[int, ...]) -> int:
    for s in supported:
        if s >= channels:
            return s
    return -1


def _nht_pad_colors(
    colors: Tensor,
    backgrounds: Optional[Tensor],
    divisor: int,
    encf: int,
    supported: Tuple[int, ...],
    device,
) -> Tuple[Tensor, Optional[Tensor], int, int]:
    """Pad NHT vertex-feature colors to a supported CDIM (input channels).

    The C++ dispatch uses ``colors.size(-1)`` (input channels) as the template
    CDIM. The output has ``(input_ch / divisor) * encf`` channels.

    Returns ``(padded_colors, padded_backgrounds, output_channels, padded_delta)``.
    """
    input_ch = colors.shape[-1]
    fvdim = input_ch // divisor
    output_ch = fvdim * encf

    padded_input = 0
    if input_ch not in supported:
        target_input = _find_next_supported(input_ch, supported)
        if target_input < 0:
            raise ValueError(f"Unsupported NHT input channels: {input_ch}")
        padded_input = target_input - input_ch
        target_fvdim = target_input // divisor
        pad_per_group = target_fvdim - fvdim
        colors = colors.unflatten(-1, (divisor, fvdim))
        colors = torch.nn.functional.pad(colors, (0, pad_per_group))
        colors = colors.flatten(-2, -1)
        fvdim = target_fvdim
        output_ch = fvdim * encf
        if backgrounds is not None:
            padded_output = output_ch - (input_ch // divisor) * encf
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(*backgrounds.shape[:-1], padded_output, device=device),
                ],
                dim=-1,
            )

    return colors, backgrounds, output_ch, padded_input


# ---------------------------------------------------------------------------
# 3DGS NHT (from-world / eval3d path) autograd
# ---------------------------------------------------------------------------
# NHT always uses world evaluation (with_eval3d + with_ut).


class _RasterizeToPixelsNHTEval3D(torch.autograd.Function):
    """Autograd bridge for the NHT world-eval CUDA ops."""

    @staticmethod
    def forward(
        ctx,
        means,
        quats,
        scales,
        colors,
        opacities,
        backgrounds,
        masks,
        viewmats,
        Ks,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        camera_model="pinhole",
        ut_params=None,
        radial_coeffs=None,
        tangential_coeffs=None,
        thin_prism_coeffs=None,
        ftheta_coeffs=None,
        lidar_coeffs=None,
        external_distortion_coeffs=None,
        rolling_shutter=RollingShutterType.GLOBAL,
        viewmats_rs=None,
        center_ray_mode=False,
        ray_dir_scale=1.0,
        depths_per_gauss=None,
        use_hit_distance=False,
        with_normals=False,
    ):
        if ut_params is None:
            ut_params = UnscentedTransformParameters()
        cm_type = _make_lazy_cuda_obj(f"CameraModelType.{camera_model.upper()}")
        ftheta_cpp = (
            ftheta_coeffs
            if ftheta_coeffs is not None
            else FThetaCameraDistortionParameters()
        )

        (
            fused_renders,
            render_alphas,
            render_depth,
            render_normals,
            last_ids,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_from_world_nht_3dgs_fwd")(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            width,
            height,
            tile_size,
            viewmats,
            viewmats_rs,
            Ks,
            cm_type,
            ut_params,
            int(rolling_shutter),
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_cpp,
            lidar_coeffs,
            external_distortion_coeffs,
            isect_offsets,
            flatten_ids,
            center_ray_mode,
            ray_dir_scale,
            depths_per_gauss,
            use_hit_distance,
            with_normals,
        )

        ctx.save_for_backward(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            viewmats,
            viewmats_rs,
            Ks,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            depths_per_gauss if depths_per_gauss is not None else torch.empty(0),
        )
        ctx.width = width
        ctx.height = height
        ctx.tile_size = tile_size
        ctx.ut_params = ut_params
        ctx.rs_type = int(rolling_shutter)
        ctx.cm_type = cm_type
        ctx.ftheta_cpp = ftheta_cpp
        ctx.lidar_coeffs = lidar_coeffs
        ctx.external_distortion_coeffs = external_distortion_coeffs
        ctx.use_hit_distance = bool(use_hit_distance)
        ctx.with_depth = render_depth.numel() > 0
        ctx.with_normals = render_normals.numel() > 0
        ctx.has_depths_per_gauss = depths_per_gauss is not None
        return fused_renders, render_alphas, render_depth, render_normals

    @staticmethod
    def backward(
        ctx,
        v_fused_renders,
        v_render_alphas,
        v_render_depth,
        v_render_normals,
    ):
        (
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            viewmats,
            viewmats_rs,
            Ks,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            isect_offsets,
            flatten_ids,
            render_alphas,
            last_ids,
            saved_depths_per_gauss,
        ) = ctx.saved_tensors

        # Strip the 3 ray-direction gradient channels (no gradient path).
        # Forward output is half; backward kernel accumulates in FP32.
        v_render_colors = v_fused_renders[..., :-3].contiguous().float()

        v_render_depth_in = None
        if ctx.with_depth and v_render_depth is not None and v_render_depth.numel() > 0:
            v_render_depth_in = v_render_depth.contiguous()
        v_render_normals_in = None
        if (
            ctx.with_normals
            and v_render_normals is not None
            and v_render_normals.numel() > 0
        ):
            v_render_normals_in = v_render_normals.contiguous()

        depths_per_gauss_in = (
            saved_depths_per_gauss if ctx.has_depths_per_gauss else None
        )

        (
            v_means,
            v_quats,
            v_scales,
            v_colors,
            v_opacities,
            v_depths_per_gauss,
        ) = _make_lazy_cuda_func("rasterize_to_pixels_from_world_nht_3dgs_bwd")(
            means,
            quats,
            scales,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.width,
            ctx.height,
            ctx.tile_size,
            viewmats,
            viewmats_rs,
            Ks,
            ctx.cm_type,
            ctx.ut_params,
            ctx.rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ctx.ftheta_cpp,
            ctx.lidar_coeffs,
            ctx.external_distortion_coeffs,
            isect_offsets,
            flatten_ids,
            depths_per_gauss_in,
            ctx.use_hit_distance,
            render_alphas,
            last_ids,
            v_render_colors,
            v_render_alphas.contiguous(),
            v_render_depth_in,
            v_render_normals_in,
        )

        v_backgrounds = None
        if ctx.needs_input_grad[5]:
            v_bg_colors = v_fused_renders[..., :-3]
            v_backgrounds = (v_bg_colors * (1.0 - render_alphas).float()).sum(
                dim=(-3, -2)
            )

        v_depths_out = (
            v_depths_per_gauss
            if (ctx.has_depths_per_gauss and v_depths_per_gauss.numel() > 0)
            else None
        )

        return (
            v_means,  # means
            v_quats,  # quats
            v_scales,  # scales
            v_colors,  # colors
            v_opacities,  # opacities
            v_backgrounds,  # backgrounds
            None,  # masks
            None,  # viewmats
            None,  # Ks
            None,  # width
            None,  # height
            None,  # tile_size
            None,  # isect_offsets
            None,  # flatten_ids
            None,  # camera_model
            None,  # ut_params
            None,  # radial_coeffs
            None,  # tangential_coeffs
            None,  # thin_prism_coeffs
            None,  # ftheta_coeffs
            None,  # lidar_coeffs
            None,  # external_distortion_coeffs
            None,  # rolling_shutter
            None,  # viewmats_rs
            None,  # center_ray_mode
            None,  # ray_dir_scale
            v_depths_out,  # depths_per_gauss
            None,  # use_hit_distance
            None,  # with_normals
        )


def rasterize_to_pixels_eval3d_nht_extra(
    *,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    camera_model: str = "pinhole",
    ut_params=None,
    rays: Optional[Tensor] = None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs=None,
    lidar_coeffs=None,
    external_distortion_coeffs=None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,
    return_sample_counts: bool = False,
    use_hit_distance: bool = False,
    return_normals: bool = False,
    nht_params=None,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """NHT variant of ``rasterize_to_pixels_eval3d_extra``.

    Returns the standard wrapper tuple. ``render_colors`` uses
    ``[encoded_features | optional depth | 3 ray_dirs]``.
    """
    if rays is not None:
        raise NotImplementedError(
            "NHT does not support per-pixel `rays` input (camera model dispatch "
            "computes rays itself). Call rasterization() with rays=None."
        )
    (
        fused_renders,
        render_alphas,
        render_depth,
        render_normals,
    ) = _rasterize_nht_features(
        means=means,
        quats=quats,
        scales=scales,
        colors=colors,
        opacities=opacities,
        viewmats=viewmats,
        Ks=Ks,
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
        nht_params=nht_params,
        use_hit_distance=use_hit_distance,
        # NHT exposes stable normals through rendering.py as depth-derived
        # surface normals. The CUDA primitive-normal output is intentionally
        # not requested here; it is kept in the kernel only for parity/debugging
        # with regular 3DGS eval3d's camera-facing normal convention.
        with_normals=False,
    )

    # Match the standard eval3d convention: depth remains the last color channel.
    if render_depth is not None and render_depth.numel() > 0:
        fused_renders = torch.cat(
            [fused_renders, render_depth],
            dim=-1,
        )

    normals_out: Optional[Tensor] = None
    if render_normals is not None and render_normals.numel() > 0:
        normals_out = render_normals

    # Keep return arity aligned with the standard wrapper.
    placeholder_last_ids = torch.zeros(
        *render_alphas.shape[:-1], dtype=torch.int32, device=render_alphas.device
    )
    return fused_renders, render_alphas, placeholder_last_ids, None, normals_out


def _rasterize_nht_features(
    *,
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    colors: Tensor,
    opacities: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    backgrounds: Optional[Tensor] = None,
    masks: Optional[Tensor] = None,
    camera_model: str = "pinhole",
    ut_params=None,
    radial_coeffs: Optional[Tensor] = None,
    tangential_coeffs: Optional[Tensor] = None,
    thin_prism_coeffs: Optional[Tensor] = None,
    ftheta_coeffs=None,
    lidar_coeffs=None,
    external_distortion_coeffs=None,
    rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
    viewmats_rs: Optional[Tensor] = None,
    nht_params=None,
    depths_per_gauss: Optional[Tensor] = None,
    use_hit_distance: bool = False,
    with_normals: bool = False,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """NHT (Neural Harmonic Textures) tetrahedral-feature rasterizer.

    Padded as needed to the next NHT template instantiation, then routed to the
    NHT CUDA kernel. Returns
    ``(fused_renders, render_alphas, render_depth, render_normals)``.
    ``fused_renders`` is ``[encoded_features | 3 ray_dirs]`` and follows the
    standard wrapper convention by returning depth separately for the caller to
    append as the last channel. ``render_depth`` has shape ``[..., H, W, 1]``
    (empty tensor when depth is not requested). ``render_normals``
    has shape ``[..., H, W, 3]`` (empty tensor when ``with_normals`` is False).
    """
    if ut_params is None:
        ut_params = UnscentedTransformParameters()
    if nht_params is None:
        from gsplat.nht._rendering import NHTParams

        nht_params = NHTParams()
    center_ray_mode = bool(nht_params.center_ray_mode)
    ray_dir_scale = float(nht_params.ray_dir_scale)
    device = means.device
    encf = get_encoding_expansion_factor()
    divisor = get_feature_divisor()

    # Standard eval3d packs projection depth (or a dummy slot for hit distance)
    # as the last color channel. NHT must not harmonically encode that scalar,
    # so strip it before feature interpolation and feed it to the fused aux
    # depth output instead.
    depth_feature = depths_per_gauss
    maybe_depth = (
        colors.shape[-1] % divisor != 0 and (colors.shape[-1] - 1) % divisor == 0
    )
    if maybe_depth:
        depth_feature = colors[..., -1].contiguous()
        colors = colors[..., :-1]
        if backgrounds is not None:
            backgrounds = backgrounds[..., :-1]
    elif use_hit_distance:
        depth_feature = None

    if colors.shape[-1] % divisor != 0:
        raise ValueError(
            f"NHT feature channels must be divisible by {divisor}; got {colors.shape[-1]}"
        )

    orig_fvdim = colors.shape[-1] // divisor
    colors, backgrounds, _, _ = _nht_pad_colors(
        colors, backgrounds, divisor, encf, NHT_SUPPORTED_CHANNELS, device
    )

    (
        fused_renders,
        render_alphas,
        render_depth,
        render_normals,
    ) = _RasterizeToPixelsNHTEval3D.apply(
        means.contiguous(),
        quats.contiguous(),
        scales.contiguous(),
        colors.contiguous(),
        opacities.contiguous(),
        backgrounds.contiguous() if backgrounds is not None else None,
        masks.contiguous() if masks is not None else None,
        viewmats.contiguous(),
        Ks.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
        camera_model,
        ut_params,
        radial_coeffs.contiguous() if radial_coeffs is not None else None,
        tangential_coeffs.contiguous() if tangential_coeffs is not None else None,
        thin_prism_coeffs.contiguous() if thin_prism_coeffs is not None else None,
        ftheta_coeffs,
        lidar_coeffs.to_cpp() if lidar_coeffs is not None else None,
        external_distortion_coeffs,
        rolling_shutter,
        viewmats_rs.contiguous() if viewmats_rs is not None else None,
        center_ray_mode,
        ray_dir_scale,
        depth_feature,
        use_hit_distance,
        with_normals,
    )
    # fused_renders: [..., C, H, W, padded_feat_ch + 3]
    # Split off the 3 ray direction channels, strip padding from features,
    # then re-concatenate.
    ray_dirs = fused_renders[..., -3:]
    feat_renders = fused_renders[..., :-3]
    orig_output = orig_fvdim * encf
    if feat_renders.shape[-1] > orig_output:
        feat_renders = feat_renders[..., :orig_output]
    fused_renders = torch.cat([feat_renders, ray_dirs], dim=-1)
    return fused_renders, render_alphas, render_depth, render_normals
