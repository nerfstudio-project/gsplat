# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Fused NHT training: differentiable single-kernel rendering.

``nht_fused_render`` runs the fully-fused forward (rasterization + encoding +
inline WMMA MLP + sigmoid) and registers the fused backward, which
backpropagates dL/dRGB and dL/dalpha to the splat parameters AND the MLP
weights in one kernel — no intermediate feature buffer round trip, no
separate tcnn forward/backward launches.

Gradients
---------
means, quats, scales, features, opacities : fp32 (same conventions as the
    unfused NHT backward)
mlp_params : returned in tcnn linear layout, matching ``backbone.params``
    directly (the native-fragment conversion for the kernel happens
    internally each call — a 40 KB permutation, negligible).

Limitations (vs the unfused training path)
------------------------------------------
- No depth/normal render modes, no backgrounds inside the kernel (composite
  in RGB space outside, alpha gradients flow via v_render_alphas), no masks,
  no packed mode.
- Binning (projection + tile intersection) is non-differentiable; means2d
  gradients for densification strategies are not produced.
- Supported configs: CDIM in {16, 32, 48}, hidden in {64, 128}, layers in {2, 3}.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..cuda._wrapper import (
    FThetaCameraDistortionParameters,
    RollingShutterType,
    UnscentedTransformParameters,
    _make_lazy_cuda_obj,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
)
from ._wrapper import (
    convert_mlp_params_to_fused_native,
    rasterize_to_pixels_from_world_nht_3dgs_fused_bwd,
    rasterize_to_pixels_from_world_nht_3dgs_fused_fwd,
)

DEFAULT_LOSS_SCALE = 16384.0

# Feature dims (CDIM) with compiled kernel instantiations.
_FUSED_FWD_FEATURE_DIMS = (4, 8, 12, 16, 24, 32, 48, 64, 96)
_FUSED_BWD_FEATURE_DIMS = (16, 32, 48)


def nht_fused_supported(shader, for_training: bool = True) -> Tuple[bool, str]:
    """Check whether a DeferredShaderModule can run on the fused NHT kernels.

    The fused kernels hardwire the Composite[Identity + SH degree-3] input
    encoding and the rgb-only sigmoid output architecture, and are compiled
    for a fixed set of feature/hidden/layer configurations.

    Returns ``(supported, reason)`` — ``reason`` is empty when supported.
    """
    if not getattr(shader, "enable_view_encoding", False):
        return False, "requires enable_view_encoding=True"
    if getattr(shader, "view_encoding_type", "") != "sh":
        return False, "requires view_encoding_type='sh'"
    if getattr(shader, "view_sh_degree", 0) != 3:
        return False, "requires sh_degree=3"
    if getattr(shader, "_architecture", "") != "rgb_only_sigmoid":
        return False, (
            "requires the rgb-only sigmoid architecture "
            "(no AOV/auxiliary outputs, tcnn output_activation='Sigmoid')"
        )
    if shader.mlp_hidden_dim not in (64, 128):
        return False, f"mlp_hidden_dim must be in {{64, 128}}, got {shader.mlp_hidden_dim}"
    if shader.mlp_num_layers not in (2, 3):
        return False, f"mlp_num_layers must be in {{2, 3}}, got {shader.mlp_num_layers}"
    dims = _FUSED_BWD_FEATURE_DIMS if for_training else _FUSED_FWD_FEATURE_DIMS
    if shader.feature_dim not in dims:
        return False, f"feature_dim must be in {dims}, got {shader.feature_dim}"
    return True, ""


class _NHTFusedRasterize(torch.autograd.Function):
    """Differentiable fused NHT rasterize+shade for a single view."""

    @staticmethod
    def forward(
        ctx,
        means: Tensor,        # [N, 3] fp32
        quats: Tensor,        # [N, 4] fp32 (normalized)
        scales: Tensor,       # [N, 3] fp32 (activated)
        features: Tensor,     # [N, CDIM] fp16/fp32
        opacities: Tensor,    # [N] fp32 (activated)
        mlp_params: Tensor,   # [n_params] tcnn linear layout
        viewmat: Tensor,      # [4, 4]
        K: Tensor,            # [3, 3]
        width: int,
        height: int,
        tile_size: int,
        tile_offsets: Tensor,
        flatten_ids: Tensor,
        ray_dir_scale: float,
        center_ray_mode: bool,
        mlp_hidden_dim: int,
        mlp_num_layers: int,
        loss_scale: float,
    ):
        features_h = features.half().contiguous()
        params_h = mlp_params.detach()
        if params_h.dtype != torch.float16:
            params_h = params_h.half()
        n_feat = (features.shape[-1] // 4) * 2
        params_native = convert_mlp_params_to_fused_native(
            params_h, n_feat, mlp_hidden_dim, mlp_num_layers
        ).contiguous()

        # NB: grad mode is always disabled inside Function.forward, so check
        # the inputs' requires_grad flags (they survive) instead of
        # torch.is_grad_enabled().
        needs_grad = any(
            t.requires_grad
            for t in (means, quats, scales, features, opacities, mlp_params)
        )

        rgb, alphas, render_feat, last_ids = (
            rasterize_to_pixels_from_world_nht_3dgs_fused_fwd(
                means=means[None, None],
                quats=quats[None, None],
                scales=scales[None, None],
                colors=features_h[None, None],
                opacities=opacities[None, None],
                image_width=width, image_height=height, tile_size=tile_size,
                viewmats0=viewmat[None, None], viewmats1=None,
                Ks=K[None, None],
                camera_model=_make_lazy_cuda_obj("CameraModelType.PINHOLE"),
                ut_params=UnscentedTransformParameters(),
                rs_type=int(RollingShutterType.GLOBAL),
                radial_coeffs=None, tangential_coeffs=None,
                thin_prism_coeffs=None,
                ftheta_coeffs=FThetaCameraDistortionParameters(),
                lidar_coeffs=None, external_distortion_params=None,
                tile_offsets=tile_offsets, flatten_ids=flatten_ids,
                center_ray_mode=center_ray_mode,
                ray_dir_scale=ray_dir_scale,
                mlp_params=params_native,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_num_layers=mlp_num_layers,
                save_state=needs_grad,
            )
        )

        ctx.save_for_backward(
            means, quats, scales, features_h, opacities, params_native,
            viewmat, K, tile_offsets, flatten_ids,
            render_feat, alphas, last_ids,
        )
        ctx.meta = (
            width, height, tile_size, ray_dir_scale, center_ray_mode,
            mlp_hidden_dim, mlp_num_layers, loss_scale,
            features.dtype, mlp_params.dtype,
        )

        # [H, W, 3] fp32, [H, W] fp32
        rgb_out = rgb.reshape(height, width, 3).float()
        alpha_out = alphas.reshape(height, width)
        return rgb_out, alpha_out

    @staticmethod
    def backward(ctx, v_rgb: Tensor, v_alphas: Tensor):
        (means, quats, scales, features_h, opacities, params_native,
         viewmat, K, tile_offsets, flatten_ids,
         render_feat, alphas, last_ids) = ctx.saved_tensors
        (width, height, tile_size, ray_dir_scale, center_ray_mode,
         mlp_hidden_dim, mlp_num_layers, loss_scale,
         feat_dtype, params_dtype) = ctx.meta

        v_rgb_c = v_rgb.float().reshape(1, 1, height, width, 3).contiguous()
        if v_alphas is None:
            v_alphas_c = torch.zeros(
                1, 1, height, width, device=v_rgb.device, dtype=torch.float32
            )
        else:
            v_alphas_c = v_alphas.float().reshape(1, 1, height, width).contiguous()

        (v_means, v_quats, v_scales, v_colors, v_opacities, v_mlp) = (
            rasterize_to_pixels_from_world_nht_3dgs_fused_bwd(
                means=means[None, None],
                quats=quats[None, None],
                scales=scales[None, None],
                colors=features_h[None, None],
                opacities=opacities[None, None],
                image_width=width, image_height=height, tile_size=tile_size,
                viewmats0=viewmat[None, None], viewmats1=None,
                Ks=K[None, None],
                camera_model=_make_lazy_cuda_obj("CameraModelType.PINHOLE"),
                ut_params=UnscentedTransformParameters(),
                rs_type=int(RollingShutterType.GLOBAL),
                radial_coeffs=None, tangential_coeffs=None,
                thin_prism_coeffs=None,
                ftheta_coeffs=FThetaCameraDistortionParameters(),
                lidar_coeffs=None, external_distortion_params=None,
                tile_offsets=tile_offsets, flatten_ids=flatten_ids,
                center_ray_mode=center_ray_mode,
                ray_dir_scale=ray_dir_scale,
                mlp_params=params_native,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_num_layers=mlp_num_layers,
                loss_scale=loss_scale,
                render_feat=render_feat,
                render_alphas=alphas,
                last_ids=last_ids,
                v_render_rgb=v_rgb_c,
                v_render_alphas=v_alphas_c,
                compute_mlp_grad=ctx.needs_input_grad[5],
            )
        )

        v_features = v_colors.reshape(features_h.shape).to(feat_dtype)
        v_mlp_params = (
            (v_mlp / loss_scale).to(params_dtype) if v_mlp.numel() > 0 else None
        )

        return (
            v_means.reshape(means.shape),
            v_quats.reshape(quats.shape),
            v_scales.reshape(scales.shape),
            v_features,
            v_opacities.reshape(opacities.shape),
            v_mlp_params,
            None, None, None, None, None, None, None,
            None, None, None, None, None,
        )


@torch.no_grad()
def _binning(
    means, quats, scales, opacities, viewmat, K, width, height, tile_size,
    near_plane, far_plane, eps2d,
):
    """Non-differentiable projection + tile intersection for the fused path."""
    tile_width = (width + tile_size - 1) // tile_size
    tile_height = (height + tile_size - 1) // tile_size
    radii, means2d, depths, _, _ = fully_fused_projection_with_ut(
        means=means, quats=quats, scales=scales, opacities=opacities,
        viewmats=viewmat[None], Ks=K[None], width=width, height=height,
        eps2d=eps2d, near_plane=near_plane, far_plane=far_plane,
        calc_compensations=False, camera_model="pinhole",
        ut_params=UnscentedTransformParameters(),
        rolling_shutter=RollingShutterType.GLOBAL,
        viewmats_rs=None, global_z_order=True,
    )
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size=tile_size,
        tile_width=tile_width, tile_height=tile_height,
        packed=False, n_images=1,
    )
    tile_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    return tile_offsets, flatten_ids


def nht_fused_render(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    features: Tensor,
    opacities: Tensor,
    mlp_params: Tensor,
    viewmat: Tensor,
    K: Tensor,
    width: int,
    height: int,
    *,
    tile_size: int = 16,
    ray_dir_scale: float = 1.0,
    center_ray_mode: bool = False,
    mlp_hidden_dim: int = 64,
    mlp_num_layers: int = 2,
    loss_scale: float = DEFAULT_LOSS_SCALE,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
) -> Tuple[Tensor, Tensor]:
    """Differentiable fused NHT render of a single view.

    All geometric inputs must already be activated (normalized quats,
    exp scales, sigmoid opacities). ``mlp_params`` is the tcnn parameter
    vector (``shader.backbone.params``), linear layout.

    Returns ``(rgb [H, W, 3] fp32, alpha [H, W] fp32)``, both differentiable.
    """
    tile_offsets, flatten_ids = _binning(
        means.detach(), quats.detach(), scales.detach(), opacities.detach(),
        viewmat, K, width, height, tile_size, near_plane, far_plane, eps2d,
    )
    return _NHTFusedRasterize.apply(
        means, quats, scales, features, opacities, mlp_params,
        viewmat, K, width, height, tile_size, tile_offsets, flatten_ids,
        ray_dir_scale, center_ray_mode, mlp_hidden_dim, mlp_num_layers,
        loss_scale,
    )
