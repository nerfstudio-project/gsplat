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

"""Autograd.Function wrappers for fused CUDA loss kernels.

Dedicated companion to :mod:`gsplat.losses_fused` — this module groups the
autograd bridges for the fused-loss CUDA kernels, mirroring the
``losses_*`` / ``cuda/_*_wrapper`` split used elsewhere in the package.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import _make_lazy_cuda_func


class _FusedGaussianLosses(torch.autograd.Function):
    """Fused forward+backward for gaussian_scale_reg, gaussian_density_reg,
    gaussian_z_scale_reg, and out_of_bound_loss via a single CUDA kernel."""

    @staticmethod
    def forward(
        ctx,
        scales: Tensor,  # [N, 3]
        densities: Tensor,  # [N]
        z_scales: Tensor,  # [N]
        positions: Tensor,  # [N, 3]
        cuboid_dims: Tensor,  # [N, 3]
        z_scale_threshold: float,
        visibility: Optional[Tensor] = None,  # [N] float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # The CUDA backward does not compute d(oob)/d(cuboid_dims); failing loudly
        # here is safer than silently returning a zero gradient that the
        # pure-PyTorch path would have propagated.
        if cuboid_dims.requires_grad:
            raise ValueError(
                "FusedGaussianLosses does not support grad w.r.t. cuboid_dims; "
                "pass cuboid_dims with requires_grad=False."
            )
        scales = scales.contiguous()
        densities = densities.contiguous()
        z_scales = z_scales.contiguous()
        positions = positions.contiguous()
        cuboid_dims = cuboid_dims.contiguous()
        if visibility is not None:
            # The pure-PyTorch path accepts visibility as [N] or [N, 1]; the
            # CUDA kernel expects [N]. Reshape here so the public API matches
            # the pure-PyTorch path.
            visibility = visibility.reshape(-1).contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps.
        loss_scale = torch.empty_like(scales)
        loss_density = torch.empty_like(densities)
        loss_z_scale = torch.empty_like(z_scales)
        loss_oob = torch.empty_like(positions)

        _make_lazy_cuda_func("gaussian_losses_fwd")(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility,
            z_scale_threshold,
            loss_scale,
            loss_density,
            loss_z_scale,
            loss_oob,
        )

        if visibility is not None:
            ctx.save_for_backward(
                scales, densities, z_scales, positions, cuboid_dims, visibility
            )
            ctx.has_visibility = True
        else:
            ctx.save_for_backward(scales, densities, z_scales, positions, cuboid_dims)
            ctx.has_visibility = False
        ctx.z_scale_threshold = z_scale_threshold

        return loss_scale, loss_density, loss_z_scale, loss_oob

    @staticmethod
    def backward(
        ctx,
        v_loss_scale: Tensor,
        v_loss_density: Tensor,
        v_loss_z_scale: Tensor,
        v_loss_oob: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        if ctx.has_visibility:
            (
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                visibility,
            ) = ctx.saved_tensors
        else:
            scales, densities, z_scales, positions, cuboid_dims = ctx.saved_tensors
            visibility = None

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_scales = torch.empty_like(scales)
        v_densities = torch.empty_like(densities)
        v_z_scales = torch.empty_like(z_scales)
        v_positions = torch.empty_like(positions)

        _make_lazy_cuda_func("gaussian_losses_bwd")(
            scales,
            densities,
            z_scales,
            positions,
            cuboid_dims,
            visibility,
            ctx.z_scale_threshold,
            v_loss_scale.contiguous(),
            v_loss_density.contiguous(),
            v_loss_z_scale.contiguous(),
            v_loss_oob.contiguous(),
            v_scales,
            v_densities,
            v_z_scales,
            v_positions,
        )

        # Gradients for: scales, densities, z_scales, positions, cuboid_dims,
        #                 z_scale_threshold, visibility
        return v_scales, v_densities, v_z_scales, v_positions, None, None, None


class _FusedCameraLosses(torch.autograd.Function):
    """Fused RGB L1 + background MSE with per-ray flag masking."""

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [N] int32
        rgb_pred: Tensor,  # [N, 3]
        rgb_gt: Tensor,  # [N, 3]
        bg_pred: Tensor,  # [N]
        rgb_factor: float,
        bg_factor: float,
    ) -> Tuple[Tensor, Tensor]:
        flags = flags.contiguous()
        rgb_pred = rgb_pred.contiguous()
        rgb_gt = rgb_gt.contiguous()
        bg_pred = bg_pred.contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps. rgb_loss
        # is per-ray [N] (the RGB channels are reduced inside the kernel), so it
        # is allocated from bg_pred's [N] shape rather than rgb_pred's [N, 3].
        rgb_loss = torch.empty_like(bg_pred)
        bg_loss = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("camera_losses_fwd")(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            rgb_factor,
            bg_factor,
            rgb_loss,
            bg_loss,
        )

        ctx.save_for_backward(flags, rgb_pred, rgb_gt, bg_pred)
        ctx.rgb_factor = rgb_factor
        ctx.bg_factor = bg_factor

        return rgb_loss, bg_loss

    @staticmethod
    def backward(
        ctx,
        v_rgb_loss: Tensor,
        v_bg_loss: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        flags, rgb_pred, rgb_gt, bg_pred = ctx.saved_tensors

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_rgb_pred = torch.empty_like(rgb_pred)
        v_bg_pred = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("camera_losses_bwd")(
            flags,
            rgb_pred,
            rgb_gt,
            bg_pred,
            ctx.rgb_factor,
            ctx.bg_factor,
            v_rgb_loss.contiguous(),
            v_bg_loss.contiguous(),
            v_rgb_pred,
            v_bg_pred,
        )

        # Gradients for: flags, rgb_pred, rgb_gt, bg_pred, rgb_factor, bg_factor
        return None, v_rgb_pred, None, v_bg_pred, None, None


class _FusedLidarLosses(torch.autograd.Function):
    """Fused distance L1 + intensity/raydrop/bg MSE with per-ray flag masking."""

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [N] int32
        distance_pred: Tensor,  # [N]
        distance_gt: Tensor,  # [N]
        intensity_pred: Tensor,  # [N]
        intensity_gt: Tensor,  # [N]
        raydrop_pred: Tensor,  # [N]
        raydrop_gt: Tensor,  # [N]
        bg_pred: Tensor,  # [N]
        distance_factor: float,
        intensity_factor: float,
        raydrop_factor: float,
        bg_factor: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        flags = flags.contiguous()
        distance_pred = distance_pred.contiguous()
        distance_gt = distance_gt.contiguous()
        intensity_pred = intensity_pred.contiguous()
        intensity_gt = intensity_gt.contiguous()
        raydrop_pred = raydrop_pred.contiguous()
        raydrop_gt = raydrop_gt.contiguous()
        bg_pred = bg_pred.contiguous()

        # Pre-allocate outputs in Python so lifetimes are explicit and the
        # caching allocator can reuse buffers across training steps. Each loss
        # is per-ray [N], matching its corresponding *_pred input.
        distance_loss = torch.empty_like(distance_pred)
        intensity_loss = torch.empty_like(intensity_pred)
        raydrop_loss = torch.empty_like(raydrop_pred)
        bg_loss = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("lidar_losses_fwd")(
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
            distance_factor,
            intensity_factor,
            raydrop_factor,
            bg_factor,
            distance_loss,
            intensity_loss,
            raydrop_loss,
            bg_loss,
        )

        ctx.save_for_backward(
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
        )
        ctx.factors = (distance_factor, intensity_factor, raydrop_factor, bg_factor)

        return distance_loss, intensity_loss, raydrop_loss, bg_loss

    @staticmethod
    def backward(
        ctx,
        v_distance_loss: Tensor,
        v_intensity_loss: Tensor,
        v_raydrop_loss: Tensor,
        v_bg_loss: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        (
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
        ) = ctx.saved_tensors
        distance_factor, intensity_factor, raydrop_factor, bg_factor = ctx.factors

        # Pre-allocate gradient buffers in Python (matches forward pattern).
        v_distance_pred = torch.empty_like(distance_pred)
        v_intensity_pred = torch.empty_like(intensity_pred)
        v_raydrop_pred = torch.empty_like(raydrop_pred)
        v_bg_pred = torch.empty_like(bg_pred)

        _make_lazy_cuda_func("lidar_losses_bwd")(
            flags,
            distance_pred,
            distance_gt,
            intensity_pred,
            intensity_gt,
            raydrop_pred,
            raydrop_gt,
            bg_pred,
            distance_factor,
            intensity_factor,
            raydrop_factor,
            bg_factor,
            v_distance_loss.contiguous(),
            v_intensity_loss.contiguous(),
            v_raydrop_loss.contiguous(),
            v_bg_loss.contiguous(),
            v_distance_pred,
            v_intensity_pred,
            v_raydrop_pred,
            v_bg_pred,
        )

        # Gradients for: flags, distance_pred, distance_gt, intensity_pred,
        #   intensity_gt, raydrop_pred, raydrop_gt, bg_pred, 4x factors
        return (
            None,
            v_distance_pred,
            None,
            v_intensity_pred,
            None,
            v_raydrop_pred,
            None,
            v_bg_pred,
            None,
            None,
            None,
            None,
        )


class _FusedBgGridLosses(torch.autograd.Function):
    """Fused forward+backward for sky-envmap TV + bilateral-grid drift + grid
    spatial TV via a single pair of CUDA kernels.

    Any of ``bg_tex`` / ``grids_camera`` / ``grids_frame`` can be ``None``
    to skip its sub-losses entirely. Per-sub-loss enable/disable is also
    available via a negative ``factor`` (loss output is written as zeros).
    """

    @staticmethod
    def forward(
        ctx,
        bg_tex: Optional[Tensor],  # [B*D, H, W, C] flat (D=1 planar, D=6 cubemap)
        bg_tex_depth: int,
        grids_camera: Optional[Tensor],  # [B*12, D, H, W]
        grids_frame: Optional[Tensor],  # [B*12, D, H, W]
        bg_tex_factor: float,
        grid_drift_camera_factor: float,
        grid_drift_frame_factor: float,
        grid_camera_tv_factor: float,
        grid_frame_tv_factor: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Unused outputs must reach backward as None rather than materialized zeros, so a
        # NaN-weighted-but-unused sub-loss can't be fed to the native op and poison a finite
        # sibling's shared-input gradient via NaN*0. backward() disables each None output.
        ctx.set_materialize_grads(False)
        bg_tex = bg_tex.contiguous() if bg_tex is not None else None
        grids_camera = grids_camera.contiguous() if grids_camera is not None else None
        grids_frame = grids_frame.contiguous() if grids_frame is not None else None

        # Reference device for allocation (first non-None input).
        ref = (
            bg_tex
            if bg_tex is not None
            else (grids_camera if grids_camera is not None else grids_frame)
        )
        if ref is None:
            raise ValueError(
                "FusedBgGridLosses.forward needs at least one of "
                "bg_tex / grids_camera / grids_frame to be non-None."
            )
        device, dtype = ref.device, ref.dtype

        # Per-cell element counts (1 scalar per (b, d, h, w) cell for grids;
        # 1 scalar per element for bg_tex).
        def _cell_numel(g: Optional[Tensor]) -> int:
            if g is None:
                return 0
            # g is [B*12, D, H, W]; cells = B * D * H * W. The 12 is the fixed
            # 3x4 affine channel count (LOSSES_GRID_NUM_CHANNELS in Config.h,
            # which is non-overrideable so this constant can't drift from it).
            return (g.shape[0] // 12) * g.shape[1] * g.shape[2] * g.shape[3]

        numel_bg = bg_tex.numel() if bg_tex is not None else 0
        numel_gc = _cell_numel(grids_camera)
        numel_gf = _cell_numel(grids_frame)

        # Pre-allocate outputs on the Python side (keeps buffer lifetime
        # explicit and lets torch's caching allocator reuse across steps).
        bg_tex_loss = torch.empty(numel_bg, device=device, dtype=dtype)
        grids_drift_loss = torch.empty(numel_gc + numel_gf, device=device, dtype=dtype)
        grid_camera_tv_loss = torch.empty(numel_gc, device=device, dtype=dtype)
        grid_frame_tv_loss = torch.empty(numel_gf, device=device, dtype=dtype)

        _make_lazy_cuda_func("bg_grid_losses_fwd")(
            bg_tex,
            int(bg_tex_depth),
            grids_camera,
            grids_frame,
            float(bg_tex_factor),
            float(grid_drift_camera_factor),
            float(grid_drift_frame_factor),
            float(grid_camera_tv_factor),
            float(grid_frame_tv_factor),
            bg_tex_loss,
            grids_drift_loss,
            grid_camera_tv_loss,
            grid_frame_tv_loss,
        )

        ctx.save_for_backward(
            bg_tex
            if bg_tex is not None
            else torch.empty(0, device=device, dtype=dtype),
            grids_camera
            if grids_camera is not None
            else torch.empty(0, device=device, dtype=dtype),
            grids_frame
            if grids_frame is not None
            else torch.empty(0, device=device, dtype=dtype),
        )
        ctx.has_bg_tex = bg_tex is not None
        ctx.has_grids_camera = grids_camera is not None
        ctx.has_grids_frame = grids_frame is not None
        ctx.bg_tex_depth = int(bg_tex_depth)
        # Output element counts so backward can build shape-correct zero placeholders
        # for any output whose cotangent arrives as None (unused).
        ctx.numel_bg = numel_bg
        ctx.numel_gc = numel_gc
        ctx.numel_gf = numel_gf
        ctx.factors = (
            float(bg_tex_factor),
            float(grid_drift_camera_factor),
            float(grid_drift_frame_factor),
            float(grid_camera_tv_factor),
            float(grid_frame_tv_factor),
        )

        return bg_tex_loss, grids_drift_loss, grid_camera_tv_loss, grid_frame_tv_loss

    @staticmethod
    def backward(
        ctx,
        v_bg_tex_loss: Optional[Tensor],
        v_grids_drift_loss: Optional[Tensor],
        v_grid_camera_tv_loss: Optional[Tensor],
        v_grid_frame_tv_loss: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], ...]:
        bg_tex, grids_camera, grids_frame = ctx.saved_tensors

        bg_tex_arg = bg_tex if ctx.has_bg_tex else None
        grids_camera_arg = grids_camera if ctx.has_grids_camera else None
        grids_frame_arg = grids_frame if ctx.has_grids_frame else None

        # set_materialize_grads(False) means an unused output arrives as None. Replace it
        # with a shape-correct zero placeholder for the native op and disable that output's
        # factor(s) (negative), so the native backward skips it — otherwise a NaN factor on
        # an unused output would contaminate a finite sibling's shared-input gradient.
        (bg_f, drift_c_f, drift_f_f, tv_c_f, tv_f_f) = ctx.factors
        _ref = next(
            t for t in (bg_tex_arg, grids_camera_arg, grids_frame_arg) if t is not None
        )
        _dev, _dt = _ref.device, _ref.dtype
        if v_bg_tex_loss is None:
            v_bg_tex_loss = torch.zeros(ctx.numel_bg, device=_dev, dtype=_dt)
            bg_f = -1.0
        if v_grids_drift_loss is None:
            v_grids_drift_loss = torch.zeros(
                ctx.numel_gc + ctx.numel_gf, device=_dev, dtype=_dt
            )
            # Combined camera+frame drift output -> disable both drift factors.
            drift_c_f = -1.0
            drift_f_f = -1.0
        if v_grid_camera_tv_loss is None:
            v_grid_camera_tv_loss = torch.zeros(ctx.numel_gc, device=_dev, dtype=_dt)
            tv_c_f = -1.0
        if v_grid_frame_tv_loss is None:
            v_grid_frame_tv_loss = torch.zeros(ctx.numel_gf, device=_dev, dtype=_dt)
            tv_f_f = -1.0

        # Gradient buffers — the gather backward writes every entry exactly once
        # (zero when a sub-loss is disabled), so no pre-zeroing is needed.
        v_bg_tex = (
            torch.empty_like(bg_tex)
            if ctx.has_bg_tex
            else torch.empty(0, device=v_bg_tex_loss.device, dtype=v_bg_tex_loss.dtype)
        )
        v_grids_camera = (
            torch.empty_like(grids_camera)
            if ctx.has_grids_camera
            else torch.empty(
                0, device=v_grids_drift_loss.device, dtype=v_grids_drift_loss.dtype
            )
        )
        v_grids_frame = (
            torch.empty_like(grids_frame)
            if ctx.has_grids_frame
            else torch.empty(
                0, device=v_grids_drift_loss.device, dtype=v_grids_drift_loss.dtype
            )
        )

        _make_lazy_cuda_func("bg_grid_losses_bwd")(
            bg_tex_arg,
            ctx.bg_tex_depth,
            grids_camera_arg,
            grids_frame_arg,
            bg_f,
            drift_c_f,
            drift_f_f,
            tv_c_f,
            tv_f_f,
            v_bg_tex_loss.contiguous(),
            v_grids_drift_loss.contiguous(),
            v_grid_camera_tv_loss.contiguous(),
            v_grid_frame_tv_loss.contiguous(),
            v_bg_tex,
            v_grids_camera,
            v_grids_frame,
        )

        return (
            v_bg_tex if ctx.has_bg_tex else None,
            None,  # bg_tex_depth
            v_grids_camera if ctx.has_grids_camera else None,
            v_grids_frame if ctx.has_grids_frame else None,
            None,
            None,
            None,
            None,
            None,  # 5 factors
        )


class _FusedSSIMLosses(torch.autograd.Function):
    """Fused invalid-pixel blend + Gaussian SSIM + masked reduction."""

    @staticmethod
    def forward(
        ctx,
        flags: Tensor,  # [B, H, W] int32
        pred: Tensor,  # [B, H, W, C]
        target: Tensor,  # [B, H, W, C]
        factor: float,
        mask_mode_target: bool,
        constant_mask_value: float,
    ) -> Tensor:
        ctx.set_materialize_grads(False)

        flags = flags.contiguous()
        pred = pred.contiguous()
        target = target.contiguous()
        B, H, W, C = pred.shape

        loss = torch.empty((B, H, W, 1), device=pred.device, dtype=pred.dtype)
        dm_dmu1 = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)
        dm_dsigma1_sq = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)
        dm_dsigma12 = torch.empty((B, C, H, W), device=pred.device, dtype=pred.dtype)

        _make_lazy_cuda_func("ssim_losses_fwd")(
            flags,
            pred,
            target,
            factor,
            mask_mode_target,
            constant_mask_value,
            loss,
            dm_dmu1,
            dm_dsigma1_sq,
            dm_dsigma12,
        )

        ctx.save_for_backward(flags, pred, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.factor = factor
        ctx.mask_mode_target = mask_mode_target
        ctx.constant_mask_value = constant_mask_value
        return loss

    @staticmethod
    def backward(ctx, v_loss: Optional[Tensor]) -> Tuple[Optional[Tensor], ...]:
        flags, pred, target, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        if v_loss is None:
            v_pred = torch.zeros_like(pred)
        else:
            v_pred = torch.empty_like(pred)
            _make_lazy_cuda_func("ssim_losses_bwd")(
                flags,
                pred,
                target,
                ctx.factor,
                ctx.mask_mode_target,
                ctx.constant_mask_value,
                v_loss.contiguous(),
                dm_dmu1,
                dm_dsigma1_sq,
                dm_dsigma12,
                v_pred,
            )

        # Gradients for: flags, pred, target, factor, mask_mode_target,
        # constant_mask_value
        return None, v_pred, None, None, None, None
