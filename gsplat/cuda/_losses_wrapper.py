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
