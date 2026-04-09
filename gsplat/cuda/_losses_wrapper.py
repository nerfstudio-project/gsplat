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
autograd bridges (Tier 2 entry points) for all Phase 9 fused-loss CUDA
kernels, mirroring the ``losses_*`` / ``cuda/_*_wrapper`` split used
elsewhere in the package.
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
        # here is safer than silently returning a zero gradient that Tier 1 would
        # have propagated.
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
            # Tier 1 accepts visibility as [N] or [N, 1]; the CUDA kernel
            # expects [N]. Reshape here so the public API matches Tier 1.
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
