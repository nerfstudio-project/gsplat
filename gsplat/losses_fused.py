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

"""Fused CUDA gaussian regularization losses with Tier 1 Python fallback."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._wrapper import has_losses
from gsplat.losses import (
    gaussian_density_reg,
    gaussian_scale_reg,
    gaussian_z_scale_reg,
    out_of_bound_loss,
)


class FusedGaussianLosses(torch.nn.Module):
    """Fused CUDA implementation of four gaussian regularization losses.

    Computes :func:`~gsplat.losses.gaussian_scale_reg`,
    :func:`~gsplat.losses.gaussian_density_reg`,
    :func:`~gsplat.losses.gaussian_z_scale_reg`, and
    :func:`~gsplat.losses.out_of_bound_loss` in a single fused CUDA kernel.

    Falls back to the Tier 1 pure-PyTorch implementations when CUDA is
    unavailable or the inputs are on CPU.

    All outputs are **unreduced** (per-element), matching the Tier 1 contract.

    .. note::
        The CUDA kernel dispatches on ``fp32`` and ``fp64`` only. AMP training
        with ``fp16``/``bf16`` inputs will raise at kernel dispatch; cast to
        ``float`` before calling, or disable AMP for this module. Extending
        to half-precision is tracked as a follow-up phase in the gsplat
        losses-migration plan.

    Args:
        z_scale_threshold: Threshold above which z-scale penalty is applied.
    """

    def __init__(self, z_scale_threshold: float = 0.0):
        super().__init__()
        self.z_scale_threshold = z_scale_threshold
        self._cuda_available = has_losses()

    def forward(
        self,
        scales: Tensor,  # [N, 3]
        densities: Tensor,  # [N]
        z_scales: Tensor,  # [N]
        positions: Tensor,  # [N, 3]
        cuboid_dims: Tensor,  # [N, 3]
        visibility: Optional[Tensor] = None,  # [N] float
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute all four gaussian regularization losses.

        Returns:
            Tuple of ``(loss_scale, loss_density, loss_z_scale, loss_oob)``,
            each unreduced per-element.
        """
        # Dispatch to CUDA only when every input is on the same GPU — otherwise
        # fall through to Tier 1 instead of tripping CHECK_INPUT deep in C++.
        all_cuda = scales.is_cuda and all(
            t.is_cuda for t in (densities, z_scales, positions, cuboid_dims)
        )
        if visibility is not None:
            all_cuda = all_cuda and visibility.is_cuda
        if self._cuda_available and all_cuda:
            from gsplat.cuda._losses_wrapper import _FusedGaussianLosses

            return _FusedGaussianLosses.apply(
                scales,
                densities,
                z_scales,
                positions,
                cuboid_dims,
                self.z_scale_threshold,
                visibility,
            )

        # Tier 1 fallback: pure-PyTorch
        return (
            gaussian_scale_reg(scales, visibility=visibility),
            gaussian_density_reg(densities, visibility=visibility),
            gaussian_z_scale_reg(z_scales, self.z_scale_threshold),
            out_of_bound_loss(positions, cuboid_dims),
        )
