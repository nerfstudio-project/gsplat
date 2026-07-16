/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace at
{
class Tensor;
}

namespace gsplat
{
// Deform-smoothness loss. Masked mean of |deformation|:
//
//   loss = sum(|deformation| * mask~) / max(sum(mask), 1)
//
// where mask~ is the mask broadcast to deformation's [N, D] shape and the
// denominator reduces the ORIGINAL (un-broadcast) mask exactly once. Without a
// mask, mask~ reads as 1 and the denominator is the element count, i.e. the
// plain mean of |deformation|.
//
// Masked-out elements (mask~ == 0) are fully inert: their deformation is
// never read by the forward or by the mask-gradient numerator, so non-finite
// residuals left behind in masked-out rows cannot poison the loss or the
// mask gradient (they also deliberately stop steering the mask).
//
// The mask is never materialized at [N, D]: the kernels map each deformation
// element to its mask source element through two virtual strides
// (mask_stride_row, mask_stride_col), 0 on broadcast dimensions.
//
// Forward: an accumulation kernel fills the [2] running-sums buffer
// (sum(|d| * mask~), sum(mask)), then a single-thread finalize kernel turns
// those into the scalar loss.
void launch_deform_losses_fwd_kernel(
    const at::Tensor &deformation, // [N, D]
    const at::Tensor *mask,        // broadcastable to [N, D], or nullptr
    int64_t mask_stride_row,       // mask element stride along N (0 = broadcast)
    int64_t mask_stride_col,       // mask element stride along D (0 = broadcast)
    at::Tensor &sums,              // [2] {loss_sum, mask_sum} (zero-initialized)
    at::Tensor &loss               // [] scalar (zero-initialized)
);

// Backward: re-reads the running sums (passed back from forward) to scatter
// gradients to the deformation and, when present, the mask. The mask gradient
// combines the numerator term (up * |d| / denom, broadcast-reduced onto the
// mask source element; exactly zero for masked-out elements per the inert
// contract above) and the denominator term (-up * loss_sum / denom^2,
// suppressed while the clamp max(mask_sum, 1) is active) into one buffer.
void launch_deform_losses_bwd_kernel(
    const at::Tensor &deformation, // [N, D]
    const at::Tensor *mask,        // broadcastable to [N, D], or nullptr
    int64_t mask_stride_row,
    int64_t mask_stride_col,
    const at::Tensor &sums,    // [2] {loss_sum, mask_sum} from forward
    const at::Tensor &v_loss,  // [] scalar upstream gradient
    at::Tensor &v_deformation, // [N, D]
    at::Tensor *v_mask         // same shape as mask (zero-initialized), or nullptr
);

// Host-side validation helpers for the deform loss family, used by the
// grouped gaussian dispatch (the host entry must apply these checks before
// reaching the launchers above; a later standalone deform op would reuse
// them unchanged). Implemented in DeformLosses.cpp.
//
// TORCH_CHECKs that `x` shares deformation's device and dtype — the kernels
// dispatch on deformation.scalar_type() and read every pointer as scalar_t on
// deformation's device.
void deform_losses_check_matches_deformation(const at::Tensor &deformation, const at::Tensor &x, const char *name);

// Right-aligns the mask shape against deformation's [N, D] and derives the
// virtual element strides (0 on broadcast dimensions) the kernels use to map
// a deformation element to its mask source element. One-way broadcast only:
// TORCH_CHECKs that the mask never forces deformation to expand (that would
// change the numerator element count).
void deform_losses_resolve_mask_broadcast(
    const at::Tensor &deformation, const at::Tensor &mask, int64_t &stride_row, int64_t &stride_col
);
} // namespace gsplat
