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

#include <cmath>
#include <cstdint>

// Deform-smoothness loss: per-element device math cores.
//
// Two-layer layout (device cores + kernel shells): every arithmetic expression
// of the loss lives here exactly once, and each kernel shell — today the
// per-dispatch kernels in DeformLossesCUDA.cu, later the fused all-losses
// kernel — is only responsible for data movement: pointer loads, the shared-
// memory block reductions that sum these per-element values, and the atomics
// that flush them. The cores take plain scalars and return plain scalars; no
// torch types, no pointers, no synchronization.
//
// Loss: sum_i |d_i| * mask~_i / max(sum_j m_j, 1), where mask~ is the mask
// broadcast to the deformation shape and m is the ORIGINAL (source) mask.
// Masked-out elements (mask~_i == 0) are fully inert: neither the loss sum
// nor the mask-gradient numerator reads their |d_i|, so non-finite residuals
// in masked-out rows cannot poison either.
// Backward:
//     d(loss)/d(d_i) = mask~_i * sign(d_i) / denom
//     d(loss)/d(m_j) = sum_{i in replicas(j), mask~_i != 0} |d_i| / denom (numerator)
//                    - [mask_sum >= 1] * loss_sum / denom^2               (denominator)

namespace gsplat
{
// sign with sign(0) = 0 — matches torch.abs' subgradient at zero, so the
// fused backward agrees with the pure-PyTorch fallback exactly.
template<typename scalar_t>
__device__ __forceinline__ scalar_t deform_sign(scalar_t x)
{
    return x > scalar_t(0) ? scalar_t(1) : (x < scalar_t(0) ? scalar_t(-1) : scalar_t(0));
}

// Flat deformation index -> flat mask source index through the virtual
// broadcast strides (0 on broadcast dimensions). With a same-shape mask
// this is the identity map.
__device__ __forceinline__ int64_t deform_mask_index(int64_t i, int64_t D, int64_t stride_row, int64_t stride_col)
{
    return (i / D) * stride_row + (i % D) * stride_col;
}

// Clamped mean denominator: max(mask_sum, 1).
template<typename scalar_t>
__device__ __forceinline__ scalar_t deform_clamped_denominator(scalar_t mask_sum)
{
    return mask_sum < scalar_t(1) ? scalar_t(1) : mask_sum;
}

// Per-element numerator contribution |d| * mask~. `weight` is the broadcast
// mask value for this element (1 when there is no mask). Masked-out elements
// (weight == 0) are fully inert: the value is never read into the loss, so a
// non-finite deformation left behind in a masked-out row cannot poison the
// scalar (masking a region is often exactly why its residuals are garbage).
template<typename scalar_t>
__device__ __forceinline__ scalar_t deform_element_loss(const scalar_t value, const scalar_t weight)
{
    return weight == scalar_t(0) ? scalar_t(0) : abs(value) * weight;
}

// Scalar finalization: loss = loss_sum / max(mask_sum, 1).
template<typename scalar_t>
__device__ __forceinline__ scalar_t deform_finalize_loss(const scalar_t loss_sum, const scalar_t mask_sum)
{
    return loss_sum / deform_clamped_denominator(mask_sum);
}

// d(loss)/d(d_i) = up * mask~_i * sign(d_i) / denom. `denom` must be the
// already clamped denominator (deform_clamped_denominator).
template<typename scalar_t>
__device__ __forceinline__ scalar_t
    deform_grad_deformation(const scalar_t up, const scalar_t value, const scalar_t weight, const scalar_t denom)
{
    return up * weight * deform_sign(value) / denom;
}

// Numerator term of d(loss)/d(m_j): up * |d_i| / denom, accumulated by the
// shell over the replicas of mask element j (atomics stay in the shell).
// Masked-out elements (weight == 0) contribute nothing: with the inert-lane
// contract their |d_i| must not reach the mask gradient either — both so a
// non-finite masked residual cannot poison it, and so a masked-out lane stops
// steering the mask (the "this residual is large" reopening signal is
// deliberately dropped; that is the contract change, not an accident).
template<typename scalar_t>
__device__ __forceinline__ scalar_t
    deform_grad_mask_numerator(const scalar_t up, const scalar_t value, const scalar_t weight, const scalar_t denom)
{
    return weight == scalar_t(0) ? scalar_t(0) : up * abs(value) / denom;
}

// Denominator term of d(loss)/d(m_j): -up * loss_sum / denom^2. The shell
// applies this only when mask_sum >= 1 — the clamp max(mask_sum, 1) has zero
// slope while active (matching torch's clamp(min=1) backward, which passes
// gradient at mask_sum == 1 exactly).
template<typename scalar_t>
__device__ __forceinline__ scalar_t
    deform_grad_mask_denominator(const scalar_t up, const scalar_t loss_sum, const scalar_t denom)
{
    return -up * loss_sum / (denom * denom);
}
} // namespace gsplat
