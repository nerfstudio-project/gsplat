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

// Masked semantic cross-entropy: per-element / per-row device math cores.
//
// Two-layer layout (device cores + kernel shells): every arithmetic expression
// of the loss lives here exactly once, and each kernel shell — today the
// per-dispatch kernels in SemanticCELossesCUDA.cu, later the fused all-losses
// kernel — is only responsible for data movement: pointer loads, the
// warp/block reductions that feed these cores, and the atomics that flush
// their results. The cores take plain scalars and return plain scalars; no
// torch types, no pointers, no synchronization.
//
// Row loss (numerically stable logsumexp form):
//     CE_i = log(sum_j exp(x_ij - m_i)) + m_i - x_i[t_i],   m_i = max_j x_ij.
// Gradient: d(loss)/d(x_ij) = (softmax_ij - [j == t_i]) * scale, with
// scale = v_loss / count for valid, non-ignored rows.

namespace gsplat
{
// Row-target classification sentinels; real class indices are >= 0.
// A valid row whose target matches `ignore_index` is excluded from the
// numerator (but still counted in the denominator). A valid row whose
// target is outside [0, n_classes) is a caller bug: the kernel shells fire
// a loud device assert on assert-enabled builds, and on assert-free builds
// the row is deterministically skipped exactly like an ignored row — zero
// numerator contribution, still counted in the denominator, exact-zero
// gradients, and its logits are never read. `ignore_index` is compared
// against the raw (int64) target before the range check, so it may itself
// name a class index, matching torch.nn.functional.cross_entropy's
// precedence.
// Classification is int64 end to end (raw target, class count, returned
// index) so no narrowing happens on the way to the class-loop comparison.
constexpr int64_t kSemanticCeTargetIgnored = -2;
constexpr int64_t kSemanticCeTargetInvalid = -1;

template<typename target_t>
__device__ __forceinline__ int64_t
    semantic_ce_classify_target(const target_t raw_target, const int64_t n_classes, const int64_t ignore_index)
{
    const int64_t raw = static_cast<int64_t>(raw_target);
    if(raw == ignore_index)
    {
        return kSemanticCeTargetIgnored;
    }
    return (raw >= 0 && raw < n_classes) ? raw : kSemanticCeTargetInvalid;
}

// Per-element shifted exponential exp(x_ij - m_i): the summand of the stable
// logsumexp in the forward pass and the softmax numerator in the backward
// pass. `row_max` must be the (already reduced) row maximum.
template<typename scalar_t>
__device__ __forceinline__ scalar_t semantic_ce_exp_shifted(const scalar_t value, const scalar_t row_max)
{
    return exp(value - row_max);
}

// Per-row CE value from fully reduced row statistics. `exp_sum`, `row_max`,
// and `target_logit` must already be warp/block-reduced by the caller.
// Non-contributing rows (ignored targets, and out-of-range targets on
// assert-free builds) contribute an exact zero to the numerator.
template<typename scalar_t>
__device__ __forceinline__ scalar_t semantic_ce_row_loss(
    const bool contributing, const scalar_t exp_sum, const scalar_t row_max, const scalar_t target_logit
)
{
    return contributing ? log(exp_sum) + row_max - target_logit : scalar_t(0);
}

// Masked-mean finalization: `count > 0 ? numerator / count : 0` is the
// clamped-denominator contract — with no valid rows the numerator is also
// zero, so this equals `numerator / max(count, 1)`. Also yields the backward
// per-element scale when called with the upstream gradient as numerator.
template<typename scalar_t>
__device__ __forceinline__ scalar_t semantic_ce_masked_mean(const scalar_t numerator, const int32_t count)
{
    return count > 0 ? numerator / static_cast<scalar_t>(count) : scalar_t(0);
}

// Per-element softmax-minus-onehot gradient:
//     d(loss)/d(x_ij) = (softmax_ij - [j == t_i]) * scale.
// `inverse_sum` is the reciprocal of the reduced exp_sum (computed once per
// row by the shell so each element multiplies instead of dividing), and
// `scale` is semantic_ce_masked_mean(v_loss, count).
template<typename scalar_t>
__device__ __forceinline__ scalar_t semantic_ce_grad_element(
    const scalar_t logit, const scalar_t row_max, const scalar_t inverse_sum, const bool is_target, const scalar_t scale
)
{
    const scalar_t probability = semantic_ce_exp_shifted(logit, row_max) * inverse_sum;
    return (probability - static_cast<scalar_t>(is_target)) * scale;
}
} // namespace gsplat
