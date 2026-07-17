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
// Masked semantic cross-entropy (row gating via a caller-provided boolean
// row mask instead of a hardcoded flag predicate).
//
// Forward: a warp-per-row kernel computes a numerically stable
// logsumexp-based cross-entropy for every valid, non-ignored row and
// accumulates the row losses and the valid-row count into a two-scalar
// workspace via atomics; a finalize kernel then writes the masked mean
// `loss = valid_count > 0 ? loss_sum / valid_count : 0`.
//
// Contract: rows with `valid[i] == false` contribute nothing; valid rows whose
// target equals `ignore_index` contribute zero to the numerator but still
// count in the denominator; a valid row with a target outside
// [0, n_classes) asserts (debug) and poisons the loss with NaN.
void launch_semantic_ce_fwd_kernel(
    const at::Tensor &logits,  // [N, C] fp32/fp64
    const at::Tensor &targets, // [N] uint8 or int64
    const at::Tensor &valid,   // [N] bool
    int64_t ignore_index,
    at::Tensor &loss_sum,    // [] accumulator, logits dtype (zero-initialized)
    at::Tensor &valid_count, // [] int32 accumulator (zero-initialized)
    at::Tensor &loss         // [] scalar output, logits dtype
);

// Backward: re-derives the per-row softmax and scatters
// `(softmax - onehot) * v_loss / valid_count` into v_logits for contributing
// rows; invalid and ignored rows receive exact zeros.
void launch_semantic_ce_bwd_kernel(
    const at::Tensor &logits,  // [N, C] fp32/fp64
    const at::Tensor &targets, // [N] uint8 or int64
    const at::Tensor &valid,   // [N] bool
    int64_t ignore_index,
    const at::Tensor &valid_count, // [] int32 (from forward)
    const at::Tensor &v_loss,      // [] scalar upstream gradient
    at::Tensor &v_logits           // [N, C]
);

// Host entries (implemented in SemanticCELosses.cpp): full input validation +
// device guard + kernel launch. Declared here so the fused camera dispatch
// (CameraLosses.cpp) can reuse them verbatim when its optional semantic-CE
// group member is enabled.
// Not registered as standalone per-loss ops — the grouped camera dispatch is
// the only entry point.
void semantic_ce_fwd(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    at::Tensor loss_sum,
    at::Tensor valid_count,
    at::Tensor loss
);

void semantic_ce_bwd(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    const at::Tensor &valid_count,
    const at::Tensor &v_loss,
    at::Tensor v_logits
);
} // namespace gsplat
