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

#include "GSplatBuildConfig.h"

#if GSPLAT_BUILD_LOSSES

#    include <ATen/Functions.h>
#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>

#    include <limits>

#    include "Common.h"
#    include "SemanticCELosses.h"

namespace gsplat
{
// Output tensors (workspace accumulators, loss, gradients) are allocated by
// the caller (Python autograd wrapper) and passed in as mutable arguments —
// keeps memory lifetime explicit on the Python side so it can be reused by
// torch's caching allocator across training steps.

namespace
{
    // CHECK_INPUT verifies CUDA/privateuseone + contiguity but not cross-tensor
    // device index or dtype. The kernels dispatch on logits.scalar_type() and
    // read every float pointer as scalar_t on logits' device, so every other
    // tensor argument must share logits' device (and, for float tensors, its
    // dtype).
    inline void check_semantic_ce_device(const at::Tensor &logits, const at::Tensor &x, const char *name)
    {
        TORCH_CHECK(
            x.device() == logits.device(),
            name,
            " must be on the same device as logits (",
            logits.device(),
            "), got ",
            x.device()
        );
    }

    inline void check_semantic_ce_float(const at::Tensor &logits, const at::Tensor &x, const char *name)
    {
        check_semantic_ce_device(logits, x, name);
        TORCH_CHECK(
            x.scalar_type() == logits.scalar_type(),
            name,
            " must have the same dtype as logits (",
            logits.scalar_type(),
            "), got ",
            x.scalar_type()
        );
    }

    // Checks shared by the semantic CE forward and backward entry points so the
    // invariants stay in sync. Both are reached with caller-provided tensors
    // (through the public fused camera dispatch), so a malformed call must not
    // read or write past the provided buffers.
    inline void check_semantic_ce_common(const at::Tensor &logits, const at::Tensor &targets, const at::Tensor &valid)
    {
        const int64_t N = logits.size(0);
        TORCH_CHECK(logits.dim() == 2 && logits.size(1) > 0, "logits must be [N, C] with C > 0, got ", logits.sizes());
        TORCH_CHECK(
            N <= std::numeric_limits<int>::max() && logits.size(1) <= std::numeric_limits<int>::max(),
            "logits rows/classes must fit an int, got ",
            logits.sizes()
        );
        TORCH_CHECK(
            targets.dim() == 1 && targets.size(0) == N, "targets must be [N], got ", targets.sizes(), " for N=", N
        );
        TORCH_CHECK(
            targets.scalar_type() == at::kByte || targets.scalar_type() == at::kLong,
            "targets must be uint8 or int64, got ",
            targets.scalar_type()
        );
        TORCH_CHECK(
            valid.dim() == 1 && valid.size(0) == N && valid.scalar_type() == at::kBool,
            "valid must be [N] bool, got ",
            valid.sizes(),
            " ",
            valid.scalar_type()
        );
        check_semantic_ce_device(logits, targets, "targets");
        check_semantic_ce_device(logits, valid, "valid");
    }
} // namespace

void semantic_ce_fwd(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    at::Tensor loss_sum,
    at::Tensor valid_count,
    at::Tensor loss
)
{
    DEVICE_GUARD(logits);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(valid);
    CHECK_INPUT(loss_sum);
    CHECK_INPUT(valid_count);
    CHECK_INPUT(loss);

    check_semantic_ce_common(logits, targets, valid);
    TORCH_CHECK(loss_sum.numel() == 1, "loss_sum must be a scalar tensor, got ", loss_sum.sizes());
    TORCH_CHECK(
        valid_count.numel() == 1 && valid_count.scalar_type() == at::kInt,
        "valid_count must be an int32 scalar tensor, got ",
        valid_count.sizes(),
        " ",
        valid_count.scalar_type()
    );
    TORCH_CHECK(loss.numel() == 1, "loss must be a scalar tensor, got ", loss.sizes());
    check_semantic_ce_float(logits, loss_sum, "loss_sum");
    check_semantic_ce_device(logits, valid_count, "valid_count");
    check_semantic_ce_float(logits, loss, "loss");

    launch_semantic_ce_fwd_kernel(logits, targets, valid, ignore_index, loss_sum, valid_count, loss);
}

void semantic_ce_bwd(
    const at::Tensor &logits,
    const at::Tensor &targets,
    const at::Tensor &valid,
    int64_t ignore_index,
    const at::Tensor &valid_count,
    const at::Tensor &v_loss,
    at::Tensor v_logits
)
{
    DEVICE_GUARD(logits);
    CHECK_INPUT(logits);
    CHECK_INPUT(targets);
    CHECK_INPUT(valid);
    CHECK_INPUT(valid_count);
    CHECK_INPUT(v_loss);
    CHECK_INPUT(v_logits);

    check_semantic_ce_common(logits, targets, valid);
    TORCH_CHECK(
        valid_count.numel() == 1 && valid_count.scalar_type() == at::kInt,
        "valid_count must be an int32 scalar tensor, got ",
        valid_count.sizes(),
        " ",
        valid_count.scalar_type()
    );
    TORCH_CHECK(v_loss.numel() == 1, "v_loss must be a scalar tensor, got ", v_loss.sizes());
    TORCH_CHECK(
        v_logits.sizes() == logits.sizes(),
        "v_logits must have same shape as logits, got ",
        v_logits.sizes(),
        " vs ",
        logits.sizes()
    );
    check_semantic_ce_device(logits, valid_count, "valid_count");
    check_semantic_ce_float(logits, v_loss, "v_loss");
    check_semantic_ce_float(logits, v_logits, "v_logits");

    launch_semantic_ce_bwd_kernel(logits, targets, valid, ignore_index, valid_count, v_loss, v_logits);
}
} // namespace gsplat

#endif
