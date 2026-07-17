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

#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <torch/library.h>

#    include <ATen/Functions.h>

#    include <initializer_list>

#    include "CameraLosses.h"
#    include "Common.h"
#    include "SemanticCELosses.h"

namespace gsplat
{
// Output tensors are allocated by the caller (Python autograd wrapper) and
// passed in as mutable arguments — keeps memory lifetime explicit on the
// Python side so it can be reused by torch's caching allocator across training
// steps.
//
// Fused camera dispatch: the host entries below launch the
// RGB/background kernels and, when the trailing optional semantic-CE
// arguments are present, additionally launch the masked semantic
// cross-entropy kernels via the validation host entries in
// SemanticCELosses.cpp — one host entry, a group of kernels. The semantic row
// count is independent of the camera ray count (the CE port carries its own
// `valid` row mask), but the semantic tensors must live on the camera
// tensors' device so the whole group runs under one device guard/stream.

namespace
{
    // The optional semantic-CE group member is enabled iff semantic_logits is
    // present; the remaining optional semantic arguments must then all be
    // present (and must all be absent otherwise).
    inline bool check_semantic_group_presence(
        const char *entry,
        const at::optional<at::Tensor> &semantic_logits,
        std::initializer_list<const at::optional<at::Tensor> *> companions
    )
    {
        const bool enabled = semantic_logits.has_value();
        for(const auto *companion: companions)
        {
            TORCH_CHECK(
                companion->has_value() == enabled,
                entry,
                ": the optional semantic-CE arguments must be passed all together or not at all "
                "(semantic_logits is ",
                enabled ? "present" : "absent",
                ")"
            );
        }
        return enabled;
    }
} // namespace

void camera_losses_fwd(
    const at::Tensor &flags,
    const at::Tensor &rgb_pred,
    const at::Tensor &rgb_gt,
    const at::Tensor &bg_pred,
    double rgb_factor,
    double bg_factor,
    at::Tensor rgb_loss,
    at::Tensor bg_loss,
    const at::optional<at::Tensor> &semantic_logits,
    const at::optional<at::Tensor> &semantic_targets,
    const at::optional<at::Tensor> &semantic_valid,
    int64_t semantic_ignore_index,
    at::optional<at::Tensor> semantic_loss_sum,
    at::optional<at::Tensor> semantic_valid_count,
    at::optional<at::Tensor> semantic_loss
)
{
    const bool semantic_enabled = check_semantic_group_presence(
        "camera_losses_fwd",
        semantic_logits,
        {&semantic_targets, &semantic_valid, &semantic_loss_sum, &semantic_valid_count, &semantic_loss}
    );
    DEVICE_GUARD(flags);
    CHECK_INPUT(flags);
    CHECK_INPUT(rgb_pred);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(rgb_loss);
    CHECK_INPUT(bg_loss);

    const int64_t N = flags.size(0);
    TORCH_CHECK(flags.dim() == 1 && flags.scalar_type() == at::kInt, "flags must be [N] int32");
    TORCH_CHECK(rgb_pred.dim() == 2 && rgb_pred.size(0) == N && rgb_pred.size(1) == 3, "rgb_pred must be [N, 3]");
    TORCH_CHECK(rgb_gt.dim() == 2 && rgb_gt.size(0) == N && rgb_gt.size(1) == 3, "rgb_gt must be [N, 3]");
    TORCH_CHECK(bg_pred.dim() == 1 && bg_pred.size(0) == N, "bg_pred must be [N]");
    TORCH_CHECK(rgb_loss.dim() == 1 && rgb_loss.size(0) == N, "rgb_loss must be [N], got ", rgb_loss.sizes());
    TORCH_CHECK(
        bg_loss.sizes() == bg_pred.sizes(),
        "bg_loss must have same shape as bg_pred, got ",
        bg_loss.sizes(),
        " vs ",
        bg_pred.sizes()
    );

    launch_camera_losses_fwd_kernel(
        flags,
        rgb_pred,
        rgb_gt,
        bg_pred,
        static_cast<float>(rgb_factor),
        static_cast<float>(bg_factor),
        rgb_loss,
        bg_loss
    );

    if(semantic_enabled)
    {
        // The semantic-CE group member shares the entry's device guard and
        // current stream; semantic_ce_fwd revalidates its own inputs (shape,
        // dtype, cross-tensor device) before launching.
        TORCH_CHECK(
            semantic_logits->device() == flags.device(),
            "semantic_logits must be on the same device as flags (",
            flags.device(),
            "), got ",
            semantic_logits->device()
        );
        semantic_ce_fwd(
            *semantic_logits,
            *semantic_targets,
            *semantic_valid,
            semantic_ignore_index,
            *semantic_loss_sum,
            *semantic_valid_count,
            *semantic_loss
        );
    }
}

void camera_losses_bwd(
    const at::Tensor &flags,
    const at::Tensor &rgb_pred,
    const at::Tensor &rgb_gt,
    const at::Tensor &bg_pred,
    double rgb_factor,
    double bg_factor,
    const at::Tensor &v_rgb_loss,
    const at::Tensor &v_bg_loss,
    at::Tensor v_rgb_pred,
    at::Tensor v_bg_pred,
    const at::optional<at::Tensor> &semantic_logits,
    const at::optional<at::Tensor> &semantic_targets,
    const at::optional<at::Tensor> &semantic_valid,
    int64_t semantic_ignore_index,
    const at::optional<at::Tensor> &semantic_valid_count,
    const at::optional<at::Tensor> &v_semantic_loss,
    at::optional<at::Tensor> v_semantic_logits
)
{
    const bool semantic_enabled = check_semantic_group_presence(
        "camera_losses_bwd",
        semantic_logits,
        {&semantic_targets, &semantic_valid, &semantic_valid_count, &v_semantic_loss, &v_semantic_logits}
    );
    DEVICE_GUARD(flags);
    CHECK_INPUT(flags);
    CHECK_INPUT(rgb_pred);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(v_rgb_loss);
    CHECK_INPUT(v_bg_loss);
    CHECK_INPUT(v_rgb_pred);
    CHECK_INPUT(v_bg_pred);
    TORCH_CHECK(
        v_rgb_loss.dim() == 1 && v_rgb_loss.size(0) == rgb_pred.size(0),
        "v_rgb_loss must be [N], got ",
        v_rgb_loss.sizes()
    );
    TORCH_CHECK(
        v_bg_loss.sizes() == bg_pred.sizes(),
        "v_bg_loss must have same shape as bg_pred, got ",
        v_bg_loss.sizes(),
        " vs ",
        bg_pred.sizes()
    );
    TORCH_CHECK(
        v_rgb_pred.sizes() == rgb_pred.sizes(),
        "v_rgb_pred must have same shape as rgb_pred, got ",
        v_rgb_pred.sizes(),
        " vs ",
        rgb_pred.sizes()
    );
    TORCH_CHECK(
        v_bg_pred.sizes() == bg_pred.sizes(),
        "v_bg_pred must have same shape as bg_pred, got ",
        v_bg_pred.sizes(),
        " vs ",
        bg_pred.sizes()
    );

    launch_camera_losses_bwd_kernel(
        flags,
        rgb_pred,
        rgb_gt,
        bg_pred,
        static_cast<float>(rgb_factor),
        static_cast<float>(bg_factor),
        v_rgb_loss,
        v_bg_loss,
        v_rgb_pred,
        v_bg_pred
    );

    if(semantic_enabled)
    {
        // Mirrors the forward: same device guard/stream, full validation
        // inside semantic_ce_bwd.
        TORCH_CHECK(
            semantic_logits->device() == flags.device(),
            "semantic_logits must be on the same device as flags (",
            flags.device(),
            "), got ",
            semantic_logits->device()
        );
        semantic_ce_bwd(
            *semantic_logits,
            *semantic_targets,
            *semantic_valid,
            semantic_ignore_index,
            *semantic_valid_count,
            *v_semantic_loss,
            *v_semantic_logits
        );
    }
}

void register_camera_losses_cuda_impl(torch::Library &m)
{
    m.impl("camera_losses_fwd", &camera_losses_fwd);
    m.impl("camera_losses_bwd", &camera_losses_bwd);
}
} // namespace gsplat

#endif
