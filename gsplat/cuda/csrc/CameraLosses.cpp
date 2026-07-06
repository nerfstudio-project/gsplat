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

#include "Config.h"

#if GSPLAT_BUILD_LOSSES

#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <torch/library.h>

#    include <ATen/Functions.h>

#    include "CameraLosses.h"
#    include "Common.h"

namespace gsplat
{
// Output tensors are allocated by the caller (Python autograd wrapper) and
// passed in as mutable arguments — keeps memory lifetime explicit on the
// Python side so it can be reused by torch's caching allocator across training
// steps.

void camera_losses_fwd(
    const at::Tensor &flags,
    const at::Tensor &rgb_pred,
    const at::Tensor &rgb_gt,
    const at::Tensor &bg_pred,
    double rgb_factor,
    double bg_factor,
    at::Tensor rgb_loss,
    at::Tensor bg_loss
)
{
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
    at::Tensor v_bg_pred
)
{
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
}

void register_camera_losses_cuda_impl(torch::Library &m)
{
    m.impl("camera_losses_fwd", &camera_losses_fwd);
    m.impl("camera_losses_bwd", &camera_losses_bwd);
}
} // namespace gsplat

#endif
