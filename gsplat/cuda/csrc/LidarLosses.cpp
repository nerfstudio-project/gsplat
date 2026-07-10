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

#    include "LidarLosses.h"
#    include "Common.h"

namespace gsplat
{
// Output tensors are allocated by the caller (Python autograd wrapper) and
// passed in as mutable arguments — keeps memory lifetime explicit on the
// Python side so it can be reused by torch's caching allocator across training
// steps.

void lidar_losses_fwd(
    const at::Tensor &flags,
    const at::Tensor &distance_pred,
    const at::Tensor &distance_gt,
    const at::Tensor &intensity_pred,
    const at::Tensor &intensity_gt,
    const at::Tensor &raydrop_pred,
    const at::Tensor &raydrop_gt,
    const at::Tensor &bg_pred,
    double distance_factor,
    double intensity_factor,
    double raydrop_factor,
    double bg_factor,
    at::Tensor distance_loss,
    at::Tensor intensity_loss,
    at::Tensor raydrop_loss,
    at::Tensor bg_loss
)
{
    DEVICE_GUARD(flags);
    CHECK_INPUT(flags);
    CHECK_INPUT(distance_pred);
    CHECK_INPUT(distance_gt);
    CHECK_INPUT(intensity_pred);
    CHECK_INPUT(intensity_gt);
    CHECK_INPUT(raydrop_pred);
    CHECK_INPUT(raydrop_gt);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(distance_loss);
    CHECK_INPUT(intensity_loss);
    CHECK_INPUT(raydrop_loss);
    CHECK_INPUT(bg_loss);

    const int64_t N = flags.size(0);
    TORCH_CHECK(flags.dim() == 1 && flags.scalar_type() == at::kInt, "flags must be [N] int32");
    TORCH_CHECK(distance_pred.dim() == 1 && distance_pred.size(0) == N, "distance_pred must be [N]");
    TORCH_CHECK(distance_gt.dim() == 1 && distance_gt.size(0) == N, "distance_gt must be [N]");
    TORCH_CHECK(intensity_pred.dim() == 1 && intensity_pred.size(0) == N, "intensity_pred must be [N]");
    TORCH_CHECK(intensity_gt.dim() == 1 && intensity_gt.size(0) == N, "intensity_gt must be [N]");
    TORCH_CHECK(raydrop_pred.dim() == 1 && raydrop_pred.size(0) == N, "raydrop_pred must be [N]");
    TORCH_CHECK(raydrop_gt.dim() == 1 && raydrop_gt.size(0) == N, "raydrop_gt must be [N]");
    TORCH_CHECK(bg_pred.dim() == 1 && bg_pred.size(0) == N, "bg_pred must be [N]");
    TORCH_CHECK(
        distance_loss.sizes() == distance_pred.sizes(),
        "distance_loss must have same shape as distance_pred, got ",
        distance_loss.sizes(),
        " vs ",
        distance_pred.sizes()
    );
    TORCH_CHECK(
        intensity_loss.sizes() == intensity_pred.sizes(),
        "intensity_loss must have same shape as intensity_pred, got ",
        intensity_loss.sizes(),
        " vs ",
        intensity_pred.sizes()
    );
    TORCH_CHECK(
        raydrop_loss.sizes() == raydrop_pred.sizes(),
        "raydrop_loss must have same shape as raydrop_pred, got ",
        raydrop_loss.sizes(),
        " vs ",
        raydrop_pred.sizes()
    );
    TORCH_CHECK(
        bg_loss.sizes() == bg_pred.sizes(),
        "bg_loss must have same shape as bg_pred, got ",
        bg_loss.sizes(),
        " vs ",
        bg_pred.sizes()
    );

    launch_lidar_losses_fwd_kernel(
        flags,
        distance_pred,
        distance_gt,
        intensity_pred,
        intensity_gt,
        raydrop_pred,
        raydrop_gt,
        bg_pred,
        static_cast<float>(distance_factor),
        static_cast<float>(intensity_factor),
        static_cast<float>(raydrop_factor),
        static_cast<float>(bg_factor),
        distance_loss,
        intensity_loss,
        raydrop_loss,
        bg_loss
    );
}

void lidar_losses_bwd(
    const at::Tensor &flags,
    const at::Tensor &distance_pred,
    const at::Tensor &distance_gt,
    const at::Tensor &intensity_pred,
    const at::Tensor &intensity_gt,
    const at::Tensor &raydrop_pred,
    const at::Tensor &raydrop_gt,
    const at::Tensor &bg_pred,
    double distance_factor,
    double intensity_factor,
    double raydrop_factor,
    double bg_factor,
    const at::Tensor &v_distance_loss,
    const at::Tensor &v_intensity_loss,
    const at::Tensor &v_raydrop_loss,
    const at::Tensor &v_bg_loss,
    at::Tensor v_distance_pred,
    at::Tensor v_intensity_pred,
    at::Tensor v_raydrop_pred,
    at::Tensor v_bg_pred
)
{
    DEVICE_GUARD(flags);
    CHECK_INPUT(flags);
    CHECK_INPUT(distance_pred);
    CHECK_INPUT(distance_gt);
    CHECK_INPUT(intensity_pred);
    CHECK_INPUT(intensity_gt);
    CHECK_INPUT(raydrop_pred);
    CHECK_INPUT(raydrop_gt);
    CHECK_INPUT(bg_pred);
    CHECK_INPUT(v_distance_loss);
    CHECK_INPUT(v_intensity_loss);
    CHECK_INPUT(v_raydrop_loss);
    CHECK_INPUT(v_bg_loss);
    CHECK_INPUT(v_distance_pred);
    CHECK_INPUT(v_intensity_pred);
    CHECK_INPUT(v_raydrop_pred);
    CHECK_INPUT(v_bg_pred);
    TORCH_CHECK(
        v_distance_loss.sizes() == distance_pred.sizes(), "v_distance_loss must have same shape as distance_pred"
    );
    TORCH_CHECK(
        v_intensity_loss.sizes() == intensity_pred.sizes(), "v_intensity_loss must have same shape as intensity_pred"
    );
    TORCH_CHECK(v_raydrop_loss.sizes() == raydrop_pred.sizes(), "v_raydrop_loss must have same shape as raydrop_pred");
    TORCH_CHECK(v_bg_loss.sizes() == bg_pred.sizes(), "v_bg_loss must have same shape as bg_pred");
    TORCH_CHECK(
        v_distance_pred.sizes() == distance_pred.sizes(),
        "v_distance_pred must have same shape as distance_pred, got ",
        v_distance_pred.sizes(),
        " vs ",
        distance_pred.sizes()
    );
    TORCH_CHECK(
        v_intensity_pred.sizes() == intensity_pred.sizes(),
        "v_intensity_pred must have same shape as intensity_pred, got ",
        v_intensity_pred.sizes(),
        " vs ",
        intensity_pred.sizes()
    );
    TORCH_CHECK(
        v_raydrop_pred.sizes() == raydrop_pred.sizes(),
        "v_raydrop_pred must have same shape as raydrop_pred, got ",
        v_raydrop_pred.sizes(),
        " vs ",
        raydrop_pred.sizes()
    );
    TORCH_CHECK(
        v_bg_pred.sizes() == bg_pred.sizes(),
        "v_bg_pred must have same shape as bg_pred, got ",
        v_bg_pred.sizes(),
        " vs ",
        bg_pred.sizes()
    );

    launch_lidar_losses_bwd_kernel(
        flags,
        distance_pred,
        distance_gt,
        intensity_pred,
        intensity_gt,
        raydrop_pred,
        raydrop_gt,
        bg_pred,
        static_cast<float>(distance_factor),
        static_cast<float>(intensity_factor),
        static_cast<float>(raydrop_factor),
        static_cast<float>(bg_factor),
        v_distance_loss,
        v_intensity_loss,
        v_raydrop_loss,
        v_bg_loss,
        v_distance_pred,
        v_intensity_pred,
        v_raydrop_pred,
        v_bg_pred
    );
}

void register_lidar_losses_cuda_impl(torch::Library &m)
{
    m.impl("lidar_losses_fwd", &lidar_losses_fwd);
    m.impl("lidar_losses_bwd", &lidar_losses_bwd);
}
} // namespace gsplat

#endif
