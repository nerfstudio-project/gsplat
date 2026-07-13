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

#    include <ATen/Functions.h>
#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <torch/library.h>

#    include "GroundGaussiansLosses.h"
#    include "Common.h"

namespace gsplat
{
// Output tensors (stats, loss, gradients) are allocated by the caller (Python
// autograd wrapper) and passed in as mutable arguments — keeps memory lifetime
// explicit on the Python side so it can be reused by torch's caching allocator
// across training steps.

namespace
{
    // CHECK_INPUT verifies CUDA/privateuseone + contiguity but not cross-tensor
    // device index or dtype. The kernels dispatch on positions.scalar_type() and
    // read every pointer as scalar_t on positions' device, so every other tensor
    // argument must share positions' device and dtype.
    inline void check_matches_positions(const at::Tensor &positions, const at::Tensor &x, const char *name)
    {
        TORCH_CHECK(
            x.device() == positions.device(),
            name,
            " must be on the same device as positions (",
            positions.device(),
            "), got ",
            x.device()
        );
        TORCH_CHECK(
            x.scalar_type() == positions.scalar_type(),
            name,
            " must have the same dtype as positions (",
            positions.scalar_type(),
            "), got ",
            x.scalar_type()
        );
    }
} // namespace

void ground_gaussians_fwd(
    const at::Tensor &positions,
    const at::Tensor &rotations,
    const at::Tensor &cam_tquat,
    const at::Tensor &random_values,
    double min_bias,
    double range_bias,
    double grid_len,
    double rotation_lambda,
    at::Tensor stats,
    at::Tensor loss
)
{
    DEVICE_GUARD(positions);
    CHECK_INPUT(positions);
    CHECK_INPUT(rotations);
    CHECK_INPUT(cam_tquat);
    CHECK_INPUT(random_values);
    CHECK_INPUT(stats);
    CHECK_INPUT(loss);

    const int64_t N = positions.size(0);
    const int64_t B = random_values.size(0);
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N, 3]");
    TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4, "rotations must be [N, 4]");
    TORCH_CHECK(cam_tquat.numel() == 7, "cam_tquat must have 7 elements, got ", cam_tquat.numel());
    TORCH_CHECK(random_values.dim() == 1, "random_values must be [B]");
    TORCH_CHECK(
        stats.dim() == 2 && stats.size(0) == B && stats.size(1) == 7, "stats must be [B, 7], got ", stats.sizes()
    );
    TORCH_CHECK(loss.numel() == 1, "loss must be a scalar tensor, got ", loss.sizes());
    check_matches_positions(positions, rotations, "rotations");
    check_matches_positions(positions, cam_tquat, "cam_tquat");
    check_matches_positions(positions, random_values, "random_values");
    check_matches_positions(positions, stats, "stats");
    check_matches_positions(positions, loss, "loss");

    launch_ground_gaussians_fwd_kernel(
        positions,
        rotations,
        cam_tquat,
        random_values,
        static_cast<float>(min_bias),
        static_cast<float>(range_bias),
        static_cast<float>(grid_len),
        static_cast<float>(rotation_lambda),
        stats,
        loss
    );
}

void ground_gaussians_bwd(
    const at::Tensor &positions,
    const at::Tensor &rotations,
    const at::Tensor &cam_tquat,
    const at::Tensor &random_values,
    const at::Tensor &stats,
    const at::Tensor &v_loss,
    double min_bias,
    double range_bias,
    double grid_len,
    double rotation_lambda,
    at::Tensor v_positions,
    at::Tensor v_rotations
)
{
    DEVICE_GUARD(positions);
    CHECK_INPUT(positions);
    CHECK_INPUT(rotations);
    CHECK_INPUT(cam_tquat);
    CHECK_INPUT(random_values);
    CHECK_INPUT(stats);
    CHECK_INPUT(v_loss);
    CHECK_INPUT(v_positions);
    CHECK_INPUT(v_rotations);

    // Mirror the forward shape checks: ground_gaussians_losses_bwd is a public op,
    // so a malformed direct call must not read stats + b*7 or write idx*3 / idx*4
    // past the provided buffers.
    const int64_t N = positions.size(0);
    const int64_t B = random_values.size(0);
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must be [N, 3]");
    TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == N && rotations.size(1) == 4, "rotations must be [N, 4]");
    TORCH_CHECK(cam_tquat.numel() == 7, "cam_tquat must have 7 elements, got ", cam_tquat.numel());
    TORCH_CHECK(random_values.dim() == 1, "random_values must be [B]");
    TORCH_CHECK(
        stats.dim() == 2 && stats.size(0) == B && stats.size(1) == 7, "stats must be [B, 7], got ", stats.sizes()
    );
    check_matches_positions(positions, rotations, "rotations");
    check_matches_positions(positions, cam_tquat, "cam_tquat");
    check_matches_positions(positions, random_values, "random_values");
    check_matches_positions(positions, stats, "stats");
    check_matches_positions(positions, v_loss, "v_loss");
    check_matches_positions(positions, v_positions, "v_positions");
    check_matches_positions(positions, v_rotations, "v_rotations");

    TORCH_CHECK(v_loss.numel() == 1, "v_loss must be a scalar tensor, got ", v_loss.sizes());
    TORCH_CHECK(
        v_positions.sizes() == positions.sizes(),
        "v_positions must have same shape as positions, got ",
        v_positions.sizes(),
        " vs ",
        positions.sizes()
    );
    TORCH_CHECK(
        v_rotations.sizes() == rotations.sizes(),
        "v_rotations must have same shape as rotations, got ",
        v_rotations.sizes(),
        " vs ",
        rotations.sizes()
    );

    launch_ground_gaussians_bwd_kernel(
        positions,
        rotations,
        cam_tquat,
        random_values,
        stats,
        v_loss,
        static_cast<float>(min_bias),
        static_cast<float>(range_bias),
        static_cast<float>(grid_len),
        static_cast<float>(rotation_lambda),
        v_positions,
        v_rotations
    );
}

void register_ground_gaussians_losses_cuda_impl(torch::Library &m)
{
    m.impl("ground_gaussians_losses_fwd", &ground_gaussians_fwd);
    m.impl("ground_gaussians_losses_bwd", &ground_gaussians_bwd);
}
} // namespace gsplat

#endif
