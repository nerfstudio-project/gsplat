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

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include "GaussianLosses.h"
#include "Common.h"
#include "Ops.h"

namespace gsplat {

// Output tensors are allocated by the caller (Python autograd wrapper) and
// passed in as mutable arguments — keeps memory lifetime explicit on the
// Python side so it can be reused by torch's caching allocator across training
// steps.

void gaussian_losses_fwd(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::optional<at::Tensor> &visibility,
    double z_scale_threshold,
    at::Tensor loss_scale,
    at::Tensor loss_density,
    at::Tensor loss_z_scale,
    at::Tensor loss_oob
) {
    DEVICE_GUARD(scales);
    CHECK_INPUT(scales);
    CHECK_INPUT(densities);
    CHECK_INPUT(z_scales);
    CHECK_INPUT(positions);
    CHECK_INPUT(cuboid_dims);
    CHECK_INPUT(loss_scale);
    CHECK_INPUT(loss_density);
    CHECK_INPUT(loss_z_scale);
    CHECK_INPUT(loss_oob);

    const int64_t N = scales.size(0);
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 3,
                "scales must be [N, 3]");
    TORCH_CHECK(densities.dim() == 1 && densities.size(0) == N,
                "densities must be [N]");
    TORCH_CHECK(z_scales.dim() == 1 && z_scales.size(0) == N,
                "z_scales must be [N]");
    TORCH_CHECK(positions.dim() == 2 && positions.size(0) == N && positions.size(1) == 3,
                "positions must be [N, 3]");
    TORCH_CHECK(cuboid_dims.dim() == 2 && cuboid_dims.size(0) == N && cuboid_dims.size(1) == 3,
                "cuboid_dims must be [N, 3]");
    TORCH_CHECK(loss_scale.sizes() == scales.sizes(),
                "loss_scale must have same shape as scales, got ",
                loss_scale.sizes(), " vs ", scales.sizes());
    TORCH_CHECK(loss_density.sizes() == densities.sizes(),
                "loss_density must have same shape as densities, got ",
                loss_density.sizes(), " vs ", densities.sizes());
    TORCH_CHECK(loss_z_scale.sizes() == z_scales.sizes(),
                "loss_z_scale must have same shape as z_scales, got ",
                loss_z_scale.sizes(), " vs ", z_scales.sizes());
    TORCH_CHECK(loss_oob.sizes() == positions.sizes(),
                "loss_oob must have same shape as positions, got ",
                loss_oob.sizes(), " vs ", positions.sizes());

    const at::Tensor *vis_ptr = nullptr;
    if (visibility.has_value()) {
        CHECK_INPUT(visibility.value());
        TORCH_CHECK(visibility.value().dim() == 1 && visibility.value().size(0) == N,
                    "visibility must be [N]");
        vis_ptr = &visibility.value();
    }

    launch_gaussian_losses_fwd_kernel(
        scales, densities, z_scales, positions, cuboid_dims,
        vis_ptr, static_cast<float>(z_scale_threshold),
        loss_scale, loss_density, loss_z_scale, loss_oob
    );
}

void gaussian_losses_bwd(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::optional<at::Tensor> &visibility,
    double z_scale_threshold,
    const at::Tensor &v_loss_scale,
    const at::Tensor &v_loss_density,
    const at::Tensor &v_loss_z_scale,
    const at::Tensor &v_loss_oob,
    at::Tensor v_scales,
    at::Tensor v_densities,
    at::Tensor v_z_scales,
    at::Tensor v_positions
) {
    DEVICE_GUARD(scales);
    CHECK_INPUT(scales);
    CHECK_INPUT(densities);
    CHECK_INPUT(z_scales);
    CHECK_INPUT(positions);
    CHECK_INPUT(cuboid_dims);
    CHECK_INPUT(v_loss_scale);
    CHECK_INPUT(v_loss_density);
    CHECK_INPUT(v_loss_z_scale);
    CHECK_INPUT(v_loss_oob);
    CHECK_INPUT(v_scales);
    CHECK_INPUT(v_densities);
    CHECK_INPUT(v_z_scales);
    CHECK_INPUT(v_positions);
    TORCH_CHECK(v_loss_scale.sizes() == scales.sizes(),
                "v_loss_scale must have same shape as scales, got ",
                v_loss_scale.sizes(), " vs ", scales.sizes());
    TORCH_CHECK(v_loss_density.sizes() == densities.sizes(),
                "v_loss_density must have same shape as densities, got ",
                v_loss_density.sizes(), " vs ", densities.sizes());
    TORCH_CHECK(v_loss_z_scale.sizes() == z_scales.sizes(),
                "v_loss_z_scale must have same shape as z_scales, got ",
                v_loss_z_scale.sizes(), " vs ", z_scales.sizes());
    TORCH_CHECK(v_loss_oob.sizes() == positions.sizes(),
                "v_loss_oob must have same shape as positions, got ",
                v_loss_oob.sizes(), " vs ", positions.sizes());
    TORCH_CHECK(v_scales.sizes() == scales.sizes(),
                "v_scales must have same shape as scales, got ",
                v_scales.sizes(), " vs ", scales.sizes());
    TORCH_CHECK(v_densities.sizes() == densities.sizes(),
                "v_densities must have same shape as densities, got ",
                v_densities.sizes(), " vs ", densities.sizes());
    TORCH_CHECK(v_z_scales.sizes() == z_scales.sizes(),
                "v_z_scales must have same shape as z_scales, got ",
                v_z_scales.sizes(), " vs ", z_scales.sizes());
    TORCH_CHECK(v_positions.sizes() == positions.sizes(),
                "v_positions must have same shape as positions, got ",
                v_positions.sizes(), " vs ", positions.sizes());

    const at::Tensor *vis_ptr = nullptr;
    if (visibility.has_value()) {
        CHECK_INPUT(visibility.value());
        TORCH_CHECK(visibility.value().dim() == 1 &&
                        visibility.value().size(0) == scales.size(0),
                    "visibility must be [N]");
        vis_ptr = &visibility.value();
    }

    launch_gaussian_losses_bwd_kernel(
        scales, densities, z_scales, positions, cuboid_dims,
        vis_ptr, static_cast<float>(z_scale_threshold),
        v_loss_scale, v_loss_density, v_loss_z_scale, v_loss_oob,
        v_scales, v_densities, v_z_scales, v_positions
    );
}

} // namespace gsplat

#endif
