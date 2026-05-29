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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace gsplat {

// ---------------------------------------------------------------------------
// Forward kernel: fuses scale_reg, density_reg, z_scale_reg, out_of_bound
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void gaussian_losses_fwd_kernel(
    const int64_t N,
    const scalar_t *__restrict__ scales,        // [N, 3]
    const scalar_t *__restrict__ densities,     // [N]
    const scalar_t *__restrict__ z_scales,      // [N]
    const scalar_t *__restrict__ positions,     // [N, 3]
    const scalar_t *__restrict__ cuboid_dims,   // [N, 3]
    const scalar_t *__restrict__ visibility,    // [N] float or nullptr
    const scalar_t z_scale_threshold,
    scalar_t *__restrict__ loss_scale,          // [N, 3]
    scalar_t *__restrict__ loss_density,        // [N]
    scalar_t *__restrict__ loss_z_scale,        // [N]
    scalar_t *__restrict__ loss_oob             // [N, 3]
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const scalar_t vis = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);

    // gaussian_scale_reg: abs(scales) * visibility
    // Since scales are post-activation (>= 0), abs is identity.
    const int64_t idx3 = idx * 3;
    loss_scale[idx3 + 0] = scales[idx3 + 0] * vis;
    loss_scale[idx3 + 1] = scales[idx3 + 1] * vis;
    loss_scale[idx3 + 2] = scales[idx3 + 2] * vis;

    // gaussian_density_reg: abs(densities) * visibility
    loss_density[idx] = densities[idx] * vis;

    // gaussian_z_scale_reg: relu(z_scales - threshold)
    const scalar_t z_diff = z_scales[idx] - z_scale_threshold;
    loss_z_scale[idx] = z_diff > static_cast<scalar_t>(0) ? z_diff : static_cast<scalar_t>(0);

    // out_of_bound_loss: relu(|positions| - cuboid_dims / 2)
    for (int d = 0; d < 3; d++) {
        const scalar_t abs_pos = abs(positions[idx3 + d]);
        const scalar_t half_dim = cuboid_dims[idx3 + d] * static_cast<scalar_t>(0.5);
        const scalar_t diff = abs_pos - half_dim;
        loss_oob[idx3 + d] = diff > static_cast<scalar_t>(0) ? diff : static_cast<scalar_t>(0);
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void gaussian_losses_bwd_kernel(
    const int64_t N,
    const scalar_t *__restrict__ scales,        // [N, 3]
    const scalar_t *__restrict__ densities,     // [N]
    const scalar_t *__restrict__ z_scales,      // [N]
    const scalar_t *__restrict__ positions,     // [N, 3]
    const scalar_t *__restrict__ cuboid_dims,   // [N, 3]
    const scalar_t *__restrict__ visibility,    // [N] float or nullptr
    const scalar_t z_scale_threshold,
    const scalar_t *__restrict__ v_loss_scale,     // [N, 3]
    const scalar_t *__restrict__ v_loss_density,   // [N]
    const scalar_t *__restrict__ v_loss_z_scale,   // [N]
    const scalar_t *__restrict__ v_loss_oob,       // [N, 3]
    scalar_t *__restrict__ v_scales,               // [N, 3]
    scalar_t *__restrict__ v_densities,            // [N]
    scalar_t *__restrict__ v_z_scales,             // [N]
    scalar_t *__restrict__ v_positions             // [N, 3]
) {
    const int64_t idx =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    const scalar_t vis = (visibility != nullptr) ? visibility[idx] : static_cast<scalar_t>(1);

    // d(scale_reg)/d(scales) = vis, relying on the post-activation contract
    // (scales >= 0 ⇒ forward is scales * vis, no abs). This keeps fwd/bwd
    // consistent; if a caller violates the contract the result is wrong in the
    // same direction on both paths.
    const int64_t idx3 = idx * 3;
    v_scales[idx3 + 0] = v_loss_scale[idx3 + 0] * vis;
    v_scales[idx3 + 1] = v_loss_scale[idx3 + 1] * vis;
    v_scales[idx3 + 2] = v_loss_scale[idx3 + 2] * vis;

    // d(density_reg)/d(densities) = vis — same contract as scales.
    v_densities[idx] = v_loss_density[idx] * vis;

    // d(z_scale_reg)/d(z_scales) = (z_scales > threshold) ? 1 : 0
    {
        const scalar_t z_diff = z_scales[idx] - z_scale_threshold;
        v_z_scales[idx] = (z_diff > static_cast<scalar_t>(0)) ? v_loss_z_scale[idx] : static_cast<scalar_t>(0);
    }

    // d(oob)/d(positions) = (|pos| > half_dim) ? sign(pos) : 0
    for (int d = 0; d < 3; d++) {
        const scalar_t pos = positions[idx3 + d];
        const scalar_t abs_pos = abs(pos);
        const scalar_t half_dim = cuboid_dims[idx3 + d] * static_cast<scalar_t>(0.5);
        if (abs_pos > half_dim) {
            const scalar_t sign = (pos > static_cast<scalar_t>(0)) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1);
            v_positions[idx3 + d] = v_loss_oob[idx3 + d] * sign;
        } else {
            v_positions[idx3 + d] = static_cast<scalar_t>(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_gaussian_losses_fwd_kernel(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::Tensor *visibility,
    float z_scale_threshold,
    at::Tensor &loss_scale,
    at::Tensor &loss_density,
    at::Tensor &loss_z_scale,
    at::Tensor &loss_oob
) {
    const int64_t N = scales.size(0);
    if (N == 0) return;

    dim3 threads(256);
    dim3 grid(static_cast<uint32_t>((N + threads.x - 1) / threads.x));

    AT_DISPATCH_FLOATING_TYPES(scales.scalar_type(), "gaussian_losses_fwd", [&]() {
        const scalar_t *vis_ptr = (visibility != nullptr)
            ? visibility->data_ptr<scalar_t>() : nullptr;
        gaussian_losses_fwd_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                scales.data_ptr<scalar_t>(),
                densities.data_ptr<scalar_t>(),
                z_scales.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                cuboid_dims.data_ptr<scalar_t>(),
                vis_ptr,
                static_cast<scalar_t>(z_scale_threshold),
                loss_scale.data_ptr<scalar_t>(),
                loss_density.data_ptr<scalar_t>(),
                loss_z_scale.data_ptr<scalar_t>(),
                loss_oob.data_ptr<scalar_t>()
            );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_gaussian_losses_bwd_kernel(
    const at::Tensor &scales,
    const at::Tensor &densities,
    const at::Tensor &z_scales,
    const at::Tensor &positions,
    const at::Tensor &cuboid_dims,
    const at::Tensor *visibility,
    float z_scale_threshold,
    const at::Tensor &v_loss_scale,
    const at::Tensor &v_loss_density,
    const at::Tensor &v_loss_z_scale,
    const at::Tensor &v_loss_oob,
    at::Tensor &v_scales,
    at::Tensor &v_densities,
    at::Tensor &v_z_scales,
    at::Tensor &v_positions
) {
    const int64_t N = scales.size(0);
    if (N == 0) return;

    dim3 threads(256);
    dim3 grid(static_cast<uint32_t>((N + threads.x - 1) / threads.x));

    AT_DISPATCH_FLOATING_TYPES(scales.scalar_type(), "gaussian_losses_bwd", [&]() {
        const scalar_t *vis_ptr = (visibility != nullptr)
            ? visibility->data_ptr<scalar_t>() : nullptr;
        gaussian_losses_bwd_kernel<scalar_t>
            <<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                scales.data_ptr<scalar_t>(),
                densities.data_ptr<scalar_t>(),
                z_scales.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                cuboid_dims.data_ptr<scalar_t>(),
                vis_ptr,
                static_cast<scalar_t>(z_scale_threshold),
                v_loss_scale.data_ptr<scalar_t>(),
                v_loss_density.data_ptr<scalar_t>(),
                v_loss_z_scale.data_ptr<scalar_t>(),
                v_loss_oob.data_ptr<scalar_t>(),
                v_scales.data_ptr<scalar_t>(),
                v_densities.data_ptr<scalar_t>(),
                v_z_scales.data_ptr<scalar_t>(),
                v_positions.data_ptr<scalar_t>()
            );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat

#endif
