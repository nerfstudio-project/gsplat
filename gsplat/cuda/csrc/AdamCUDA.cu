/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#if GSPLAT_BUILD_ADAM

#include <ATen/Dispatch.h> // AT_DISPATCH_XXX
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream
#include <cooperative_groups.h>

#include "Null.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void adam_kernel(
    const uint32_t N,
    const uint32_t D,
    scalar_t *__restrict__ param,
    const scalar_t *__restrict__ param_grad,
    scalar_t *__restrict__ exp_avg,
    scalar_t *__restrict__ exp_avg_sq,
    const bool *valid,
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / D;

    if (g_idx >= N)
        return;
    if (valid != nullptr && !valid[g_idx])
        return;

    float register_param_grad = param_grad[p_idx];
    float register_exp_avg = exp_avg[p_idx];
    float register_exp_avg_sq = exp_avg_sq[p_idx];
    register_exp_avg =
        b1 * register_exp_avg + (1.0f - b1) * register_param_grad;
    register_exp_avg_sq = b2 * register_exp_avg_sq + (1.0f - b2) *
                                                         register_param_grad *
                                                         register_param_grad;
    float step = -lr * register_exp_avg / (sqrt(register_exp_avg_sq) + eps);

    param[p_idx] += step;
    exp_avg[p_idx] = register_exp_avg;
    exp_avg_sq[p_idx] = register_exp_avg_sq;
}

void launch_adam_kernel(
    at::Tensor &param,                    // [N, ...]
    const at::Tensor &param_grad,         // [N, ...]
    at::Tensor &exp_avg,                  // [N, ...]
    at::Tensor &exp_avg_sq,               // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    const uint32_t N = param.size(0);
    const uint32_t D = param.numel() / N;

    // parallel over [N, ...]
    int64_t n_elements = N * D;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_kernel", [&]() {
        adam_kernel<scalar_t>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                N,
                D,
                param.data_ptr<scalar_t>(),
                param_grad.data_ptr<scalar_t>(),
                exp_avg.data_ptr<scalar_t>(),
                exp_avg_sq.data_ptr<scalar_t>(),
                valid.has_value() ? valid.value().data_ptr<bool>() : nullptr,
                lr,
                b1,
                b2,
                eps
            );
    });
}

} // namespace gsplat

#endif
