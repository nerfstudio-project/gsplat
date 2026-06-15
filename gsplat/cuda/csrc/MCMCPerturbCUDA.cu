/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <cmath>
#include <limits>

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "MCMCPerturb.h"
#include "Utils.cuh"

namespace gsplat {

inline __device__ float sigmoid_standard(float x) {
    return 1.f / (1.f + expf(-x));
}

inline __device__ float sigmoid_steep(float x, float k, float x0) {
    return 1.f / (1.f + expf(-k * (x - x0)));
}

__global__ void mcmc_perturb_positions_kernel(
    const uint32_t N,
    float *__restrict__ positions,
    const float *__restrict__ quats,
    const float *__restrict__ scales_log,
    const float *__restrict__ opacities_logit,
    const float *__restrict__ noise,
    const float scaler
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    vec4 quat = glm::make_vec4(quats + idx * 4);
    vec3 scale = glm::make_vec3(scales_log + idx * 3);
    scale = vec3(expf(scale.x), expf(scale.y), expf(scale.z));

    mat3 covar;
    quat_scale_to_covar_preci(quat, scale, &covar, nullptr);

    const float density = sigmoid_standard(opacities_logit[idx]);
    const float w = sigmoid_steep(1.f - density, 100.f, 0.995f) * scaler;

    vec3 n = glm::make_vec3(noise + idx * 3) * w;
    vec3 transformed = covar * n;

    positions[idx * 3 + 0] += transformed.x;
    positions[idx * 3 + 1] += transformed.y;
    positions[idx * 3 + 2] += transformed.z;
}

void launch_mcmc_perturb_positions_kernel(
    at::Tensor positions,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &noise,
    float scaler
) {
    const int64_t N = positions.size(0);
    if (N == 0) {
        return;
    }
    TORCH_CHECK(
        N <= std::numeric_limits<uint32_t>::max(),
        "mcmc_perturb_positions: N exceeds uint32_t"
    );

    const uint32_t N_u32 = static_cast<uint32_t>(N);
    dim3 threads(256);
    dim3 grid((N_u32 + threads.x - 1) / threads.x);

    mcmc_perturb_positions_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        N_u32,
        positions.data_ptr<float>(),
        quats.const_data_ptr<float>(),
        scales.const_data_ptr<float>(),
        opacities.const_data_ptr<float>(),
        noise.const_data_ptr<float>(),
        scaler
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace gsplat

#endif // GSPLAT_BUILD_3DGS
