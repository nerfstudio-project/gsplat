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

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>

#    include "LossFlags.h"

namespace gsplat
{
// ---------------------------------------------------------------------------
// Forward: fused distance L1 + intensity MSE + raydrop MSE + BG MSE
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void lidar_losses_fwd_kernel(
    const uint32_t N,
    const int32_t *__restrict__ flags,           // [N]
    const scalar_t *__restrict__ distance_pred,  // [N]
    const scalar_t *__restrict__ distance_gt,    // [N]
    const scalar_t *__restrict__ intensity_pred, // [N]
    const scalar_t *__restrict__ intensity_gt,   // [N]
    const scalar_t *__restrict__ raydrop_pred,   // [N]
    const scalar_t *__restrict__ raydrop_gt,     // [N]
    const scalar_t *__restrict__ bg_pred,        // [N]
    const scalar_t distance_factor,
    const scalar_t intensity_factor,
    const scalar_t raydrop_factor,
    const scalar_t bg_factor,
    scalar_t *__restrict__ distance_loss,  // [N]
    scalar_t *__restrict__ intensity_loss, // [N]
    scalar_t *__restrict__ raydrop_loss,   // [N]
    scalar_t *__restrict__ bg_loss         // [N]
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N)
    {
        return;
    }

    const int32_t f              = flags[idx];
    const bool valid             = !(f & GSPLAT_LOSS_FLAG_INVALID);
    const bool valid_not_dropped = valid && !(f & GSPLAT_LOSS_FLAG_DROPPED);

    // --- Distance L1: !(INVALID) && !(DROPPED) ---
    {
        scalar_t dl = static_cast<scalar_t>(0);
        if(distance_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            dl = abs(distance_pred[idx] - distance_gt[idx]) * distance_factor;
        }
        distance_loss[idx] = dl;
    }

    // --- Intensity MSE: !(INVALID) && !(DROPPED) ---
    {
        scalar_t il = static_cast<scalar_t>(0);
        if(intensity_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            const scalar_t diff = intensity_pred[idx] - intensity_gt[idx];
            il                  = diff * diff * intensity_factor;
        }
        intensity_loss[idx] = il;
    }

    // --- Raydrop MSE: !(INVALID) only (no DROPPED check) ---
    {
        scalar_t rl = static_cast<scalar_t>(0);
        if(raydrop_factor > static_cast<scalar_t>(0) && valid)
        {
            const scalar_t diff = raydrop_pred[idx] - raydrop_gt[idx];
            rl                  = diff * diff * raydrop_factor;
        }
        raydrop_loss[idx] = rl;
    }

    // --- BG LiDAR MSE: !(INVALID) && !(DROPPED) ---
    {
        scalar_t bl = static_cast<scalar_t>(0);
        if(bg_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            scalar_t p = bg_pred[idx];
            p          = (p < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
                       : (p > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1)
                                                        : p;
            const scalar_t target
                = (f & GSPLAT_LOSS_FLAG_SKY_SEMANTIC) ? static_cast<scalar_t>(0) : static_cast<scalar_t>(1);
            const scalar_t diff = p - target;
            bl                  = diff * diff * bg_factor;
        }
        bg_loss[idx] = bl;
    }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void lidar_losses_bwd_kernel(
    const uint32_t N,
    const int32_t *__restrict__ flags,
    const scalar_t *__restrict__ distance_pred,
    const scalar_t *__restrict__ distance_gt,
    const scalar_t *__restrict__ intensity_pred,
    const scalar_t *__restrict__ intensity_gt,
    const scalar_t *__restrict__ raydrop_pred,
    const scalar_t *__restrict__ raydrop_gt,
    const scalar_t *__restrict__ bg_pred,
    const scalar_t distance_factor,
    const scalar_t intensity_factor,
    const scalar_t raydrop_factor,
    const scalar_t bg_factor,
    const scalar_t *__restrict__ v_distance_loss,
    const scalar_t *__restrict__ v_intensity_loss,
    const scalar_t *__restrict__ v_raydrop_loss,
    const scalar_t *__restrict__ v_bg_loss,
    scalar_t *__restrict__ v_distance_pred,
    scalar_t *__restrict__ v_intensity_pred,
    scalar_t *__restrict__ v_raydrop_pred,
    scalar_t *__restrict__ v_bg_pred
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N)
    {
        return;
    }

    const int32_t f              = flags[idx];
    const bool valid             = !(f & GSPLAT_LOSS_FLAG_INVALID);
    const bool valid_not_dropped = valid && !(f & GSPLAT_LOSS_FLAG_DROPPED);

    // --- Distance L1 backward: sign(pred - gt) * factor ---
    {
        scalar_t g = static_cast<scalar_t>(0);
        if(distance_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            const scalar_t diff = distance_pred[idx] - distance_gt[idx];
            const scalar_t sign = (diff > static_cast<scalar_t>(0)) ? static_cast<scalar_t>(1)
                                : (diff < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(-1)
                                                                    : static_cast<scalar_t>(0);
            g                   = sign * distance_factor * v_distance_loss[idx];
        }
        v_distance_pred[idx] = g;
    }

    // --- Intensity MSE backward: 2*(pred - gt) * factor ---
    {
        scalar_t g = static_cast<scalar_t>(0);
        if(intensity_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            g = static_cast<scalar_t>(2)
              * (intensity_pred[idx] - intensity_gt[idx])
              * intensity_factor
              * v_intensity_loss[idx];
        }
        v_intensity_pred[idx] = g;
    }

    // --- Raydrop MSE backward: 2*(pred - gt) * factor ---
    {
        scalar_t g = static_cast<scalar_t>(0);
        if(raydrop_factor > static_cast<scalar_t>(0) && valid)
        {
            g = static_cast<scalar_t>(2) * (raydrop_pred[idx] - raydrop_gt[idx]) * raydrop_factor * v_raydrop_loss[idx];
        }
        v_raydrop_pred[idx] = g;
    }

    // --- BG LiDAR MSE backward: 2*(clamp(pred,0,1) - target) * factor * clamp_grad ---
    {
        scalar_t g = static_cast<scalar_t>(0);
        if(bg_factor > static_cast<scalar_t>(0) && valid_not_dropped)
        {
            const scalar_t p_raw = bg_pred[idx];
            const scalar_t p     = (p_raw < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
                                 : (p_raw > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1)
                                                                      : p_raw;
            const scalar_t target
                = (f & GSPLAT_LOSS_FLAG_SKY_SEMANTIC) ? static_cast<scalar_t>(0) : static_cast<scalar_t>(1);
            const scalar_t clamp_grad = (p_raw >= static_cast<scalar_t>(0) && p_raw <= static_cast<scalar_t>(1))
                                          ? static_cast<scalar_t>(1)
                                          : static_cast<scalar_t>(0);
            g = static_cast<scalar_t>(2) * (p - target) * bg_factor * v_bg_loss[idx] * clamp_grad;
        }
        v_bg_pred[idx] = g;
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_lidar_losses_fwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &distance_pred,
    const at::Tensor &distance_gt,
    const at::Tensor &intensity_pred,
    const at::Tensor &intensity_gt,
    const at::Tensor &raydrop_pred,
    const at::Tensor &raydrop_gt,
    const at::Tensor &bg_pred,
    float distance_factor,
    float intensity_factor,
    float raydrop_factor,
    float bg_factor,
    at::Tensor &distance_loss,
    at::Tensor &intensity_loss,
    at::Tensor &raydrop_loss,
    at::Tensor &bg_loss
)
{
    const uint32_t N = flags.size(0);
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        distance_pred.scalar_type(),
        "lidar_losses_fwd",
        [&]()
        {
            lidar_losses_fwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                flags.data_ptr<int32_t>(),
                distance_pred.data_ptr<scalar_t>(),
                distance_gt.data_ptr<scalar_t>(),
                intensity_pred.data_ptr<scalar_t>(),
                intensity_gt.data_ptr<scalar_t>(),
                raydrop_pred.data_ptr<scalar_t>(),
                raydrop_gt.data_ptr<scalar_t>(),
                bg_pred.data_ptr<scalar_t>(),
                static_cast<scalar_t>(distance_factor),
                static_cast<scalar_t>(intensity_factor),
                static_cast<scalar_t>(raydrop_factor),
                static_cast<scalar_t>(bg_factor),
                distance_loss.data_ptr<scalar_t>(),
                intensity_loss.data_ptr<scalar_t>(),
                raydrop_loss.data_ptr<scalar_t>(),
                bg_loss.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_lidar_losses_bwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &distance_pred,
    const at::Tensor &distance_gt,
    const at::Tensor &intensity_pred,
    const at::Tensor &intensity_gt,
    const at::Tensor &raydrop_pred,
    const at::Tensor &raydrop_gt,
    const at::Tensor &bg_pred,
    float distance_factor,
    float intensity_factor,
    float raydrop_factor,
    float bg_factor,
    const at::Tensor &v_distance_loss,
    const at::Tensor &v_intensity_loss,
    const at::Tensor &v_raydrop_loss,
    const at::Tensor &v_bg_loss,
    at::Tensor &v_distance_pred,
    at::Tensor &v_intensity_pred,
    at::Tensor &v_raydrop_pred,
    at::Tensor &v_bg_pred
)
{
    const uint32_t N = flags.size(0);
    if(N == 0)
    {
        return;
    }

    dim3 threads(256);
    dim3 grid((N + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(
        distance_pred.scalar_type(),
        "lidar_losses_bwd",
        [&]()
        {
            lidar_losses_bwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                flags.data_ptr<int32_t>(),
                distance_pred.data_ptr<scalar_t>(),
                distance_gt.data_ptr<scalar_t>(),
                intensity_pred.data_ptr<scalar_t>(),
                intensity_gt.data_ptr<scalar_t>(),
                raydrop_pred.data_ptr<scalar_t>(),
                raydrop_gt.data_ptr<scalar_t>(),
                bg_pred.data_ptr<scalar_t>(),
                static_cast<scalar_t>(distance_factor),
                static_cast<scalar_t>(intensity_factor),
                static_cast<scalar_t>(raydrop_factor),
                static_cast<scalar_t>(bg_factor),
                v_distance_loss.data_ptr<scalar_t>(),
                v_intensity_loss.data_ptr<scalar_t>(),
                v_raydrop_loss.data_ptr<scalar_t>(),
                v_bg_loss.data_ptr<scalar_t>(),
                v_distance_pred.data_ptr<scalar_t>(),
                v_intensity_pred.data_ptr<scalar_t>(),
                v_raydrop_pred.data_ptr<scalar_t>(),
                v_bg_pred.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat

#endif
