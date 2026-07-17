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
// Forward: fused RGB L1 + background MSE per pixel
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void camera_losses_fwd_kernel(
    const uint32_t N,
    const int32_t *__restrict__ flags,     // [N]
    const scalar_t *__restrict__ rgb_pred, // [N, 3]
    const scalar_t *__restrict__ rgb_gt,   // [N, 3]
    const scalar_t *__restrict__ bg_pred,  // [N]
    const scalar_t rgb_factor,
    const scalar_t bg_factor,
    scalar_t *__restrict__ rgb_loss, // [N]
    scalar_t *__restrict__ bg_loss   // [N]
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N)
    {
        return;
    }

    const int32_t f = flags[idx];

    // --- RGB L1: active when (f & RGB_LABEL) && !(f & INVALID) ---
    {
        scalar_t rgb_l = static_cast<scalar_t>(0);
        if(rgb_factor > static_cast<scalar_t>(0) && (f & GSPLAT_LOSS_FLAG_RGB_LABEL) && !(f & GSPLAT_LOSS_FLAG_INVALID))
        {
            const uint32_t idx3 = idx * 3;
            rgb_l               = (abs(rgb_pred[idx3 + 0] - rgb_gt[idx3 + 0])
                                   + abs(rgb_pred[idx3 + 1] - rgb_gt[idx3 + 1])
                                   + abs(rgb_pred[idx3 + 2] - rgb_gt[idx3 + 2]))
                                * rgb_factor;
        }
        rgb_loss[idx] = rgb_l;
    }

    // --- BG MSE: active when !(f & INVALID) && !(f & DIFIXED) && !(f & SYNTHETIC) ---
    {
        scalar_t bg_l = static_cast<scalar_t>(0);
        if(bg_factor > static_cast<scalar_t>(0)
           && !(f & GSPLAT_LOSS_FLAG_INVALID)
           && !(f & GSPLAT_LOSS_FLAG_DIFIXED)
           && !(f & GSPLAT_LOSS_FLAG_SYNTHETIC))
        {
            scalar_t p = bg_pred[idx];
            p          = (p < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
                       : (p > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1)
                                                        : p;
            const scalar_t target
                = (f & GSPLAT_LOSS_FLAG_SKY_SEMANTIC) ? static_cast<scalar_t>(0) : static_cast<scalar_t>(1);
            const scalar_t diff = p - target;
            bg_l                = diff * diff * bg_factor;
        }
        bg_loss[idx] = bg_l;
    }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void camera_losses_bwd_kernel(
    const uint32_t N,
    const int32_t *__restrict__ flags,
    const scalar_t *__restrict__ rgb_pred,
    const scalar_t *__restrict__ rgb_gt,
    const scalar_t *__restrict__ bg_pred,
    const scalar_t rgb_factor,
    const scalar_t bg_factor,
    const scalar_t *__restrict__ v_rgb_loss,
    const scalar_t *__restrict__ v_bg_loss,
    scalar_t *__restrict__ v_rgb_pred, // [N, 3]
    scalar_t *__restrict__ v_bg_pred   // [N]
)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N)
    {
        return;
    }

    const int32_t f = flags[idx];

    // --- RGB L1 backward: d/d(pred) = sign(pred - gt) * factor ---
    {
        const uint32_t idx3 = idx * 3;
        scalar_t g0         = static_cast<scalar_t>(0);
        scalar_t g1         = static_cast<scalar_t>(0);
        scalar_t g2         = static_cast<scalar_t>(0);
        if(rgb_factor > static_cast<scalar_t>(0) && (f & GSPLAT_LOSS_FLAG_RGB_LABEL) && !(f & GSPLAT_LOSS_FLAG_INVALID))
        {
            const scalar_t vrl = v_rgb_loss[idx];
            const scalar_t rf  = rgb_factor * vrl;
            const scalar_t d0  = rgb_pred[idx3 + 0] - rgb_gt[idx3 + 0];
            const scalar_t d1  = rgb_pred[idx3 + 1] - rgb_gt[idx3 + 1];
            const scalar_t d2  = rgb_pred[idx3 + 2] - rgb_gt[idx3 + 2];
            g0 = (d0 > static_cast<scalar_t>(0) ? rf : d0 < static_cast<scalar_t>(0) ? -rf : static_cast<scalar_t>(0));
            g1 = (d1 > static_cast<scalar_t>(0) ? rf : d1 < static_cast<scalar_t>(0) ? -rf : static_cast<scalar_t>(0));
            g2 = (d2 > static_cast<scalar_t>(0) ? rf : d2 < static_cast<scalar_t>(0) ? -rf : static_cast<scalar_t>(0));
        }
        v_rgb_pred[idx3 + 0] = g0;
        v_rgb_pred[idx3 + 1] = g1;
        v_rgb_pred[idx3 + 2] = g2;
    }

    // --- BG MSE backward: d/d(pred) = 2*(clamp(pred,0,1) - target) * factor * clamp_grad ---
    {
        scalar_t g = static_cast<scalar_t>(0);
        if(bg_factor > static_cast<scalar_t>(0)
           && !(f & GSPLAT_LOSS_FLAG_INVALID)
           && !(f & GSPLAT_LOSS_FLAG_DIFIXED)
           && !(f & GSPLAT_LOSS_FLAG_SYNTHETIC))
        {
            const scalar_t p_raw = bg_pred[idx];
            const scalar_t p     = (p_raw < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0)
                                 : (p_raw > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1)
                                                                      : p_raw;
            const scalar_t target
                = (f & GSPLAT_LOSS_FLAG_SKY_SEMANTIC) ? static_cast<scalar_t>(0) : static_cast<scalar_t>(1);
            // clamp gradient: 1 if 0 <= p_raw <= 1, else 0
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

void launch_camera_losses_fwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &rgb_pred,
    const at::Tensor &rgb_gt,
    const at::Tensor &bg_pred,
    float rgb_factor,
    float bg_factor,
    at::Tensor &rgb_loss,
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
        rgb_pred.scalar_type(),
        "camera_losses_fwd",
        [&]()
        {
            camera_losses_fwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                flags.data_ptr<int32_t>(),
                rgb_pred.data_ptr<scalar_t>(),
                rgb_gt.data_ptr<scalar_t>(),
                bg_pred.data_ptr<scalar_t>(),
                static_cast<scalar_t>(rgb_factor),
                static_cast<scalar_t>(bg_factor),
                rgb_loss.data_ptr<scalar_t>(),
                bg_loss.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_camera_losses_bwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &rgb_pred,
    const at::Tensor &rgb_gt,
    const at::Tensor &bg_pred,
    float rgb_factor,
    float bg_factor,
    const at::Tensor &v_rgb_loss,
    const at::Tensor &v_bg_loss,
    at::Tensor &v_rgb_pred,
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
        rgb_pred.scalar_type(),
        "camera_losses_bwd",
        [&]()
        {
            camera_losses_bwd_kernel<scalar_t><<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                N,
                flags.data_ptr<int32_t>(),
                rgb_pred.data_ptr<scalar_t>(),
                rgb_gt.data_ptr<scalar_t>(),
                bg_pred.data_ptr<scalar_t>(),
                static_cast<scalar_t>(rgb_factor),
                static_cast<scalar_t>(bg_factor),
                v_rgb_loss.data_ptr<scalar_t>(),
                v_bg_loss.data_ptr<scalar_t>(),
                v_rgb_pred.data_ptr<scalar_t>(),
                v_bg_pred.data_ptr<scalar_t>()
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat

#endif
