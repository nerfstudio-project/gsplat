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

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>

#    include <cmath>
#    include <cstdint>

#    include "cores/DeformCore.cuh"

namespace gsplat
{
// Running-sums layout: numerator sum(|d| * mask~) and denominator sum(mask).
enum
{
    kDeformLossSum   = 0,
    kDeformMaskSum   = 1,
    kDeformSumsCount = 2
};

// The per-element arithmetic (|d| * mask~ value, sign/denominator gradients,
// broadcast index math) lives in cores/DeformCore.cuh (shared with the fused
// all-losses kernel); the kernels below are shells that own the pointer
// loads, the shared-memory block reductions, and the accumulator atomics.

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

// Pass 1: accumulate the numerator over the M deformation elements and the
// denominator over the S ORIGINAL mask elements (or 1 per deformation element
// when there is no mask). Block-local tree reduction keeps global atomics down
// to one flush per (block, field).
template<typename scalar_t>
__global__ void deform_losses_accumulate_kernel(
    const int64_t M, // deformation elements
    const int64_t S, // mask source elements (ignored when mask == nullptr)
    const int64_t D, // deformation inner extent
    const int64_t mask_stride_row,
    const int64_t mask_stride_col,
    const scalar_t *__restrict__ deformation, // [M]
    const scalar_t *__restrict__ mask,        // [S] or nullptr
    scalar_t *__restrict__ sums               // [2] {loss_sum, mask_sum}
)
{
    extern __shared__ char shared_raw[];
    scalar_t *partial_loss = reinterpret_cast<scalar_t *>(shared_raw);
    scalar_t *partial_mask = partial_loss + blockDim.x;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    scalar_t thread_loss = scalar_t(0);
    scalar_t thread_mask = scalar_t(0);
    if(idx < M)
    {
        const scalar_t value = deformation[idx];
        const scalar_t weight
            = (mask != nullptr) ? mask[deform_mask_index(idx, D, mask_stride_row, mask_stride_col)] : scalar_t(1);
        thread_loss = deform_element_loss(value, weight);
    }
    if(mask != nullptr)
    {
        if(idx < S)
        {
            thread_mask = mask[idx];
        }
    }
    else if(idx < M)
    {
        thread_mask = scalar_t(1);
    }

    partial_loss[threadIdx.x] = thread_loss;
    partial_mask[threadIdx.x] = thread_mask;
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            partial_loss[threadIdx.x] += partial_loss[threadIdx.x + offset];
            partial_mask[threadIdx.x] += partial_mask[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        if(partial_loss[0] != scalar_t(0))
        {
            atomicAdd(&sums[kDeformLossSum], partial_loss[0]);
        }
        if(partial_mask[0] != scalar_t(0))
        {
            atomicAdd(&sums[kDeformMaskSum], partial_mask[0]);
        }
    }
}

// Pass 2: scalar loss = loss_sum / max(mask_sum, 1).
template<typename scalar_t>
__global__ void deform_losses_finalize_kernel(const scalar_t *__restrict__ sums, scalar_t *__restrict__ loss)
{
    loss[0] = deform_finalize_loss(sums[kDeformLossSum], sums[kDeformMaskSum]);
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------

// One thread per element of the larger of the two flattened inputs.
//   d(loss)/d(d_i)  = mask~_i * sign(d_i) / denom
//   d(loss)/d(m_j)  = sum_{i in replicas(j), mask~_i != 0} |d_i| / denom (numerator)
//                   - [mask_sum >= 1] * loss_sum / denom^2              (denominator)
// Masked-out elements (mask~_i == 0) are fully inert per the forward's
// contract: they contribute neither to the loss nor to the mask-gradient
// numerator, so non-finite residuals in masked-out rows cannot poison
// either. The clamp max(mask_sum, 1) has zero slope while active, so the
// denominator term is suppressed when mask_sum < 1 (matching torch's
// clamp(min=1) backward, which passes gradient at mask_sum == 1 exactly).
// The two mask terms are combined into one v_mask buffer via atomics instead
// of being split across broadcast-view and source-mask inputs — the same
// total gradient.
template<typename scalar_t>
__global__ void deform_losses_bwd_kernel(
    const int64_t M,
    const int64_t S,
    const int64_t D,
    const int64_t mask_stride_row,
    const int64_t mask_stride_col,
    const scalar_t *__restrict__ deformation, // [M]
    const scalar_t *__restrict__ mask,        // [S] or nullptr
    const scalar_t *__restrict__ sums,        // [2] {loss_sum, mask_sum}
    const scalar_t *__restrict__ v_loss,      // [] scalar upstream gradient
    scalar_t *__restrict__ v_deformation,     // [M]
    scalar_t *__restrict__ v_mask             // [S] (zero-initialized) or nullptr
)
{
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = (mask != nullptr && S > M) ? S : M;
    if(idx >= total)
    {
        return;
    }

    const scalar_t up       = v_loss[0];
    const scalar_t loss_sum = sums[kDeformLossSum];
    const scalar_t mask_sum = sums[kDeformMaskSum];
    const scalar_t denom    = deform_clamped_denominator(mask_sum);

    if(idx < M)
    {
        const int64_t midx    = (mask != nullptr) ? deform_mask_index(idx, D, mask_stride_row, mask_stride_col) : 0;
        const scalar_t value  = deformation[idx];
        const scalar_t weight = (mask != nullptr) ? mask[midx] : scalar_t(1);
        v_deformation[idx]    = deform_grad_deformation(up, value, weight, denom);
        if(v_mask != nullptr)
        {
            atomicAdd(&v_mask[midx], deform_grad_mask_numerator(up, value, weight, denom));
        }
    }
    if(v_mask != nullptr && idx < S && mask_sum >= scalar_t(1))
    {
        atomicAdd(&v_mask[idx], deform_grad_mask_denominator(up, loss_sum, denom));
    }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_deform_losses_fwd_kernel(
    const at::Tensor &deformation,
    const at::Tensor *mask,
    int64_t mask_stride_row,
    int64_t mask_stride_col,
    at::Tensor &sums,
    at::Tensor &loss
)
{
    const int64_t M     = deformation.numel();
    const int64_t S     = (mask != nullptr) ? mask->numel() : int64_t(0);
    const int64_t D     = (deformation.dim() == 2) ? deformation.size(1) : int64_t(1);
    const int64_t total = (S > M) ? S : M;
    if(total == 0)
    {
        // sums and loss are caller-zero-initialized: the empty reduction is 0
        // and the clamped denominator keeps the loss at 0 (not 0/0).
        return;
    }

    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>((total + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(
        deformation.scalar_type(),
        "deform_losses_fwd",
        [&]()
        {
            const scalar_t *mask_ptr = (mask != nullptr) ? mask->data_ptr<scalar_t>() : nullptr;
            const size_t accum_shmem = static_cast<size_t>(2 * threads) * sizeof(scalar_t);
            deform_losses_accumulate_kernel<scalar_t><<<grid, threads, accum_shmem, stream>>>(
                M,
                S,
                D,
                mask_stride_row,
                mask_stride_col,
                deformation.data_ptr<scalar_t>(),
                mask_ptr,
                sums.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            deform_losses_finalize_kernel<scalar_t>
                <<<1, 1, 0, stream>>>(sums.data_ptr<scalar_t>(), loss.data_ptr<scalar_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

void launch_deform_losses_bwd_kernel(
    const at::Tensor &deformation,
    const at::Tensor *mask,
    int64_t mask_stride_row,
    int64_t mask_stride_col,
    const at::Tensor &sums,
    const at::Tensor &v_loss,
    at::Tensor &v_deformation,
    at::Tensor *v_mask
)
{
    const int64_t M     = deformation.numel();
    const int64_t S     = (mask != nullptr) ? mask->numel() : int64_t(0);
    const int64_t D     = (deformation.dim() == 2) ? deformation.size(1) : int64_t(1);
    const int64_t total = (S > M) ? S : M;
    if(total == 0)
    {
        return;
    }

    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>((total + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(
        deformation.scalar_type(),
        "deform_losses_bwd",
        [&]()
        {
            const scalar_t *mask_ptr = (mask != nullptr) ? mask->data_ptr<scalar_t>() : nullptr;
            scalar_t *v_mask_ptr     = (v_mask != nullptr) ? v_mask->data_ptr<scalar_t>() : nullptr;
            deform_losses_bwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                M,
                S,
                D,
                mask_stride_row,
                mask_stride_col,
                deformation.data_ptr<scalar_t>(),
                mask_ptr,
                sums.data_ptr<scalar_t>(),
                v_loss.data_ptr<scalar_t>(),
                v_deformation.data_ptr<scalar_t>(),
                v_mask_ptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}
} // namespace gsplat

#endif
