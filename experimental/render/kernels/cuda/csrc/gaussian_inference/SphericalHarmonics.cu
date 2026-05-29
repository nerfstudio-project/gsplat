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

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <type_traits>

#include "Common.h"
#include "SHCommon.h"
#include "SHCompression.h"
#include "SphericalHarmonics.h"
#include "Utils.h"

namespace higs {

static constexpr bool SHEVAL_COMPACTION = true;
static constexpr int SHEVAL_CTA_SIZE    = 256;
static constexpr int SHEVAL_MIN_CTAS    = 4;

static_assert(SHEVAL_CTA_SIZE % 32 == 0 && SHEVAL_CTA_SIZE <= 1024,
              "SHEVAL_CTA_SIZE must be a multiple of 32 and at most 1024");

// Compact valid gaussian indices for SHEVAL_CTA_SIZE bits of the mask.
// Warp 0 loads mask words and computes exclusive prefix offsets via Kogge-Stone,
// then all warps scatter set-bit indices into smem_compact[] using a single __popc
// per active lane (no serial loop). Returns the total valid count.
// All threads must call this (two barriers inside).
__device__ int32_t CompactMask(const uint32_t *__restrict__ masks, int32_t N, int32_t block_base,
                               int32_t *__restrict__ smem_compact, int32_t &smem_total)
{
    constexpr int32_t NUM_WARPS = SHEVAL_CTA_SIZE / 32;

    // let NVCC know that warp_id is a warp uniform (doh!)
    const int32_t warp_id = __shfl_sync(~0u, threadIdx.x >> 5, 0);
    const int32_t lane_id = threadIdx.x & 31;

    // broadcast arrays: warp 0 writes, all warps read after barrier
    __shared__ uint32_t smem_words[NUM_WARPS];
    __shared__ int32_t smem_offsets[NUM_WARPS];

    if (warp_id == 0)
    {
        // All 32 lanes participate to keep the warp converged; lanes >= NUM_WARPS
        // contribute count=0 so the prefix sum for lanes 0..NUM_WARPS-1 is correct.
        // This avoids partial-mask __shfl_up_sync which causes codegen issues.
        const int32_t first_bit = block_base + lane_id * 32;
        const uint32_t word     = (lane_id < NUM_WARPS && first_bit < N) ? masks[first_bit >> 5] : 0u;
        const int32_t count     = __popc(word);

        // inclusive Kogge-Stone prefix sum — full warp mask, clean codegen
        int32_t prefix = count;
#pragma unroll
        for (int32_t delta = 1; delta < NUM_WARPS; delta <<= 1)
        {
            const int32_t n = __shfl_up_sync(~0u, prefix, delta);
            if (lane_id >= delta)
            {
                prefix += n;
            }
        }

        // only lanes with real data write to smem
        if (lane_id < NUM_WARPS)
        {
            smem_words[lane_id]   = word;
            smem_offsets[lane_id] = prefix - count; // exclusive prefix
        }

        if (lane_id == NUM_WARPS - 1)
        {
            smem_total = prefix; // total valid count
        }
    }
    __syncthreads();

    // all warps: read own word + offset, scatter with popc
    const uint32_t word       = smem_words[warp_id];
    const int32_t warp_offset = smem_offsets[warp_id];

    if (word & (1u << lane_id))
    {
        smem_compact[warp_offset + __popc(word & ((1u << lane_id) - 1))] = block_base + warp_id * 32 + lane_id;
    }

    __syncthreads();
    return smem_total;
}

__global__ void spherical_harmonics_fwd_kernel(const int N, const int K, const int degrees_to_use,
                                               const float *__restrict__ means, // [3, N] planar
                                               const float cam_x, const float cam_y, const float cam_z,
                                               const float *__restrict__ coeffs,   // [N, K, 3]
                                               const uint32_t *__restrict__ masks, // [(N+31)/32] packed bitfield
                                               float bias, float min_value,
                                               __half *__restrict__ colors // [N, 4] half {R,G,B,0}
)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * 3)
    {
        return;
    }
    const int elem_id = idx / 3;
    const int c       = idx % 3;
    if (masks != nullptr && !(masks[elem_id >> 5] & (1u << (elem_id & 0x1fu))))
    {
        return;
    }
    float3 dir_n{};
    if (degrees_to_use > 0)
    {
        dir_n = GetViewDir(means, N, elem_id, cam_x, cam_y, cam_z);
    }
    float val;
    EvaluteSHCoeffs(degrees_to_use, c, dir_n, coeffs + elem_id * K * 3, bias, min_value, val);
    colors[elem_id * 4 + c] = __float2half(val);
}

// Unified K=16 SH forward kernel for raw (float/half) and compressed (32B/16B) inputs.
// Uses compile-time SHInputMode to select the coefficient loading path; all other logic
// (compaction, SH evaluation, output packing) is shared.
template<SHInputMode MODE>
__global__ void __launch_bounds__(SHEVAL_CTA_SIZE, SHEVAL_MIN_CTAS)
    spherical_harmonics_fwd_kernel_k16(int N, int degrees_to_use,
                                       const float *__restrict__ means, // [3, N] planar
                                       float cam_x, float cam_y, float cam_z, const void *__restrict__ input_data,
                                       const uint32_t *__restrict__ masks, // [(N+31)/32] packed bitfield
                                       float bias, float min_value, SHDecodeParams decode_params,
                                       __half *__restrict__ colors // [N, 4] half {R,G,B,0}
    )
{
    const int idx = blockIdx.x * SHEVAL_CTA_SIZE + threadIdx.x;

    int work_idx;

    if constexpr (SHEVAL_COMPACTION)
    {
        if (masks != nullptr)
        {
            __shared__ int32_t smem_compact[SHEVAL_CTA_SIZE];
            __shared__ int32_t smem_total;

            const int32_t total = CompactMask(masks, N, blockIdx.x * SHEVAL_CTA_SIZE, smem_compact, smem_total);
            if ((int32_t)threadIdx.x >= total)
            {
                return;
            }
            work_idx = smem_compact[threadIdx.x];
        }
        else
        {
            if (idx >= N)
            {
                return;
            }
            work_idx = idx;
        }
    }
    else
    {
        if (idx >= N)
        {
            return;
        }
        if (masks != nullptr && !(masks[idx >> 5] & (1u << (idx & 0x1fu))))
        {
            return;
        }
        work_idx = idx;
    }

    const float3 dir_n = GetViewDir(means, N, work_idx, cam_x, cam_y, cam_z);
    EvalSHForGaussian<MODE>(dir_n, work_idx, N, degrees_to_use, input_data, bias, min_value, decode_params, colors);
}

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    int32_t degrees_to_use,
    const at::Tensor means, // [3, N] float — gaussian centers (planar)
    float cam_x, float cam_y, float cam_z,
    const at::Tensor coeffs,              // [..., K, 3] OR packed int32 when compressed
    const at::optional<at::Tensor> masks, // [...]
    const float bias, const float min_value,
    // outputs
    at::Tensor colors, // [..., 4] half {R,G,B,0}
    SHCompressionMode mode, const SHDecodeParams *decode_params)
{
    const int N = means.numel() / 3;

    if (N == 0)
    {
        return;
    }

    const float *means_ptr = means.data_ptr<float>();
    const uint32_t *masks_ptr =
        masks.has_value() ? reinterpret_cast<const uint32_t *>(masks.value().data_ptr<int32_t>()) : nullptr;
    __half *colors_ptr = reinterpret_cast<__half *>(colors.data_ptr<at::Half>());
    auto stream        = at::cuda::getCurrentCUDAStream();

    const bool compressed = (mode != SHCompressionMode::NONE);
    const int K           = compressed ? 16 : static_cast<int>(coeffs.size(-2));

    if (K == 16)
    {
        if (compressed)
        {
            TORCH_CHECK(decode_params != nullptr, "compressed SH requires non-null decode_params pointer");
            TORCH_CHECK(degrees_to_use == 3, "compressed SH requires degrees_to_use=3, got ", degrees_to_use);
            TORCH_CHECK(coeffs.dim() == 2 && coeffs.size(1) == 4 && coeffs.scalar_type() == at::kInt,
                        "compressed SH expects [M, 4] int32 tensor, got shape ", coeffs.sizes());
            TORCH_CHECK(colors.scalar_type() == at::kHalf, "compressed SH requires fp16 output, got ",
                        colors.scalar_type());
            const int64_t expected_rows = (mode == SHCompressionMode::COMPRESS_32B) ? 2 * N : N;
            TORCH_CHECK(coeffs.size(0) == expected_rows, "compressed SH packed tensor row count mismatch: expected ",
                        expected_rows, ", got ", coeffs.size(0));
        }

        const dim3 threads(SHEVAL_CTA_SIZE);
        const dim3 grid((N + SHEVAL_CTA_SIZE - 1) / SHEVAL_CTA_SIZE);
        const SHDecodeParams dp = decode_params ? *decode_params : SHDecodeParams{};

        if (mode == SHCompressionMode::COMPRESS_32B)
        {
            spherical_harmonics_fwd_kernel_k16<SHInputMode::COMPRESS_32B>
                <<<grid, threads, 0, stream>>>(N, degrees_to_use, means_ptr, cam_x, cam_y, cam_z,
                                               coeffs.data_ptr<int32_t>(), masks_ptr, bias, min_value, dp, colors_ptr);
        }
        else if (mode == SHCompressionMode::COMPRESS_16B)
        {
            spherical_harmonics_fwd_kernel_k16<SHInputMode::COMPRESS_16B>
                <<<grid, threads, 0, stream>>>(N, degrees_to_use, means_ptr, cam_x, cam_y, cam_z,
                                               coeffs.data_ptr<int32_t>(), masks_ptr, bias, min_value, dp, colors_ptr);
        }
        else if (coeffs.scalar_type() == at::kFloat)
        {
            spherical_harmonics_fwd_kernel_k16<SHInputMode::RAW_FLOAT>
                <<<grid, threads, 0, stream>>>(N, degrees_to_use, means_ptr, cam_x, cam_y, cam_z,
                                               coeffs.data_ptr<float>(), masks_ptr, bias, min_value, dp, colors_ptr);
        }
        else if (coeffs.scalar_type() == at::kHalf)
        {
            spherical_harmonics_fwd_kernel_k16<SHInputMode::RAW_HALF>
                <<<grid, threads, 0, stream>>>(N, degrees_to_use, means_ptr, cam_x, cam_y, cam_z,
                                               coeffs.data_ptr<at::Half>(), masks_ptr, bias, min_value, dp, colors_ptr);
        }
        else
        {
            TORCH_CHECK(false, "spherical_harmonics_fwd_kernel_k16: unsupported dtype ", coeffs.scalar_type());
        }
    }
    else
    {
        TORCH_CHECK(coeffs.scalar_type() == at::kFloat,
                    "spherical_harmonics_fwd_kernel: generic path only supports float, got ", coeffs.scalar_type());
        const int32_t n_elements = N * 3;
        const dim3 threads(256);
        const dim3 grid((n_elements + threads.x - 1) / threads.x);
        spherical_harmonics_fwd_kernel<<<grid, threads, 0, stream>>>(N, K, degrees_to_use, means_ptr, cam_x, cam_y,
                                                                     cam_z, coeffs.data_ptr<float>(), masks_ptr, bias,
                                                                     min_value, colors_ptr);
    }
}

} // namespace higs
