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

// CUDA kernel wrappers for SH compression encode/decode (32B and 16B modes).
// The actual encode/decode device functions live in SHCompression.h.
//
// 32B packed memory layout: [2, N] uint4 — two 16-byte blocks per gaussian,
// stored as block0 for all N gaussians followed by block1 for all N.
// 16B packed memory layout: [N] uint4 — one 16-byte block per gaussian.
// Both layouts ensure fully coalesced 16-byte warp transactions.

#include "SHCommon.h"
#include "SHCompression.h"

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace higs {

static constexpr int CODEC_CTA_SIZE   = 256;
static constexpr int CODEC_MIN_BLOCKS = 4;

template<SHCompressionMode MODE, typename T>
__global__ void __launch_bounds__(CODEC_CTA_SIZE, CODEC_MIN_BLOCKS)
    sh_encode_kernel(int32_t N, const T *__restrict__ coeffs_in, uint4 *__restrict__ packed_out,
                     SHEncodeParams encode_params)
{
    const int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }

    // Load coefficients into channel-first local array [3][16]
    // Input layout: [N, 16, 3] T → stride: gaussian=48, basis=3, channel=1
    T coeffs[3][16];
    const T *src = coeffs_in + (int64_t)idx * 48;
#pragma unroll
    for (int k = 0; k < 16; k++)
    {
        coeffs[0][k] = src[k * 3 + 0];
        coeffs[1][k] = src[k * 3 + 1];
        coeffs[2][k] = src[k * 3 + 2];
    }

    if constexpr (MODE == SHCompressionMode::COMPRESS_32B)
    {
        uint32_t block[8];
        sh_encode_32b(coeffs, block, encode_params.inv_scales);

        // store as two uint4 at stride N for coalesced warp access
        uint4 b0, b1;
        AssignAs<uint4>(b0, block[0]);
        AssignAs<uint4>(b1, block[4]);
        packed_out[idx]              = b0;
        packed_out[(int64_t)N + idx] = b1;
    }
    else
    {
        uint32_t block[4];
        sh_encode_16b(coeffs, block, encode_params.inv_scales);

        // store one uint4 per gaussian
        uint4 b;
        AssignAs<uint4>(b, block[0]);
        packed_out[idx] = b;
    }
}

// Unified host functions

at::Tensor launch_sh_encode(const at::Tensor &coeffs, const SHEncodeParams &encode_params, SHCompressionMode mode)
{
    TORCH_CHECK(coeffs.dim() == 3 && coeffs.size(1) == 16 && coeffs.size(2) == 3, "coeffs must be [N, 16, 3]");
    TORCH_CHECK(coeffs.scalar_type() == at::kFloat || coeffs.scalar_type() == at::kHalf,
                "coeffs must be float32 or float16");
    TORCH_CHECK(coeffs.is_cuda() && coeffs.is_contiguous());

    const int32_t N   = static_cast<int32_t>(coeffs.size(0));
    const bool is_32b = (mode == SHCompressionMode::COMPRESS_32B);
    auto packed       = at::empty({is_32b ? 2 * N : N, 4}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));

    if (N > 0)
    {
        dim3 threads(CODEC_CTA_SIZE);
        dim3 blocks((N + CODEC_CTA_SIZE - 1) / CODEC_CTA_SIZE);
        auto stream = at::cuda::getCurrentCUDAStream();
        auto *dst   = reinterpret_cast<uint4 *>(packed.data_ptr<int32_t>());

        auto launch = [&](auto mode_tag) {
            constexpr SHCompressionMode MODE = decltype(mode_tag)::value;
            if (coeffs.scalar_type() == at::kFloat)
            {
                sh_encode_kernel<MODE, float>
                    <<<blocks, threads, 0, stream>>>(N, coeffs.data_ptr<float>(), dst, encode_params);
            }
            else
            {
                sh_encode_kernel<MODE, __half><<<blocks, threads, 0, stream>>>(
                    N, reinterpret_cast<const __half *>(coeffs.data_ptr<at::Half>()), dst, encode_params);
            }
        };

        if (is_32b)
        {
            launch(std::integral_constant<SHCompressionMode, SHCompressionMode::COMPRESS_32B>{});
        }
        else
        {
            launch(std::integral_constant<SHCompressionMode, SHCompressionMode::COMPRESS_16B>{});
        }
    }

    return packed;
}

// Full compression pipeline: fp32 coefficients → compressed + scales.
// Computes per-basis-per-channel p99.99 scales in YCoCg space (excluding low-opacity
// gaussians from scale computation), encodes all gaussians, and returns the packed
// bitstream + decoder scale structure.
SHCompressed launch_sh_compress(const at::Tensor &coeffs, const at::Tensor &opacities, SHCompressionMode mode)
{
    TORCH_CHECK(coeffs.dim() == 3 && coeffs.size(1) == 16 && coeffs.size(2) == 3);
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == coeffs.size(0));
    TORCH_CHECK(coeffs.is_cuda() && opacities.is_cuda());

    const int64_t N   = coeffs.size(0);
    const bool is_32b = (mode == SHCompressionMode::COMPRESS_32B);

    // Convert to half for the non-compressed render path
    auto coeffs_half = coeffs.to(at::kHalf).contiguous();

    SHEncodeParams encode_params{};
    float scales_f[3][16]{}; // [channel][basis] — raw scales for building SHDecodeParams

    // Need to use scope to free unused tensors early
    {
        // Extract HO in float, convert to YCoCg for scale computation
        at::Tensor Y, Co, Cg;
        {
            auto ho = coeffs_half.narrow(1, 1, 15).to(at::kFloat); // [N, 15, 3]
            auto R  = ho.select(2, 0);
            auto G  = ho.select(2, 1);
            auto B  = ho.select(2, 2);
            Y       = R.mul(0.25f).add(G.mul(0.5f)).add(B.mul(0.25f));
            Co      = R.mul(0.5f).sub(B.mul(0.5f));
            Cg      = R.mul(-0.25f).add(G.mul(0.5f)).add(B.mul(-0.25f));
        }

        // Filter to visible gaussians (opacity >= 0.5%) for scale computation.
        // Low-opacity outliers don't visibly affect rendering but can inflate scales.
        constexpr float OPACITY_THRESHOLD = 0.005f;
        auto visible_mask                 = opacities.to(at::kFloat).ge(OPACITY_THRESHOLD);   // [N] bool
        auto vis_indices                  = visible_mask.nonzero().squeeze(1); // [M] long

        // Compute per-basis p99.99 scales from visible gaussians only
        constexpr float PERCENTILE = 0.9999f;

        // DC slots: 1.0 (unused by encoder's DC path, but well-defined)
        encode_params.inv_scales[0][0] = encode_params.inv_scales[1][0] = encode_params.inv_scales[2][0] = 1.0f;
        scales_f[0][0] = scales_f[1][0] = scales_f[2][0] = 1.0f;

        at::Tensor ycocg_channels[] = {Y, Co, Cg};
        float *inv_arrays[] = {encode_params.inv_scales[0], encode_params.inv_scales[1], encode_params.inv_scales[2]};
        int64_t M           = vis_indices.size(0);

        // When all Gaussians are below the opacity threshold, fall back to
        // unit scales so the encoder/decoder round-trips zeros cleanly.
        if (M == 0) {
            for (int ch = 0; ch < 3; ch++) {
                for (int j = 0; j < 15; j++) {
                    scales_f[ch][j + 1] = 1.0f;
                    encode_params.inv_scales[ch][j + 1] = 1.0f;
                }
            }
        } else {
            // For 16B mode, only Y channel scales are needed (CoCg is dropped)
            const int n_channels = is_32b ? 3 : 1;

            for (int ch = 0; ch < n_channels; ch++)
            {
                for (int j = 0; j < 15; j++)
                {
                    // Select only visible gaussians for this basis/channel
                    auto col        = ycocg_channels[ch].select(1, j).index_select(0, vis_indices); // [M]
                    auto abs_sorted = std::get<0>(col.abs().sort());
                    int64_t idx     = std::min((int64_t)(PERCENTILE * M), M - 1);
                    float scale     = abs_sorted[idx].item<float>();

                    scales_f[ch][j + 1] = scale;
                    // If scale is zero (all coefficients zero for this basis/channel), set reciprocal
                    // to zero so val * 0 = 0 quantizes to the midpoint. Avoids inf/NaN.
                    inv_arrays[ch][j + 1] = (scale > 0.f) ? 1.0f / scale : 0.f;
                }
            }
        }
    }

    // Encode
    auto packed = launch_sh_encode(coeffs_half, encode_params, mode);

    // build decoder params: pack per-basis scales to half2
    SHDecodeParams decode_params{};
    for (int p = 0; p < 8; p++)
    {
        int k_lo                   = p * 2;
        int k_hi                   = p * 2 + 1;
        decode_params.scales[0][p] = __floats2half2_rn(scales_f[0][k_lo], scales_f[0][k_hi]);
        decode_params.scales[1][p] = __floats2half2_rn(scales_f[1][k_lo], scales_f[1][k_hi]);
        decode_params.scales[2][p] = __floats2half2_rn(scales_f[2][k_lo], scales_f[2][k_hi]);
    }

    return {packed, decode_params};
}

} // namespace higs
