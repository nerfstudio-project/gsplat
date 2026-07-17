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

#    include <ATen/core/Tensor.h>
#    include <bitset>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>
#    include <cmath>
#    include <cooperative_groups.h>

#    include "LossFlags.h"

namespace gsplat
{
namespace
{
    constexpr float SSIM_SIGMA          = 1.5f;
    constexpr int SSIM_BX               = 32;
    constexpr int SSIM_BY               = 32;
    constexpr int SSIM_MAX_CUDA_DEVICES = 64;
    constexpr int SSIM_RADIUS           = 5;
    constexpr int SSIM_SX               = SSIM_BX + 2 * SSIM_RADIUS;
    constexpr int SSIM_SY               = SSIM_BY + 2 * SSIM_RADIUS;
    constexpr int SSIM_CX               = SSIM_BX;
    constexpr int SSIM_CY               = SSIM_BY + 2 * SSIM_RADIUS;
    constexpr float SSIM_C1             = 0.01f * 0.01f;
    constexpr float SSIM_C2             = 0.03f * 0.03f;

    namespace cg = cooperative_groups;

    inline uint32_t div_round_up(uint32_t val, uint32_t divisor)
    {
        return (val + divisor - 1) / divisor;
    }

    __device__ __forceinline__ bool ssim_flag_valid(int32_t flags)
    {
        return (flags & GSPLAT_LOSS_FLAG_INVALID) == 0;
    }

    __device__ void load_blended_pair_into_shared(
        float pred_sh[SSIM_SY][SSIM_SX],
        float target_sh[SSIM_SY][SSIM_SX],
        const float *__restrict__ pred,
        const float *__restrict__ target,
        const int32_t *__restrict__ flags,
        int H,
        int W,
        int C,
        int c,
        bool mask_mode_target,
        float constant_mask_value
    )
    {
        auto block        = cg::this_thread_block();
        const int batch   = block.group_index().z;
        const int start_y = block.group_index().y * SSIM_BY;
        const int start_x = block.group_index().x * SSIM_BX;

        const int cnt        = SSIM_SY * SSIM_SX;
        const int num_blocks = (cnt + SSIM_BX * SSIM_BY - 1) / (SSIM_BX * SSIM_BY);
        for(int b = 0; b < num_blocks; ++b)
        {
            int tid = b * (SSIM_BX * SSIM_BY) + block.thread_rank();
            if(tid < cnt)
            {
                int local_y = tid / SSIM_SX;
                int local_x = tid % SSIM_SX;
                int y       = start_y + local_y - SSIM_RADIUS;
                int x       = start_x + local_x - SSIM_RADIUS;

                float pred_v   = 0.0f;
                float target_v = 0.0f;
                if(x >= 0 && y >= 0 && x < W && y < H)
                {
                    int pixel = (batch * H + y) * W + x;
                    pred_v    = pred[pixel * C + c];
                    target_v  = target[pixel * C + c];
                    if(!ssim_flag_valid(flags[pixel]))
                    {
                        float mv = mask_mode_target ? target_v : constant_mask_value;
                        pred_v   = mv;
                        target_v = mv;
                    }
                }
                pred_sh[local_y][local_x]   = pred_v;
                target_sh[local_y][local_x] = target_v;
            }
        }
    }

    __device__ void load_channel_into_shared(
        float sh[SSIM_SY][SSIM_SX],
        const float *__restrict__ buf_bchw,
        int H,
        int W,
        int C,
        int c,
        float dmap[SSIM_SY][SSIM_SX] = nullptr
    )
    {
        auto block        = cg::this_thread_block();
        const int batch   = block.group_index().z;
        const int start_y = block.group_index().y * SSIM_BY;
        const int start_x = block.group_index().x * SSIM_BX;

        const int cnt        = SSIM_SY * SSIM_SX;
        const int num_blocks = (cnt + SSIM_BX * SSIM_BY - 1) / (SSIM_BX * SSIM_BY);
        for(int b = 0; b < num_blocks; ++b)
        {
            int tid = b * (SSIM_BX * SSIM_BY) + block.thread_rank();
            if(tid < cnt)
            {
                int local_y = tid / SSIM_SX;
                int local_x = tid % SSIM_SX;
                int y       = start_y + local_y - SSIM_RADIUS;
                int x       = start_x + local_x - SSIM_RADIUS;

                float val = 0.0f;
                if(x >= 0 && y >= 0 && x < W && y < H)
                {
                    val = buf_bchw[((batch * C + c) * H + y) * W + x];
                }
                sh[local_y][local_x] = dmap ? val * dmap[local_y][local_x] : val;
            }
        }
    }

    __device__ void load_dLdmap_into_shared(
        float sh[SSIM_SY][SSIM_SX],
        const float *__restrict__ v_loss,
        const int32_t *__restrict__ flags,
        int H,
        int W,
        float factor,
        float neg_inv_C
    )
    {
        auto block        = cg::this_thread_block();
        const int batch   = block.group_index().z;
        const int start_y = block.group_index().y * SSIM_BY;
        const int start_x = block.group_index().x * SSIM_BX;

        const int cnt        = SSIM_SY * SSIM_SX;
        const int num_blocks = (cnt + SSIM_BX * SSIM_BY - 1) / (SSIM_BX * SSIM_BY);
        for(int b = 0; b < num_blocks; ++b)
        {
            int tid = b * (SSIM_BX * SSIM_BY) + block.thread_rank();
            if(tid < cnt)
            {
                int local_y = tid / SSIM_SX;
                int local_x = tid % SSIM_SX;
                int y       = start_y + local_y - SSIM_RADIUS;
                int x       = start_x + local_x - SSIM_RADIUS;

                float val = 0.0f;
                if(x >= 0 && y >= 0 && x < W && y < H)
                {
                    int pixel = (batch * H + y) * W + x;
                    if(ssim_flag_valid(flags[pixel]))
                    {
                        val = v_loss[pixel] * factor * neg_inv_C;
                    }
                }
                sh[local_y][local_x] = val;
            }
        }
    }

    __constant__ float c_ssim_gauss[SSIM_RADIUS + 1];

    void ensure_ssim_gauss()
    {
        int device = 0;
        C10_CUDA_CHECK(cudaGetDevice(&device));
        static std::bitset<SSIM_MAX_CUDA_DEVICES> uploaded;
        if(device >= 0 && device < SSIM_MAX_CUDA_DEVICES && uploaded.test(device))
        {
            return;
        }

        double w[SSIM_RADIUS + 1];
        double sum = 0.0;
        for(int d = 0; d <= SSIM_RADIUS; ++d)
        {
            w[d]  = std::exp(-static_cast<double>(d * d) / (2.0 * SSIM_SIGMA * SSIM_SIGMA));
            sum  += (d == 0) ? w[d] : 2.0 * w[d];
        }
        float wf[SSIM_RADIUS + 1];
        for(int d = 0; d <= SSIM_RADIUS; ++d)
        {
            wf[d] = static_cast<float>(w[d] / sum);
        }
        C10_CUDA_CHECK(cudaMemcpyToSymbol(c_ssim_gauss, wf, sizeof(wf)));
        if(device >= 0 && device < SSIM_MAX_CUDA_DEVICES)
        {
            uploaded.set(device);
        }
    }

    template<bool SQ>
    __device__ __forceinline__ float ssim_sym_tap(float v)
    {
        return SQ ? v * v : v;
    }

    template<bool SQ>
    __device__ __forceinline__ float conv_x_sym_row(float pixels[SSIM_SY][SSIM_SX], int row, int tx)
    {
        float acc = c_ssim_gauss[0] * ssim_sym_tap<SQ>(pixels[row][tx + SSIM_RADIUS]);
#    pragma unroll
        for(int d = 1; d <= SSIM_RADIUS; ++d)
        {
            acc += c_ssim_gauss[d]
                 * (ssim_sym_tap<SQ>(pixels[row][tx + SSIM_RADIUS - d])
                    + ssim_sym_tap<SQ>(pixels[row][tx + SSIM_RADIUS + d]));
        }
        return acc;
    }

    template<bool SQ>
    __device__ void conv_x_clean(float pixels[SSIM_SY][SSIM_SX], float opt[SSIM_CY][SSIM_CX])
    {
        auto block   = cg::this_thread_block();
        const int tx = block.thread_index().x;
        int row      = block.thread_index().y;
        opt[row][tx] = conv_x_sym_row<SQ>(pixels, row, tx);
        row          = block.thread_index().y + SSIM_BY;
        if(row < SSIM_CY)
        {
            opt[row][tx] = conv_x_sym_row<SQ>(pixels, row, tx);
        }
    }

    __device__ void conv_x_prod_clean(float a[SSIM_SY][SSIM_SX], float b[SSIM_SY][SSIM_SX], float opt[SSIM_CY][SSIM_CX])
    {
        auto block   = cg::this_thread_block();
        const int tx = block.thread_index().x;
        for(int pass = 0; pass < 2; ++pass)
        {
            int row = block.thread_index().y + pass * SSIM_BY;
            if(row >= SSIM_CY)
            {
                continue;
            }
            float acc = c_ssim_gauss[0] * (a[row][tx + SSIM_RADIUS] * b[row][tx + SSIM_RADIUS]);
#    pragma unroll
            for(int d = 1; d <= SSIM_RADIUS; ++d)
            {
                const int l  = tx + SSIM_RADIUS - d;
                const int r  = tx + SSIM_RADIUS + d;
                acc         += c_ssim_gauss[d] * (a[row][l] * b[row][l] + a[row][r] * b[row][r]);
            }
            opt[row][tx] = acc;
        }
    }

    __device__ __forceinline__ float conv_y_clean(float opt[SSIM_CY][SSIM_CX])
    {
        auto block   = cg::this_thread_block();
        const int tx = block.thread_index().x;
        const int ty = block.thread_index().y;
        float acc    = c_ssim_gauss[0] * opt[ty + SSIM_RADIUS][tx];
#    pragma unroll
        for(int d = 1; d <= SSIM_RADIUS; ++d)
        {
            acc += c_ssim_gauss[d] * (opt[ty + SSIM_RADIUS - d][tx] + opt[ty + SSIM_RADIUS + d][tx]);
        }
        return acc;
    }

    __global__ void ssim_losses_fwd_kernel(
        int H,
        int W,
        int C,
        bool mask_mode_target,
        float constant_mask_value,
        float factor,
        const int32_t *__restrict__ flags,
        const float *__restrict__ pred,
        const float *__restrict__ target,
        float *__restrict__ loss,
        float *__restrict__ dm_dmu1,
        float *__restrict__ dm_dsigma1_sq,
        float *__restrict__ dm_dsigma12
    )
    {
        auto block           = cg::this_thread_block();
        const int pix_y      = block.group_index().y * SSIM_BY + block.thread_index().y;
        const int pix_x      = block.group_index().x * SSIM_BX + block.thread_index().x;
        const int batch      = block.group_index().z;
        const bool in_bounds = (pix_x < W && pix_y < H);

        __shared__ float pred_sh[SSIM_SY][SSIM_SX];
        __shared__ float target_sh[SSIM_SY][SSIM_SX];
        __shared__ float cx_mu1[SSIM_CY][SSIM_CX];
        __shared__ float cx_sigma1_sq[SSIM_CY][SSIM_CX];
        __shared__ float cx_mu2[SSIM_CY][SSIM_CX];
        __shared__ float cx_sigma2_sq[SSIM_CY][SSIM_CX];
        __shared__ float cx_sigma12[SSIM_CY][SSIM_CX];

        float ssim_accum = 0.0f;
        for(int c = 0; c < C; ++c)
        {
            load_blended_pair_into_shared(
                pred_sh, target_sh, pred, target, flags, H, W, C, c, mask_mode_target, constant_mask_value
            );
            block.sync();

            conv_x_clean<false>(pred_sh, cx_mu1);
            conv_x_clean<true>(pred_sh, cx_sigma1_sq);
            conv_x_clean<false>(target_sh, cx_mu2);
            conv_x_clean<true>(target_sh, cx_sigma2_sq);
            conv_x_prod_clean(pred_sh, target_sh, cx_sigma12);
            block.sync();

            float mu1       = conv_y_clean(cx_mu1);
            float mu2       = conv_y_clean(cx_mu2);
            float sigma1_sq = conv_y_clean(cx_sigma1_sq) - mu1 * mu1;
            float sigma2_sq = conv_y_clean(cx_sigma2_sq) - mu2 * mu2;
            float sigma12   = conv_y_clean(cx_sigma12) - mu1 * mu2;
            block.sync();

            const float mu1_sq  = mu1 * mu1;
            const float mu2_sq  = mu2 * mu2;
            const float mu1_mu2 = mu1 * mu2;
            const float Cc      = 2.0f * mu1_mu2 + SSIM_C1;
            const float Dd      = 2.0f * sigma12 + SSIM_C2;
            const float Aa      = mu1_sq + mu2_sq + SSIM_C1;
            const float Bb      = sigma1_sq + sigma2_sq + SSIM_C2;
            const float m       = (Cc * Dd) / (Aa * Bb);

            if(in_bounds)
            {
                ssim_accum             += m;
                const int out_idx       = ((batch * C + c) * H + pix_y) * W + pix_x;
                dm_dmu1[out_idx]        = (mu2 * 2.0f * Dd) / (Aa * Bb)
                                        - (mu2 * 2.0f * Cc) / (Aa * Bb)
                                        - (mu1 * 2.0f * Cc * Dd) / (Aa * Aa * Bb)
                                        + (mu1 * 2.0f * Cc * Dd) / (Aa * Bb * Bb);
                dm_dsigma1_sq[out_idx]  = (-Cc * Dd) / (Aa * Bb * Bb);
                dm_dsigma12[out_idx]    = (2.0f * Cc) / (Aa * Bb);
            }
        }

        if(in_bounds)
        {
            const int pixel       = (batch * H + pix_y) * W + pix_x;
            const bool valid      = ssim_flag_valid(flags[pixel]);
            const float ssim_mean = ssim_accum / static_cast<float>(C);
            loss[pixel]           = valid ? (1.0f - ssim_mean) * factor : 0.0f;
        }
    }

    __global__ void ssim_losses_bwd_kernel(
        int H,
        int W,
        int C,
        bool mask_mode_target,
        float constant_mask_value,
        float factor,
        const int32_t *__restrict__ flags,
        const float *__restrict__ pred,
        const float *__restrict__ target,
        const float *__restrict__ v_loss,
        const float *__restrict__ dm_dmu1,
        const float *__restrict__ dm_dsigma1_sq,
        const float *__restrict__ dm_dsigma12,
        float *__restrict__ v_pred
    )
    {
        auto block            = cg::this_thread_block();
        const int pix_y       = block.group_index().y * SSIM_BY + block.thread_index().y;
        const int pix_x       = block.group_index().x * SSIM_BX + block.thread_index().x;
        const int batch       = block.group_index().z;
        const bool in_bounds  = (pix_x < W && pix_y < H);
        const float neg_inv_C = -1.0f / static_cast<float>(C);

        __shared__ float dmap_sh[SSIM_SY][SSIM_SX];
        __shared__ float aux_sh[SSIM_SY][SSIM_SX];
        __shared__ float conv_sh[SSIM_CY][SSIM_CX];

        const int pixel         = (batch * H + pix_y) * W + pix_x;
        const bool center_valid = in_bounds && ssim_flag_valid(flags[pixel]);

        load_dLdmap_into_shared(dmap_sh, v_loss, flags, H, W, factor, neg_inv_C);
        block.sync();

        for(int c = 0; c < C; ++c)
        {
            float pix1 = 0.0f;
            float pix2 = 0.0f;
            if(in_bounds)
            {
                pix1 = pred[pixel * C + c];
                pix2 = target[pixel * C + c];
                if(!center_valid)
                {
                    const float mv = mask_mode_target ? pix2 : constant_mask_value;
                    pix1           = mv;
                    pix2           = mv;
                }
            }

            float dL_dpix = 0.0f;
            load_channel_into_shared(aux_sh, dm_dmu1, H, W, C, c, dmap_sh);
            block.sync();
            conv_x_clean<false>(aux_sh, conv_sh);
            block.sync();
            dL_dpix += conv_y_clean(conv_sh);

            load_channel_into_shared(aux_sh, dm_dsigma1_sq, H, W, C, c, dmap_sh);
            block.sync();
            conv_x_clean<false>(aux_sh, conv_sh);
            block.sync();
            dL_dpix += pix1 * 2.0f * conv_y_clean(conv_sh);

            load_channel_into_shared(aux_sh, dm_dsigma12, H, W, C, c, dmap_sh);
            block.sync();
            conv_x_clean<false>(aux_sh, conv_sh);
            block.sync();
            dL_dpix += pix2 * conv_y_clean(conv_sh);

            if(in_bounds)
            {
                if(!center_valid)
                {
                    dL_dpix = 0.0f;
                }
                v_pred[pixel * C + c] = dL_dpix;
            }
        }
    }
} // namespace

void launch_ssim_losses_fwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &pred,
    const at::Tensor &target,
    float factor,
    bool mask_mode_target,
    float constant_mask_value,
    at::Tensor &loss,
    at::Tensor &dm_dmu1,
    at::Tensor &dm_dsigma1_sq,
    at::Tensor &dm_dsigma12
)
{
    const int B = pred.size(0);
    const int H = pred.size(1);
    const int W = pred.size(2);
    const int C = pred.size(3);
    if(B == 0 || H == 0 || W == 0)
    {
        return;
    }

    ensure_ssim_gauss();
    dim3 threads(SSIM_BX, SSIM_BY, 1);
    dim3 blocks(div_round_up(W, SSIM_BX), div_round_up(H, SSIM_BY), B);
    ssim_losses_fwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        H,
        W,
        C,
        mask_mode_target,
        constant_mask_value,
        factor,
        flags.data_ptr<int32_t>(),
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        loss.data_ptr<float>(),
        dm_dmu1.data_ptr<float>(),
        dm_dsigma1_sq.data_ptr<float>(),
        dm_dsigma12.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_ssim_losses_bwd_kernel(
    const at::Tensor &flags,
    const at::Tensor &pred,
    const at::Tensor &target,
    float factor,
    bool mask_mode_target,
    float constant_mask_value,
    const at::Tensor &v_loss,
    const at::Tensor &dm_dmu1,
    const at::Tensor &dm_dsigma1_sq,
    const at::Tensor &dm_dsigma12,
    at::Tensor &v_pred
)
{
    const int B = pred.size(0);
    const int H = pred.size(1);
    const int W = pred.size(2);
    const int C = pred.size(3);
    if(B == 0 || H == 0 || W == 0)
    {
        return;
    }

    ensure_ssim_gauss();
    dim3 threads(SSIM_BX, SSIM_BY, 1);
    dim3 blocks(div_round_up(W, SSIM_BX), div_round_up(H, SSIM_BY), B);
    ssim_losses_bwd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        H,
        W,
        C,
        mask_mode_target,
        constant_mask_value,
        factor,
        flags.data_ptr<int32_t>(),
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        v_loss.data_ptr<float>(),
        dm_dmu1.data_ptr<float>(),
        dm_dsigma1_sq.data_ptr<float>(),
        dm_dsigma12.data_ptr<float>(),
        v_pred.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace gsplat

#endif
