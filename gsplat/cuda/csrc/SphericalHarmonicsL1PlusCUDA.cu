/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <array>
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template<typename scalar_t>
__device__ void sh_l1_plus_coeffs_to_color_fast(
    const uint32_t degree,  // degree of SH to be evaluated
    const uint32_t D,       // channels (coeffs last dim)
    const uint32_t c,       // output channel
    const vec3 &dir,        // [3]
    const scalar_t *coeffs, // [K - 1, D]
    // output
    at::opmath_type<scalar_t> *colors // [D]
)
{
    using opmath_t  = at::opmath_type<scalar_t>;
    opmath_t result = 0;
    if(degree >= 1)
    {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x     = dir.x * inorm;
        float y     = dir.y * inorm;
        float z     = dir.z * inorm;

        result += 0.48860251190292f
                * (-y * static_cast<opmath_t>(coeffs[c])
                   + z * static_cast<opmath_t>(coeffs[1 * D + c])
                   - x * static_cast<opmath_t>(coeffs[2 * D + c]));
        if(degree >= 2)
        {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1    = x * x - y * y;
            float fS1    = 2.f * x * y;
            float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7   = fTmp0B * x;
            float pSH5   = fTmp0B * y;
            float pSH8   = 0.5462742152960395f * fC1;
            float pSH4   = 0.5462742152960395f * fS1;

            result += pSH4 * static_cast<opmath_t>(coeffs[3 * D + c])
                    + pSH5 * static_cast<opmath_t>(coeffs[4 * D + c])
                    + pSH6 * static_cast<opmath_t>(coeffs[5 * D + c])
                    + pSH7 * static_cast<opmath_t>(coeffs[6 * D + c])
                    + pSH8 * static_cast<opmath_t>(coeffs[7 * D + c]);
            if(degree >= 3)
            {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2    = x * fC1 - y * fS1;
                float fS2    = x * fS1 + y * fC1;
                float pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13  = fTmp0C * x;
                float pSH11  = fTmp0C * y;
                float pSH14  = fTmp1B * fC1;
                float pSH10  = fTmp1B * fS1;
                float pSH15  = -0.5900435899266435f * fC2;
                float pSH9   = -0.5900435899266435f * fS2;

                result += pSH9 * static_cast<opmath_t>(coeffs[8 * D + c])
                        + pSH10 * static_cast<opmath_t>(coeffs[9 * D + c])
                        + pSH11 * static_cast<opmath_t>(coeffs[10 * D + c])
                        + pSH12 * static_cast<opmath_t>(coeffs[11 * D + c])
                        + pSH13 * static_cast<opmath_t>(coeffs[12 * D + c])
                        + pSH14 * static_cast<opmath_t>(coeffs[13 * D + c])
                        + pSH15 * static_cast<opmath_t>(coeffs[14 * D + c]);

                if(degree >= 4)
                {
                    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3    = x * fC2 - y * fS2;
                    float fS3    = x * fS2 + y * fC2;
                    float pSH20  = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    float pSH21  = fTmp0D * x;
                    float pSH19  = fTmp0D * y;
                    float pSH22  = fTmp1C * fC1;
                    float pSH18  = fTmp1C * fS1;
                    float pSH23  = fTmp2B * fC2;
                    float pSH17  = fTmp2B * fS2;
                    float pSH24  = 0.6258357354491763f * fC3;
                    float pSH16  = 0.6258357354491763f * fS3;

                    result += pSH16 * static_cast<opmath_t>(coeffs[15 * D + c])
                            + pSH17 * static_cast<opmath_t>(coeffs[16 * D + c])
                            + pSH18 * static_cast<opmath_t>(coeffs[17 * D + c])
                            + pSH19 * static_cast<opmath_t>(coeffs[18 * D + c])
                            + pSH20 * static_cast<opmath_t>(coeffs[19 * D + c])
                            + pSH21 * static_cast<opmath_t>(coeffs[20 * D + c])
                            + pSH22 * static_cast<opmath_t>(coeffs[21 * D + c])
                            + pSH23 * static_cast<opmath_t>(coeffs[22 * D + c])
                            + pSH24 * static_cast<opmath_t>(coeffs[23 * D + c]);
                }
            }
        }
    }

    colors[c] = result;
}

template<typename scalar_t>
__device__ void sh_l1_plus_coeffs_to_color_fast_vjp(
    const uint32_t degree,                     // degree of SH to be evaluated
    const uint32_t D,                          // channels (coeffs last dim)
    const uint32_t c,                          // output channel
    const vec3 &dir,                           // [3]
    const scalar_t *coeffs,                    // [K - 1, D]
    const at::opmath_type<scalar_t> *v_colors, // [D]
    // output
    // acc is a per-thread fp32 accumulator indexed by coefficient k (channel c is
    // fixed for this thread). The caller sums contributions from every batch here
    // in registers and writes the result to global memory once (casting to the
    // coeff dtype), so no global atomics or fp32 scratch buffer are needed.
    at::opmath_type<scalar_t> *acc, // [MAX_K]
    vec3 *v_dir                     // [3] optional
)
{
    using opmath_t          = at::opmath_type<scalar_t>;
    opmath_t v_colors_local = v_colors[c];

    if(degree < 1)
    {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x     = dir.x * inorm;
    float y     = dir.y * inorm;
    float z     = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    acc[0] += -0.48860251190292f * y * v_colors_local;
    acc[1] += 0.48860251190292f * z * v_colors_local;
    acc[2] += -0.48860251190292f * x * v_colors_local;

    if(v_dir != nullptr)
    {
        v_x += -0.48860251190292f * static_cast<opmath_t>(coeffs[2 * D + c]) * v_colors_local;
        v_y += -0.48860251190292f * static_cast<opmath_t>(coeffs[c]) * v_colors_local;
        v_z += 0.48860251190292f * static_cast<opmath_t>(coeffs[1 * D + c]) * v_colors_local;
    }
    if(degree < 2)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2      = z * z;
    float fTmp0B  = -1.092548430592079f * z;
    float fC1     = x * x - y * y;
    float fS1     = 2.f * x * y;
    float pSH6    = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7    = fTmp0B * x;
    float pSH5    = fTmp0B * y;
    float pSH8    = 0.5462742152960395f * fC1;
    float pSH4    = 0.5462742152960395f * fS1;
    acc[3]       += pSH4 * v_colors_local;
    acc[4]       += pSH5 * v_colors_local;
    acc[5]       += pSH6 * v_colors_local;
    acc[6]       += pSH7 * v_colors_local;
    acc[7]       += pSH8 * v_colors_local;

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if(v_dir != nullptr)
    {
        fTmp0B_z = -1.092548430592079f;
        fC1_x    = 2.f * x;
        fC1_y    = -2.f * y;
        fS1_x    = 2.f * y;
        fS1_y    = 2.f * x;
        pSH6_z   = 2.f * 0.9461746957575601f * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = 0.5462742152960395f * fC1_x;
        pSH8_y   = 0.5462742152960395f * fC1_y;
        pSH4_x   = 0.5462742152960395f * fS1_x;
        pSH4_y   = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local
             * (pSH4_x * static_cast<opmath_t>(coeffs[3 * D + c])
                + pSH8_x * static_cast<opmath_t>(coeffs[7 * D + c])
                + pSH7_x * static_cast<opmath_t>(coeffs[6 * D + c]));
        v_y += v_colors_local
             * (pSH4_y * static_cast<opmath_t>(coeffs[3 * D + c])
                + pSH8_y * static_cast<opmath_t>(coeffs[7 * D + c])
                + pSH5_y * static_cast<opmath_t>(coeffs[4 * D + c]));
        v_z += v_colors_local
             * (pSH6_z * static_cast<opmath_t>(coeffs[5 * D + c])
                + pSH7_z * static_cast<opmath_t>(coeffs[6 * D + c])
                + pSH5_z * static_cast<opmath_t>(coeffs[4 * D + c]));
    }

    if(degree < 3)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0C  = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B  = 1.445305721320277f * z;
    float fC2     = x * fC1 - y * fS1;
    float fS2     = x * fS1 + y * fC1;
    float pSH12   = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13   = fTmp0C * x;
    float pSH11   = fTmp0C * y;
    float pSH14   = fTmp1B * fC1;
    float pSH10   = fTmp1B * fS1;
    float pSH15   = -0.5900435899266435f * fC2;
    float pSH9    = -0.5900435899266435f * fS2;
    acc[8]       += pSH9 * v_colors_local;
    acc[9]       += pSH10 * v_colors_local;
    acc[10]      += pSH11 * v_colors_local;
    acc[11]      += pSH12 * v_colors_local;
    acc[12]      += pSH13 * v_colors_local;
    acc[13]      += pSH14 * v_colors_local;
    acc[14]      += pSH15 * v_colors_local;

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y,
        pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if(v_dir != nullptr)
    {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x  = fTmp0C;
        pSH13_z  = fTmp0C_z * x;
        pSH11_y  = fTmp0C;
        pSH11_z  = fTmp0C_z * y;
        pSH14_x  = fTmp1B * fC1_x;
        pSH14_y  = fTmp1B * fC1_y;
        pSH14_z  = fTmp1B_z * fC1;
        pSH10_x  = fTmp1B * fS1_x;
        pSH10_y  = fTmp1B * fS1_y;
        pSH10_z  = fTmp1B_z * fS1;
        pSH15_x  = -0.5900435899266435f * fC2_x;
        pSH15_y  = -0.5900435899266435f * fC2_y;
        pSH9_x   = -0.5900435899266435f * fS2_x;
        pSH9_y   = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local
             * (pSH9_x * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH15_x * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH10_x * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH14_x * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH13_x * static_cast<opmath_t>(coeffs[12 * D + c]));

        v_y += v_colors_local
             * (pSH9_y * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH15_y * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH10_y * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH14_y * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH11_y * static_cast<opmath_t>(coeffs[10 * D + c]));

        v_z += v_colors_local
             * (pSH12_z * static_cast<opmath_t>(coeffs[11 * D + c])
                + pSH13_z * static_cast<opmath_t>(coeffs[12 * D + c])
                + pSH11_z * static_cast<opmath_t>(coeffs[10 * D + c])
                + pSH14_z * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH10_z * static_cast<opmath_t>(coeffs[9 * D + c]));
    }

    if(degree < 4)
    {
        if(v_dir != nullptr)
        {
            vec3 dir_n   = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0D  = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C  = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B  = -1.770130769779931f * z;
    float fC3     = x * fC2 - y * fS2;
    float fS3     = x * fS2 + y * fC2;
    float pSH20   = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21   = fTmp0D * x;
    float pSH19   = fTmp0D * y;
    float pSH22   = fTmp1C * fC1;
    float pSH18   = fTmp1C * fS1;
    float pSH23   = fTmp2B * fC2;
    float pSH17   = fTmp2B * fS2;
    float pSH24   = 0.6258357354491763f * fC3;
    float pSH16   = 0.6258357354491763f * fS3;
    acc[15]      += pSH16 * v_colors_local;
    acc[16]      += pSH17 * v_colors_local;
    acc[17]      += pSH18 * v_colors_local;
    acc[18]      += pSH19 * v_colors_local;
    acc[19]      += pSH20 * v_colors_local;
    acc[20]      += pSH21 * v_colors_local;
    acc[21]      += pSH22 * v_colors_local;
    acc[22]      += pSH23 * v_colors_local;
    acc[23]      += pSH24 * v_colors_local;

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y, pSH19_z,
        pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if(v_dir != nullptr)
    {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x    = fC2 + x * fC2_x - y * fS2_x;
        fC3_y    = x * fC2_y - fS2 - y * fS2_y;
        fS3_x    = fS2 + y * fC2_x + x * fS2_x;
        fS3_y    = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z  = 1.984313483298443f * (pSH12 + z * pSH12_z) + -1.006230589874905f * pSH6_z;
        pSH21_x  = fTmp0D;
        pSH21_z  = fTmp0D_z * x;
        pSH19_y  = fTmp0D;
        pSH19_z  = fTmp0D_z * y;
        pSH22_x  = fTmp1C * fC1_x;
        pSH22_y  = fTmp1C * fC1_y;
        pSH22_z  = fTmp1C_z * fC1;
        pSH18_x  = fTmp1C * fS1_x;
        pSH18_y  = fTmp1C * fS1_y;
        pSH18_z  = fTmp1C_z * fS1;
        pSH23_x  = fTmp2B * fC2_x;
        pSH23_y  = fTmp2B * fC2_y;
        pSH23_z  = fTmp2B_z * fC2;
        pSH17_x  = fTmp2B * fS2_x;
        pSH17_y  = fTmp2B * fS2_y;
        pSH17_z  = fTmp2B_z * fS2;
        pSH24_x  = 0.6258357354491763f * fC3_x;
        pSH24_y  = 0.6258357354491763f * fC3_y;
        pSH16_x  = 0.6258357354491763f * fS3_x;
        pSH16_y  = 0.6258357354491763f * fS3_y;

        v_x += v_colors_local
             * (pSH16_x * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH24_x * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH17_x * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH23_x * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH18_x * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH22_x * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH21_x * static_cast<opmath_t>(coeffs[20 * D + c]));
        v_y += v_colors_local
             * (pSH16_y * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH24_y * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH17_y * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH23_y * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH18_y * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH22_y * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH19_y * static_cast<opmath_t>(coeffs[18 * D + c]));
        v_z += v_colors_local
             * (pSH20_z * static_cast<opmath_t>(coeffs[19 * D + c])
                + pSH21_z * static_cast<opmath_t>(coeffs[20 * D + c])
                + pSH19_z * static_cast<opmath_t>(coeffs[18 * D + c])
                + pSH22_z * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH18_z * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH23_z * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH17_z * static_cast<opmath_t>(coeffs[16 * D + c]));

        vec3 dir_n   = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x, v_y, v_z);
        vec3 v_d     = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        v_dir->x = v_d.x;
        v_dir->y = v_d.y;
        v_dir->z = v_d.z;
    }
}

// Generic forward kernel: any K, one thread per (Gaussian, channel).
// scalar_t is fp16 or fp32; reads are upcast to opmath_t (fp32) inside the helper.
template<typename scalar_t, typename opmath_t>
__global__ void spherical_harmonics_l1_plus_fwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,       // [..., N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K - 1, D]
    const bool *__restrict__ masks,      // [..., N]
    opmath_t *__restrict__ colors        // [..., N, D]
)
{
    // parallelize over B * gaussian_count * D
    auto idx            = cg::this_grid().thread_rank();
    const int64_t count = static_cast<int64_t>(B) * gaussian_count * D;
    if(idx >= count)
    {
        return;
    }
    const int64_t local_elem_id = idx / D;
    const int64_t batch_id      = local_elem_id / gaussian_count;
    const int64_t gaussian_id   = local_elem_id % gaussian_count + gaussian_offset;
    const int64_t elem_id       = batch_id * N + gaussian_id;
    const uint32_t c            = idx % D; // output channel
    if(masks != nullptr && !masks[elem_id])
    {
        return;
    }
    if(degrees_to_use == 0)
    {
        colors[elem_id * D + c] = 0;
        return;
    }
    sh_l1_plus_coeffs_to_color_fast<scalar_t>(
        degrees_to_use, D, c, dirs[elem_id], coeffs + gaussian_id * K * D, colors + elem_id * D
    );
}

void launch_spherical_harmonics_l1_plus_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    // outputs
    at::Tensor colors // [..., N, D]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    // parallelize over B * N * D
    int64_t n_elements             = static_cast<int64_t>(B) * N * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch on the coeff dtype (fp16/fp32); dirs/colors read as opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_fwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t   = at::opmath_type<scalar_t>;
                auto *dirs_ptr   = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *colors_ptr = colors.data_ptr<opmath_t>();

                spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    0, N, B, N, K, D, degrees_to_use, dirs_ptr, coeffs.const_data_ptr<scalar_t>(), masks_ptr, colors_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        ),
        at::kFloat,
        at::kHalf
    );
}

void launch_spherical_harmonics_l1_plus_fwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    // outputs
    at::Tensor colors // [..., N, D]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    constexpr unsigned int threads = 256;

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = static_cast<int64_t>(B) * gaussian_count * D;
        unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
        if(blocks > 0)
        {
            AT_DISPATCH_V2(
                coeffs.scalar_type(),
                "spherical_harmonics_l1_plus_fwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t   = at::opmath_type<scalar_t>;
                        auto *dirs_ptr   = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                        auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *colors_ptr = colors.data_ptr<opmath_t>();

                        spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                            gaussian_offset,
                            gaussian_count,
                            B,
                            N,
                            K,
                            D,
                            degrees_to_use,
                            dirs_ptr,
                            coeffs.const_data_ptr<scalar_t>(),
                            masks_ptr,
                            colors_ptr
                        );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                ),
                at::kFloat,
                at::kHalf
            );
        }
    }
    merge_streams();
}

template<typename scalar_t, typename opmath_t>
__global__ void spherical_harmonics_l1_plus_bwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,         // [..., N, 3]
    const scalar_t *__restrict__ coeffs,   // [N, K - 1, D]
    const bool *__restrict__ masks,        // [..., N]
    const opmath_t *__restrict__ v_colors, // [..., N, D]
    scalar_t *__restrict__ v_coeffs,       // [N, K - 1, D] (coeff dtype)
    opmath_t *__restrict__ v_dirs          // [..., N, 3] optional
)
{
    // One thread per (Gaussian, channel). Reduce the coeff gradient over all B
    // batches in fp32 registers, then write the K-1 results once in the coeff
    // dtype. Avoids the fp32 scratch buffer, zero-init and fp32->coeff cast.
    auto idx            = cg::this_grid().thread_rank();
    const int64_t count = gaussian_count * static_cast<int64_t>(D);
    if(idx >= count)
    {
        return;
    }
    const int64_t gaussian_id = idx / D + gaussian_offset;
    const uint32_t c          = static_cast<uint32_t>(idx % D); // output channel

    // fp32 register accumulator, one slot per l1+ coefficient (K-1 <= 24).
    constexpr int MAX_K = SH_MAX_COEFFS;

    if(degrees_to_use == 0)
    {
        // No l1+ coefficients contribute; write zeros so the output is defined.
        scalar_t *v_coeffs_ptr = v_coeffs + gaussian_id * K * D;
#pragma unroll
        for(int k = 0; k < MAX_K; ++k)
        {
            if(static_cast<uint32_t>(k) < K)
            {
                v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
            }
        }
        for(uint32_t k = MAX_K; k < K; ++k)
        {
            v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
        }
        return;
    }

    std::array<opmath_t, MAX_K> acc{};

    const scalar_t *coeffs_ptr = coeffs + gaussian_id * K * D;
    for(uint32_t batch_id = 0; batch_id < B; ++batch_id)
    {
        const int64_t elem_id = static_cast<int64_t>(batch_id) * N + gaussian_id;
        if(masks != nullptr && !masks[elem_id])
        {
            continue;
        }

        vec3 v_dir = {0.f, 0.f, 0.f};
        sh_l1_plus_coeffs_to_color_fast_vjp<scalar_t>(
            degrees_to_use,
            D,
            c,
            dirs[elem_id],
            coeffs_ptr,
            v_colors + elem_id * D,
            acc.data(),
            v_dirs == nullptr ? nullptr : &v_dir
        );

        if(v_dirs != nullptr)
        {
            gpuAtomicAdd(v_dirs + elem_id * 3, v_dir.x);
            gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
            gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
        }
    }

    scalar_t *v_coeffs_ptr = v_coeffs + gaussian_id * K * D;
#pragma unroll
    for(int k = 0; k < MAX_K; ++k)
    {
        if(static_cast<uint32_t>(k) < K)
        {
            v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(acc[k]);
        }
    }
    // Padded bases may allocate K > MAX_K; zero the tail so at::empty outputs are defined.
    for(uint32_t k = MAX_K; k < K; ++k)
    {
        v_coeffs_ptr[k * D + c] = static_cast<scalar_t>(0.f);
    }
}

void launch_spherical_harmonics_l1_plus_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::Tensor v_colors,            // [..., N, D]
    // outputs
    at::Tensor v_coeffs,            // [N, K - 1, D]
    at::optional<at::Tensor> v_dirs // [..., N, 3]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    auto stream = at::cuda::getCurrentCUDAStream();

    // One thread per (Gaussian, channel); the kernel loops over the B batches
    // internally and reduces in registers.
    int64_t n_elements             = static_cast<int64_t>(N) * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    // Dispatch on the coeff dtype (fp16/fp32). v_coeffs is written in the coeff
    // dtype (register accumulation is fp32); dirs/v_colors/v_dirs read as opmath_t.
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_bwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t     = at::opmath_type<scalar_t>;
                auto *dirs_ptr     = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();
                auto *v_dirs_ptr   = v_dirs.has_value() ? v_dirs.value().data_ptr<opmath_t>() : nullptr;

                spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    0,
                    N,
                    B,
                    N,
                    K,
                    D,
                    degrees_to_use,
                    dirs_ptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    v_colors_ptr,
                    v_coeffs.data_ptr<scalar_t>(),
                    v_dirs_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        ),
        at::kFloat,
        at::kHalf
    );
}

void launch_spherical_harmonics_l1_plus_bwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::Tensor v_colors,            // [..., N, D]
    // outputs
    at::Tensor v_coeffs,            // [N, K - 1, D]
    at::optional<at::Tensor> v_dirs // [..., N, 3]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    constexpr unsigned int threads = 256;

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = gaussian_count * static_cast<int64_t>(D);
        unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
        if(blocks > 0)
        {
            AT_DISPATCH_V2(
                coeffs.scalar_type(),
                "spherical_harmonics_l1_plus_bwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t     = at::opmath_type<scalar_t>;
                        auto *dirs_ptr     = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                        auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();
                        auto *v_dirs_ptr   = v_dirs.has_value() ? v_dirs.value().data_ptr<opmath_t>() : nullptr;

                        spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                            gaussian_offset,
                            gaussian_count,
                            B,
                            N,
                            K,
                            D,
                            degrees_to_use,
                            dirs_ptr,
                            coeffs.const_data_ptr<scalar_t>(),
                            masks_ptr,
                            v_colors_ptr,
                            v_coeffs.data_ptr<scalar_t>(),
                            v_dirs_ptr
                        );
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    }
                ),
                at::kFloat,
                at::kHalf
            );
        }
    }
    merge_streams();
}
} // namespace gsplat
