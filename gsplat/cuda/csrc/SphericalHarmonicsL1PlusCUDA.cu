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
#include <cooperative_groups.h>

#include "Common.h"
#include "SphericalHarmonics.h"
#include "SphericalHarmonics.cuh"
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
    // v_coeffs is fp32; caller casts back to the coeff dtype after all atomic adds.
    float *v_coeffs, // [K - 1, D]
    vec3 *v_dir      // [3] optional
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

    gpuAtomicAdd(&v_coeffs[c], -0.48860251190292f * y * v_colors_local);
    gpuAtomicAdd(&v_coeffs[1 * D + c], 0.48860251190292f * z * v_colors_local);
    gpuAtomicAdd(&v_coeffs[2 * D + c], -0.48860251190292f * x * v_colors_local);

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

    float z2     = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1    = x * x - y * y;
    float fS1    = 2.f * x * y;
    float pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7   = fTmp0B * x;
    float pSH5   = fTmp0B * y;
    float pSH8   = 0.5462742152960395f * fC1;
    float pSH4   = 0.5462742152960395f * fS1;
    gpuAtomicAdd(&v_coeffs[3 * D + c], pSH4 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[4 * D + c], pSH5 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[5 * D + c], pSH6 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[6 * D + c], pSH7 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[7 * D + c], pSH8 * v_colors_local);

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
    gpuAtomicAdd(&v_coeffs[8 * D + c], pSH9 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[9 * D + c], pSH10 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[10 * D + c], pSH11 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[11 * D + c], pSH12 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[12 * D + c], pSH13 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[13 * D + c], pSH14 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[14 * D + c], pSH15 * v_colors_local);

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

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3    = x * fC2 - y * fS2;
    float fS3    = x * fS2 + y * fC2;
    float pSH20  = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21  = fTmp0D * x;
    float pSH19  = fTmp0D * y;
    float pSH22  = fTmp1C * fC1;
    float pSH18  = fTmp1C * fS1;
    float pSH23  = fTmp2B * fC2;
    float pSH17  = fTmp2B * fS2;
    float pSH24  = 0.6258357354491763f * fC3;
    float pSH16  = 0.6258357354491763f * fS3;
    gpuAtomicAdd(&v_coeffs[15 * D + c], pSH16 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[16 * D + c], pSH17 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[17 * D + c], pSH18 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[18 * D + c], pSH19 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[19 * D + c], pSH20 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[20 * D + c], pSH21 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[21 * D + c], pSH22 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[22 * D + c], pSH23 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[23 * D + c], pSH24 * v_colors_local);

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
    const uint32_t C,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ viewmats_rs,
    const scalar_t *__restrict__ coeffs, // [N, K - 1, D]
    const bool *__restrict__ masks,      // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    opmath_t *__restrict__ colors // [..., N, D]
)
{
    // parallelize over B * gaussian_count * D
    auto idx            = cg::this_grid().thread_rank();
    const bool packed   = batch_ids != nullptr;
    const int64_t count = (packed ? gaussian_count : static_cast<int64_t>(B) * C * gaussian_count) * D;
    if(idx >= count)
    {
        return;
    }
    const int64_t local_elem_id = idx / D;
    const int64_t image_id      = packed ? 0 : local_elem_id / gaussian_count;
    const int64_t output_id = packed ? local_elem_id : image_id * N + local_elem_id % gaussian_count + gaussian_offset;
    const int64_t batch_id  = packed ? batch_ids[output_id] : image_id / C;
    const int64_t camera_id = packed ? camera_ids[output_id] : image_id % C;
    const int64_t gaussian_id = packed ? gaussian_ids[output_id] : local_elem_id % gaussian_count + gaussian_offset;
    const int64_t coeff_id    = packed ? output_id : gaussian_id;
    const uint32_t c          = idx % D; // output channel
    if(masks != nullptr && !masks[output_id])
    {
        return;
    }
    if(degrees_to_use == 0)
    {
        colors[output_id * D + c] = 0;
        return;
    }
    const float *viewmat    = viewmats + (batch_id * C + camera_id) * 16;
    const float *viewmat_rs = viewmats_rs == nullptr ? nullptr : viewmats_rs + (batch_id * C + camera_id) * 16;
    const vec3 dir = view_direction_from_world_to_camera(means + (batch_id * N + gaussian_id) * 3, viewmat, viewmat_rs);
    sh_l1_plus_coeffs_to_color_fast<scalar_t>(
        degrees_to_use, D, c, dir, coeffs + coeff_id * K * D, colors + output_id * D
    );
}

void launch_spherical_harmonics_l1_plus_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    const uint32_t E = batch_ids.has_value() ? coeffs.size(0) : B * C * N;

    // parallelize over B * N * D
    int64_t n_elements             = static_cast<int64_t>(E) * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // Dispatch on the coeff dtype (fp16/fp32); colors use opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_fwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t   = at::opmath_type<scalar_t>;
                auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *colors_ptr = colors.data_ptr<opmath_t>();

                spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    0,
                    batch_ids.has_value() ? E : N,
                    B,
                    C,
                    N,
                    K,
                    D,
                    degrees_to_use,
                    means.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    viewmats_rs.has_value() ? viewmats_rs.value().const_data_ptr<float>() : nullptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    colors_ptr
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
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, D]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    TORCH_INTERNAL_ASSERT(!batch_ids.has_value() && !camera_ids.has_value() && !gaussian_ids.has_value());

    constexpr unsigned int threads = 256;

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = static_cast<int64_t>(B) * C * gaussian_count * D;
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
                        auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *colors_ptr = colors.data_ptr<opmath_t>();

                        spherical_harmonics_l1_plus_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                            gaussian_offset,
                            gaussian_count,
                            B,
                            C,
                            N,
                            K,
                            D,
                            degrees_to_use,
                            means.const_data_ptr<float>(),
                            viewmats.const_data_ptr<float>(),
                            nullptr,
                            coeffs.const_data_ptr<scalar_t>(),
                            masks_ptr,
                            nullptr,
                            nullptr,
                            nullptr,
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
    const uint32_t C,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const float *__restrict__ means,
    const float *__restrict__ viewmats,
    const float *__restrict__ viewmats_rs,
    const scalar_t *__restrict__ coeffs, // [N, K - 1, D]
    const bool *__restrict__ masks,      // [..., N]
    const int64_t *__restrict__ batch_ids,
    const int64_t *__restrict__ camera_ids,
    const int64_t *__restrict__ gaussian_ids,
    const opmath_t *__restrict__ v_colors, // [..., N, D]
    float *__restrict__ v_coeffs,          // [N, K - 1, D]
    float *__restrict__ v_means,
    float *__restrict__ v_viewmats,
    float *__restrict__ v_viewmats_rs
)
{
    // parallelize over B * gaussian_count * D
    auto idx            = cg::this_grid().thread_rank();
    const bool packed   = batch_ids != nullptr;
    const int64_t count = (packed ? gaussian_count : static_cast<int64_t>(B) * C * gaussian_count) * D;
    if(idx >= count)
    {
        return;
    }
    const int64_t local_elem_id = idx / D;
    const int64_t image_id      = packed ? 0 : local_elem_id / gaussian_count;
    const int64_t output_id = packed ? local_elem_id : image_id * N + local_elem_id % gaussian_count + gaussian_offset;
    const int64_t batch_id  = packed ? batch_ids[output_id] : image_id / C;
    const int64_t camera_id = packed ? camera_ids[output_id] : image_id % C;
    const int64_t gaussian_id = packed ? gaussian_ids[output_id] : local_elem_id % gaussian_count + gaussian_offset;
    const int64_t coeff_id    = packed ? output_id : gaussian_id;
    const uint32_t c          = idx % D; // output channel
    if(masks != nullptr && !masks[output_id])
    {
        return;
    }
    if(degrees_to_use == 0)
    {
        return;
    }

    const float *viewmat    = viewmats + (batch_id * C + camera_id) * 16;
    const float *viewmat_rs = viewmats_rs == nullptr ? nullptr : viewmats_rs + (batch_id * C + camera_id) * 16;
    const vec3 dir = view_direction_from_world_to_camera(means + (batch_id * N + gaussian_id) * 3, viewmat, viewmat_rs);
    vec3 v_dir     = {0.f, 0.f, 0.f};
    sh_l1_plus_coeffs_to_color_fast_vjp<scalar_t>(
        degrees_to_use,
        D,
        c,
        dir,
        coeffs + coeff_id * K * D,
        v_colors + output_id * D,
        v_coeffs + coeff_id * K * D,
        v_means == nullptr && v_viewmats == nullptr && v_viewmats_rs == nullptr ? nullptr : &v_dir
    );

    if(v_means != nullptr || v_viewmats != nullptr || v_viewmats_rs != nullptr)
    {
        accumulate_view_direction_vjp(
            v_dir,
            viewmat,
            v_means == nullptr ? nullptr : v_means + (batch_id * N + gaussian_id) * 3,
            v_viewmats == nullptr ? nullptr : v_viewmats + (batch_id * C + camera_id) * 16,
            viewmat_rs,
            v_viewmats_rs == nullptr ? nullptr : v_viewmats_rs + (batch_id * C + camera_id) * 16
        );
    }
}

void launch_spherical_harmonics_l1_plus_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats,
    at::optional<at::Tensor> v_viewmats_rs
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    const uint32_t E = batch_ids.has_value() ? coeffs.size(0) : B * C * N;

    auto stream = at::cuda::getCurrentCUDAStream();

    // parallelize over B * N * D
    int64_t n_elements             = static_cast<int64_t>(E) * D;
    constexpr unsigned int threads = 256;
    unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

    if(n_elements == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    // Dispatch on the coeff dtype (fp16/fp32). v_coeffs accumulates in fp32;
    // v_colors uses opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_l1_plus_bwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t     = at::opmath_type<scalar_t>;
                auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();

                spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                    0,
                    batch_ids.has_value() ? E : N,
                    B,
                    C,
                    N,
                    K,
                    D,
                    degrees_to_use,
                    means.const_data_ptr<float>(),
                    viewmats.const_data_ptr<float>(),
                    viewmats_rs.has_value() ? viewmats_rs.value().const_data_ptr<float>() : nullptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr,
                    batch_ids.has_value() ? batch_ids.value().const_data_ptr<int64_t>() : nullptr,
                    camera_ids.has_value() ? camera_ids.value().const_data_ptr<int64_t>() : nullptr,
                    gaussian_ids.has_value() ? gaussian_ids.value().const_data_ptr<int64_t>() : nullptr,
                    v_colors_ptr,
                    v_coeffs.data_ptr<float>(),
                    v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                    v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>() : nullptr,
                    v_viewmats_rs.has_value() ? v_viewmats_rs.value().data_ptr<float>() : nullptr
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
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::Tensor coeffs,              // [N, K - 1, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, D]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = means.size(-2);
    const uint32_t C = viewmats.size(-3);
    const uint32_t B = c10::multiply_integers(means.sizes().slice(0, means.dim() - 2));
    TORCH_INTERNAL_ASSERT(!batch_ids.has_value() && !camera_ids.has_value() && !gaussian_ids.has_value());

    constexpr unsigned int threads = 256;

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = static_cast<int64_t>(B) * C * gaussian_count * D;
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
                        auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();

                        spherical_harmonics_l1_plus_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
                            gaussian_offset,
                            gaussian_count,
                            B,
                            C,
                            N,
                            K,
                            D,
                            degrees_to_use,
                            means.const_data_ptr<float>(),
                            viewmats.const_data_ptr<float>(),
                            nullptr,
                            coeffs.const_data_ptr<scalar_t>(),
                            masks_ptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            v_colors_ptr,
                            v_coeffs.data_ptr<float>(),
                            v_means.has_value() ? v_means.value().data_ptr<float>() : nullptr,
                            v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>() : nullptr,
                            nullptr
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
