/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
#include <type_traits>
#include <variant>
#include <vector>

#include "Common.h"
#include "Dispatch.h"
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
__device__ void sh_coeffs_to_color_fast(
    const uint32_t degree,  // degree of SH to be evaluated
    const uint32_t D,       // channels (coeffs last dim)
    const uint32_t c,       // output channel
    const vec3 &dir,        // [3]
    const scalar_t *coeffs, // [K, D]
    // output
    at::opmath_type<scalar_t> *colors // [D]
)
{
    using opmath_t  = at::opmath_type<scalar_t>;
    opmath_t result = 0.2820947917738781f * static_cast<opmath_t>(coeffs[c]);
    if(degree >= 1)
    {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x     = dir.x * inorm;
        float y     = dir.y * inorm;
        float z     = dir.z * inorm;

        result += 0.48860251190292f
                * (-y * static_cast<opmath_t>(coeffs[1 * D + c])
                   + z * static_cast<opmath_t>(coeffs[2 * D + c])
                   - x * static_cast<opmath_t>(coeffs[3 * D + c]));
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

            result += pSH4 * static_cast<opmath_t>(coeffs[4 * D + c])
                    + pSH5 * static_cast<opmath_t>(coeffs[5 * D + c])
                    + pSH6 * static_cast<opmath_t>(coeffs[6 * D + c])
                    + pSH7 * static_cast<opmath_t>(coeffs[7 * D + c])
                    + pSH8 * static_cast<opmath_t>(coeffs[8 * D + c]);
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

                result += pSH9 * static_cast<opmath_t>(coeffs[9 * D + c])
                        + pSH10 * static_cast<opmath_t>(coeffs[10 * D + c])
                        + pSH11 * static_cast<opmath_t>(coeffs[11 * D + c])
                        + pSH12 * static_cast<opmath_t>(coeffs[12 * D + c])
                        + pSH13 * static_cast<opmath_t>(coeffs[13 * D + c])
                        + pSH14 * static_cast<opmath_t>(coeffs[14 * D + c])
                        + pSH15 * static_cast<opmath_t>(coeffs[15 * D + c]);

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

                    result += pSH16 * static_cast<opmath_t>(coeffs[16 * D + c])
                            + pSH17 * static_cast<opmath_t>(coeffs[17 * D + c])
                            + pSH18 * static_cast<opmath_t>(coeffs[18 * D + c])
                            + pSH19 * static_cast<opmath_t>(coeffs[19 * D + c])
                            + pSH20 * static_cast<opmath_t>(coeffs[20 * D + c])
                            + pSH21 * static_cast<opmath_t>(coeffs[21 * D + c])
                            + pSH22 * static_cast<opmath_t>(coeffs[22 * D + c])
                            + pSH23 * static_cast<opmath_t>(coeffs[23 * D + c])
                            + pSH24 * static_cast<opmath_t>(coeffs[24 * D + c]);
                }
            }
        }
    }

    colors[c] = result;
}

template<typename scalar_t>
__device__ void sh_coeffs_to_color_fast_vjp(
    const uint32_t degree,                     // degree of SH to be evaluated
    const uint32_t D,                          // channels (coeffs last dim)
    const uint32_t c,                          // output channel
    const vec3 &dir,                           // [3]
    const scalar_t *coeffs,                    // [K, D]
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

    acc[0] += 0.2820947917738781f * v_colors_local;
    if(degree < 1)
    {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x     = dir.x * inorm;
    float y     = dir.y * inorm;
    float z     = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    acc[1] += -0.48860251190292f * y * v_colors_local;
    acc[2] += 0.48860251190292f * z * v_colors_local;
    acc[3] += -0.48860251190292f * x * v_colors_local;

    if(v_dir != nullptr)
    {
        v_x += -0.48860251190292f * static_cast<opmath_t>(coeffs[3 * D + c]) * v_colors_local;
        v_y += -0.48860251190292f * static_cast<opmath_t>(coeffs[1 * D + c]) * v_colors_local;
        v_z += 0.48860251190292f * static_cast<opmath_t>(coeffs[2 * D + c]) * v_colors_local;
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
    acc[4]       += pSH4 * v_colors_local;
    acc[5]       += pSH5 * v_colors_local;
    acc[6]       += pSH6 * v_colors_local;
    acc[7]       += pSH7 * v_colors_local;
    acc[8]       += pSH8 * v_colors_local;

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
             * (pSH4_x * static_cast<opmath_t>(coeffs[4 * D + c])
                + pSH8_x * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH7_x * static_cast<opmath_t>(coeffs[7 * D + c]));
        v_y += v_colors_local
             * (pSH4_y * static_cast<opmath_t>(coeffs[4 * D + c])
                + pSH8_y * static_cast<opmath_t>(coeffs[8 * D + c])
                + pSH5_y * static_cast<opmath_t>(coeffs[5 * D + c]));
        v_z += v_colors_local
             * (pSH6_z * static_cast<opmath_t>(coeffs[6 * D + c])
                + pSH7_z * static_cast<opmath_t>(coeffs[7 * D + c])
                + pSH5_z * static_cast<opmath_t>(coeffs[5 * D + c]));
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
    acc[9]       += pSH9 * v_colors_local;
    acc[10]      += pSH10 * v_colors_local;
    acc[11]      += pSH11 * v_colors_local;
    acc[12]      += pSH12 * v_colors_local;
    acc[13]      += pSH13 * v_colors_local;
    acc[14]      += pSH14 * v_colors_local;
    acc[15]      += pSH15 * v_colors_local;

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
             * (pSH9_x * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH15_x * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH10_x * static_cast<opmath_t>(coeffs[10 * D + c])
                + pSH14_x * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH13_x * static_cast<opmath_t>(coeffs[13 * D + c]));

        v_y += v_colors_local
             * (pSH9_y * static_cast<opmath_t>(coeffs[9 * D + c])
                + pSH15_y * static_cast<opmath_t>(coeffs[15 * D + c])
                + pSH10_y * static_cast<opmath_t>(coeffs[10 * D + c])
                + pSH14_y * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH11_y * static_cast<opmath_t>(coeffs[11 * D + c]));

        v_z += v_colors_local
             * (pSH12_z * static_cast<opmath_t>(coeffs[12 * D + c])
                + pSH13_z * static_cast<opmath_t>(coeffs[13 * D + c])
                + pSH11_z * static_cast<opmath_t>(coeffs[11 * D + c])
                + pSH14_z * static_cast<opmath_t>(coeffs[14 * D + c])
                + pSH10_z * static_cast<opmath_t>(coeffs[10 * D + c]));
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
    acc[16]      += pSH16 * v_colors_local;
    acc[17]      += pSH17 * v_colors_local;
    acc[18]      += pSH18 * v_colors_local;
    acc[19]      += pSH19 * v_colors_local;
    acc[20]      += pSH20 * v_colors_local;
    acc[21]      += pSH21 * v_colors_local;
    acc[22]      += pSH22 * v_colors_local;
    acc[23]      += pSH23 * v_colors_local;
    acc[24]      += pSH24 * v_colors_local;

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
             * (pSH16_x * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH24_x * static_cast<opmath_t>(coeffs[24 * D + c])
                + pSH17_x * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH23_x * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH18_x * static_cast<opmath_t>(coeffs[18 * D + c])
                + pSH22_x * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH21_x * static_cast<opmath_t>(coeffs[21 * D + c]));
        v_y += v_colors_local
             * (pSH16_y * static_cast<opmath_t>(coeffs[16 * D + c])
                + pSH24_y * static_cast<opmath_t>(coeffs[24 * D + c])
                + pSH17_y * static_cast<opmath_t>(coeffs[17 * D + c])
                + pSH23_y * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH18_y * static_cast<opmath_t>(coeffs[18 * D + c])
                + pSH22_y * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH19_y * static_cast<opmath_t>(coeffs[19 * D + c]));
        v_z += v_colors_local
             * (pSH20_z * static_cast<opmath_t>(coeffs[20 * D + c])
                + pSH21_z * static_cast<opmath_t>(coeffs[21 * D + c])
                + pSH19_z * static_cast<opmath_t>(coeffs[19 * D + c])
                + pSH22_z * static_cast<opmath_t>(coeffs[22 * D + c])
                + pSH18_z * static_cast<opmath_t>(coeffs[18 * D + c])
                + pSH23_z * static_cast<opmath_t>(coeffs[23 * D + c])
                + pSH17_z * static_cast<opmath_t>(coeffs[17 * D + c]));

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
__global__ void spherical_harmonics_fwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,       // [..., N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, D]
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
    sh_coeffs_to_color_fast<scalar_t>(
        degrees_to_use, D, c, dirs[elem_id], coeffs + gaussian_id * K * D, colors + elem_id * D
    );
}

// K=16, D=3 forward kernel for the RGB SH hot path. One thread per
// (batch, Gaussian), processes all 3 color channels with uint4 / ushort4
// wide loads. DEGREE is a template parameter so each instantiation sizes
// sh_coeffs[] to exactly what it needs.
template<typename scalar_t, typename opmath_t, int DEGREE>
__global__ void __launch_bounds__(256, 4) spherical_harmonics_fwd_kernel_k16_3channel(
    const uint32_t B,
    const uint32_t N,
    const vec3 *__restrict__ dirs,       // [..., N, 3]
    const scalar_t *__restrict__ coeffs, // [N, 16, 3]
    const bool *__restrict__ masks,      // [..., N]
    opmath_t *__restrict__ colors        // [..., N, 3]
)
{
    uint32_t idx = cg::this_grid().thread_rank();
    if(idx >= B * N)
    {
        return;
    }
    uint32_t gaussian_id = idx % N;
    if(masks != nullptr && !masks[idx])
    {
        return;
    }

    coeffs += gaussian_id * 16 * 3;

    constexpr bool COEFFS_FP32 = std::is_same_v<scalar_t, float>;

    // Wide-load shape per (DEGREE, coeff dtype):
    //  DEGREE 0/1 (small payload): ushort4 for fp16, uint4 for fp32, PACK_SIZE = 4.
    //  DEGREE 2/3 (larger payload): always uint4, PACK_SIZE = 4 (fp32) or 8 (fp16).
    constexpr int PACK_SIZE  = (DEGREE <= 1) ? 4 : (COEFFS_FP32 ? 4 : 8);
    constexpr int LOAD_COUNT = (DEGREE == 0) ? 1
                             : (DEGREE == 1) ? 3
                             : (DEGREE == 2) ? (COEFFS_FP32 ? 7 : 4)
                                             : (COEFFS_FP32 ? 12 : 6);
    constexpr int N_COEFFS   = LOAD_COUNT * PACK_SIZE;

    using V = std::conditional_t<(DEGREE <= 1) && !COEFFS_FP32, ushort4, uint4>;

    opmath_t out_color[3];
    opmath_t sh_coeffs[N_COEFFS];

#pragma unroll
    for(int i = 0; i < LOAD_COUNT; ++i)
    {
        alignas(alignof(V)) scalar_t mem[PACK_SIZE];
        reinterpret_cast<V *>(mem)[0] = reinterpret_cast<const V *>(coeffs)[i];
#pragma unroll
        for(int j = 0; j < PACK_SIZE; ++j)
        {
            sh_coeffs[i * PACK_SIZE + j] = static_cast<opmath_t>(mem[j]);
        }
    }

    // K=16 path is gated on D=3 in the host launcher; pass D=3 explicitly here.
    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 0, dirs[idx], sh_coeffs, out_color);
    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 1, dirs[idx], sh_coeffs, out_color);
    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 2, dirs[idx], sh_coeffs, out_color);

    colors[idx * 3 + 0] = out_color[0];
    colors[idx * 3 + 1] = out_color[1];
    colors[idx * 3 + 2] = out_color[2];
}

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    // outputs
    at::Tensor colors // [..., N, D]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    if(B * N == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // K=16 wide loads need D == 3 and natural alignment; misaligned tensors
    // fall through to the scalar generic kernel.
    const bool uses_ushort4      = (coeffs.scalar_type() == at::kHalf) && (degrees_to_use <= 1);
    const size_t coeffs_align    = uses_ushort4 ? alignof(ushort4) : alignof(uint4);
    const bool wide_load_aligned = (reinterpret_cast<uintptr_t>(coeffs.const_data_ptr()) % coeffs_align) == 0;

    if(K == 16 && D == 3 && wide_load_aligned)
    {
        int64_t n_elements             = B * N;
        constexpr unsigned int threads = 256;
        unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

        auto *masks_ptr = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;

        std::variant<float, at::Half> coeff_variant;
        switch(coeffs.scalar_type())
        {
        case at::kFloat: coeff_variant.emplace<float>(); break;
        case at::kHalf:  coeff_variant.emplace<at::Half>(); break;
        default:
            TORCH_CHECK(
                false, "spherical_harmonics_fwd_kernel_k16_3channel: unsupported coeffs dtype ", coeffs.scalar_type()
            );
        }

        const int deg = std::min<int>(degrees_to_use, 3);

        // coeff dtype and DEGREE resolved at compile time; colors/dirs use
        // opmath_t = at::opmath_type<coeff dtype> (float for both fp16 and fp32).
        const bool dispatched = dispatch::dispatch(
            dispatch::IntParam<0, 1, 2, 3>{deg},
            dispatch::TypeParam<float, at::Half>{coeff_variant},
            [&]<typename DegConst, typename CoeffT>()
            {
                using opmath_t       = at::opmath_type<CoeffT>;
                constexpr int DEGREE = DegConst::value;
                auto *dirs_ptr       = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                auto *colors_ptr     = colors.data_ptr<opmath_t>();
                spherical_harmonics_fwd_kernel_k16_3channel<CoeffT, opmath_t, DEGREE><<<blocks, threads, 0, stream>>>(
                    B, N, dirs_ptr, coeffs.const_data_ptr<CoeffT>(), masks_ptr, colors_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        );
        TORCH_CHECK(
            dispatched, "spherical_harmonics_fwd_kernel_k16_3channel: dispatch failed for sh_degree=", degrees_to_use
        );
    }
    else
    {
        int64_t n_elements             = static_cast<int64_t>(B) * N * D;
        constexpr unsigned int threads = 256;
        unsigned int blocks            = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));

        // Dispatch on the coeff dtype (fp16/fp32); dirs/colors read as opmath_t (float).
        AT_DISPATCH_V2(
            coeffs.scalar_type(),
            "spherical_harmonics_fwd_kernel",
            AT_WRAP(
                [&]()
                {
                    using opmath_t   = at::opmath_type<scalar_t>;
                    auto *dirs_ptr   = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                    auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                    auto *colors_ptr = colors.data_ptr<opmath_t>();

                    spherical_harmonics_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
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

void launch_spherical_harmonics_fwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
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
                "spherical_harmonics_fwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t   = at::opmath_type<scalar_t>;
                        auto *dirs_ptr   = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                        auto *masks_ptr  = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *colors_ptr = colors.data_ptr<opmath_t>();

                        spherical_harmonics_fwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
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
__global__ void spherical_harmonics_bwd_kernel(
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,         // [..., N, 3]
    const scalar_t *__restrict__ coeffs,   // [N, K, D]
    const bool *__restrict__ masks,        // [..., N]
    const opmath_t *__restrict__ v_colors, // [..., N, D]
    scalar_t *__restrict__ v_coeffs,       // [N, K, D] (coeff dtype)
    opmath_t *__restrict__ v_dirs          // [..., N, 3] optional
)
{
    // One thread per (Gaussian, channel). Each thread reduces the coeff gradient
    // over all B batches in registers (accumulating in fp32) and writes the K
    // results to global memory once, cast to the coeff dtype. This removes the
    // global atomic accumulation into a fp32 scratch buffer as well as the
    // separate fp32 zero-init and fp32->coeff-dtype cast the batch-parallel
    // formulation required.
    auto idx            = cg::this_grid().thread_rank();
    const int64_t count = gaussian_count * static_cast<int64_t>(D);
    if(idx >= count)
    {
        return;
    }
    const int64_t gaussian_id = idx / D + gaussian_offset;
    const uint32_t c          = static_cast<uint32_t>(idx % D); // output channel

    // fp32 register accumulator, one slot per coefficient. Degree <= 4 -> K <= 25.
    constexpr int MAX_K = SH_MAX_COEFFS;
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
        sh_coeffs_to_color_fast_vjp<scalar_t>(
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
            // v_dirs is [B, N, 3]: each (batch, Gaussian) slot is summed across the
            // D channel-threads, so it still uses atomics. It is tiny relative to
            // v_coeffs, so this is not on the critical path.
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

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::Tensor v_colors,            // [..., N, D]
    // outputs
    at::Tensor v_coeffs,            // [N, K, D]
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
        "spherical_harmonics_bwd_kernel",
        AT_WRAP(
            [&]()
            {
                using opmath_t     = at::opmath_type<scalar_t>;
                auto *dirs_ptr     = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();
                auto *v_dirs_ptr   = v_dirs.has_value() ? v_dirs.value().data_ptr<opmath_t>() : nullptr;

                spherical_harmonics_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
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

// ===========================================================================
// Fused proj_features assembly (forward)
// ===========================================================================
// Fused post-transform applied to each SH output element before it is written,
// matching the per-feature adjustment the caller used to do as separate strided
// elementwise kernels. 0: none; 1: shift (+0.5); 2: shift_relu (max(x+0.5, 0)).
enum SHPostOp : uint32_t
{
    SH_POST_NONE       = 0,
    SH_POST_SHIFT      = 1,
    SH_POST_SHIFT_RELU = 2
};

template<typename T>
__device__ __forceinline__ T apply_sh_post(T v, const uint32_t post_op)
{
    if(post_op == SH_POST_SHIFT)
    {
        return v + T(0.5);
    }
    if(post_op == SH_POST_SHIFT_RELU)
    {
        T s = v + T(0.5);
        return s > T(0) ? s : T(0);
    }
    return v;
}

// K=16, D=3 SH evaluation for the RGB hot path: wide uint4 / ushort4 loads of a
// single Gaussian's 48 coefficients, then evaluate all 3 channels. DEGREE is a
// template parameter so sh_coeffs[] is sized to exactly what it needs. Shared by
// the standalone SH kernel and the fused proj-features assembly kernel.
template<typename scalar_t, typename opmath_t, int DEGREE>
__device__ __forceinline__ void sh16_3channel_eval(
    const scalar_t *__restrict__ coeffs_gaussian, // coeffs + gaussian_id * 16 * 3
    const vec3 &dir,
    opmath_t out_color[3]
)
{
    constexpr bool COEFFS_FP32 = std::is_same_v<scalar_t, float>;

    // Wide-load shape per (DEGREE, coeff dtype):
    //  DEGREE 0/1 (small payload): ushort4 for fp16, uint4 for fp32, PACK_SIZE = 4.
    //  DEGREE 2/3 (larger payload): always uint4, PACK_SIZE = 4 (fp32) or 8 (fp16).
    constexpr int PACK_SIZE  = (DEGREE <= 1) ? 4 : (COEFFS_FP32 ? 4 : 8);
    constexpr int LOAD_COUNT = (DEGREE == 0) ? 1
                             : (DEGREE == 1) ? 3
                             : (DEGREE == 2) ? (COEFFS_FP32 ? 7 : 4)
                                             : (COEFFS_FP32 ? 12 : 6);
    constexpr int N_COEFFS   = LOAD_COUNT * PACK_SIZE;

    using V = std::conditional_t<(DEGREE <= 1) && !COEFFS_FP32, ushort4, uint4>;

    opmath_t sh_coeffs[N_COEFFS];
#pragma unroll
    for(int i = 0; i < LOAD_COUNT; ++i)
    {
        alignas(alignof(V)) scalar_t mem[PACK_SIZE];
        reinterpret_cast<V *>(mem)[0] = reinterpret_cast<const V *>(coeffs_gaussian)[i];
#pragma unroll
        for(int j = 0; j < PACK_SIZE; ++j)
        {
            sh_coeffs[i * PACK_SIZE + j] = static_cast<opmath_t>(mem[j]);
        }
    }

    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 0, dir, sh_coeffs, out_color);
    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 1, dir, sh_coeffs, out_color);
    sh_coeffs_to_color_fast<opmath_t>(DEGREE, 3, 2, dir, sh_coeffs, out_color);
}

// Unified fused assembler: proj_features = [SH colors | extra | (depth)] for the
// unpacked rasterization path. A block stages TILE_ROWS consecutive Gaussians of
// one (batch, camera) pair through shared memory so every global access is fully
// coalesced. This single kernel subsumes the generic / k16 variants via
// compile-time specialization:
//   * K16FAST=true  -> wide-load RGB SH (sh16_3channel_eval), K==16, Dc==3.
//   * K16FAST=false -> generic per-channel SH (sh_coeffs_to_color_fast), any K/Dc.
// The optional tensors are lifted to compile-time "soft booleans" (HAS_DEPTH,
// DEPTH_ZERO, HAS_MASK, EMIT_RELU) so an absent tensor's branch and its global
// traffic vanish entirely from the corresponding instantiation. `scalar_t` is the
// coeff dtype (fp16/fp32); means/campos/extra/depths/out are opmath_t (fp32).
template<
    typename scalar_t,
    typename opmath_t,
    int DEGREE,
    bool K16FAST,
    bool HAS_DEPTH,
    bool DEPTH_ZERO,
    bool HAS_MASK,
    bool EMIT_RELU
>
__global__ void __launch_bounds__(256) assemble_proj_features_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t degrees_to_use,       // runtime SH degree (generic path only)
    const uint32_t K,                    // SH bands (coeffs dim -2); 16 when K16FAST
    const uint32_t Dc,                   // SH color channels; 3 when K16FAST
    const uint32_t E,                    // extra-signal channels
    const uint32_t width,                // dc + E + (HAS_DEPTH ? 1 : 0)
    const uint32_t color_post,           // SHPostOp applied to colors
    const uint32_t extra_post,           // SHPostOp applied to extra
    const bool extra_has_c,              // extra indexed by (b,c,n) vs broadcast (b,n)
    const float *__restrict__ means,     // [B, N, 3]
    const float *__restrict__ campos,    // [B, C, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, Dc]
    const opmath_t *__restrict__ extra,  // [B, C, N, E] or [B, N, E]; null if E == 0
    const opmath_t *__restrict__ depths, // [B, C, N]; null when DEPTH_ZERO or !HAS_DEPTH
    const bool *__restrict__ masks,      // [B, C, N]; null when !HAS_MASK
    opmath_t *__restrict__ out,          // [B, C, N, width]
    bool *__restrict__ relu_mask         // [B, C, N, dc]; null when !EMIT_RELU
)
{
    constexpr uint32_t TILE_ROWS = 64;
    constexpr uint32_t THREADS   = 256;
    // dc / cstride collapse to compile-time constants on the K16FAST path so the
    // color loops fully unroll; on the generic path they track the runtime K/Dc.
    const uint32_t dc            = K16FAST ? 3u : Dc;
    const uint32_t cstride       = K16FAST ? 48u : (K * Dc); // coeff elems per gaussian
    const uint32_t tid           = threadIdx.x;

    // blockIdx.x -> n-tile within a (b,c); blockIdx.y -> flattened (b*C + c).
    const uint32_t bc      = blockIdx.y;
    const uint32_t b       = bc / C;
    const uint32_t rowBase = blockIdx.x * TILE_ROWS;
    if(rowBase >= N)
    {
        return;
    }
    const uint32_t rows = (N - rowBase < TILE_ROWS) ? (N - rowBase) : TILE_ROWS;

    // Dynamic shared layout (16B-aligned base; per-coeff-row alignment preserved):
    //   [coeffs: TILE_ROWS*cstride scalar_t][means: TILE_ROWS*3 float][out: TILE_ROWS*width float]
    extern __shared__ __align__(16) char smem_raw[];
    scalar_t *smem_coeffs = reinterpret_cast<scalar_t *>(smem_raw);
    float *smem_means     = reinterpret_cast<float *>(smem_coeffs + TILE_ROWS * cstride);
    float *smem_out       = smem_means + TILE_ROWS * 3;

    // --- coalesced loads into shared ---
    // coeffs are indexed by gaussian n only (shared across cameras).
    for(uint32_t i = tid; i < rows * cstride; i += THREADS)
    {
        smem_coeffs[i] = coeffs[static_cast<size_t>(rowBase) * cstride + i];
    }
    for(uint32_t i = tid; i < rows * 3u; i += THREADS)
    {
        smem_means[i] = means[(static_cast<size_t>(b) * N + rowBase) * 3 + i];
    }
    if(E > 0)
    {
        const opmath_t eshift = (extra_post == SH_POST_SHIFT) ? opmath_t(0.5) : opmath_t(0);
        const size_t ebase
            = extra_has_c ? (static_cast<size_t>(bc) * N + rowBase) * E : (static_cast<size_t>(b) * N + rowBase) * E;
        for(uint32_t i = tid; i < rows * E; i += THREADS)
        {
            const uint32_t lr             = i / E;
            const uint32_t e              = i % E;
            smem_out[lr * width + dc + e] = extra[ebase + i] + eshift;
        }
    }
    if constexpr(HAS_DEPTH)
    {
        for(uint32_t i = tid; i < rows; i += THREADS)
        {
            opmath_t d;
            if constexpr(DEPTH_ZERO)
            {
                d = opmath_t(0);
            }
            else
            {
                d = depths[static_cast<size_t>(bc) * N + rowBase + i];
            }
            smem_out[i * width + dc + E] = d;
        }
    }
    __syncthreads();

    // --- per-row SH evaluation out of shared ---
    const float cx = campos[static_cast<size_t>(bc) * 3 + 0];
    const float cy = campos[static_cast<size_t>(bc) * 3 + 1];
    const float cz = campos[static_cast<size_t>(bc) * 3 + 2];
    for(uint32_t lr = tid; lr < rows; lr += THREADS)
    {
        const uint32_t n  = rowBase + lr;
        const size_t gidx = static_cast<size_t>(bc) * N + n;
        float *dst        = smem_out + lr * width;
        bool active       = true;
        if constexpr(HAS_MASK)
        {
            active = masks[gidx];
        }
        if(active)
        {
            // View direction = gaussian center - camera position (un-normalized;
            // SH normalizes internally). Folds compute_directions in.
            const float *mp = smem_means + lr * 3;
            const vec3 dir  = vec3(mp[0] - cx, mp[1] - cy, mp[2] - cz);
            if constexpr(K16FAST)
            {
                opmath_t col[3];
                sh16_3channel_eval<scalar_t, opmath_t, DEGREE>(smem_coeffs + lr * 48, dir, col);
                dst[0] = apply_sh_post<opmath_t>(col[0], color_post);
                dst[1] = apply_sh_post<opmath_t>(col[1], color_post);
                dst[2] = apply_sh_post<opmath_t>(col[2], color_post);
            }
            else
            {
                const scalar_t *crow = smem_coeffs + lr * cstride;
                for(uint32_t ch = 0; ch < dc; ++ch)
                {
                    sh_coeffs_to_color_fast<scalar_t>(degrees_to_use, dc, ch, dir, crow, dst);
                }
                if(color_post != SH_POST_NONE)
                {
                    for(uint32_t ch = 0; ch < dc; ++ch)
                    {
                        dst[ch] = apply_sh_post<opmath_t>(dst[ch], color_post);
                    }
                }
            }
            // Emit the relu Jacobian mask inline (clamped value already in a
            // register), folding away the separate strided compare kernel.
            if constexpr(EMIT_RELU)
            {
                bool *m = relu_mask + gidx * dc;
                for(uint32_t ch = 0; ch < dc; ++ch)
                {
                    m[ch] = dst[ch] > opmath_t(0);
                }
            }
        }
        else
        {
            for(uint32_t ch = 0; ch < dc; ++ch)
            {
                dst[ch] = opmath_t(0);
            }
        }
    }
    __syncthreads();

    // --- coalesced bulk store ---
    const size_t obase = (static_cast<size_t>(bc) * N + rowBase) * width;
    for(uint32_t i = tid; i < rows * width; i += THREADS)
    {
        out[obase + i] = smem_out[i];
    }
}

void launch_assemble_proj_features_unpacked_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t degrees_to_use,
    const uint32_t Dc,
    const uint32_t E,
    const uint32_t color_post,
    const uint32_t extra_post,
    const bool has_depth,
    const bool depth_is_zero,
    const bool extra_has_c,
    const at::Tensor means,                  // [B, N, 3]
    const at::Tensor campos,                 // [B, C, 3]
    const at::Tensor coeffs,                 // [N, K, Dc]
    const at::optional<at::Tensor> extra,    // [B, C, N, E] or [B, N, E]
    const at::optional<at::Tensor> depths,   // [B, C, N]
    const at::optional<at::Tensor> masks,    // [B, C, N]
    at::Tensor out,                          // [B, C, N, width]
    const at::optional<at::Tensor> relu_mask // [B, C, N, Dc]
)
{
    const int64_t n_threads = static_cast<int64_t>(B) * C * N;
    if(n_threads == 0)
    {
        return;
    }
    const uint32_t K     = coeffs.size(-2);
    const uint32_t width = static_cast<uint32_t>(out.size(-1));
    auto stream          = at::cuda::getCurrentCUDAStream();

    auto *masks_ptr     = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
    auto *extra_ptr     = extra.has_value() ? extra.value().const_data_ptr<float>() : nullptr;
    auto *depths_ptr    = depths.has_value() ? depths.value().const_data_ptr<float>() : nullptr;
    auto *means_ptr     = means.const_data_ptr<float>();
    auto *campos_ptr    = campos.const_data_ptr<float>();
    auto *out_ptr       = out.data_ptr<float>();
    auto *relu_mask_ptr = relu_mask.has_value() ? relu_mask.value().data_ptr<bool>() : nullptr;

    // Single tiled kernel for every shape: a block stages one (b,c) n-tile
    // through shared memory, so global coeff alignment is irrelevant (the wide
    // SH loads come from shared, which is always 16B aligned per row).
    constexpr uint32_t TILE_ROWS = 64;
    dim3 threads(256);
    dim3 grid((N + TILE_ROWS - 1) / TILE_ROWS, B * C);

    std::variant<float, at::Half> coeff_variant;
    switch(coeffs.scalar_type())
    {
    case at::kFloat: coeff_variant.emplace<float>(); break;
    case at::kHalf:  coeff_variant.emplace<at::Half>(); break;
    default:         TORCH_CHECK(false, "assemble_proj_features: unsupported coeffs dtype ", coeffs.scalar_type());
    }

    // Lift the optional tensors to compile-time "soft booleans". depth_mode
    // folds (has_depth, depth_is_zero) into one 3-state so the impossible
    // (!has_depth && depth_is_zero) instantiation is never generated.
    const int depth_mode = !has_depth ? 0 : (depth_is_zero ? 2 : 1);
    const int mask_on    = (masks_ptr != nullptr) ? 1 : 0;
    const int relu_on    = (relu_mask_ptr != nullptr) ? 1 : 0;
    const bool k16       = (K == 16 && Dc == 3);

    if(k16)
    {
        // RGB SH hot path: DEGREE is specialized so the wide-load evaluator
        // unrolls to exactly its coefficient count.
        const int deg         = std::min<int>(degrees_to_use, 3);
        const bool dispatched = dispatch::dispatch(
            dispatch::TypeParam<float, at::Half>{coeff_variant},
            dispatch::IntParam<0, 1, 2, 3>{deg},
            dispatch::IntParam<0, 1, 2>{depth_mode},
            dispatch::IntParam<0, 1>{mask_on},
            dispatch::IntParam<0, 1>{relu_on},
            [&]<typename CoeffT, typename Deg, typename Dm, typename Mk, typename Rl>()
            {
                constexpr int DEGREE       = Deg::value;
                constexpr bool HAS_DEPTH   = Dm::value != 0;
                constexpr bool DEPTH_ZERO  = Dm::value == 2;
                constexpr bool HAS_MASK    = Mk::value != 0;
                constexpr bool EMIT_RELU   = Rl::value != 0;
                const size_t smem          = static_cast<size_t>(TILE_ROWS) * 48 * sizeof(CoeffT)
                                           + static_cast<size_t>(TILE_ROWS) * 3 * sizeof(float)
                                           + static_cast<size_t>(TILE_ROWS) * width * sizeof(float);
                // Opt in to >48KB dynamic shared memory (and surface an actionable
                // error if the shape exceeds the device budget). The attribute is
                // set once per kernel specialization: the request only grows with
                // width, which is constant across a run, so the per-launch driver
                // roundtrip is amortized to a single call.
                static int configured_smem = -1;
                if(static_cast<int>(smem) > configured_smem)
                {
                    if(cudaFuncSetAttribute(
                           assemble_proj_features_kernel<
                               CoeffT,
                               float,
                               DEGREE,
                               /*K16FAST=*/true,
                               HAS_DEPTH,
                               DEPTH_ZERO,
                               HAS_MASK,
                               EMIT_RELU
                           >,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)
                       )
                       != cudaSuccess)
                    {
                        AT_ERROR(
                            "assemble_proj_features: failed to set dynamic shared "
                            "memory (requested ",
                            smem,
                            " bytes; width=",
                            width,
                            "); shape exceeds this device's shared-memory budget."
                        );
                    }
                    configured_smem = static_cast<int>(smem);
                }
                assemble_proj_features_kernel<
                    CoeffT,
                    float,
                    DEGREE,
                    /*K16FAST=*/true,
                    HAS_DEPTH,
                    DEPTH_ZERO,
                    HAS_MASK,
                    EMIT_RELU
                ><<<grid, threads, smem, stream>>>(
                    B,
                    C,
                    N,
                    degrees_to_use,
                    K,
                    Dc,
                    E,
                    width,
                    color_post,
                    extra_post,
                    extra_has_c,
                    means_ptr,
                    campos_ptr,
                    coeffs.const_data_ptr<CoeffT>(),
                    extra_ptr,
                    depths_ptr,
                    masks_ptr,
                    out_ptr,
                    relu_mask_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        );
        TORCH_CHECK(dispatched, "assemble_proj_features: dispatch failed for sh_degree=", degrees_to_use);
    }
    else
    {
        // Generic path: any K/Dc; the SH degree stays a runtime argument so this
        // is not specialized per degree.
        const bool dispatched = dispatch::dispatch(
            dispatch::TypeParam<float, at::Half>{coeff_variant},
            dispatch::IntParam<0, 1, 2>{depth_mode},
            dispatch::IntParam<0, 1>{mask_on},
            dispatch::IntParam<0, 1>{relu_on},
            [&]<typename CoeffT, typename Dm, typename Mk, typename Rl>()
            {
                constexpr bool HAS_DEPTH   = Dm::value != 0;
                constexpr bool DEPTH_ZERO  = Dm::value == 2;
                constexpr bool HAS_MASK    = Mk::value != 0;
                constexpr bool EMIT_RELU   = Rl::value != 0;
                const size_t smem          = static_cast<size_t>(TILE_ROWS) * K * Dc * sizeof(CoeffT)
                                           + static_cast<size_t>(TILE_ROWS) * 3 * sizeof(float)
                                           + static_cast<size_t>(TILE_ROWS) * width * sizeof(float);
                // Set the dynamic-smem opt-in once per kernel specialization
                // (see the k16 branch above for the amortization rationale).
                static int configured_smem = -1;
                if(static_cast<int>(smem) > configured_smem)
                {
                    if(cudaFuncSetAttribute(
                           assemble_proj_features_kernel<
                               CoeffT,
                               float,
                               /*DEGREE=*/0,
                               /*K16FAST=*/false,
                               HAS_DEPTH,
                               DEPTH_ZERO,
                               HAS_MASK,
                               EMIT_RELU
                           >,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)
                       )
                       != cudaSuccess)
                    {
                        AT_ERROR(
                            "assemble_proj_features: failed to set dynamic shared "
                            "memory (requested ",
                            smem,
                            " bytes; width=",
                            width,
                            ", K=",
                            K,
                            ", Dc=",
                            Dc,
                            "); shape exceeds this device's "
                            "shared-memory budget."
                        );
                    }
                    configured_smem = static_cast<int>(smem);
                }
                assemble_proj_features_kernel<
                    CoeffT,
                    float,
                    /*DEGREE=*/0,
                    /*K16FAST=*/false,
                    HAS_DEPTH,
                    DEPTH_ZERO,
                    HAS_MASK,
                    EMIT_RELU
                ><<<grid, threads, smem, stream>>>(
                    B,
                    C,
                    N,
                    degrees_to_use,
                    K,
                    Dc,
                    E,
                    width,
                    color_post,
                    extra_post,
                    extra_has_c,
                    means_ptr,
                    campos_ptr,
                    coeffs.const_data_ptr<CoeffT>(),
                    extra_ptr,
                    depths_ptr,
                    masks_ptr,
                    out_ptr,
                    relu_mask_ptr
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        );
        TORCH_CHECK(dispatched, "assemble_proj_features: dispatch failed for sh_degree=", degrees_to_use);
    }
}

void launch_spherical_harmonics_bwd_kernels(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::Tensor v_colors,            // [..., N, D]
    // outputs
    at::Tensor v_coeffs,            // [N, K, D]
    at::optional<at::Tensor> v_dirs // [..., N, 3]
)
{
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B = c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    constexpr unsigned int threads = 256;

    if(N == 0)
    {
        return;
    }

    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreate(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getStreamFromPool(false, device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);

        if(gaussian_count > 0)
        {
            std::vector<void *> prefetch_ptrs;
            std::vector<size_t> prefetch_sizes;
            auto append_prefetch_range
                = [&prefetch_ptrs,
                   &prefetch_sizes](const at::Tensor &tensor, int64_t row_offset, int64_t row_count, int64_t row_stride)
            {
                prefetch_ptrs.emplace_back(
                    static_cast<char *>(tensor.data_ptr()) + row_offset * row_stride * tensor.element_size()
                );
                prefetch_sizes.emplace_back(row_count * row_stride * tensor.element_size());
            };

            append_prefetch_range(v_coeffs, gaussian_offset, gaussian_count, v_coeffs.stride(-3));
            for(const auto batch_id: c10::irange(B))
            {
                const int64_t row_offset = static_cast<int64_t>(batch_id) * N + gaussian_offset;
                if(v_dirs.has_value())
                {
                    append_prefetch_range(v_dirs.value(), row_offset, gaussian_count, v_dirs.value().stride(-2));
                }
                append_prefetch_range(v_colors, row_offset, gaussian_count, v_colors.stride(-2));
            }

#if CUDART_VERSION < 13000
            for(const auto range_id: c10::irange(prefetch_ptrs.size()))
            {
                C10_CUDA_CHECK(
                    cudaMemPrefetchAsync(prefetch_ptrs[range_id], prefetch_sizes[range_id], device_id, stream)
                );
            }
#else
            const cudaMemLocation location                  = {cudaMemLocationTypeDevice, device_id};
            std::vector<cudaMemLocation> prefetch_locations = {location};
            std::vector<size_t> prefetch_location_indices   = {0};
            C10_CUDA_CHECK(cudaMemPrefetchBatchAsync(
                prefetch_ptrs.data(),
                prefetch_sizes.data(),
                prefetch_ptrs.size(),
                prefetch_locations.data(),
                prefetch_location_indices.data(),
                prefetch_locations.size(),
                0,
                stream
            ));
#endif

            // v_coeffs is fully written by the kernel (one thread per (Gaussian,
            // channel) stores all K coefficients), so no zero-init is needed.
            if(v_dirs.has_value())
            {
                for(const auto batch_id: c10::irange(B))
                {
                    const int64_t row_offset = static_cast<int64_t>(batch_id) * N + gaussian_offset;
                    C10_CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char *>(v_dirs.value().data_ptr())
                            + row_offset * v_dirs.value().stride(-2) * v_dirs.value().element_size(),
                        0,
                        gaussian_count * v_dirs.value().stride(-2) * v_dirs.value().element_size(),
                        stream
                    ));
                }
            }
        }
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[device_id]));
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));

        int64_t gaussian_offset, gaussian_count;
        std::tie(gaussian_offset, gaussian_count) = chunk(N, device_id);
        int64_t n_elements                        = gaussian_count * static_cast<int64_t>(D);
        unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(n_elements, threads));
        if(blocks > 0)
        {
            AT_DISPATCH_V2(
                coeffs.scalar_type(),
                "spherical_harmonics_bwd_kernel",
                AT_WRAP(
                    [&]()
                    {
                        using opmath_t     = at::opmath_type<scalar_t>;
                        auto *dirs_ptr     = reinterpret_cast<const vec3 *>(dirs.const_data_ptr<opmath_t>());
                        auto *masks_ptr    = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;
                        auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();
                        auto *v_dirs_ptr   = v_dirs.has_value() ? v_dirs.value().data_ptr<opmath_t>() : nullptr;

                        spherical_harmonics_bwd_kernel<scalar_t, opmath_t><<<blocks, threads, 0, stream>>>(
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
