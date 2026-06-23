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
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <type_traits>
#include <variant>

#include "Common.h"
#include "Dispatch.h"
#include "SphericalHarmonics.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation

template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast(
    const uint32_t degree,  // degree of SH to be evaluated
    const uint32_t D,       // channels (coeffs last dim)
    const uint32_t c,       // output channel
    const vec3 &dir,        // [3]
    const scalar_t *coeffs, // [K, D]
    // output
    at::opmath_type<scalar_t> *colors // [D]
) {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t result = 0.2820947917738781f * static_cast<opmath_t>(coeffs[c]);
    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        float x = dir.x * inorm;
        float y = dir.y * inorm;
        float z = dir.z * inorm;

        result +=
            0.48860251190292f * (-y * static_cast<opmath_t>(coeffs[1 * D + c]) +
                                 z * static_cast<opmath_t>(coeffs[2 * D + c]) - x * static_cast<opmath_t>(coeffs[3 * D + c]));
        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.f * x * y;
            float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            result += pSH4 * static_cast<opmath_t>(coeffs[4 * D + c]) + pSH5 * static_cast<opmath_t>(coeffs[5 * D + c]) +
                      pSH6 * static_cast<opmath_t>(coeffs[6 * D + c]) + pSH7 * static_cast<opmath_t>(coeffs[7 * D + c]) +
                      pSH8 * static_cast<opmath_t>(coeffs[8 * D + c]);
            if (degree >= 3) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                result +=
                    pSH9 * static_cast<opmath_t>(coeffs[9 * D + c]) + pSH10 * static_cast<opmath_t>(coeffs[10 * D + c]) +
                    pSH11 * static_cast<opmath_t>(coeffs[11 * D + c]) + pSH12 * static_cast<opmath_t>(coeffs[12 * D + c]) +
                    pSH13 * static_cast<opmath_t>(coeffs[13 * D + c]) + pSH14 * static_cast<opmath_t>(coeffs[14 * D + c]) +
                    pSH15 * static_cast<opmath_t>(coeffs[15 * D + c]);

                if (degree >= 4) {
                    float fTmp0D =
                        z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    result += pSH16 * static_cast<opmath_t>(coeffs[16 * D + c]) +
                              pSH17 * static_cast<opmath_t>(coeffs[17 * D + c]) +
                              pSH18 * static_cast<opmath_t>(coeffs[18 * D + c]) +
                              pSH19 * static_cast<opmath_t>(coeffs[19 * D + c]) +
                              pSH20 * static_cast<opmath_t>(coeffs[20 * D + c]) +
                              pSH21 * static_cast<opmath_t>(coeffs[21 * D + c]) +
                              pSH22 * static_cast<opmath_t>(coeffs[22 * D + c]) +
                              pSH23 * static_cast<opmath_t>(coeffs[23 * D + c]) +
                              pSH24 * static_cast<opmath_t>(coeffs[24 * D + c]);
                }
            }
        }
    }

    colors[c] = result;
}

template <typename scalar_t>
__device__ void sh_coeffs_to_color_fast_vjp(
    const uint32_t degree,                     // degree of SH to be evaluated
    const uint32_t D,                          // channels (coeffs last dim)
    const uint32_t c,                          // output channel
    const vec3 &dir,                           // [3]
    const scalar_t *coeffs,                    // [K, D]
    const at::opmath_type<scalar_t> *v_colors, // [D]
    // output
    // v_coeffs is fp32; caller casts back to the coeff dtype after all atomic adds.
    float *v_coeffs,   // [K, D]
    vec3 *v_dir        // [3] optional
) {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t v_colors_local = v_colors[c];

    gpuAtomicAdd(&v_coeffs[c], 0.2820947917738781f * v_colors_local);
    if (degree < 1) {
        return;
    }
    float inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    float x = dir.x * inorm;
    float y = dir.y * inorm;
    float z = dir.z * inorm;
    float v_x = 0.f, v_y = 0.f, v_z = 0.f;

    gpuAtomicAdd(&v_coeffs[1 * D + c], -0.48860251190292f * y * v_colors_local);
    gpuAtomicAdd(&v_coeffs[2 * D + c], 0.48860251190292f * z * v_colors_local);
    gpuAtomicAdd(&v_coeffs[3 * D + c], -0.48860251190292f * x * v_colors_local);

    if (v_dir != nullptr) {
        v_x += -0.48860251190292f * static_cast<opmath_t>(coeffs[3 * D + c]) * v_colors_local;
        v_y += -0.48860251190292f * static_cast<opmath_t>(coeffs[1 * D + c]) * v_colors_local;
        v_z += 0.48860251190292f * static_cast<opmath_t>(coeffs[2 * D + c]) * v_colors_local;
    }
    if (degree < 2) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;
    gpuAtomicAdd(&v_coeffs[4 * D + c], pSH4 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[5 * D + c], pSH5 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[6 * D + c], pSH6 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[7 * D + c], pSH7 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[8 * D + c], pSH8 * v_colors_local);

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x = 2.f * x;
        fC1_y = -2.f * y;
        fS1_x = 2.f * y;
        fS1_y = 2.f * x;
        pSH6_z = 2.f * 0.9461746957575601f * z;
        pSH7_x = fTmp0B;
        pSH7_z = fTmp0B_z * x;
        pSH5_y = fTmp0B;
        pSH5_z = fTmp0B_z * y;
        pSH8_x = 0.5462742152960395f * fC1_x;
        pSH8_y = 0.5462742152960395f * fC1_y;
        pSH4_x = 0.5462742152960395f * fS1_x;
        pSH4_y = 0.5462742152960395f * fS1_y;

        v_x += v_colors_local *
               (pSH4_x * static_cast<opmath_t>(coeffs[4 * D + c]) + pSH8_x * static_cast<opmath_t>(coeffs[8 * D + c]) +
                pSH7_x * static_cast<opmath_t>(coeffs[7 * D + c]));
        v_y += v_colors_local *
               (pSH4_y * static_cast<opmath_t>(coeffs[4 * D + c]) + pSH8_y * static_cast<opmath_t>(coeffs[8 * D + c]) +
                pSH5_y * static_cast<opmath_t>(coeffs[5 * D + c]));
        v_z += v_colors_local *
               (pSH6_z * static_cast<opmath_t>(coeffs[6 * D + c]) + pSH7_z * static_cast<opmath_t>(coeffs[7 * D + c]) +
                pSH5_z * static_cast<opmath_t>(coeffs[5 * D + c]));
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    gpuAtomicAdd(&v_coeffs[9 * D + c], pSH9 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[10 * D + c], pSH10 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[11 * D + c], pSH11 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[12 * D + c], pSH12 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[13 * D + c], pSH13 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[14 * D + c], pSH14 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[15 * D + c], pSH15 * v_colors_local);

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x = fC1 + x * fC1_x - y * fS1_x;
        fC2_y = x * fC1_y - fS1 - y * fS1_y;
        fS2_x = fS1 + x * fS1_x + y * fC1_x;
        fS2_y = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x = fTmp0C;
        pSH13_z = fTmp0C_z * x;
        pSH11_y = fTmp0C;
        pSH11_z = fTmp0C_z * y;
        pSH14_x = fTmp1B * fC1_x;
        pSH14_y = fTmp1B * fC1_y;
        pSH14_z = fTmp1B_z * fC1;
        pSH10_x = fTmp1B * fS1_x;
        pSH10_y = fTmp1B * fS1_y;
        pSH10_z = fTmp1B_z * fS1;
        pSH15_x = -0.5900435899266435f * fC2_x;
        pSH15_y = -0.5900435899266435f * fC2_y;
        pSH9_x = -0.5900435899266435f * fS2_x;
        pSH9_y = -0.5900435899266435f * fS2_y;

        v_x += v_colors_local *
               (pSH9_x * static_cast<opmath_t>(coeffs[9 * D + c]) + pSH15_x * static_cast<opmath_t>(coeffs[15 * D + c]) +
                pSH10_x * static_cast<opmath_t>(coeffs[10 * D + c]) + pSH14_x * static_cast<opmath_t>(coeffs[14 * D + c]) +
                pSH13_x * static_cast<opmath_t>(coeffs[13 * D + c]));

        v_y += v_colors_local *
               (pSH9_y * static_cast<opmath_t>(coeffs[9 * D + c]) + pSH15_y * static_cast<opmath_t>(coeffs[15 * D + c]) +
                pSH10_y * static_cast<opmath_t>(coeffs[10 * D + c]) + pSH14_y * static_cast<opmath_t>(coeffs[14 * D + c]) +
                pSH11_y * static_cast<opmath_t>(coeffs[11 * D + c]));

        v_z += v_colors_local *
               (pSH12_z * static_cast<opmath_t>(coeffs[12 * D + c]) + pSH13_z * static_cast<opmath_t>(coeffs[13 * D + c]) +
                pSH11_z * static_cast<opmath_t>(coeffs[11 * D + c]) + pSH14_z * static_cast<opmath_t>(coeffs[14 * D + c]) +
                pSH10_z * static_cast<opmath_t>(coeffs[10 * D + c]));
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            vec3 dir_n = vec3(x, y, z);
            vec3 v_dir_n = vec3(v_x, v_y, v_z);
            vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

            v_dir->x = v_d.x;
            v_dir->y = v_d.y;
            v_dir->z = v_d.z;
        }
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    gpuAtomicAdd(&v_coeffs[16 * D + c], pSH16 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[17 * D + c], pSH17 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[18 * D + c], pSH18 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[19 * D + c], pSH19 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[20 * D + c], pSH20 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[21 * D + c], pSH21 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[22 * D + c], pSH22 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[23 * D + c], pSH23 * v_colors_local);
    gpuAtomicAdd(&v_coeffs[24 * D + c], pSH24 * v_colors_local);

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x = fC2 + x * fC2_x - y * fS2_x;
        fC3_y = x * fC2_y - fS2 - y * fS2_y;
        fS3_x = fS2 + y * fC2_x + x * fS2_x;
        fS3_y = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
                  -1.006230589874905f * pSH6_z;
        pSH21_x = fTmp0D;
        pSH21_z = fTmp0D_z * x;
        pSH19_y = fTmp0D;
        pSH19_z = fTmp0D_z * y;
        pSH22_x = fTmp1C * fC1_x;
        pSH22_y = fTmp1C * fC1_y;
        pSH22_z = fTmp1C_z * fC1;
        pSH18_x = fTmp1C * fS1_x;
        pSH18_y = fTmp1C * fS1_y;
        pSH18_z = fTmp1C_z * fS1;
        pSH23_x = fTmp2B * fC2_x;
        pSH23_y = fTmp2B * fC2_y;
        pSH23_z = fTmp2B_z * fC2;
        pSH17_x = fTmp2B * fS2_x;
        pSH17_y = fTmp2B * fS2_y;
        pSH17_z = fTmp2B_z * fS2;
        pSH24_x = 0.6258357354491763f * fC3_x;
        pSH24_y = 0.6258357354491763f * fC3_y;
        pSH16_x = 0.6258357354491763f * fS3_x;
        pSH16_y = 0.6258357354491763f * fS3_y;

        v_x += v_colors_local *
               (pSH16_x * static_cast<opmath_t>(coeffs[16 * D + c]) + pSH24_x * static_cast<opmath_t>(coeffs[24 * D + c]) +
                pSH17_x * static_cast<opmath_t>(coeffs[17 * D + c]) + pSH23_x * static_cast<opmath_t>(coeffs[23 * D + c]) +
                pSH18_x * static_cast<opmath_t>(coeffs[18 * D + c]) + pSH22_x * static_cast<opmath_t>(coeffs[22 * D + c]) +
                pSH21_x * static_cast<opmath_t>(coeffs[21 * D + c]));
        v_y += v_colors_local *
               (pSH16_y * static_cast<opmath_t>(coeffs[16 * D + c]) + pSH24_y * static_cast<opmath_t>(coeffs[24 * D + c]) +
                pSH17_y * static_cast<opmath_t>(coeffs[17 * D + c]) + pSH23_y * static_cast<opmath_t>(coeffs[23 * D + c]) +
                pSH18_y * static_cast<opmath_t>(coeffs[18 * D + c]) + pSH22_y * static_cast<opmath_t>(coeffs[22 * D + c]) +
                pSH19_y * static_cast<opmath_t>(coeffs[19 * D + c]));
        v_z += v_colors_local *
               (pSH20_z * static_cast<opmath_t>(coeffs[20 * D + c]) + pSH21_z * static_cast<opmath_t>(coeffs[21 * D + c]) +
                pSH19_z * static_cast<opmath_t>(coeffs[19 * D + c]) + pSH22_z * static_cast<opmath_t>(coeffs[22 * D + c]) +
                pSH18_z * static_cast<opmath_t>(coeffs[18 * D + c]) + pSH23_z * static_cast<opmath_t>(coeffs[23 * D + c]) +
                pSH17_z * static_cast<opmath_t>(coeffs[17 * D + c]));

        vec3 dir_n = vec3(x, y, z);
        vec3 v_dir_n = vec3(v_x, v_y, v_z);
        vec3 v_d = (v_dir_n - glm::dot(v_dir_n, dir_n) * dir_n) * inorm;

        v_dir->x = v_d.x;
        v_dir->y = v_d.y;
        v_dir->z = v_d.z;
    }
}

// Generic forward kernel: any K, one thread per (Gaussian, channel).
// scalar_t is fp16 or fp32; reads are upcast to opmath_t (fp32) inside the helper.
template <typename scalar_t, typename opmath_t>
__global__ void spherical_harmonics_fwd_kernel(
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,       // [..., N, 3]
    const scalar_t *__restrict__ coeffs, // [N, K, D]
    const bool *__restrict__ masks,      // [..., N]
    opmath_t *__restrict__ colors        // [..., N, D]
) {
    // parallelize over B * N * D
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * N * D) {
        return;
    }
    uint32_t elem_id = idx / D;
    uint32_t gaussian_id = elem_id % N;
    uint32_t c = idx % D; // output channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }
    sh_coeffs_to_color_fast<scalar_t>(
        degrees_to_use,
        D,
        c,
        dirs[elem_id],
        coeffs + gaussian_id * K * D,
        colors + elem_id * D
    );
}

// K=16, D=3 forward kernel for the RGB SH hot path. One thread per
// (batch, Gaussian), processes all 3 color channels with uint4 / ushort4
// wide loads. DEGREE is a template parameter so each instantiation sizes
// sh_coeffs[] to exactly what it needs.
template <typename scalar_t, typename opmath_t, int DEGREE>
__global__ void __launch_bounds__(256, 4)
spherical_harmonics_fwd_kernel_k16_3channel(
    const uint32_t B,
    const uint32_t N,
    const vec3 *__restrict__ dirs,          // [..., N, 3]
    const scalar_t *__restrict__ coeffs,    // [N, 16, 3]
    const bool *__restrict__ masks,         // [..., N]
    opmath_t *__restrict__ colors           // [..., N, 3]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * N) {
        return;
    }
    uint32_t gaussian_id = idx % N;
    if (masks != nullptr && !masks[idx]) {
        return;
    }

    coeffs += gaussian_id * 16 * 3;

    constexpr bool COEFFS_FP32 = std::is_same_v<scalar_t, float>;

    // Wide-load shape per (DEGREE, coeff dtype):
    //  DEGREE 0/1 (small payload): ushort4 for fp16, uint4 for fp32, PACK_SIZE = 4.
    //  DEGREE 2/3 (larger payload): always uint4, PACK_SIZE = 4 (fp32) or 8 (fp16).
    constexpr int PACK_SIZE = (DEGREE <= 1) ? 4 : (COEFFS_FP32 ? 4 : 8);
    constexpr int LOAD_COUNT =
        (DEGREE == 0) ? 1 :
        (DEGREE == 1) ? 3 :
        (DEGREE == 2) ? (COEFFS_FP32 ? 7 : 4) :
                        (COEFFS_FP32 ? 12 : 6);
    constexpr int N_COEFFS = LOAD_COUNT * PACK_SIZE;

    using V = std::conditional_t<(DEGREE <= 1) && !COEFFS_FP32, ushort4, uint4>;

    opmath_t out_color[3];
    opmath_t sh_coeffs[N_COEFFS];

    #pragma unroll
    for (int i = 0; i < LOAD_COUNT; ++i) {
        alignas(alignof(V)) scalar_t mem[PACK_SIZE];
        reinterpret_cast<V *>(mem)[0] = reinterpret_cast<const V *>(coeffs)[i];
        #pragma unroll
        for (int j = 0; j < PACK_SIZE; ++j) {
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
) {
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B =
        c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    if (B * N == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    // K=16 wide loads need D == 3 and natural alignment; misaligned tensors
    // fall through to the scalar generic kernel.
    const bool uses_ushort4 =
        (coeffs.scalar_type() == at::kHalf) && (degrees_to_use <= 1);
    const size_t coeffs_align =
        uses_ushort4 ? alignof(ushort4) : alignof(uint4);
    const bool wide_load_aligned =
        (reinterpret_cast<uintptr_t>(coeffs.const_data_ptr()) %
         coeffs_align) == 0;

    if (K == 16 && D == 3 && wide_load_aligned) {
        int64_t n_threads = B * N;
        dim3 threads(256);
        dim3 grid((n_threads + threads.x - 1) / threads.x);

        auto *masks_ptr = masks.has_value()
            ? masks.value().const_data_ptr<bool>()
            : nullptr;

        std::variant<float, at::Half> coeff_variant;
        switch (coeffs.scalar_type()) {
        case at::kFloat: coeff_variant.emplace<float>(); break;
        case at::kHalf:  coeff_variant.emplace<at::Half>(); break;
        default:
            TORCH_CHECK(
                false,
                "spherical_harmonics_fwd_kernel_k16_3channel: unsupported coeffs dtype ",
                coeffs.scalar_type()
            );
        }

        const int deg = std::min<int>(degrees_to_use, 3);

        // coeff dtype and DEGREE resolved at compile time; colors/dirs use
        // opmath_t = at::opmath_type<coeff dtype> (float for both fp16 and fp32).
        const bool dispatched = dispatch::dispatch(
            dispatch::IntParam<0, 1, 2, 3>{deg},
            dispatch::TypeParam<float, at::Half>{coeff_variant},
            [&]<typename DegConst, typename CoeffT>() {
                using opmath_t = at::opmath_type<CoeffT>;
                constexpr int DEGREE = DegConst::value;
                auto *dirs_ptr = reinterpret_cast<const vec3 *>(
                    dirs.const_data_ptr<opmath_t>()
                );
                auto *colors_ptr = colors.data_ptr<opmath_t>();
                spherical_harmonics_fwd_kernel_k16_3channel<CoeffT, opmath_t, DEGREE>
                    <<<grid, threads, 0, stream>>>(
                        B, N, dirs_ptr,
                        coeffs.const_data_ptr<CoeffT>(),
                        masks_ptr, colors_ptr
                    );
            }
        );
        TORCH_CHECK(
            dispatched,
            "spherical_harmonics_fwd_kernel_k16_3channel: dispatch failed for sh_degree=",
            degrees_to_use
        );
    } else {
        int64_t n_elements = static_cast<int64_t>(B) * N * D;
        dim3 threads(256);
        dim3 grid((n_elements + threads.x - 1) / threads.x);
        int64_t shmem_size = 0; // No shared memory used in this kernel

        // Dispatch on the coeff dtype (fp16/fp32); dirs/colors read as opmath_t (float).
        AT_DISPATCH_V2(
            coeffs.scalar_type(),
            "spherical_harmonics_fwd_kernel",
            AT_WRAP([&]() {
                using opmath_t = at::opmath_type<scalar_t>;
                auto *dirs_ptr = reinterpret_cast<const vec3 *>(
                    dirs.const_data_ptr<opmath_t>()
                );
                auto *masks_ptr = masks.has_value()
                    ? masks.value().const_data_ptr<bool>()
                    : nullptr;
                auto *colors_ptr = colors.data_ptr<opmath_t>();

                spherical_harmonics_fwd_kernel<scalar_t, opmath_t>
                    <<<grid, threads, shmem_size, stream>>>(
                        B, N, K, D, degrees_to_use, dirs_ptr,
                        coeffs.const_data_ptr<scalar_t>(),
                        masks_ptr, colors_ptr
                    );
            }),
            at::kFloat,
            at::kHalf
        );
    }
}

template <typename scalar_t, typename opmath_t>
__global__ void spherical_harmonics_bwd_kernel(
    const uint32_t B,
    const uint32_t N,
    const uint32_t K,
    const uint32_t D,
    const uint32_t degrees_to_use,
    const vec3 *__restrict__ dirs,          // [..., N, 3]
    const scalar_t *__restrict__ coeffs,    // [N, K, D]
    const bool *__restrict__ masks,         // [..., N]
    const opmath_t *__restrict__ v_colors,  // [..., N, D]
    float *__restrict__ v_coeffs,           // [N, K, D]
    opmath_t *__restrict__ v_dirs           // [..., N, 3] optional
) {
    // parallelize over B * N * D
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * N * D) {
        return;
    }
    uint32_t elem_id = idx / D;
    uint32_t gaussian_id = elem_id % N;
    uint32_t c = idx % D; // output channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    vec3 v_dir = {0.f, 0.f, 0.f};
    sh_coeffs_to_color_fast_vjp<scalar_t>(
        degrees_to_use,
        D,
        c,
        dirs[elem_id],
        coeffs + gaussian_id * K * D,
        v_colors + elem_id * D,
        v_coeffs + gaussian_id * K * D,
        v_dirs == nullptr ? nullptr : &v_dir
    );

    if (v_dirs != nullptr) {
        gpuAtomicAdd(v_dirs + elem_id * 3, v_dir.x);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
        gpuAtomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
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
) {
    const uint32_t D = coeffs.size(-1);
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = coeffs.size(-3);
    const uint32_t B =
        c10::multiply_integers(dirs.sizes().slice(0, dirs.dim() - 2));

    auto stream = at::cuda::getCurrentCUDAStream();

    // parallelize over B * N * D
    int64_t n_elements = static_cast<int64_t>(B) * N * D;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // Dispatch on the coeff dtype (fp16/fp32). v_coeffs accumulates in fp32;
    // dirs/v_colors/v_dirs read as opmath_t (float).
    AT_DISPATCH_V2(
        coeffs.scalar_type(),
        "spherical_harmonics_bwd_kernel",
        AT_WRAP([&]() {
            using opmath_t = at::opmath_type<scalar_t>;
            auto *dirs_ptr = reinterpret_cast<const vec3 *>(
                dirs.const_data_ptr<opmath_t>()
            );
            auto *masks_ptr = masks.has_value()
                ? masks.value().const_data_ptr<bool>()
                : nullptr;
            auto *v_colors_ptr = v_colors.const_data_ptr<opmath_t>();
            auto *v_dirs_ptr = v_dirs.has_value()
                ? v_dirs.value().data_ptr<opmath_t>()
                : nullptr;

            spherical_harmonics_bwd_kernel<scalar_t, opmath_t>
                <<<grid, threads, shmem_size, stream>>>(
                    B, N, K, D, degrees_to_use, dirs_ptr,
                    coeffs.const_data_ptr<scalar_t>(),
                    masks_ptr, v_colors_ptr,
                    v_coeffs.data_ptr<float>(),
                    v_dirs_ptr
                );
        }),
        at::kFloat,
        at::kHalf
    );
}

} // namespace gsplat
