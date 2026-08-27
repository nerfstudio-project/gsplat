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

#pragma once

#include "Common.h"

// Per-(gaussian, pixel) alpha-blending math for 3DGS rasterization, shared by
// the dense and sparse rasterizers so the numerically sensitive forward and
// backward steps live in one place. These describe the contribution of a single
// gaussian to a single pixel; tile iteration, pixel addressing, shared-memory
// batching and warp reductions stay in the kernels.

namespace gsplat
{
// Per-(gaussian, pixel) gaussian-weight evaluation shared by every 3DGS
// conic-based kernel. Computes the Mahalanobis exponent from the conic and the
// pixel offset (dx, dy) = (mean - pixel), the resulting alpha (clamped to
// MAX_ALPHA), and whether the sample contributes. `valid == false` means the
// caller skips this gaussian (negative exponent or sub-threshold alpha). `vis`
// (== exp(-sigma)) is retained for the backward pass, which needs it directly.
struct GaussianWeight
{
    float vis;   // __expf(-sigma)
    float alpha; // min(MAX_ALPHA, opac * vis)
    bool valid;  // sigma >= 0 and alpha >= ALPHA_THRESHOLD
};

__device__ __forceinline__ GaussianWeight
    eval_gaussian_weight(const vec3 &conic, const float dx, const float dy, const float opac)
{
    const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
    const float vis   = __expf(-sigma);
    const float alpha = min(MAX_ALPHA, opac * vis);
    GaussianWeight out;
    out.vis   = vis;
    out.alpha = alpha;
    out.valid = !(sigma < 0.f || alpha < ALPHA_THRESHOLD);
    return out;
}

// Forward: blend one gaussian into one pixel's running color/transmittance.
// Returns true when the pixel has saturated (transmittance fell to the
// threshold) and should stop accumulating -- the caller marks it done and the
// current gaussian is excluded, matching the dense kernel's behavior.
template<uint32_t CDIM>
__device__ __forceinline__ bool rasterize_to_pixels_3dgs_blend_fwd(
    const vec3 &conic,
    const vec3 &xy_opacity, // (mean.x, mean.y, opacity)
    const float px,
    const float py,
    const float *__restrict__ color_ptr, // colors + g * CDIM
    const uint32_t gaussian_idx,         // running index, stored as cur_idx on hit
    float &T,                            // transmittance, updated on accumulate
    float (&pix_out)[CDIM],              // per-channel accumulator, updated
    uint32_t &cur_idx                    // last contributing gaussian, updated
)
{
    const float opac        = xy_opacity.z;
    const float dx          = xy_opacity.x - px;
    const float dy          = xy_opacity.y - py;
    const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
    if(!gw.valid)
    {
        return false; // negligible contribution; pixel not done
    }
    const float alpha  = gw.alpha;
    const float next_T = T * (1.0f - alpha);
    if(next_T <= TRANSMITTANCE_THRESHOLD)
    {
        return true; // pixel saturated; exclusive of this gaussian
    }
    const float vis = alpha * T;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        pix_out[k] += color_ptr[k] * vis;
    }
    cur_idx = gaussian_idx;
    T       = next_T;
    return false;
}

// Backward: gradient contribution of one gaussian to one pixel. Walks the
// blend in reverse (the caller iterates gaussians back-to-front), updating the
// running transmittance `T` and color `buffer`, and producing this lane's local
// gradients. The caller zero-initializes the `*_local` outputs and reduces them
// across the warp before the atomic scatter. `compute_abs` mirrors absgrad.
template<uint32_t CDIM>
__device__ __forceinline__ void rasterize_to_pixels_3dgs_blend_bwd(
    const vec3 &conic,
    const vec2 &delta, // (mean.x - px, mean.y - py)
    const float opac,
    const float vis, // __expf(-sigma)
    const float alpha,
    const float *__restrict__ rgbs,       // rgbs_batch + t * CDIM
    const float *__restrict__ v_render_c, // [CDIM]
    const float v_render_a,
    const float T_final,
    const float *__restrict__ backgrounds, // [CDIM] or nullptr
    const bool compute_abs,
    float &T,              // running transmittance, updated
    float (&buffer)[CDIM], // running color buffer, updated
    float (&v_rgb_local)[CDIM],
    vec3 &v_conic_local,
    vec2 &v_xy_local,
    vec2 &v_xy_abs_local,
    float &v_opacity_local
)
{
    // compute the current T for this gaussian
    const float ra   = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
    T               *= ra;
    // update v_rgb for this gaussian
    const float fac  = alpha * T;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_rgb_local[k] = fac * v_render_c[k];
    }
    // contribution from this pixel
    float v_alpha = 0.f;
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_alpha += (rgbs[k] * T - buffer[k] * ra) * v_render_c[k];
    }
    v_alpha += T_final * ra * v_render_a;
    // contribution from background pixel
    if(backgrounds != nullptr)
    {
        float accum = 0.f;
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            accum += backgrounds[k] * v_render_c[k];
        }
        v_alpha += -T_final * ra * accum;
    }
    if(opac * vis <= MAX_ALPHA)
    {
        const float v_sigma = -opac * vis * v_alpha;
        v_conic_local
            = {0.5f * v_sigma * delta.x * delta.x, v_sigma * delta.x * delta.y, 0.5f * v_sigma * delta.y * delta.y};
        v_xy_local
            = {v_sigma * (conic.x * delta.x + conic.y * delta.y), v_sigma * (conic.y * delta.x + conic.z * delta.y)};
        if(compute_abs)
        {
            v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
        }
        v_opacity_local = vis * v_alpha;
    }
#pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        buffer[k] += rgbs[k] * fac;
    }
}
} // namespace gsplat
