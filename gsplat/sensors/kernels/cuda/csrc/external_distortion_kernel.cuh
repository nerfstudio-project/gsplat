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

// Device-only helpers for bivariate windshield distortion. The coefficients are
// packed as [h(6), v(15), h_inv(6), v_inv(15)]; `reference_polynomial` decides
// which 21-float half is the active distort or undistort map.
// Keep this header out of host-only translation units.

#pragma once

#include "external_distortion_params.h"
#include "math.cuh"

#include <cuda_runtime.h>

// ===========================================================================
// Polynomial capacities encoded by the coefficient tensor ABI.
// ===========================================================================

// Degree-2 triangular basis: 1, x, x^2, y, xy, y^2.
constexpr int BIVARIATE_H_POLY_TERMS    = 6;
// Degree-4 triangular basis: (4 + 1) * (4 + 2) / 2.
constexpr int BIVARIATE_V_POLY_TERMS    = 15;
// One active direction pair contributes h + v coefficients.
constexpr int BIVARIATE_NUM_DIFF_PARAMS = BIVARIATE_H_POLY_TERMS + BIVARIATE_V_POLY_TERMS;

// Register copy avoids repeatedly indexing the flat global coefficient buffer.
struct BivariateWindshieldParams
{
    float h_poly[BIVARIATE_H_POLY_TERMS]; // horizontal: degree <= 2
    float v_poly[BIVARIATE_V_POLY_TERMS]; // vertical:   degree <= 4
    uint32_t h_poly_degree;
    uint32_t v_poly_degree;
};

// Match BivariateWindshieldParams so callers can scatter the active slice with
// the same indexing used by forward.
struct BivariateParamGrads
{
    float h_poly[BIVARIATE_H_POLY_TERMS];
    float v_poly[BIVARIATE_V_POLY_TERMS];
};

__device__ __forceinline__ BivariateWindshieldParams
    load_bivariate_windshield_params(BivariateWindshieldDistortion_KernelParameters distortion, bool is_undistort);

__device__ __forceinline__ float3 apply_bivariate_distortion(float3 camera_ray, const BivariateWindshieldParams &p);

__device__ __forceinline__ void apply_bivariate_distortion_bwd(
    float3 camera_ray, const BivariateWindshieldParams &p, float3 d_out, float3 &d_camera_ray, BivariateParamGrads &d_p
);

__device__ __forceinline__ uint32_t bivariate_coeff_base(int reference_polynomial, bool is_undistort);

template<int BlockThreads>
__device__ __forceinline__ void reduce_bivariate_param_grads(
    const BivariateParamGrads &local,
    BivariateWindshieldDistortion_KernelParameters distortion,
    bool is_undistort,
    float *__restrict__ grad_distortion_coeffs
)
{
    static_assert(BlockThreads > 0 && BlockThreads <= 1024, "BlockThreads must be between 1 and 1024");
    static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size");
    constexpr int kNumWarps = BlockThreads / 32;
    constexpr int kSlots    = BIVARIATE_NUM_DIFF_PARAMS;
    __shared__ float warp_sums[kNumWarps][kSlots];

    float values[kSlots];
#pragma unroll
    for(int i = 0; i < BIVARIATE_H_POLY_TERMS; ++i)
    {
        values[i] = local.h_poly[i];
    }
#pragma unroll
    for(int i = 0; i < BIVARIATE_V_POLY_TERMS; ++i)
    {
        values[BIVARIATE_H_POLY_TERMS + i] = local.v_poly[i];
    }

    unsigned int mask = 0xFFFFFFFFu;
    int lane          = threadIdx.x & 31;
    int warp          = threadIdx.x >> 5;
#pragma unroll
    for(int i = 0; i < kSlots; ++i)
    {
        float value = values[i];
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            value += __shfl_xor_sync(mask, value, offset);
        }
        if(lane == 0)
        {
            warp_sums[warp][i] = value;
        }
    }
    __syncthreads();

    if(warp == 0 && lane < kSlots)
    {
        float total = 0.0f;
#pragma unroll
        for(int w = 0; w < kNumWarps; ++w)
        {
            total += warp_sums[w][lane];
        }
        if(grad_distortion_coeffs != nullptr)
        {
            uint32_t base = bivariate_coeff_base(distortion.reference_polynomial, is_undistort);
            atomicAdd(&grad_distortion_coeffs[base + lane], total);
        }
    }
}

struct NoExternalDistortionPolicy
{
    using Tag              = NoExternalDistortionPolicyTag;
    using KernelParameters = NoExternalDistortion_KernelParameters;
    using ParamGrads       = BivariateParamGrads;

    struct Params
    {
    };

    static constexpr bool kHasDistortion = false;

    static __device__ Params load(KernelParameters, bool)
    {
        return {};
    }

    static __device__ float3 apply_fwd(float3 ray, const Params &)
    {
        return ray;
    }

    static __device__ float3 apply_inverse(float3 ray, const Params &, float *, int64_t)
    {
        return ray;
    }

    static __device__ float3 inverse_bwd_input(float3 fallback_primal, const float *, int64_t)
    {
        return fallback_primal;
    }

    static __device__ void apply_bwd(float3, const Params &, float3 d_out, float3 &d_in, ParamGrads &)
    {
        d_in.x += d_out.x;
        d_in.y += d_out.y;
        d_in.z += d_out.z;
    }

    template<int BlockThreads>
    static __device__ void reduce_param_grads(const ParamGrads &, KernelParameters, bool, float *)
    {
    }
};

struct BivariateWindshieldPolicy
{
    using Tag              = BivariateWindshieldPolicyTag;
    using KernelParameters = BivariateWindshieldDistortion_KernelParameters;
    using Params           = BivariateWindshieldParams;
    using ParamGrads       = BivariateParamGrads;

    static constexpr bool kHasDistortion = true;

    static __device__ Params load(KernelParameters distortion, bool is_undistort)
    {
        return load_bivariate_windshield_params(distortion, is_undistort);
    }

    static __device__ float3 apply_fwd(float3 ray, const Params &params)
    {
        return apply_bivariate_distortion(ray, params);
    }

    static __device__ float3 apply_inverse(float3 ray, const Params &params, float *scratch, int64_t stash_offset)
    {
        float3 unnorm_out = apply_bivariate_distortion(ray, params);
        // Finish normalization before the inverse stash extends these values.
        float3 camera_ray = normalize3(unnorm_out);
        if(scratch != nullptr && stash_offset >= 0)
        {
            scratch[stash_offset + 0] = unnorm_out.x;
            scratch[stash_offset + 1] = unnorm_out.y;
            scratch[stash_offset + 2] = unnorm_out.z;
        }
        return camera_ray;
    }

    static __device__ float3 inverse_bwd_input(float3, const float *scratch, int64_t stash_offset)
    {
        return make_float3(scratch[stash_offset + 0], scratch[stash_offset + 1], scratch[stash_offset + 2]);
    }

    static __device__ void apply_bwd(float3 ray, const Params &params, float3 d_out, float3 &d_in, ParamGrads &d_params)
    {
        apply_bivariate_distortion_bwd(ray, params, d_out, d_in, d_params);
    }

    template<int BlockThreads>
    static __device__ void reduce_param_grads(
        const ParamGrads &local,
        KernelParameters distortion,
        bool is_undistort,
        float *__restrict__ grad_distortion_coeffs
    )
    {
        reduce_bivariate_param_grads<BlockThreads>(local, distortion, is_undistort, grad_distortion_coeffs);
    }
};

// ===========================================================================
// Coefficient-slice selection
// ===========================================================================

// Selects the 21-float half that acts as the caller's current map. A BACKWARD
// reference polynomial swaps the semantic roles of the stored pairs.
__device__ __forceinline__ uint32_t bivariate_coeff_base(int reference_polynomial, bool is_undistort)
{
    if(is_undistort)
    {
        return reference_polynomial == 0 ? BIVARIATE_NUM_DIFF_PARAMS : 0u;
    }
    return reference_polynomial == 0 ? 0u : BIVARIATE_NUM_DIFF_PARAMS;
}

// Load the active map once per thread; callers pass is_undistort so the same
// policy can serve project and backproject paths.
__device__ __forceinline__ BivariateWindshieldParams
    load_bivariate_windshield_params(BivariateWindshieldDistortion_KernelParameters distortion, bool is_undistort)
{
    uint32_t base = bivariate_coeff_base(distortion.reference_polynomial, is_undistort);
    BivariateWindshieldParams p;
    for(int i = 0; i < BIVARIATE_H_POLY_TERMS; ++i)
    {
        p.h_poly[i] = distortion.distortion_coeffs[base + i];
    }
    for(int i = 0; i < BIVARIATE_V_POLY_TERMS; ++i)
    {
        p.v_poly[i] = distortion.distortion_coeffs[base + BIVARIATE_H_POLY_TERMS + i];
    }
    p.h_poly_degree = distortion.h_poly_degree;
    p.v_poly_degree = distortion.v_poly_degree;
    return p;
}

// ===========================================================================
// Bivariate polynomial evaluation (forward)
// ===========================================================================

// Horner evaluation over y-degree rows of the triangular basis:
// [1, x, x^2, ..., y, xy, ..., y^2, ...]. There is no runtime bounds guard:
// h_poly requires order <= 2 and v_poly requires order <= 4.
__device__ __forceinline__ float eval_poly_2d(float x, float y, const float *__restrict__ c, uint32_t order)
{
    if(order == 0u)
    {
        return c[0];
    }
    if(order == 1u)
    {
        float y_coeff0 = c[0] + x * c[1];
        float y_coeff1 = c[2];
        return y_coeff0 + y * y_coeff1;
    }
    if(order == 2u)
    {
        float y_coeff0 = c[0] + x * (c[1] + x * c[2]);
        float y_coeff1 = c[3] + x * c[4];
        float y_coeff2 = c[5];
        return y_coeff0 + y * (y_coeff1 + y * y_coeff2);
    }
    if(order == 3u)
    {
        float y_coeff0 = c[0] + x * (c[1] + x * (c[2] + x * c[3]));
        float y_coeff1 = c[4] + x * (c[5] + x * c[6]);
        float y_coeff2 = c[7] + x * c[8];
        float y_coeff3 = c[9];
        return y_coeff0 + y * (y_coeff1 + y * (y_coeff2 + y * y_coeff3));
    }
    float y_coeff0 = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])));
    float y_coeff1 = c[5] + x * (c[6] + x * (c[7] + x * c[8]));
    float y_coeff2 = c[9] + x * (c[10] + x * c[11]);
    float y_coeff3 = c[12] + x * c[13];
    float y_coeff4 = c[14];
    return y_coeff0 + y * (y_coeff1 + y * (y_coeff2 + y * (y_coeff3 + y * y_coeff4)));
}

// Partial derivative dP/dx; branches mirror eval_poly_2d's basis ordering.
__device__ __forceinline__ float eval_poly_2d_dfdx(float x, float y, const float *__restrict__ c, uint32_t order)
{
    if(order == 0u)
    {
        return 0.0f;
    }
    if(order == 1u)
    {
        return c[1];
    }
    if(order == 2u)
    {
        float dc0 = c[1] + x * (2.0f * c[2]);
        float dc1 = c[4];
        return dc0 + y * dc1;
    }
    if(order == 3u)
    {
        float dc0 = c[1] + x * (2.0f * c[2] + x * (3.0f * c[3]));
        float dc1 = c[5] + x * (2.0f * c[6]);
        float dc2 = c[8];
        return dc0 + y * (dc1 + y * dc2);
    }
    float dc0 = c[1] + x * (2.0f * c[2] + x * (3.0f * c[3] + x * (4.0f * c[4])));
    float dc1 = c[6] + x * (2.0f * c[7] + x * (3.0f * c[8]));
    float dc2 = c[10] + x * (2.0f * c[11]);
    float dc3 = c[13];
    return dc0 + y * (dc1 + y * (dc2 + y * dc3));
}

// Partial derivative dP/dy; branches mirror eval_poly_2d's basis ordering.
__device__ __forceinline__ float eval_poly_2d_dfdy(float x, float y, const float *__restrict__ c, uint32_t order)
{
    if(order == 0u)
    {
        return 0.0f;
    }
    if(order == 1u)
    {
        return c[2];
    }
    if(order == 2u)
    {
        float yc1 = c[3] + x * c[4];
        return yc1 + y * (2.0f * c[5]);
    }
    if(order == 3u)
    {
        float yc1 = c[4] + x * (c[5] + x * c[6]);
        float yc2 = c[7] + x * c[8];
        return yc1 + y * (2.0f * yc2 + y * (3.0f * c[9]));
    }
    float yc1 = c[5] + x * (c[6] + x * (c[7] + x * c[8]));
    float yc2 = c[9] + x * (c[10] + x * c[11]);
    float yc3 = c[12] + x * c[13];
    return yc1 + y * (2.0f * yc2 + y * (3.0f * yc3 + y * (4.0f * c[14])));
}

// ===========================================================================
// Bivariate polynomial backward pass
// ===========================================================================

// MaxTerms bounds the register accumulator for the active h or v polynomial.
template<uint32_t MaxTerms>
__device__ __forceinline__ void eval_poly_2d_bwd(
    float x,
    float y,
    const float *__restrict__ c,
    uint32_t order,
    float d_result,
    float &d_x,
    float &d_y,
    float (&d_c)[MaxTerms]
)
{
    d_x += d_result * eval_poly_2d_dfdx(x, y, c, order);
    d_y += d_result * eval_poly_2d_dfdy(x, y, c, order);

    float px2 = x * x;
    float py2 = y * y;
    if(order == 0u)
    {
        d_c[0] += d_result;
        return;
    }
    if(order == 1u)
    {
        d_c[0] += d_result;
        d_c[1] += d_result * x;
        d_c[2] += d_result * y;
        return;
    }
    if(order == 2u)
    {
        d_c[0] += d_result;
        d_c[1] += d_result * x;
        d_c[2] += d_result * px2;
        d_c[3] += d_result * y;
        d_c[4] += d_result * x * y;
        d_c[5] += d_result * py2;
        return;
    }
    if(order == 3u)
    {
        float px3  = px2 * x;
        float py3  = py2 * y;
        d_c[0]    += d_result;
        d_c[1]    += d_result * x;
        d_c[2]    += d_result * px2;
        d_c[3]    += d_result * px3;
        d_c[4]    += d_result * y;
        d_c[5]    += d_result * x * y;
        d_c[6]    += d_result * px2 * y;
        d_c[7]    += d_result * py2;
        d_c[8]    += d_result * x * py2;
        d_c[9]    += d_result * py3;
        return;
    }
    float px3  = px2 * x;
    float px4  = px3 * x;
    float py3  = py2 * y;
    float py4  = py3 * y;
    d_c[0]    += d_result;
    d_c[1]    += d_result * x;
    d_c[2]    += d_result * px2;
    d_c[3]    += d_result * px3;
    d_c[4]    += d_result * px4;
    d_c[5]    += d_result * y;
    d_c[6]    += d_result * x * y;
    d_c[7]    += d_result * px2 * y;
    d_c[8]    += d_result * px3 * y;
    d_c[9]    += d_result * py2;
    d_c[10]   += d_result * x * py2;
    d_c[11]   += d_result * px2 * py2;
    d_c[12]   += d_result * py3;
    d_c[13]   += d_result * x * py3;
    d_c[14]   += d_result * py4;
}

// ===========================================================================
// Forward and backward distortion application
// ===========================================================================

// The polynomial maps incident angles to a distorted unit direction. Preserve
// the input z sign so rear-facing rays remain rear-facing after reconstruction.
__device__ __forceinline__ float3 apply_bivariate_distortion(float3 camera_ray, const BivariateWindshieldParams &p)
{
    float inv_len   = rsqrtf(camera_ray.x * camera_ray.x + camera_ray.y * camera_ray.y + camera_ray.z * camera_ray.z);
    float3 ray_norm = make_float3(camera_ray.x * inv_len, camera_ray.y * inv_len, camera_ray.z * inv_len);
    float phi       = asinf(ray_norm.x);
    float theta     = asinf(ray_norm.y);
    float adj_phi   = eval_poly_2d(phi, theta, p.h_poly, p.h_poly_degree);
    float adj_theta = eval_poly_2d(phi, theta, p.v_poly, p.v_poly_degree);
    float x         = sinf(adj_phi);
    float y         = sinf(adj_theta);
    float xy_norm   = x * x + y * y;
    float clamped   = fminf(fmaxf(1.0f - xy_norm, 0.0f), 1.0f);
    float z_sign    = ray_norm.z >= 0.0f ? 1.0f : -1.0f;
    float z         = sqrtf(clamped) * z_sign;
    return make_float3(x, y, z);
}

// Gradients accumulate locally first, then callers scatter with atomicAdd.
// Degenerate rays are skipped so one bad pixel cannot poison global coeff grads.
__device__ __forceinline__ void apply_bivariate_distortion_bwd(
    float3 camera_ray, const BivariateWindshieldParams &p, float3 d_out, float3 &d_camera_ray, BivariateParamGrads &d_p
)
{
    float len_sq = camera_ray.x * camera_ray.x + camera_ray.y * camera_ray.y + camera_ray.z * camera_ray.z;
    // NaN len_sq is rejected here too because comparisons with NaN are false.
    if(!(len_sq > 0.0f))
    {
        return;
    }
    float inv_len   = rsqrtf(len_sq);
    float3 ray_norm = make_float3(camera_ray.x * inv_len, camera_ray.y * inv_len, camera_ray.z * inv_len);
    float phi       = asinf(ray_norm.x);
    float theta     = asinf(ray_norm.y);
    float adj_phi   = eval_poly_2d(phi, theta, p.h_poly, p.h_poly_degree);
    float adj_theta = eval_poly_2d(phi, theta, p.v_poly, p.v_poly_degree);
    float x_out     = sinf(adj_phi);
    float y_out     = sinf(adj_theta);
    float xy_norm   = x_out * x_out + y_out * y_out;
    float clamp_arg = 1.0f - xy_norm;
    bool clamp_open = clamp_arg > 0.0f && clamp_arg < 1.0f;
    float clamped   = fminf(fmaxf(clamp_arg, 0.0f), 1.0f);
    float z_sign    = ray_norm.z >= 0.0f ? 1.0f : -1.0f;

    float d_x = d_out.x;
    float d_y = d_out.y;
    if(clamp_open)
    {
        float sqrt_clamped  = sqrtf(clamped);
        float d_clamped     = 0.5f * z_sign * d_out.z / fmaxf(sqrt_clamped, 1.0e-30f);
        float d_xy_norm     = -d_clamped;
        d_x                += d_xy_norm * 2.0f * x_out;
        d_y                += d_xy_norm * 2.0f * y_out;
    }
    float d_adj_phi   = d_x * cosf(adj_phi);
    float d_adj_theta = d_y * cosf(adj_theta);

    float d_phi   = 0.0f;
    float d_theta = 0.0f;
    eval_poly_2d_bwd<BIVARIATE_V_POLY_TERMS>(
        phi, theta, p.v_poly, p.v_poly_degree, d_adj_theta, d_phi, d_theta, d_p.v_poly
    );
    eval_poly_2d_bwd<BIVARIATE_H_POLY_TERMS>(
        phi, theta, p.h_poly, p.h_poly_degree, d_adj_phi, d_phi, d_theta, d_p.h_poly
    );

    float denom_x      = sqrtf(fmaxf(1.0f - ray_norm.x * ray_norm.x, 1.0e-30f));
    float denom_y      = sqrtf(fmaxf(1.0f - ray_norm.y * ray_norm.y, 1.0e-30f));
    float3 d_ray_norm  = make_float3(d_phi / denom_x, d_theta / denom_y, 0.0f);
    float3 d_ray       = normalize3_bwd(camera_ray, d_ray_norm);
    d_camera_ray.x    += d_ray.x;
    d_camera_ray.y    += d_ray.y;
    d_camera_ray.z    += d_ray.z;
}
