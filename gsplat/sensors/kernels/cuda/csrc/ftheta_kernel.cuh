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

#pragma once

// Device-side math helpers for the FTheta camera CUDA kernels.
// Provides the forward/backward PODs, polynomial evaluators, Newton solvers,
// and project/backproject primitives (with matching adjoints). Shared
// pose/quat/lerp helpers come from camera_kernel.cuh.

#include "camera_kernel.cuh"
#include "camera_params.h"
#include "math.cuh"

#include <cuda_runtime.h>

#include <cstdint>

// Newton-iteration convergence epsilon: short-circuits the solver when |delta|
// or |df| collapses to numerical noise.
constexpr float kFThetaNewtonEpsilon = 1.0e-10f;

// IFT one-step det-guard epsilon. Shared by the forward IFT correction step
// and the backward adjoint so they zero gradient on the same set of
// near-singular rays. Loosening below 1e-10 produces systematic 1-step phase
// offsets under Adam.
constexpr float kFThetaIftDfEpsilon = 1.0e-10f;

// Reference-polynomial selector. Values must match the Python IntEnum.
//   Forward  (0): r = fw_poly(theta) is the direct evaluation; backproject
//                 must Newton-invert fw_poly.
//   Backward (1): theta = bw_poly(rdist) is the direct evaluation; project
//                 must Newton-invert bw_poly.
enum class FThetaReferencePolynomial : int64_t
{
    Forward  = 0,
    Backward = 1,
};

// Number of differentiable scalar slots in the FTheta intrinsic surface:
//   2 (pp) + 6 (fw_poly) + 6 (bw_poly) + 4 (A) + 4 (Ainv) = 22.
constexpr int kFThetaNumDiffParams = 2 + 2 * kFThetaMaxPolynomialTerms + 4 + 4;

// Register-resident parameter cache loaded once per thread. Mirrors the
// OpenCVPinholeParams pattern from camera_kernel.cuh.
struct FThetaParams
{
    float2 principal_point;
    float fw_poly[kFThetaMaxPolynomialTerms];
    float bw_poly[kFThetaMaxPolynomialTerms];
    float A[2][2];
    float Ainv[2][2];
    FThetaReferencePolynomial reference_polynomial;
    int64_t fw_poly_degree;
    int64_t bw_poly_degree;
    int64_t newton_iterations;
    float max_angle;
    float min_2d_norm;
    int64_t width;
    int64_t height;
};

// Forward intermediate state captured per-thread in ftheta_project_ray.
// The bool flags are bit-packed into a single float scratch slot by the
// per-kernel save layer.
struct FThetaProjectState
{
    float3 ray_norm;
    float theta;
    float r;
    float xy_norm;
    bool behind_camera;
    bool angle_clamped;
    bool min2d_clamped;
};

// Backproject intermediate state. `min2d_clamped` short-circuits to (0,0,1)
// with no upstream gradient. `ray_raw` is the pre-normalize3 vector; the bwd
// chains normalize3_bwd through it.
struct FThetaBackprojectState
{
    float2 transformed;
    float rdist;
    float theta;
    float3 ray_raw;
    bool min2d_clamped;
};

// Unpacks the flat KernelParameters into a register-resident FThetaParams.
// FTheta has no fx/fy denominators; the IFT det-guard at kFThetaIftDfEpsilon
// plays the equivalent role on the polynomial Newton path.
__device__ __forceinline__ FThetaParams load_ftheta_params(const FThetaProjection_KernelParameters &projection)
{
    FThetaParams p;
    p.principal_point.x = projection.principal_point[0];
    p.principal_point.y = projection.principal_point[1];
#pragma unroll
    for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
    {
        p.fw_poly[i] = projection.fw_poly[i];
        p.bw_poly[i] = projection.bw_poly[i];
    }
    p.A[0][0]              = projection.A[0];
    p.A[0][1]              = projection.A[1];
    p.A[1][0]              = projection.A[2];
    p.A[1][1]              = projection.A[3];
    p.Ainv[0][0]           = projection.Ainv[0];
    p.Ainv[0][1]           = projection.Ainv[1];
    p.Ainv[1][0]           = projection.Ainv[2];
    p.Ainv[1][1]           = projection.Ainv[3];
    p.reference_polynomial = static_cast<FThetaReferencePolynomial>(projection.reference_polynomial);
    p.fw_poly_degree       = projection.fw_poly_degree;
    p.bw_poly_degree       = projection.bw_poly_degree;
    p.newton_iterations    = projection.newton_iterations;
    p.max_angle            = projection.max_angle;
    p.min_2d_norm          = projection.min_2d_norm;
    p.width                = projection.width;
    p.height               = projection.height;
    return p;
}

// ---- Polynomial helpers ---------------------------------------------------

__device__ __forceinline__ int64_t clamp_ftheta_poly_degree(const int64_t degree)
{
    return degree < kFThetaMaxPolynomialTerms ? degree : kFThetaMaxPolynomialTerms - 1;
}

// Evaluates the power-basis polynomial c[0] + c[1]*x + ... + c[degree]*x^degree.
// The `i <= d` predicate inside the unrolled loop folds into the FMA chain.
__device__ __forceinline__ float ftheta_poly_eval(const float x, const float *__restrict__ c, const int64_t degree)
{
    const int64_t d = clamp_ftheta_poly_degree(degree);
    float result    = 0.0f;
    float x_pow     = 1.0f;
#pragma unroll
    for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
    {
        if(i <= d)
        {
            result += c[i] * x_pow;
        }
        x_pow *= x;
    }
    return result;
}

// Evaluates the polynomial derivative sum_{i>=1} i * c[i] * x^(i-1).
// Returns 0 when degree == 0.
__device__ __forceinline__ float ftheta_poly_eval_derivative(
    const float x, const float *__restrict__ c, const int64_t degree
)
{
    const int64_t d = clamp_ftheta_poly_degree(degree);
    if(d == 0)
    {
        return 0.0f;
    }
    float result = 0.0f;
    float x_pow  = 1.0f;
#pragma unroll
    for(int i = 1; i < kFThetaMaxPolynomialTerms; ++i)
    {
        if(i <= d)
        {
            result += static_cast<float>(i) * c[i] * x_pow;
        }
        x_pow *= x;
    }
    return result;
}

// ---- Newton solvers (non-diff) --------------------------------------------

__device__ __forceinline__ float ftheta_solve_newton(
    const float target,
    const float initial_guess,
    const float *__restrict__ poly,
    const int64_t degree,
    const int64_t newton_iterations
)
{
    float solution = initial_guess;
    for(int64_t i = 0; i < newton_iterations; ++i)
    {
        const float df = ftheta_poly_eval_derivative(solution, poly, degree);
        if(fabsf(df) < kFThetaNewtonEpsilon)
        {
            break;
        }
        const float f      = ftheta_poly_eval(solution, poly, degree);
        const float delta  = (f - target) / df;
        solution          -= delta;
        if(fabsf(delta) < kFThetaNewtonEpsilon)
        {
            break;
        }
    }
    return fmaxf(solution, 0.0f);
}

// Solves bw_poly(r) = theta for r on the BACKWARD reference branch of
// ftheta_project_ray. Initial guess fw_poly(theta) is exact when fw_poly is
// the algebraic inverse of bw_poly; typical convergence is 2-3 iterations.
// Returns max(r, 0) since negative roots are non-physical for radial distance.
__device__ __forceinline__ float ftheta_solve_bw_newton(const float theta, const FThetaParams &p)
{
    const float initial = ftheta_poly_eval(theta, p.fw_poly, p.fw_poly_degree);
    return ftheta_solve_newton(theta, initial, p.bw_poly, p.bw_poly_degree, p.newton_iterations);
}

// Solves fw_poly(theta) = r for theta on the FORWARD reference branch of
// ftheta_backproject_image_point. Initial guess: bw_poly(r).
__device__ __forceinline__ float ftheta_solve_fw_newton(const float r, const FThetaParams &p)
{
    const float initial = ftheta_poly_eval(r, p.bw_poly, p.bw_poly_degree);
    return ftheta_solve_newton(r, initial, p.fw_poly, p.fw_poly_degree, p.newton_iterations);
}

// ---- Bit-packed flag helpers (used by per-kernel scratch save) ------------

// Packs the three singularity flags into one float scratch slot via the
// IEEE-754 bit pattern; the bwd recovers them with ftheta_unpack_flags.
// The slot is only ever read back through __float_as_uint (no float
// arithmetic, no comparisons), so storing a non-canonical bit pattern in
// the float buffer is safe end-to-end.
__device__ __forceinline__ float ftheta_pack_flags(bool behind_camera, bool angle_clamped, bool min2d_clamped)
{
    uint32_t bits = 0u;
    if(behind_camera)
    {
        bits |= 1u;
    }
    if(angle_clamped)
    {
        bits |= 2u;
    }
    if(min2d_clamped)
    {
        bits |= 4u;
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ void ftheta_unpack_flags(
    float packed, bool &behind_camera, bool &angle_clamped, bool &min2d_clamped
)
{
    uint32_t bits = __float_as_uint(packed);
    behind_camera = (bits & 1u) != 0u;
    angle_clamped = (bits & 2u) != 0u;
    min2d_clamped = (bits & 4u) != 0u;
}

// One-flag variant for the backproject path; only `min2d_clamped` is meaningful.
__device__ __forceinline__ float ftheta_bp_pack_flags(bool min2d_clamped)
{
    return __uint_as_float(min2d_clamped ? 1u : 0u);
}

__device__ __forceinline__ bool ftheta_bp_unpack_flags(float packed)
{
    return (__float_as_uint(packed) & 1u) != 0u;
}

// Projects a camera-space ray to an image coordinate using the FTheta model.
// `valid_out` reports projection-success only (ray in front of camera and
// theta <= max_angle). The finite/in-frame check is applied separately at the
// write site via ftheta_image_point_in_frame so rolling-shutter loops can
// recover from mid-iteration poses that briefly fall outside the frame.
//
// Singular cases (in order):
//   1. ray.z <= 0           -> (0, 0), valid=false, behind_camera=true.
//   2. theta > max_angle    -> (0, 0), valid=false, angle_clamped=true.
//   3. xy_norm < min_2d_norm -> principal_point exactly, valid=true,
//                               min2d_clamped=true.
//
// Composition: image_point = A @ (ray_norm.xy * (r / xy_norm)) + pp.
//
// IFT correction (BACKWARD ref branch): non-diff Newton to r_star, then one
// differentiable correction step r = r_star - (bw_poly(r_star) - theta)/safe_df,
// where safe_df = bw_poly'(r_star) when |bw_poly'| > kFThetaIftDfEpsilon and
// 1.0 otherwise. The near-singular sanitization keeps r finite; the matching
// bwd uses the same safe_df branch, including the saturated safe_df = 1.0f
// local derivative and polynomial coefficient adjoint.
__device__ __forceinline__ float2
    ftheta_project_ray(const float3 camera_ray, const FThetaParams &p, FThetaProjectState &state, bool &valid_out)
{
    state.ray_norm      = make_float3(0.0f, 0.0f, 0.0f);
    state.theta         = 0.0f;
    state.r             = 0.0f;
    state.xy_norm       = 0.0f;
    state.behind_camera = false;
    state.angle_clamped = false;
    state.min2d_clamped = false;

    if(camera_ray.z <= 0.0f)
    {
        state.behind_camera = true;
        valid_out           = false;
        return make_float2(0.0f, 0.0f);
    }

    state.ray_norm = normalize3(camera_ray);
    state.xy_norm  = sqrtf(state.ray_norm.x * state.ray_norm.x + state.ray_norm.y * state.ray_norm.y);
    state.theta    = atan2f(state.xy_norm, state.ray_norm.z);
    if(state.theta > p.max_angle)
    {
        state.angle_clamped = true;
        valid_out           = false;
        return make_float2(0.0f, 0.0f);
    }

    if(p.reference_polynomial == FThetaReferencePolynomial::Forward)
    {
        state.r = ftheta_poly_eval(state.theta, p.fw_poly, p.fw_poly_degree);
    }
    else
    {
        // BACKWARD ref: non-diff Newton to r_star, then one-step IFT correction.
        const float r_star  = ftheta_solve_bw_newton(state.theta, p);
        const float f_val   = ftheta_poly_eval(r_star, p.bw_poly, p.bw_poly_degree);
        const float df_val  = ftheta_poly_eval_derivative(r_star, p.bw_poly, p.bw_poly_degree);
        const float safe_df = (fabsf(df_val) > kFThetaIftDfEpsilon) ? df_val : 1.0f;
        state.r             = r_star - (f_val - state.theta) / safe_df;
    }

    // `<=` because min_2d_norm may be configured to 0; the bwd divides by
    // xy_norm, so xy_norm == 0 must take the clamp path.
    if(state.xy_norm <= p.min_2d_norm)
    {
        state.min2d_clamped = true;
        valid_out           = true;
        return p.principal_point;
    }

    const float scale = state.r / state.xy_norm;
    const float2 off  = make_float2(state.ray_norm.x * scale, state.ray_norm.y * scale);
    const float2 xf   = make_float2(p.A[0][0] * off.x + p.A[0][1] * off.y, p.A[1][0] * off.x + p.A[1][1] * off.y);
    valid_out         = true;
    return make_float2(xf.x + p.principal_point.x, xf.y + p.principal_point.y);
}

// Lifts an image-plane pixel to a unit camera-space ray via the FTheta model.
// Composition order:
//   1. offset      = image_point - pp.
//   2. transformed = Ainv @ offset.
//   3. rdist       = sqrt(transformed.x^2 + transformed.y^2).
//   4. rdist < min_2d_norm -> (0, 0, 1) short-circuit, min2d_clamped=true.
//   5. theta       = bw_poly(rdist) (BACKWARD ref) OR Newton-invert fw_poly
//                    on rdist (FORWARD ref) with IFT correction.
//   6. ray_raw     = (transformed.xy * sin(theta)/rdist, cos(theta)).
//   7. return normalize3(ray_raw).
__device__ __forceinline__ float3
    ftheta_backproject_image_point(const float2 image_point, const FThetaParams &p, FThetaBackprojectState &state)
{
    state.transformed   = make_float2(0.0f, 0.0f);
    state.rdist         = 0.0f;
    state.theta         = 0.0f;
    state.ray_raw       = make_float3(0.0f, 0.0f, 1.0f);
    state.min2d_clamped = false;

    const float2 offset = make_float2(image_point.x - p.principal_point.x, image_point.y - p.principal_point.y);
    state.transformed   = make_float2(
        p.Ainv[0][0] * offset.x + p.Ainv[0][1] * offset.y, p.Ainv[1][0] * offset.x + p.Ainv[1][1] * offset.y
    );
    state.rdist = sqrtf(state.transformed.x * state.transformed.x + state.transformed.y * state.transformed.y);

    // `<=` because min_2d_norm may be 0; the bwd divides by rdist, so
    // rdist == 0 must take the clamp path.
    if(state.rdist <= p.min_2d_norm)
    {
        state.min2d_clamped = true;
        return make_float3(0.0f, 0.0f, 1.0f);
    }

    if(p.reference_polynomial == FThetaReferencePolynomial::Backward)
    {
        state.theta = ftheta_poly_eval(state.rdist, p.bw_poly, p.bw_poly_degree);
    }
    else
    {
        // FORWARD ref: Newton-invert fw_poly(theta) = rdist + IFT correction.
        const float theta_star = ftheta_solve_fw_newton(state.rdist, p);
        const float f_val      = ftheta_poly_eval(theta_star, p.fw_poly, p.fw_poly_degree);
        const float df_val     = ftheta_poly_eval_derivative(theta_star, p.fw_poly, p.fw_poly_degree);
        const float safe_df    = (fabsf(df_val) > kFThetaIftDfEpsilon) ? df_val : 1.0f;
        state.theta            = theta_star - (f_val - state.rdist) / safe_df;
    }

    const float sin_theta = sinf(state.theta);
    const float cos_theta = cosf(state.theta);
    const float scale     = sin_theta / state.rdist;
    state.ray_raw         = make_float3(state.transformed.x * scale, state.transformed.y * scale, cos_theta);
    return normalize3(state.ray_raw);
}

// ---- Bounds helpers (FTheta variants) -------------------------------------

// Returns true iff image_point falls inside [0, width) x [0, height).
// FTheta produces pixel-space output directly via the A warp, so width/height
// are read straight off the POD.
__device__ __forceinline__ bool is_in_bounds_ftheta(float2 image_point, int64_t width, int64_t height)
{
    return image_point.x >= 0.0f
        && image_point.x < static_cast<float>(width)
        && image_point.y >= 0.0f
        && image_point.y < static_cast<float>(height);
}

// Final-validity predicate applied at kernel write sites: image_point must be
// finite and inside [0, width) x [0, height). Split out of ftheta_project_ray
// so rolling-shutter loops can iterate back in from a brief out-of-frame pose.
__device__ __forceinline__ bool ftheta_image_point_in_frame(float2 image_point, const FThetaParams &p)
{
    return isfinite(image_point.x) && isfinite(image_point.y) && is_in_bounds_ftheta(image_point, p.width, p.height);
}

// =============================================================================
// Backward POD + device functions
// =============================================================================

// Per-thread intrinsic-grad accumulator. Default-initialized so a thread that
// bails (behind_camera, angle_clamped, etc.) contributes zero to the block
// reduction without touching uninitialized memory. Slot count matches
// kFThetaNumDiffParams (2 + 6 + 6 + 4 + 4 = 22).
struct FThetaParamGrads
{
    float pp[2]                              = {};
    float fw_poly[kFThetaMaxPolynomialTerms] = {};
    float bw_poly[kFThetaMaxPolynomialTerms] = {};
    float A[4]                               = {}; // row-major 2x2
    float Ainv[4]                            = {}; // row-major 2x2
};

// ---- Polynomial bwd helpers ------------------------------------------------

// Backward of ftheta_poly_eval. For y = sum c[i]*x^i:
//   dy/dx    = sum_{i>=1} i * c[i] * x^(i-1) = ftheta_poly_eval_derivative.
//   dy/dc[i] = x^i.
__device__ __forceinline__ void ftheta_poly_eval_bwd(
    const float x,
    const float *__restrict__ c,
    const int64_t degree,
    const float d_y,
    float &__restrict__ d_x,
    float *__restrict__ d_c
)
{
    d_x             += d_y * ftheta_poly_eval_derivative(x, c, degree);
    const int64_t d  = clamp_ftheta_poly_degree(degree);
    float x_pow      = 1.0f;
#pragma unroll
    for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
    {
        if(i <= d)
        {
            d_c[i] += d_y * x_pow;
        }
        x_pow *= x;
    }
}

// Backward of ftheta_poly_eval_derivative. For dy = sum_{i>=1} i*c[i]*x^(i-1):
//   d(dy)/dx    = sum_{i>=2} i*(i-1)*c[i]*x^(i-2).
//   d(dy)/dc[i] = i * x^(i-1) for i >= 1.
__device__ __forceinline__ void ftheta_poly_eval_derivative_bwd(
    const float x,
    const float *__restrict__ c,
    const int64_t degree,
    const float d_dy,
    float &__restrict__ d_x,
    float *__restrict__ d_c
)
{
    const int64_t d = clamp_ftheta_poly_degree(degree);
    if(d == 0)
    {
        return;
    }
    // d(dy)/dx contribution
    float second_deriv = 0.0f;
    {
        float x_pow = 1.0f;
#pragma unroll
        for(int i = 2; i < kFThetaMaxPolynomialTerms; ++i)
        {
            if(i <= d)
            {
                second_deriv += static_cast<float>(i) * static_cast<float>(i - 1) * c[i] * x_pow;
            }
            x_pow *= x;
        }
    }
    d_x += d_dy * second_deriv;
    // d(dy)/dc[i] contribution: i * x^(i-1) for i in [1..d].
    {
        float x_pow = 1.0f; // x^(i-1) starting at i=1
#pragma unroll
        for(int i = 1; i < kFThetaMaxPolynomialTerms; ++i)
        {
            if(i <= d)
            {
                d_c[i] += d_dy * static_cast<float>(i) * x_pow;
            }
            x_pow *= x;
        }
    }
}

// ---- Forward project bwd (camera ray -> image point) ----------------------
//
// Backward of ftheta_project_ray. Recovers d_camera_ray and accumulates into
// d_params. IFT adjoint:
//
//   * Det-guard `|df_val| > kFThetaIftDfEpsilon` zeroes the gradient on the
//     same set of near-singular rays as the forward IFT correction step.
//   * Both terms of the polynomial-coefficient adjoint are emitted: the
//     direct path through f_val and the indirect path through df_val.
//   * r_star is recomputed deterministically via ftheta_solve_bw_newton; the
//     re-solve avoids paying for an extra scratch slot.
__device__ __forceinline__ void ftheta_project_ray_bwd(
    const float3 camera_ray,
    const FThetaParams &p,
    const FThetaProjectState &state,
    const float2 d_img,
    float3 &__restrict__ d_camera_ray,
    FThetaParamGrads &__restrict__ d_params
)
{
    if(state.behind_camera || state.angle_clamped)
    {
        return;
    }
    // d_pp gets the full d_img regardless of the min2d_clamped branch
    // (that branch returns pp exactly).
    d_params.pp[0] += d_img.x;
    d_params.pp[1] += d_img.y;
    if(state.min2d_clamped)
    {
        return;
    }

    // Recover off from saved state (off = ray_norm.xy * (r / xy_norm)).
    const float inv_xy = 1.0f / state.xy_norm;
    const float scale  = state.r * inv_xy;
    const float off_x  = state.ray_norm.x * scale;
    const float off_y  = state.ray_norm.y * scale;

    // d_A row-major.
    d_params.A[0] += d_img.x * off_x;
    d_params.A[1] += d_img.x * off_y;
    d_params.A[2] += d_img.y * off_x;
    d_params.A[3] += d_img.y * off_y;

    // d_off = A^T @ d_img.
    const float d_off_x = p.A[0][0] * d_img.x + p.A[1][0] * d_img.y;
    const float d_off_y = p.A[0][1] * d_img.x + p.A[1][1] * d_img.y;

    // off = ray_norm.xy * scale -> d_scale and d_ray_norm via off chain.
    const float d_scale = state.ray_norm.x * d_off_x + state.ray_norm.y * d_off_y;
    float d_ray_norm_x  = d_off_x * scale;
    float d_ray_norm_y  = d_off_y * scale;

    // scale = r / xy_norm; the d_xy_norm path folds an extra 1/xy_norm
    // through to d_ray_norm via xy_norm = sqrt(ray_norm.x^2 + ray_norm.y^2),
    // giving the inv_xy^3 pre-factor below.
    const float d_r                 = d_scale * inv_xy;
    const float inv_xy3             = inv_xy * inv_xy * inv_xy;
    const float d_xy_norm_via_scale = -d_scale * state.r * inv_xy3;

    // xy_norm = sqrt(ray_norm.x^2 + ray_norm.y^2).
    d_ray_norm_x += d_xy_norm_via_scale * state.ray_norm.x;
    d_ray_norm_y += d_xy_norm_via_scale * state.ray_norm.y;

    // r path: FORWARD ref = direct fw_poly eval; BACKWARD ref = IFT.
    float d_theta = 0.0f;
    if(p.reference_polynomial == FThetaReferencePolynomial::Forward)
    {
        ftheta_poly_eval_bwd(state.theta, p.fw_poly, p.fw_poly_degree, d_r, d_theta, d_params.fw_poly);
    }
    else
    {
        // Recompute r_star deterministically rather than spending a scratch slot.
        const float r_star = ftheta_solve_bw_newton(state.theta, p);
        const float f_val  = ftheta_poly_eval(r_star, p.bw_poly, p.bw_poly_degree);
        const float df_val = ftheta_poly_eval_derivative(r_star, p.bw_poly, p.bw_poly_degree);
        if(fabsf(df_val) > kFThetaIftDfEpsilon)
        {
            const float inv_df    = 1.0f / df_val;
            // r = r_star - (f_val - theta) * inv_df. Treat r_star as constant
            // per the IFT trick, so d_f_val/d_r_star contributes nothing.
            d_theta              += d_r * inv_df;
            const float d_f_val   = -d_r * inv_df;
            const float d_df_val  = d_r * (f_val - state.theta) * inv_df * inv_df;
            // Fused accumulation over d_params.bw_poly[i]: f_val term (all i)
            // and df_val term (i>=1) collapse to one slot-touch each.
            const int64_t bdeg    = clamp_ftheta_poly_degree(p.bw_poly_degree);
            float r_pow           = 1.0f;
            float r_pow_prev      = 0.0f; // r_star^(i-1); unused at i=0
#pragma unroll
            for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
            {
                if(i <= bdeg)
                {
                    float grad = d_f_val * r_pow;
                    if(i >= 1)
                    {
                        grad += d_df_val * static_cast<float>(i) * r_pow_prev;
                    }
                    d_params.bw_poly[i] += grad;
                }
                r_pow_prev  = r_pow;
                r_pow      *= r_star;
            }
        }
        else
        {
            // Saturated branch: safe_df was set to 1.0f in the forward, so
            //   state.r = r_star - f_val + state.theta
            // Local derivatives (r_star treated constant per IFT trick):
            //   d_r/d_theta       = +1
            //   d_r/d_bw_poly[i]  = -r_star^i
            d_theta            += d_r;
            const int64_t bdeg  = clamp_ftheta_poly_degree(p.bw_poly_degree);
            float r_pow         = 1.0f;
#pragma unroll
            for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
            {
                if(i <= bdeg)
                {
                    d_params.bw_poly[i] += -d_r * r_pow;
                }
                r_pow *= r_star;
            }
        }
    }

    // ray_norm is unit length, so the atan2 denominator
    // xy_norm^2 + ray_norm.z^2 is 1.
    const float d_xy_norm_via_theta  = d_theta * state.ray_norm.z;
    d_ray_norm_x                    += d_xy_norm_via_theta * state.ray_norm.x * inv_xy;
    d_ray_norm_y                    += d_xy_norm_via_theta * state.ray_norm.y * inv_xy;

    float3 d_ray_norm  = make_float3(d_ray_norm_x, d_ray_norm_y, -d_theta * state.xy_norm);
    float3 d_in        = normalize3_bwd(camera_ray, d_ray_norm);
    d_camera_ray.x    += d_in.x;
    d_camera_ray.y    += d_in.y;
    d_camera_ray.z    += d_in.z;
}

// ---- Inverse backproject bwd (image point -> camera ray) ------------------
//
// Backward of ftheta_backproject_image_point. Recovers d_image_point from
// d_camera_ray and accumulates into d_params. Mirrors the IFT adjoint of
// ftheta_project_ray_bwd, with theta_star as the converged Newton point on
// the FORWARD-ref branch.
__device__ __forceinline__ void ftheta_backproject_image_point_bwd(
    const float2 image_point,
    const FThetaParams &p,
    const FThetaBackprojectState &state,
    const float3 d_camera_ray,
    float2 &__restrict__ d_image_point,
    FThetaParamGrads &__restrict__ d_params
)
{
    if(state.min2d_clamped)
    {
        // ray = (0, 0, 1) constant; no upstream flow.
        return;
    }

    // d_ray_raw = normalize3_bwd(ray_raw, d_camera_ray).
    const float3 d_ray_raw = normalize3_bwd(state.ray_raw, d_camera_ray);

    // ray_raw.xy = transformed.xy * (sin(theta) / rdist); ray_raw.z = cos(theta).
    const float inv_rdist = 1.0f / state.rdist;
    const float sin_theta = sinf(state.theta);
    const float cos_theta = cosf(state.theta);
    const float scale     = sin_theta * inv_rdist;

    // d_transformed direct contribution from ray_raw.xy = transformed.xy * scale.
    float d_transformed_x = d_ray_raw.x * scale;
    float d_transformed_y = d_ray_raw.y * scale;
    const float d_scale   = state.transformed.x * d_ray_raw.x + state.transformed.y * d_ray_raw.y;

    // scale = sin_theta / rdist.
    const float d_sin_theta = d_scale * inv_rdist;
    float d_rdist           = -d_scale * sin_theta * inv_rdist * inv_rdist;
    const float d_cos_theta = d_ray_raw.z;

    // d_theta from sin/cos.
    float d_theta = d_sin_theta * cos_theta - d_cos_theta * sin_theta;

    // theta path: BACKWARD ref = direct bw_poly(rdist); FORWARD ref = IFT.
    if(p.reference_polynomial == FThetaReferencePolynomial::Backward)
    {
        ftheta_poly_eval_bwd(state.rdist, p.bw_poly, p.bw_poly_degree, d_theta, d_rdist, d_params.bw_poly);
    }
    else
    {
        const float theta_star = ftheta_solve_fw_newton(state.rdist, p);
        const float f_val      = ftheta_poly_eval(theta_star, p.fw_poly, p.fw_poly_degree);
        const float df_val     = ftheta_poly_eval_derivative(theta_star, p.fw_poly, p.fw_poly_degree);
        if(fabsf(df_val) > kFThetaIftDfEpsilon)
        {
            const float inv_df    = 1.0f / df_val;
            d_rdist              += d_theta * inv_df;
            const float d_f_val   = -d_theta * inv_df;
            const float d_df_val  = d_theta * (f_val - state.rdist) * inv_df * inv_df;
            // Fused accumulation over d_params.fw_poly[i] -- mirrors the
            // bw_poly counterpart in ftheta_project_ray_bwd.
            const int64_t fdeg    = clamp_ftheta_poly_degree(p.fw_poly_degree);
            float t_pow           = 1.0f;
            float t_pow_prev      = 0.0f; // theta_star^(i-1); unused at i=0
#pragma unroll
            for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
            {
                if(i <= fdeg)
                {
                    float grad = d_f_val * t_pow;
                    if(i >= 1)
                    {
                        grad += d_df_val * static_cast<float>(i) * t_pow_prev;
                    }
                    d_params.fw_poly[i] += grad;
                }
                t_pow_prev  = t_pow;
                t_pow      *= theta_star;
            }
        }
        else
        {
            // Saturated branch: safe_df was set to 1.0f, so
            //   state.theta = theta_star - f_val + state.rdist
            // Local derivatives (theta_star treated constant):
            //   d_theta/d_rdist      = +1
            //   d_theta/d_fw_poly[i] = -theta_star^i
            d_rdist            += d_theta;
            const int64_t fdeg  = clamp_ftheta_poly_degree(p.fw_poly_degree);
            float t_pow         = 1.0f;
#pragma unroll
            for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
            {
                if(i <= fdeg)
                {
                    d_params.fw_poly[i] += -d_theta * t_pow;
                }
                t_pow *= theta_star;
            }
        }
    }

    // rdist = sqrt(transformed.x^2 + transformed.y^2); chain into d_transformed.
    d_transformed_x += d_rdist * state.transformed.x * inv_rdist;
    d_transformed_y += d_rdist * state.transformed.y * inv_rdist;

    // transformed = Ainv @ offset; d_offset = Ainv^T @ d_transformed.
    const float d_offset_x = p.Ainv[0][0] * d_transformed_x + p.Ainv[1][0] * d_transformed_y;
    const float d_offset_y = p.Ainv[0][1] * d_transformed_x + p.Ainv[1][1] * d_transformed_y;

    // d_Ainv: row-major.
    const float2 offset  = make_float2(image_point.x - p.principal_point.x, image_point.y - p.principal_point.y);
    d_params.Ainv[0]    += d_transformed_x * offset.x;
    d_params.Ainv[1]    += d_transformed_x * offset.y;
    d_params.Ainv[2]    += d_transformed_y * offset.x;
    d_params.Ainv[3]    += d_transformed_y * offset.y;

    // offset = image_point - pp.
    d_image_point.x += d_offset_x;
    d_image_point.y += d_offset_y;
    d_params.pp[0]  += -d_offset_x;
    d_params.pp[1]  += -d_offset_y;
}

struct FThetaProjectionPolicy
{
    using KernelParameters = FThetaProjection_KernelParameters;
    using Params           = FThetaParams;
    using ProjectState     = FThetaProjectState;
    using BackprojectState = FThetaBackprojectState;
    using ParamGrads       = FThetaParamGrads;

    struct IntrinsicGradOutputs
    {
        float *principal_point;
        float *forward_poly;
        float *backward_poly;
        float *A;
        float *Ainv;
    };

    template<DistortionOpFamily Op>
    struct ScratchIO;

    static constexpr DistortionSensor kScratchSensor     = DistortionSensor::FTheta;
    static constexpr bool kGatePosePointBeforeDistortion = true;
    static constexpr bool kNormalizePoseProjectInput     = false;

    static __device__ Params load(const KernelParameters &projection)
    {
        return load_ftheta_params(projection);
    }

    static __device__ float2 project(float3 camera_ray, const Params &params, ProjectState &state, bool &valid)
    {
        return ftheta_project_ray(camera_ray, params, state, valid);
    }

    static __device__ void project_bwd(
        float3 camera_ray,
        const Params &params,
        const ProjectState &state,
        float2 d_image_point,
        float3 &d_camera_ray,
        ParamGrads &d_params
    )
    {
        ftheta_project_ray_bwd(camera_ray, params, state, d_image_point, d_camera_ray, d_params);
    }

    static __device__ float3 backproject(float2 image_point, const Params &params, BackprojectState &state)
    {
        return ftheta_backproject_image_point(image_point, params, state);
    }

    static __device__ float3 backproject_output(const BackprojectState &state)
    {
        return normalize3(state.ray_raw);
    }

    static __device__ void finalize_backproject_state_for_scratch(const Params &, BackprojectState &) { }

    static __device__ void backproject_bwd(
        float2 image_point,
        const Params &params,
        const BackprojectState &state,
        float3 d_camera_ray,
        float2 &d_image_point,
        ParamGrads &d_params
    )
    {
        ftheta_backproject_image_point_bwd(image_point, params, state, d_camera_ray, d_image_point, d_params);
    }

    static __device__ __forceinline__ void backproject_bwd_from_kernel_parameters(
        const KernelParameters &projection,
        float2 image_point,
        const BackprojectState &state,
        float3 d_camera_ray,
        float2 &__restrict__ d_image_point,
        ParamGrads &__restrict__ d_params
    )
    {
        const Params params = load(projection);
        backproject_bwd(image_point, params, state, d_camera_ray, d_image_point, d_params);
    }

    static __device__ bool final_project_valid(float2 image_point, const Params &params, bool projection_valid)
    {
        return projection_valid && ftheta_image_point_in_frame(image_point, params);
    }

    static __device__ void set_rejected_pose_project_state(ProjectState &state)
    {
        state.ray_norm      = make_float3(0.0f, 0.0f, 0.0f);
        state.theta         = 0.0f;
        state.r             = 0.0f;
        state.xy_norm       = 0.0f;
        state.behind_camera = true;
        state.angle_clamped = false;
        state.min2d_clamped = false;
    }

    static __device__ void prepare_pose_project_state_for_backward(ProjectState &state, float3 projected_ray)
    {
        state.ray_norm = normalize3(projected_ray);
    }

    static __device__ bool pose_project_backward_enabled(const ProjectState &state)
    {
        return !state.behind_camera;
    }

    template<int BlockThreads>
    static __device__ void reduce_intrinsic_grads(const ParamGrads &local, const IntrinsicGradOutputs &outputs)
    {
        const float pp0 = block_sum<BlockThreads>(local.pp[0]);
        const float pp1 = block_sum<BlockThreads>(local.pp[1]);
        float forward[kFThetaMaxPolynomialTerms];
        float backward[kFThetaMaxPolynomialTerms];
#pragma unroll
        for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
        {
            forward[i]  = block_sum<BlockThreads>(local.fw_poly[i]);
            backward[i] = block_sum<BlockThreads>(local.bw_poly[i]);
        }
        float affine[4];
        float inverse_affine[4];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            affine[i]         = block_sum<BlockThreads>(local.A[i]);
            inverse_affine[i] = block_sum<BlockThreads>(local.Ainv[i]);
        }

        if(threadIdx.x == 0)
        {
            if(outputs.principal_point != nullptr)
            {
                atomicAdd(&outputs.principal_point[0], pp0);
                atomicAdd(&outputs.principal_point[1], pp1);
            }
            if(outputs.forward_poly != nullptr)
            {
#pragma unroll
                for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
                {
                    atomicAdd(&outputs.forward_poly[i], forward[i]);
                }
            }
            if(outputs.backward_poly != nullptr)
            {
#pragma unroll
                for(int i = 0; i < kFThetaMaxPolynomialTerms; ++i)
                {
                    atomicAdd(&outputs.backward_poly[i], backward[i]);
                }
            }
            if(outputs.A != nullptr)
            {
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    atomicAdd(&outputs.A[i], affine[i]);
                }
            }
            if(outputs.Ainv != nullptr)
            {
#pragma unroll
                for(int i = 0; i < 4; ++i)
                {
                    atomicAdd(&outputs.Ainv[i], inverse_affine[i]);
                }
            }
        }
    }
};

template<>
struct FThetaProjectionPolicy::ScratchIO<DistortionOpFamily::CameraRaysToImagePoints>
{
    template<typename Scratch>
    static __device__ void validate()
    {
        static_assert(Scratch::kScratchStride >= 8);
        static_assert(Scratch::kInverseStashOffset == -1);
    }

    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const ProjectState &state)
    {
        validate<Scratch>();
        scratch[off + 0] = state.ray_norm.x;
        scratch[off + 1] = state.ray_norm.y;
        scratch[off + 2] = state.ray_norm.z;
        scratch[off + 3] = state.theta;
        scratch[off + 4] = state.r;
        scratch[off + 5] = state.xy_norm;
        scratch[off + 6] = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
        scratch[off + 7] = 0.0f;
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, ProjectState &state)
    {
        validate<Scratch>();
        state.ray_norm = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        state.theta    = scratch[off + 3];
        state.r        = scratch[off + 4];
        state.xy_norm  = scratch[off + 5];
        ftheta_unpack_flags(scratch[off + 6], state.behind_camera, state.angle_clamped, state.min2d_clamped);
    }
};

struct FThetaBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        static_assert(Scratch::kScratchStride >= 8);
        static_assert(
            Scratch::kInverseStashOffset == -1
            || (Scratch::kInverseStashOffset >= 8 && Scratch::kInverseStashOffset + 3 <= Scratch::kScratchStride)
        );
    }

    template<typename Scratch>
    static __device__ void save_state(float *scratch, int64_t off, const FThetaBackprojectState &state)
    {
        validate<Scratch>();
        scratch[off + 0] = state.transformed.x;
        scratch[off + 1] = state.transformed.y;
        scratch[off + 2] = state.rdist;
        scratch[off + 3] = state.theta;
        scratch[off + 4] = state.ray_raw.x;
        scratch[off + 5] = state.ray_raw.y;
        scratch[off + 6] = state.ray_raw.z;
        scratch[off + 7] = ftheta_bp_pack_flags(state.min2d_clamped);
    }

    template<typename Scratch>
    static __device__ void load_state(const float *scratch, int64_t off, FThetaBackprojectState &state)
    {
        validate<Scratch>();
        state.transformed   = make_float2(scratch[off + 0], scratch[off + 1]);
        state.rdist         = scratch[off + 2];
        state.theta         = scratch[off + 3];
        state.ray_raw       = make_float3(scratch[off + 4], scratch[off + 5], scratch[off + 6]);
        state.min2d_clamped = ftheta_bp_unpack_flags(scratch[off + 7]);
    }
};

template<>
struct FThetaProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToCameraRays> : FThetaBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const BackprojectState &state)
    {
        save_state<Scratch>(scratch, off, state);
        if constexpr(Scratch::kInverseStashOffset >= 0)
        {
            scratch[off + Scratch::kScratchStride - 1] = 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, BackprojectState &state)
    {
        load_state<Scratch>(scratch, off, state);
    }
};

template<>
struct FThetaProjectionPolicy::ScratchIO<DistortionOpFamily::ProjectWorldPointsMeanPose>
{
    template<typename Scratch>
    static __device__ void validate()
    {
        static_assert(Scratch::kScratchStride >= 10);
        static_assert(Scratch::kInverseStashOffset == -1);
    }

    template<typename Scratch>
    static __device__ void save_forward(
        float *scratch, int64_t off, float3 p_rel, float3 camera_point, const ProjectState &state
    )
    {
        validate<Scratch>();
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = camera_point.x;
        scratch[off + 4] = camera_point.y;
        scratch[off + 5] = camera_point.z;
        scratch[off + 6] = state.theta;
        scratch[off + 7] = state.r;
        scratch[off + 8] = state.xy_norm;
        scratch[off + 9] = ftheta_pack_flags(state.behind_camera, state.angle_clamped, state.min2d_clamped);
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, float3 &p_rel, float3 &camera_point, ProjectState &state
    )
    {
        validate<Scratch>();
        p_rel          = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        camera_point   = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
        state.ray_norm = make_float3(0.0f, 0.0f, 0.0f);
        state.theta    = scratch[off + 6];
        state.r        = scratch[off + 7];
        state.xy_norm  = scratch[off + 8];
        ftheta_unpack_flags(scratch[off + 9], state.behind_camera, state.angle_clamped, state.min2d_clamped);
    }
};

template<>
struct FThetaProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToWorldRaysStaticPose>
    : FThetaBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const BackprojectState &state)
    {
        save_state<Scratch>(scratch, off, state);
        if constexpr(Scratch::kInverseStashOffset >= 0)
        {
            scratch[off + Scratch::kScratchStride - 1] = 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, BackprojectState &state)
    {
        load_state<Scratch>(scratch, off, state);
    }
};

template<>
struct FThetaProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToWorldRaysShutterPose>
    : FThetaBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        FThetaBackprojectScratchIO::validate<Scratch>();
        constexpr int kAlphaOffset = Scratch::kScratchStride - 1;
        static_assert(kAlphaOffset >= 8);
        static_assert(Scratch::kInverseStashOffset == -1 || kAlphaOffset >= Scratch::kInverseStashOffset + 3);
    }

    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const BackprojectState &state, float alpha)
    {
        validate<Scratch>();
        save_state<Scratch>(scratch, off, state);
        scratch[off + Scratch::kScratchStride - 1] = alpha;
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, BackprojectState &state, float &alpha)
    {
        validate<Scratch>();
        load_state<Scratch>(scratch, off, state);
        alpha = scratch[off + Scratch::kScratchStride - 1];
    }
};
