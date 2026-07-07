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

// Device-side math helpers for the OpenCV equidistant fisheye CUDA kernels:
// the forward odd-power radial series and its derivative, the project/backproject
// primitives with matching adjoints, and the fixed-iteration forward Newton
// solver inverted by the IFT one-step backward.
//
// nvcc-only: this header pulls in no torch/ATen headers. Backproject reuses the
// floored normalize3 / normalize3_bwd from math.cuh; the floor never engages on
// the ~unit reconstructed ray.

#include "camera_params.h"
#include "math.cuh"

#include <cuda_runtime.h>

// Numerical floor on ray_xy_norm used by the forward to keep `delta /
// ray_xy_norm` finite. The forward takes max(norm, eps) BEFORE invoking atan2,
// and the backward honors the same clamp so gradients zero out on the
// degenerate near-axis branch. Not a tunable knob.
constexpr float FISHEYE_RAY_XY_NORM_EPS = 1e-6f;

// IFT det-guard for the one-step backward.
constexpr float FISHEYE_IFT_DF_EPSILON = 1e-10f;

// Register-resident parameter cache loaded once per thread. The fisheye surface
// is reduced: anisotropic focal, an odd-power radial series, and a Newton
// initial-guess factor; no A/Ainv affine and no backward polynomial.
struct OpenCVFisheyeParams
{
    float2 principal_point;
    float2 focal_length;
    float forward_poly[kFisheyeForwardPolyTerms]; // forward_poly coefficients k1..k4
    float approx_bw_factor;                       // approx_backward_factor: initial guess for Newton
    float max_angle;
    int64_t newton_iterations;
    float min_2d_norm;
    int64_t width;  // bounds check / shutter timing only; non-diff
    int64_t height; // bounds check / shutter timing only; non-diff
};

// Per-thread accumulator for the 9 differentiable fisheye params:
//   [0..2)  pp        (cx, cy)
//   [2..4)  focal     (fx, fy)
//   [4..8)  forward_poly  (k1, k2, k3, k4)
//   [8]     ab        (approx_backward_factor) — always 0, no gradient
struct OpenCVFisheyeParamGrads
{
    float pp[2];
    float focal[2];
    float forward_poly[kFisheyeForwardPolyTerms];
    float ab;
};

// Unpacks the per-component KernelParameters into a register-resident
// OpenCVFisheyeParams. Reads off the four component pointers rather than a flat
// packed tensor; the resolution scalars are integer config fields.
__device__ __forceinline__ OpenCVFisheyeParams
    load_opencv_fisheye_params(const OpenCVFisheyeProjection_KernelParameters &projection)
{
    OpenCVFisheyeParams p;
    p.principal_point.x = projection.principal_point[0];
    p.principal_point.y = projection.principal_point[1];
    p.focal_length.x    = projection.focal_length[0];
    p.focal_length.y    = projection.focal_length[1];
#pragma unroll
    for(int i = 0; i < kFisheyeForwardPolyTerms; ++i)
    {
        p.forward_poly[i] = projection.forward_poly[i];
    }
    p.approx_bw_factor  = projection.approx_backward_factor[0];
    p.max_angle         = projection.max_angle;
    p.newton_iterations = projection.newton_iterations;
    p.min_2d_norm       = projection.min_2d_norm;
    p.width             = projection.width;
    p.height            = projection.height;
    return p;
}

// Forward polynomial: delta = theta * (1 + k1*θ² + k2*θ⁴ + k3*θ⁶ + k4*θ⁸).
__device__ __forceinline__ float fisheye_eval_forward_poly(const float theta, const float k[4])
{
    const float t2 = theta * theta;
    return theta * (1.0f + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * k[3]))));
}

// Derivative of forward poly w.r.t. theta:
//   1 + 3k1*θ² + 5k2*θ⁴ + 7k3*θ⁶ + 9k4*θ⁸.
__device__ __forceinline__ float fisheye_eval_dforward_poly(const float theta, const float k[4])
{
    const float t2 = theta * theta;
    return 1.0f + t2 * (3.0f * k[0] + t2 * (5.0f * k[1] + t2 * (7.0f * k[2] + t2 * (9.0f * k[3]))));
}

// Saved forward state for camera_rays_to_image_points_opencv_fisheye.
// Project state packs theta then delta (the backproject state packs them in the
// opposite order — do not copy one unpack across directions).
struct FisheyeProjectState
{
    float ray_xy_norm;
    float theta;
    float delta;
    bool behind_camera;   // camera_ray.z <= 0
    bool angle_clamped;   // atan2(...) > max_angle
    bool oob;             // image_point outside resolution bounds
    bool xy_norm_clamped; // raw_xy_norm < eps (forces image_point = pp)
};

// Forward: OpenCV fisheye camera_ray -> image_point. theta is taken from the
// RAW ray via atan2 (no normalize3) and the focal scale is
// anisotropic (independent fx / fy).
__device__ __forceinline__ float2 fisheye_project_ray(
    const float3 camera_ray, const OpenCVFisheyeParams &p, FisheyeProjectState &state, bool &valid_out
)
{
    state.ray_xy_norm     = 0.0f;
    state.theta           = 0.0f;
    state.delta           = 0.0f;
    state.behind_camera   = false;
    state.angle_clamped   = false;
    state.oob             = false;
    state.xy_norm_clamped = false;

    if(camera_ray.z <= 0.0f)
    {
        state.behind_camera = true;
        valid_out           = false;
        return make_float2(0.0f, 0.0f);
    }

    const float raw_xy_norm = sqrtf(camera_ray.x * camera_ray.x + camera_ray.y * camera_ray.y);
    if(raw_xy_norm < FISHEYE_RAY_XY_NORM_EPS)
    {
        state.xy_norm_clamped = true;
        state.ray_xy_norm     = FISHEYE_RAY_XY_NORM_EPS;
    }
    else
    {
        state.ray_xy_norm = raw_xy_norm;
    }

    const float raw_theta = atan2f(state.ray_xy_norm, camera_ray.z);
    if(raw_theta > p.max_angle)
    {
        state.angle_clamped = true;
        state.theta         = p.max_angle;
    }
    else
    {
        state.theta = raw_theta;
    }

    state.delta = fisheye_eval_forward_poly(state.theta, p.forward_poly);

    const float scale   = p.focal_length.x * state.delta / state.ray_xy_norm;
    const float scale_y = p.focal_length.y * state.delta / state.ray_xy_norm;
    const float2 img_point
        = make_float2(scale * camera_ray.x + p.principal_point.x, scale_y * camera_ray.y + p.principal_point.y);

    if(img_point.x < 0.0f
       || img_point.x >= static_cast<float>(p.width)
       || img_point.y < 0.0f
       || img_point.y >= static_cast<float>(p.height))
    {
        state.oob = true;
        valid_out = false;
        return make_float2(0.0f, 0.0f);
    }

    valid_out = true;
    return img_point;
}

// Backward of fisheye_project_ray. Accumulates into d_camera_ray and d_params.
// Adjoints flow through atan2 and the raw sqrt(x²+y²); there is no normalize3_bwd
// and approx_backward_factor receives no gradient.
__device__ __forceinline__ void fisheye_project_ray_bwd(
    const float3 camera_ray,
    const OpenCVFisheyeParams &p,
    const FisheyeProjectState &state,
    const float2 d_image_point,
    float3 &d_camera_ray, // accumulated
    OpenCVFisheyeParamGrads &d_params
)
{ // accumulated
    if(state.behind_camera || state.oob)
    {
        return;
    }

    d_params.pp[0] += d_image_point.x;
    d_params.pp[1] += d_image_point.y;

    const float inv_xy_norm = 1.0f / state.ray_xy_norm;
    const float q           = state.delta * inv_xy_norm; // delta / ray_xy_norm

    d_params.focal[0] += d_image_point.x * q * camera_ray.x;
    d_params.focal[1] += d_image_point.y * q * camera_ray.y;

    const float dA = d_image_point.x * p.focal_length.x;
    const float dB = d_image_point.y * p.focal_length.y;

    const float d_q  = dA * camera_ray.x + dB * camera_ray.y;
    d_camera_ray.x  += dA * q;
    d_camera_ray.y  += dB * q;

    const float d_delta = d_q * inv_xy_norm;
    float d_ray_xy_norm = -d_q * state.delta * inv_xy_norm * inv_xy_norm;

    float d_theta             = d_delta * fisheye_eval_dforward_poly(state.theta, p.forward_poly);
    const float t2            = state.theta * state.theta;
    const float t3            = state.theta * t2;
    const float t5            = t3 * t2;
    const float t7            = t5 * t2;
    const float t9            = t7 * t2;
    d_params.forward_poly[0] += d_delta * t3;
    d_params.forward_poly[1] += d_delta * t5;
    d_params.forward_poly[2] += d_delta * t7;
    d_params.forward_poly[3] += d_delta * t9;

    // approx_backward_factor is never read by the forward, so its adjoint is 0.
    (void)p.approx_bw_factor;
    (void)d_params.ab;

    if(state.angle_clamped)
    {
        d_theta = 0.0f;
    }

    // atan2(u, v): d/du = v / (u² + v²),  d/dv = -u / (u² + v²).
    const float u          = state.ray_xy_norm;
    const float v          = camera_ray.z;
    const float denom      = u * u + v * v;
    const float inv_denom  = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    d_ray_xy_norm         += d_theta * (v * inv_denom);
    d_camera_ray.z        += d_theta * (-u * inv_denom);

    if(state.xy_norm_clamped)
    {
        // raw_xy_norm < eps: the max() stops the subgradient back to ray.xy.
        return;
    }

    const float inv_raw  = 1.0f / state.ray_xy_norm;
    d_camera_ray.x      += d_ray_xy_norm * camera_ray.x * inv_raw;
    d_camera_ray.y      += d_ray_xy_norm * camera_ray.y * inv_raw;
}

// Non-diff Newton solver for fw_poly(theta) = delta. Fixed iteration count with
// an unguarded derivative division: the differentiable Newton it mirrors does
// not guard df_x == 0, and the fixture keeps poly' strictly positive on the
// operating band.
__device__ __forceinline__ float fisheye_solve_fw_newton(const float delta, const OpenCVFisheyeParams &p)
{
    float x = p.approx_bw_factor * delta;
    for(int i = 0; i < p.newton_iterations; ++i)
    {
        const float df_x     = fisheye_eval_dforward_poly(x, p.forward_poly);
        const float residual = fisheye_eval_forward_poly(x, p.forward_poly) - delta;
        x                    = x - residual / df_x;
    }
    return x;
}

// Saved forward state for image_points_to_camera_rays_opencv_fisheye.
// Backproject state packs delta then theta (opposite to the project state).
struct FisheyeBackprojectState
{
    float2 normalized;  // (image - pp) / focal
    float delta;        // sqrt(normalized.x² + normalized.y²)
    float theta;        // Newton solve for fw_poly(theta) = delta
    float3 ray_raw;     // pre-normalize ray
    bool min2d_clamped; // delta <= min_2d_norm -> returns (0,0,1) with no flow
};

// Forward: fisheye image_point -> camera_ray. Returns the normalized ray
// (backproject normalizes the ray) and writes `state` for backward.
__device__ __forceinline__ float3 fisheye_backproject_image_point(
    const float2 image_point, const OpenCVFisheyeParams &p, FisheyeBackprojectState &state
)
{
    state.normalized    = make_float2(0.0f, 0.0f);
    state.delta         = 0.0f;
    state.theta         = 0.0f;
    state.ray_raw       = make_float3(0.0f, 0.0f, 1.0f);
    state.min2d_clamped = false;

    state.normalized = make_float2(
        (image_point.x - p.principal_point.x) / p.focal_length.x,
        (image_point.y - p.principal_point.y) / p.focal_length.y
    );
    state.delta = sqrtf(state.normalized.x * state.normalized.x + state.normalized.y * state.normalized.y);
    // `<=` because min_2d_norm may be configured to 0; the bwd divides by
    // delta, so delta == 0 must take the clamp path.
    if(state.delta <= p.min_2d_norm)
    {
        state.min2d_clamped = true;
        return make_float3(0.0f, 0.0f, 1.0f);
    }

    // No post-loop IFT correction: return Newton's x_final directly.
    state.theta = fisheye_solve_fw_newton(state.delta, p);

    const float sin_theta = sinf(state.theta);
    const float cos_theta = cosf(state.theta);

    const float scale = sin_theta / state.delta;
    state.ray_raw     = make_float3(state.normalized.x * scale, state.normalized.y * scale, cos_theta);
    return normalize3(state.ray_raw);
}

// Backward of fisheye_backproject_image_point. theta = Newton(delta) is handled
// as a pure IFT one-step adjoint re-solved at x_star, det-guarded by
// FISHEYE_IFT_DF_EPSILON (whole-contribution skip on failure, NOT a safe_df=1.0
// fallback). approx_backward_factor stays 0.
__device__ __forceinline__ void fisheye_backproject_image_point_bwd(
    const float2 image_point,
    const OpenCVFisheyeParams &p,
    const FisheyeBackprojectState &state,
    const float3 d_camera_ray_out,
    float2 &d_image_point, // accumulated
    OpenCVFisheyeParamGrads &d_params
)
{ // accumulated
    if(state.min2d_clamped)
    {
        return;
    }

    const float3 d_ray_raw = normalize3_bwd(state.ray_raw, d_camera_ray_out);

    const float sin_theta = sinf(state.theta);
    const float cos_theta = cosf(state.theta);

    float d_theta = -sin_theta * d_ray_raw.z;

    const float inv_delta = 1.0f / state.delta;
    const float scale     = sin_theta * inv_delta;
    float2 d_normalized   = make_float2(d_ray_raw.x * scale, d_ray_raw.y * scale);
    const float d_scale   = d_ray_raw.x * state.normalized.x + d_ray_raw.y * state.normalized.y;

    const float d_sin_theta  = d_scale * inv_delta;
    float d_delta            = -d_scale * sin_theta * inv_delta * inv_delta;
    d_theta                 += d_sin_theta * cos_theta;

    // theta = invert_forward_poly_newton(delta): pure IFT adjoint at x_star.
    //   dtheta/ddelta = 1 / f'(x_star)
    //   dtheta/dk_i    = -x_star^(2i+1) / f'(x_star)
    // No (f - delta) residual term: the forward returns x_star directly.
    // x_star is the converged root the forward already saved into state.theta.
    const float x_star = state.theta;
    const float df_val = fisheye_eval_dforward_poly(x_star, p.forward_poly);
    if(fabsf(df_val) > FISHEYE_IFT_DF_EPSILON)
    {
        const float inv_df        = 1.0f / df_val;
        d_delta                  += d_theta * inv_df;
        const float x2            = x_star * x_star;
        const float x3            = x_star * x2;
        const float x5            = x3 * x2;
        const float x7            = x5 * x2;
        const float x9            = x7 * x2;
        d_params.forward_poly[0] += -d_theta * x3 * inv_df;
        d_params.forward_poly[1] += -d_theta * x5 * inv_df;
        d_params.forward_poly[2] += -d_theta * x7 * inv_df;
        d_params.forward_poly[3] += -d_theta * x9 * inv_df;
    }

    // approx_backward_factor drops out under IFT at a converged root: stays 0.

    d_normalized.x += d_delta * state.normalized.x * inv_delta;
    d_normalized.y += d_delta * state.normalized.y * inv_delta;

    const float inv_fx  = 1.0f / p.focal_length.x;
    const float inv_fy  = 1.0f / p.focal_length.y;
    d_image_point.x    += d_normalized.x * inv_fx;
    d_image_point.y    += d_normalized.y * inv_fy;
    d_params.pp[0]     += -d_normalized.x * inv_fx;
    d_params.pp[1]     += -d_normalized.y * inv_fy;
    d_params.focal[0]  += -d_normalized.x * state.normalized.x * inv_fx;
    d_params.focal[1]  += -d_normalized.y * state.normalized.y * inv_fy;
}

__device__ __forceinline__ float opencv_fisheye_pack_flags(
    bool behind_camera, bool angle_clamped, bool oob, bool xy_norm_clamped
)
{
    uint32_t flags  = 0u;
    flags          |= behind_camera ? 1u : 0u;
    flags          |= angle_clamped ? 2u : 0u;
    flags          |= oob ? 4u : 0u;
    flags          |= xy_norm_clamped ? 8u : 0u;
    return static_cast<float>(flags);
}

__device__ __forceinline__ void opencv_fisheye_unpack_flags(
    float packed, bool &behind_camera, bool &angle_clamped, bool &oob, bool &xy_norm_clamped
)
{
    const uint32_t flags = static_cast<uint32_t>(packed);
    behind_camera        = (flags & 1u) != 0u;
    angle_clamped        = (flags & 2u) != 0u;
    oob                  = (flags & 4u) != 0u;
    xy_norm_clamped      = (flags & 8u) != 0u;
}

struct OpenCVFisheyeProjectionPolicy
{
    using KernelParameters = OpenCVFisheyeProjection_KernelParameters;
    using Params           = OpenCVFisheyeParams;
    using ProjectState     = FisheyeProjectState;
    using BackprojectState = FisheyeBackprojectState;
    using ParamGrads       = OpenCVFisheyeParamGrads;

    struct IntrinsicGradOutputs
    {
        float *principal_point;
        float *focal_length;
        float *forward_poly;
    };

    template<DistortionOpFamily Op>
    struct ScratchIO;

    static constexpr DistortionSensor kScratchSensor     = DistortionSensor::OpenCVFisheye;
    static constexpr bool kGatePosePointBeforeDistortion = false;
    static constexpr bool kNormalizePoseProjectInput     = false;

    static __device__ Params load(const KernelParameters &projection)
    {
        return load_opencv_fisheye_params(projection);
    }

    static __device__ float2 project(float3 camera_ray, const Params &params, ProjectState &state, bool &valid)
    {
        return fisheye_project_ray(camera_ray, params, state, valid);
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
        fisheye_project_ray_bwd(camera_ray, params, state, d_image_point, d_camera_ray, d_params);
    }

    static __device__ float3 backproject(float2 image_point, const Params &params, BackprojectState &state)
    {
        return fisheye_backproject_image_point(image_point, params, state);
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
        fisheye_backproject_image_point_bwd(image_point, params, state, d_camera_ray, d_image_point, d_params);
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

    static __device__ bool final_project_valid(float2, const Params &, bool projection_valid)
    {
        return projection_valid;
    }

    static __device__ void set_rejected_pose_project_state(ProjectState &state)
    {
        state.ray_xy_norm     = 0.0f;
        state.theta           = 0.0f;
        state.delta           = 0.0f;
        state.behind_camera   = true;
        state.angle_clamped   = false;
        state.oob             = false;
        state.xy_norm_clamped = false;
    }

    static __device__ void prepare_pose_project_state_for_backward(ProjectState &, float3) { }

    static __device__ bool pose_project_backward_enabled(const ProjectState &state)
    {
        return !state.behind_camera && !state.oob;
    }

    template<int BlockThreads>
    static __device__ void reduce_intrinsic_grads(const ParamGrads &local, const IntrinsicGradOutputs &outputs)
    {
        static_assert(kFisheyeForwardPolyTerms == 4);
        const float pp0    = block_sum<BlockThreads>(local.pp[0]);
        const float pp1    = block_sum<BlockThreads>(local.pp[1]);
        const float focal0 = block_sum<BlockThreads>(local.focal[0]);
        const float focal1 = block_sum<BlockThreads>(local.focal[1]);
        float forward[kFisheyeForwardPolyTerms];
#pragma unroll
        for(int i = 0; i < kFisheyeForwardPolyTerms; ++i)
        {
            forward[i] = block_sum<BlockThreads>(local.forward_poly[i]);
        }

        if(threadIdx.x == 0)
        {
            if(outputs.principal_point != nullptr)
            {
                atomicAdd(&outputs.principal_point[0], pp0);
                atomicAdd(&outputs.principal_point[1], pp1);
            }
            if(outputs.focal_length != nullptr)
            {
                atomicAdd(&outputs.focal_length[0], focal0);
                atomicAdd(&outputs.focal_length[1], focal1);
            }
            if(outputs.forward_poly != nullptr)
            {
#pragma unroll
                for(int i = 0; i < kFisheyeForwardPolyTerms; ++i)
                {
                    atomicAdd(&outputs.forward_poly[i], forward[i]);
                }
            }
        }
    }
};

template<>
struct OpenCVFisheyeProjectionPolicy::ScratchIO<DistortionOpFamily::CameraRaysToImagePoints>
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
        scratch[off + 0] = state.ray_xy_norm;
        scratch[off + 1] = state.theta;
        scratch[off + 2] = state.delta;
        scratch[off + 3]
            = opencv_fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 4] = 0.0f;
        scratch[off + 5] = 0.0f;
        scratch[off + 6] = 0.0f;
        scratch[off + 7] = 0.0f;
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, ProjectState &state)
    {
        validate<Scratch>();
        state.ray_xy_norm = scratch[off + 0];
        state.theta       = scratch[off + 1];
        state.delta       = scratch[off + 2];
        opencv_fisheye_unpack_flags(
            scratch[off + 3], state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped
        );
    }
};

struct OpenCVFisheyeBackprojectScratchIO
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
    static __device__ void save_state(float *scratch, int64_t off, const FisheyeBackprojectState &state)
    {
        validate<Scratch>();
        scratch[off + 0] = state.normalized.x;
        scratch[off + 1] = state.normalized.y;
        scratch[off + 2] = state.delta;
        scratch[off + 3] = state.theta;
        scratch[off + 4] = state.ray_raw.x;
        scratch[off + 5] = state.ray_raw.y;
        scratch[off + 6] = state.ray_raw.z;
        scratch[off + 7] = state.min2d_clamped ? 1.0f : 0.0f;
    }

    template<typename Scratch>
    static __device__ void load_state(const float *scratch, int64_t off, FisheyeBackprojectState &state)
    {
        validate<Scratch>();
        state.normalized    = make_float2(scratch[off + 0], scratch[off + 1]);
        state.delta         = scratch[off + 2];
        state.theta         = scratch[off + 3];
        state.ray_raw       = make_float3(scratch[off + 4], scratch[off + 5], scratch[off + 6]);
        state.min2d_clamped = static_cast<uint32_t>(scratch[off + 7]) != 0u;
    }
};

template<>
struct OpenCVFisheyeProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToCameraRays>
    : OpenCVFisheyeBackprojectScratchIO
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
struct OpenCVFisheyeProjectionPolicy::ScratchIO<DistortionOpFamily::ProjectWorldPointsMeanPose>
{
    template<typename Scratch>
    static __device__ void validate()
    {
        static_assert(Scratch::kScratchStride >= 14);
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
        scratch[off + 6] = state.ray_xy_norm;
        scratch[off + 7] = state.theta;
        scratch[off + 8] = state.delta;
        scratch[off + 9]
            = opencv_fisheye_pack_flags(state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped);
        scratch[off + 10] = 0.0f;
        scratch[off + 11] = 0.0f;
        scratch[off + 12] = 0.0f;
        scratch[off + 13] = 0.0f;
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, float3 &p_rel, float3 &camera_point, ProjectState &state
    )
    {
        validate<Scratch>();
        p_rel             = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        camera_point      = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
        state.ray_xy_norm = scratch[off + 6];
        state.theta       = scratch[off + 7];
        state.delta       = scratch[off + 8];
        opencv_fisheye_unpack_flags(
            scratch[off + 9], state.behind_camera, state.angle_clamped, state.oob, state.xy_norm_clamped
        );
    }
};

template<>
struct OpenCVFisheyeProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToWorldRaysStaticPose>
    : OpenCVFisheyeBackprojectScratchIO
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
struct OpenCVFisheyeProjectionPolicy::ScratchIO<DistortionOpFamily::ImagePointsToWorldRaysShutterPose>
    : OpenCVFisheyeBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        OpenCVFisheyeBackprojectScratchIO::validate<Scratch>();
        static_assert(Scratch::kScratchStride > 8);
        static_assert(Scratch::kInverseStashOffset == -1 || Scratch::kInverseStashOffset >= 9);
    }

    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const BackprojectState &state, float alpha)
    {
        validate<Scratch>();
        save_state<Scratch>(scratch, off, state);
        scratch[off + 8] = alpha;
    }

    template<typename Scratch>
    static __device__ void load_backward(const float *scratch, int64_t off, BackprojectState &state, float &alpha)
    {
        validate<Scratch>();
        load_state<Scratch>(scratch, off, state);
        alpha = scratch[off + 8];
    }
};
