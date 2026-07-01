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

// Device-side math helpers for the OpenCV pinhole camera CUDA kernels.
// Provides thin private adapters that delegate quaternion math to gsplat/geometry
// (quaternion.cuh, pose.cuh), OpenCV pinhole projection/distortion helpers, and
// a rolling-shutter relative-time helper. Also declares the shared structs
// (OpenCVPinholeParams, DistortionResult, etc.) and the compile-time guard constants.

#pragma once

#include "camera_params.h"
#include "external_distortion_kernel.cuh"
#include "math.cuh"
#include "pose.cuh"
#include "quaternion.cuh"
#include "shutter_type.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdint.h>
#include <type_traits>

// ===========================================================================
// Compile-time constants
// ===========================================================================

// kDenominatorEpsilon — guard against zero denominators in the OpenCV
// radial-distortion factor on both the forward (via safe_nonzero in
// compute_distortion / distortion_den) and backward (compute_distortion_bwd,
// compute_dF_dxy) paths.
constexpr float kDenominatorEpsilon = 1.0e-12f;
// kMinRadialDist / kMaxRadialDist — valid range for the combined radial factor
// icD = num/den. Points outside this band are clamped to the sensor boundary
// rather than distorted, preventing runaway projections.
constexpr float kMinRadialDist      = 0.8f;
constexpr float kMaxRadialDist      = 1.2f;

// ===========================================================================
// Shared structs and type aliases
// ===========================================================================

struct OpenCVPinholeParams
{
    float fx;
    float fy;
    float cx;
    float cy;
    float k[6];
    float p[2];
    float s[4];
    float res_x;
    float res_y;
};

// Keeps focal and principal-point loads outside the distortion VJP live range.
struct OpenCVPinholeLensParams
{
    float k[6];
    float p[2];
    float s[4];
};

struct DistortionResult
{
    float icD;
    float delta_x;
    float delta_y;
    float r2;
    float den;
};

struct DistortionParamGrads
{
    float k[6] = {};
    float p[2] = {};
    float s[4] = {};
};

struct OpenCVPinholeProjectState
{
    float3 projection_input{};
    float x                 = 0.0f;
    float y                 = 0.0f;
    float inv_z             = 0.0f;
    float r2                = 0.0f;
    float icD               = 0.0f;
    float den               = 0.0f;
    bool projection_forward = false;
    bool pose_point_forward = false;
};

struct OpenCVPinholeBackprojectState
{
    float2 xy{};
    float r2  = 0.0f;
    float icD = 0.0f;
    float den = 0.0f;
};

struct OpenCVPinholeParamGrads
{
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    DistortionParamGrads distortion{};
};

struct OpenCVPinholeIntrinsicGradOutputs
{
    float *focal_length;
    float *principal_point;
    float *radial_coeffs;
    float *tangential_coeffs;
    float *thin_prism_coeffs;
};

static_assert(std::is_trivially_copyable_v<OpenCVPinholeProjectState>);
static_assert(std::is_trivially_copyable_v<OpenCVPinholeBackprojectState>);
static_assert(std::is_trivially_copyable_v<OpenCVPinholeParamGrads>);
static_assert(std::is_trivially_copyable_v<OpenCVPinholeIntrinsicGradOutputs>);
static_assert(std::is_trivially_copyable_v<OpenCVPinholeLensParams>);

template<DistortionOpFamily Op>
struct OpenCVPinholeScratchIO;

// ===========================================================================
// Scalar utilities
// ===========================================================================

// Returns `value` if |value| >= kDenominatorEpsilon, otherwise returns
// ±kDenominatorEpsilon with the original sign. Used wherever a float appears
// in a denominator to prevent division-by-zero without branching on a predicate.
__device__ __forceinline__ float safe_nonzero(float value)
{
    if(fabsf(value) >= kDenominatorEpsilon)
    {
        return value;
    }
    return value < 0.0f ? -kDenominatorEpsilon : kDenominatorEpsilon;
}

// ===========================================================================
// Parameter loaders
// ===========================================================================

// Unpacks the flat KernelParameters into OpenCVPinholeParams and applies
// safe_nonzero to fx/fy so downstream reciprocals are always finite.
__device__ __forceinline__ OpenCVPinholeParams
    load_opencv_pinhole_params(OpenCVPinholeProjection_KernelParameters projection)
{
    OpenCVPinholeParams p;
    p.fx = safe_nonzero(projection.focal_length[0]);
    p.fy = safe_nonzero(projection.focal_length[1]);
    p.cx = projection.principal_point[0];
    p.cy = projection.principal_point[1];
    for(int i = 0; i < 6; ++i)
    {
        p.k[i] = projection.radial_coeffs[i];
    }
    p.p[0] = projection.tangential_coeffs[0];
    p.p[1] = projection.tangential_coeffs[1];
    for(int i = 0; i < 4; ++i)
    {
        p.s[i] = projection.thin_prism_coeffs[i];
    }
    p.res_x = static_cast<float>(projection.width);
    p.res_y = static_cast<float>(projection.height);
    return p;
}

__device__ __forceinline__ OpenCVPinholeLensParams
    load_opencv_pinhole_lens_params(OpenCVPinholeProjection_KernelParameters projection)
{
    OpenCVPinholeLensParams params;
    for(int i = 0; i < 6; ++i)
    {
        params.k[i] = projection.radial_coeffs[i];
    }
    params.p[0] = projection.tangential_coeffs[0];
    params.p[1] = projection.tangential_coeffs[1];
    for(int i = 0; i < 4; ++i)
    {
        params.s[i] = projection.thin_prism_coeffs[i];
    }
    return params;
}

// ===========================================================================
// Row-major 3x3 matrix-vector helpers
// ===========================================================================

// M * v where rows are r0, r1, r2 (column-vector convention).
__device__ __forceinline__ float3 mat_vec(float3 r0, float3 r1, float3 r2, float3 v)
{
    return make_float3(dot3(r0, v), dot3(r1, v), dot3(r2, v));
}

// M^T * v — transposed multiply used to apply the inverse rotation (R^T = R^-1 for SO(3)).
__device__ __forceinline__ float3 mat_t_vec(float3 r0, float3 r1, float3 r2, float3 v)
{
    return make_float3(
        r0.x * v.x + r1.x * v.y + r2.x * v.z, r0.y * v.x + r1.y * v.y + r2.y * v.z, r0.z * v.x + r1.z * v.y + r2.z * v.z
    );
}

// ===========================================================================
// Contiguous-buffer read/write helpers
// ===========================================================================

// The read_*/write_* helpers below reinterpret a contiguous float* buffer as
// float3*/float4* and index by element. Callers must pass a pointer obtained
// from tensor.data_ptr<float>() on a contiguous tensor; PyTorch guarantees
// 16-byte alignment, which is sufficient for both float3 (alignof 4) and
// float4 (alignof 16). Do not call these on offset/strided pointers.
__device__ __forceinline__ float3 read_vec3(const float *data, int64_t idx)
{
    return reinterpret_cast<const float3 *>(data)[idx];
}

__device__ __forceinline__ float4 read_quat(const float *data, int64_t idx)
{
    return reinterpret_cast<const float4 *>(data)[idx];
}

// ===========================================================================
// Quaternion component-order swizzles
// ===========================================================================

// Storage convention in gsplat tensors is wxyz; gsplat/geometry uses xyzw.
// These two helpers convert at the call boundary so the rest of the code
// stays in the gsplat/geometry xyzw convention.
__device__ __forceinline__ float4 wxyz_to_xyzw(float4 q)
{
    return make_float4(q.y, q.z, q.w, q.x);
}

__device__ __forceinline__ float4 xyzw_to_wxyz(float4 q)
{
    return make_float4(q.w, q.x, q.y, q.z);
}

// ===========================================================================
// Quaternion adapters (delegate to gsplat/geometry)
// ===========================================================================

// Private call-boundary glue for gsplat/geometry quaternion primitives. These
// helpers keep sensorlib callsites float3/float4-shaped while delegating the
// actual quaternion math to gsplat/geometry/kernels/cuda/csrc/quaternion.cuh.

// Rotates v by q (xyzw order). Wraps quat_rotate_vector_fwd_impl<float>
// from gsplat/geometry; inputs and outputs stay in float3/float4 shapes.
__device__ __forceinline__ float3 quat_rotate_xyzw_geom(float4 q, float3 v)
{
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    gsplat_geometry::quat_rotate_vector_fwd_impl<float>(q.x, q.y, q.z, q.w, v.x, v.y, v.z, &ox, &oy, &oz);
    return make_float3(ox, oy, oz);
}

// Inverse-rotates v by q (xyzw order). Negates the vector part to form the
// conjugate quaternion then delegates to quat_rotate_xyzw_geom.
__device__ __forceinline__ float3 quat_inverse_rotate_xyzw_geom(float4 q, float3 v)
{
    return quat_rotate_xyzw_geom(make_float4(-q.x, -q.y, -q.z, q.w), v);
}

// Fills r0/r1/r2 with the rows of the rotation matrix for q (xyzw order).
// Wraps quat_to_matrix_fwd_write<float> from gsplat/geometry.
__device__ __forceinline__ void quat_to_matrix_xyzw_geom(float4 q, float3 &r0, float3 &r1, float3 &r2)
{
    float r[9];
    gsplat_geometry::quat_to_matrix_fwd_write<float>(q.x, q.y, q.z, q.w, r);
    r0 = make_float3(r[0], r[1], r[2]);
    r1 = make_float3(r[3], r[4], r[5]);
    r2 = make_float3(r[6], r[7], r[8]);
}

// Same as quat_to_matrix_xyzw_geom but accepts the tensor-storage wxyz order.
// The wxyz↔xyzw swizzle is applied at this call boundary via wxyz_to_xyzw.
__device__ __forceinline__ void quat_to_matrix_wxyz_geom(float4 q_wxyz, float3 &r0, float3 &r1, float3 &r2)
{
    quat_to_matrix_xyzw_geom(wxyz_to_xyzw(q_wxyz), r0, r1, r2);
}

// Normalizes q (xyzw order) to unit length. Wraps quat_normalize_safe_fwd_write<float>
// from gsplat/geometry; safe against the zero quaternion.
__device__ __forceinline__ float4 normalize_quat_geom(float4 q)
{
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    float ow = 0.0f;
    gsplat_geometry::quat_normalize_safe_fwd_write<float>(q.x, q.y, q.z, q.w, &ox, &oy, &oz, &ow);
    return make_float4(ox, oy, oz, ow);
}

// Backward (VJP) call-boundary glue for gsplat/geometry quaternion primitives.
// Same shape as the forward `_geom` helpers above, but accumulating into
// caller-supplied float4&/float3& grad slots.
__device__ __forceinline__ void quat_rotate_bwd_xyzw_geom(float4 q, float3 v, float3 d_v_out, float4 &d_q, float3 &d_v)
{
    float gqx = 0.0f;
    float gqy = 0.0f;
    float gqz = 0.0f;
    float gqw = 0.0f;
    float gvx = 0.0f;
    float gvy = 0.0f;
    float gvz = 0.0f;
    gsplat_geometry::quat_rotate_vector_bwd_impl<float>(
        q.x, q.y, q.z, q.w, v.x, v.y, v.z, d_v_out.x, d_v_out.y, d_v_out.z, &gqx, &gqy, &gqz, &gqw, &gvx, &gvy, &gvz
    );
    d_q.x += gqx;
    d_q.y += gqy;
    d_q.z += gqz;
    d_q.w += gqw;
    d_v.x += gvx;
    d_v.y += gvy;
    d_v.z += gvz;
}

__device__ __forceinline__ void quat_inverse_rotate_bwd_xyzw_geom(
    float4 q, float3 v, float3 d_v_out, float4 &d_q, float3 &d_v
)
{
    float gqx = 0.0f;
    float gqy = 0.0f;
    float gqz = 0.0f;
    float gqw = 0.0f;
    float gvx = 0.0f;
    float gvy = 0.0f;
    float gvz = 0.0f;
    gsplat_geometry::quat_rotate_vector_bwd_impl<float>(
        -q.x, -q.y, -q.z, q.w, v.x, v.y, v.z, d_v_out.x, d_v_out.y, d_v_out.z, &gqx, &gqy, &gqz, &gqw, &gvx, &gvy, &gvz
    );
    // Conjugate chain rule: imaginary parts negate, real part unchanged.
    d_q.x += -gqx;
    d_q.y += -gqy;
    d_q.z += -gqz;
    d_q.w += gqw;
    d_v.x += gvx;
    d_v.y += gvy;
    d_v.z += gvz;
}

// Reads a wxyz quaternion from the tensor buffer and converts it to xyzw for
// use in the gsplat/geometry adapter layer; the swizzle is applied here.
__device__ __forceinline__ float4 read_quat_xyzw_from_wxyz(const float *data, int64_t idx)
{
    return wxyz_to_xyzw(read_quat(data, idx));
}

__device__ __forceinline__ void write_vec3(float *data, int64_t idx, float3 v)
{
    reinterpret_cast<float3 *>(data)[idx] = v;
}

__device__ __forceinline__ void write_quat(float *data, int64_t idx, float4 q)
{
    reinterpret_cast<float4 *>(data)[idx] = q;
}

// Converts q from xyzw to wxyz order and writes it to the tensor buffer.
__device__ __forceinline__ void write_quat_wxyz_from_xyzw(float *data, int64_t idx, float4 q)
{
    write_quat(data, idx, xyzw_to_wxyz(q));
}

// ===========================================================================
// float3 interpolation
// ===========================================================================

__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t)
{
    return make_float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}

// ===========================================================================
// OpenCV pinhole distortion helpers
// ===========================================================================

// Computes the OpenCV pinhole distortion at normalised image-plane coordinates xy.
// Returns the combined radial factor icD = num/den, tangential/thin-prism
// offsets (delta_x, delta_y), squared radius r2, and the denominator den for
// reuse in the backward. xy is in normalised camera-space (x/z, y/z).
__device__ __forceinline__ DistortionResult compute_distortion(float2 xy, const OpenCVPinholeParams &p)
{
    float x       = xy.x;
    float y       = xy.y;
    float r2      = x * x + y * y;
    float xy_prod = x * y;
    float a1      = 2.0f * xy_prod;
    float a2      = r2 + 2.0f * x * x;
    float a3      = r2 + 2.0f * y * y;
    float num     = 1.0f + r2 * (p.k[0] + r2 * (p.k[1] + r2 * p.k[2]));
    float den     = safe_nonzero(1.0f + r2 * (p.k[3] + r2 * (p.k[4] + r2 * p.k[5])));
    DistortionResult result;
    result.icD     = num * __frcp_rn(den);
    result.delta_x = p.p[0] * a1 + p.p[1] * a2 + r2 * (p.s[0] + r2 * p.s[1]);
    result.delta_y = p.p[0] * a3 + p.p[1] * a1 + r2 * (p.s[2] + r2 * p.s[3]);
    result.r2      = r2;
    result.den     = den;
    return result;
}

// Returns the denominator polynomial den(r2), clamped away from zero.
__device__ __forceinline__ float distortion_den(float r2, const OpenCVPinholeParams &p)
{
    return safe_nonzero(1.0f + r2 * (p.k[3] + r2 * (p.k[4] + r2 * p.k[5])));
}

inline __device__ float2 opencv_pinhole_undistort_step(float2 xy0, float2 xy, const OpenCVPinholeParams &params)
{
    const DistortionResult distortion = compute_distortion(xy, params);
    return make_float2((xy0.x - distortion.delta_x) / distortion.icD, (xy0.y - distortion.delta_y) / distortion.icD);
}

// Returns true iff image_point falls inside [0, res_x) x [0, res_y).
__device__ __forceinline__ bool is_in_bounds(float2 image_point, const OpenCVPinholeParams &p)
{
    return image_point.x >= 0.0f && image_point.x < p.res_x && image_point.y >= 0.0f && image_point.y < p.res_y;
}

// ===========================================================================
// Rolling-shutter timing helpers
// ===========================================================================

// Converts a per-pixel relative scan time in [0, 1] to an absolute timestamp
// in microseconds by linearly interpolating between start and end. Uses double
// precision internally to avoid rounding error on large timestamps.
__device__ __forceinline__ int64_t
    timestamp_from_relative_time(float relative_time, int64_t start_timestamp_us, int64_t end_timestamp_us)
{
    double start    = static_cast<double>(start_timestamp_us);
    double duration = static_cast<double>(end_timestamp_us - start_timestamp_us);
    return static_cast<int64_t>(start + static_cast<double>(relative_time) * duration);
}

// Returns the relative scan time in [0, 1] for image_point given the shutter
// direction. GLOBAL shutter always returns 0. For rolling shutters the scanline
// (or column) index is floored/ceiled to the nearest integer scan boundary
// before dividing by (dim - 1).
__device__ __forceinline__ float compute_relative_frame_time_opencv(
    float2 image_point, int64_t width, int64_t height, int64_t shutter_type
)
{
    float safe_w  = static_cast<float>(std::max(width, static_cast<int64_t>(2)) - 1);
    float safe_h  = static_cast<float>(std::max(height, static_cast<int64_t>(2)) - 1);
    const auto st = static_cast<gsplat_sensors::ShutterType>(shutter_type);
    if(st == gsplat_sensors::ShutterType::ROLLING_TOP_TO_BOTTOM)
    {
        return floorf(image_point.y) / safe_h;
    }
    if(st == gsplat_sensors::ShutterType::ROLLING_BOTTOM_TO_TOP)
    {
        return (static_cast<float>(height) - ceilf(image_point.y)) / safe_h;
    }
    if(st == gsplat_sensors::ShutterType::ROLLING_LEFT_TO_RIGHT)
    {
        return floorf(image_point.x) / safe_w;
    }
    if(st == gsplat_sensors::ShutterType::ROLLING_RIGHT_TO_LEFT)
    {
        return (static_cast<float>(width) - ceilf(image_point.x)) / safe_w;
    }
    return 0.0f;
}

// ===========================================================================
// Distortion backward and Jacobian helpers
// ===========================================================================

// Backward of compute_distortion. d_out packs upstream gradients as
// (d_icD, d_delta_x, d_delta_y, d_r2). Reads cached primal scalars (x, y, r2,
// icD, den) produced in the forward; returns early without writing to d_xy or
// d_params if |den| < kDenominatorEpsilon to avoid propagating degenerate grads.
template<typename LensParams>
__device__ __forceinline__ void compute_distortion_bwd(
    float x,
    float y,
    float r2,
    float icD,
    float den,
    const LensParams &params,
    float4 d_out,
    float2 &__restrict__ d_xy,
    DistortionParamGrads &__restrict__ d_params
)
{
    if(fabsf(den) < kDenominatorEpsilon)
    {
        return;
    }
    float d_icD     = d_out.x;
    float d_delta_x = d_out.y;
    float d_delta_y = d_out.z;
    float d_r2_out  = d_out.w;
    float r2_2      = r2 * r2;
    float r2_3      = r2_2 * r2;
    float inv_den   = __frcp_rn(den);
    float dnum_dr2  = params.k[0] + r2 * (2.0f * params.k[1] + r2 * 3.0f * params.k[2]);
    float dden_dr2  = params.k[3] + r2 * (2.0f * params.k[4] + r2 * 3.0f * params.k[5]);
    float dicD_dr2  = (dnum_dr2 - icD * dden_dr2) * inv_den;

    d_params.k[0] += d_icD * (r2 * inv_den);
    d_params.k[1] += d_icD * (r2_2 * inv_den);
    d_params.k[2] += d_icD * (r2_3 * inv_den);
    d_params.k[3] += d_icD * (-icD * r2 * inv_den);
    d_params.k[4] += d_icD * (-icD * r2_2 * inv_den);
    d_params.k[5] += d_icD * (-icD * r2_3 * inv_den);

    float d_r2_accum = d_icD * dicD_dr2 + d_r2_out;
    float two_x      = 2.0f * x;
    float two_y      = 2.0f * y;
    float two_xy     = 2.0f * x * y;
    float a2         = r2 + 2.0f * x * x;
    float a3         = r2 + 2.0f * y * y;

    d_params.p[0] += d_delta_x * two_xy + d_delta_y * a3;
    d_params.p[1] += d_delta_x * a2 + d_delta_y * two_xy;
    d_params.s[0] += d_delta_x * r2;
    d_params.s[1] += d_delta_x * r2_2;
    d_params.s[2] += d_delta_y * r2;
    d_params.s[3] += d_delta_y * r2_2;

    float d_x   = 0.0f;
    float d_y   = 0.0f;
    d_x        += d_delta_x * (params.p[0] * two_y + params.p[1] * (4.0f * x));
    d_y        += d_delta_x * (params.p[0] * two_x);
    d_r2_accum += d_delta_x * (params.p[1] + params.s[0] + 2.0f * r2 * params.s[1]);
    d_x        += d_delta_y * (params.p[1] * two_y);
    d_y        += d_delta_y * (params.p[0] * (4.0f * y) + params.p[1] * two_x);
    d_r2_accum += d_delta_y * (params.p[0] + params.s[2] + 2.0f * r2 * params.s[3]);
    d_x        += d_r2_accum * two_x;
    d_y        += d_r2_accum * two_y;
    d_xy.x     += d_x;
    d_xy.y     += d_y;
}

// Fills M[2][2] with the 2x2 Jacobian dF/dxy of the distortion fixed-point
// F(xy) = icD*xy + delta used by the iterative undistort. The IFT backward
// solves M^T * d_xy_dist = d_xy_undist using solve_2x2_transposed.
// Zeroes M and returns early if |den| < kDenominatorEpsilon (degenerate).
template<typename LensParams>
__device__ __forceinline__ void compute_dF_dxy(
    float x, float y, float r2, float icD, float den, const LensParams &params, float M[2][2]
)
{
    if(fabsf(den) < kDenominatorEpsilon)
    {
        M[0][0] = 0.0f;
        M[0][1] = 0.0f;
        M[1][0] = 0.0f;
        M[1][1] = 0.0f;
        return;
    }
    float inv_den  = __frcp_rn(den);
    float dnum_dr2 = params.k[0] + r2 * (2.0f * params.k[1] + r2 * 3.0f * params.k[2]);
    float dden_dr2 = params.k[3] + r2 * (2.0f * params.k[4] + r2 * 3.0f * params.k[5]);
    float dicD_dr2 = (dnum_dr2 - icD * dden_dr2) * inv_den;
    float dicD_dx  = dicD_dr2 * (2.0f * x);
    float dicD_dy  = dicD_dr2 * (2.0f * y);
    float d_deltaX_dx
        = params.p[0] * (2.0f * y) + params.p[1] * (6.0f * x) + (params.s[0] + 2.0f * r2 * params.s[1]) * (2.0f * x);
    float d_deltaX_dy
        = params.p[0] * (2.0f * x) + params.p[1] * (2.0f * y) + (params.s[0] + 2.0f * r2 * params.s[1]) * (2.0f * y);
    float d_deltaY_dx
        = params.p[0] * (2.0f * x) + params.p[1] * (2.0f * y) + (params.s[2] + 2.0f * r2 * params.s[3]) * (2.0f * x);
    float d_deltaY_dy
        = params.p[0] * (6.0f * y) + params.p[1] * (2.0f * x) + (params.s[2] + 2.0f * r2 * params.s[3]) * (2.0f * y);
    M[0][0] = icD + x * dicD_dx + d_deltaX_dx;
    M[0][1] = x * dicD_dy + d_deltaX_dy;
    M[1][0] = y * dicD_dx + d_deltaY_dx;
    M[1][1] = icD + y * dicD_dy + d_deltaY_dy;
}

// Solves M^T * x = b for x. Returns (0, 0) if |det(M)| < 1e-8 (ill-conditioned
// IFT Jacobian); returning zero gradient is safer than propagating a 1e+20-scale
// result that would pass isfinite() and poison Adam moments.
__device__ __forceinline__ float2 solve_2x2_transposed(float M[2][2], float2 b)
{
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if(fabsf(det) < 1.0e-8f)
    {
        return make_float2(0.0f, 0.0f);
    }
    float inv_det = 1.0f / det;
    return make_float2(inv_det * (M[1][1] * b.x - M[1][0] * b.y), inv_det * (-M[0][1] * b.x + M[0][0] * b.y));
}

enum class OpenCVPinholeUndistortStrategyId
{
    EarlyExit,
    FixedTen,
};

struct EarlyExitIntrinsicUndistort
{
    static constexpr OpenCVPinholeUndistortStrategyId kId = OpenCVPinholeUndistortStrategyId::EarlyExit;

    static __device__ float2 solve(float2 xy0, const OpenCVPinholeParams &params)
    {
        float2 xy = xy0;
        for(int i = 0; i < 10; ++i)
        {
            const float2 next     = opencv_pinhole_undistort_step(xy0, xy, params);
            const float max_delta = fmaxf(fabsf(next.x - xy.x), fabsf(next.y - xy.y));
            xy                    = next;
            if(max_delta < 1.0e-8f)
            {
                break;
            }
        }
        return xy;
    }
};

struct FixedTenIntrinsicUndistort
{
    static constexpr OpenCVPinholeUndistortStrategyId kId = OpenCVPinholeUndistortStrategyId::FixedTen;

    static __device__ float2 solve(float2 xy0, const OpenCVPinholeParams &params)
    {
        // Unconditional unrolled iterations keep every lane on one schedule.
        float2 xy = xy0;
#pragma unroll
        for(int i = 0; i < 10; ++i)
        {
            xy = opencv_pinhole_undistort_step(xy0, xy, params);
        }
        return xy;
    }
};

__device__ __forceinline__ float2 opencv_pinhole_project_compact(
    float3 camera_ray, const OpenCVPinholeParams &params, float *__restrict__ compact_state, bool &__restrict__ valid
)
{
    const bool projection_forward = camera_ray.z > 0.0f;
    float x                       = 0.0f;
    float y                       = 0.0f;
    float inv_z                   = 0.0f;
    float r2                      = 0.0f;
    float icD                     = 0.0f;
    float den                     = 0.0f;
    valid                         = false;
    float2 image_point            = make_float2(0.0f, 0.0f);
    if(projection_forward)
    {
        inv_z                             = 1.0f / camera_ray.z;
        x                                 = camera_ray.x * inv_z;
        y                                 = camera_ray.y * inv_z;
        const DistortionResult distortion = compute_distortion(make_float2(x, y), params);
        r2                                = distortion.r2;
        icD                               = distortion.icD;
        den                               = distortion_den(r2, params);
        if(icD < kMinRadialDist || icD > kMaxRadialDist)
        {
            const float resolution_length = sqrtf(params.res_x * params.res_x + params.res_y * params.res_y);
            const float inv_radius        = r2 > 0.0f ? rsqrtf(r2) : 0.0f;
            image_point.x                 = x * resolution_length * inv_radius + params.cx;
            image_point.y                 = y * resolution_length * inv_radius + params.cy;
        }
        else
        {
            const float distorted_x = x * icD + distortion.delta_x;
            const float distorted_y = y * icD + distortion.delta_y;
            image_point.x           = distorted_x * params.fx + params.cx;
            image_point.y           = distorted_y * params.fy + params.cy;
        }
        valid = is_in_bounds(image_point, params);
    }

    if(compact_state != nullptr)
    {
        compact_state[0] = x;
        compact_state[1] = y;
        compact_state[2] = inv_z;
        compact_state[3] = r2;
        compact_state[4] = icD;
        compact_state[5] = den;
    }
    return image_point;
}

inline __device__ float2 opencv_pinhole_project(
    float3 camera_ray,
    const OpenCVPinholeParams &params,
    OpenCVPinholeProjectState &__restrict__ state,
    bool &__restrict__ valid
)
{
    float compact_state[6];
    const float2 image_point = opencv_pinhole_project_compact(camera_ray, params, compact_state, valid);
    state                    = {};
    state.projection_input   = camera_ray;
    state.x                  = compact_state[0];
    state.y                  = compact_state[1];
    state.inv_z              = compact_state[2];
    state.r2                 = compact_state[3];
    state.icD                = compact_state[4];
    state.den                = compact_state[5];
    state.projection_forward = camera_ray.z > 0.0f;
    return image_point;
}

inline __device__ void prepare_opencv_pinhole_project_state_for_backward(
    OpenCVPinholeProjectState &__restrict__ state, float3 projected_ray
)
{
    state.projection_input   = projected_ray;
    state.projection_forward = projected_ray.z > 0.0f;
    state.x                  = 0.0f;
    state.y                  = 0.0f;
    state.inv_z              = 0.0f;
    if(state.projection_forward)
    {
        state.inv_z = 1.0f / projected_ray.z;
        state.x     = projected_ray.x * state.inv_z;
        state.y     = projected_ray.y * state.inv_z;
    }
}

inline __device__ float2 opencv_pinhole_image_plane_bwd(
    const OpenCVPinholeParams &params,
    const OpenCVPinholeProjectState &state,
    float2 d_image_point,
    OpenCVPinholeParamGrads &__restrict__ d_params
)
{
    d_params.cx += d_image_point.x;
    d_params.cy += d_image_point.y;
    float2 d_xy  = make_float2(0.0f, 0.0f);

    if(state.icD < kMinRadialDist || state.icD > kMaxRadialDist)
    {
        const float resolution_length  = sqrtf(params.res_x * params.res_x + params.res_y * params.res_y);
        const float radius             = sqrtf(fmaxf(state.r2, 1.0e-30f));
        const float factor             = resolution_length / radius;
        d_xy.x                        += d_image_point.x * factor;
        d_xy.y                        += d_image_point.y * factor;
        const float d_r2               = (d_image_point.x * state.x + d_image_point.y * state.y)
                                       * (-0.5f * resolution_length / (radius * state.r2));
        compute_distortion_bwd(
            state.x,
            state.y,
            state.r2,
            state.icD,
            state.den,
            params,
            make_float4(0.0f, 0.0f, 0.0f, d_r2),
            d_xy,
            d_params.distortion
        );
        return d_xy;
    }

    const float xy      = 2.0f * state.x * state.y;
    const float x_term  = state.r2 + 2.0f * state.x * state.x;
    const float y_term  = state.r2 + 2.0f * state.y * state.y;
    const float delta_x = params.p[0] * xy + params.p[1] * x_term + state.r2 * (params.s[0] + state.r2 * params.s[1]);
    const float delta_y = params.p[0] * y_term + params.p[1] * xy + state.r2 * (params.s[2] + state.r2 * params.s[3]);
    const float distorted_x  = state.x * state.icD + delta_x;
    const float distorted_y  = state.y * state.icD + delta_y;
    d_params.fx             += d_image_point.x * distorted_x;
    d_params.fy             += d_image_point.y * distorted_y;

    const float d_distorted_x  = d_image_point.x * params.fx;
    const float d_distorted_y  = d_image_point.y * params.fy;
    d_xy.x                    += d_distorted_x * state.icD;
    d_xy.y                    += d_distorted_y * state.icD;
    const float d_icD          = d_distorted_x * state.x + d_distorted_y * state.y;
    compute_distortion_bwd(
        state.x,
        state.y,
        state.r2,
        state.icD,
        state.den,
        params,
        make_float4(d_icD, d_distorted_x, d_distorted_y, 0.0f),
        d_xy,
        d_params.distortion
    );
    return d_xy;
}

inline __device__ void opencv_pinhole_project_bwd(
    float3 camera_ray,
    const OpenCVPinholeParams &params,
    const OpenCVPinholeProjectState &state,
    float2 d_image_point,
    float3 &__restrict__ d_camera_ray,
    OpenCVPinholeParamGrads &__restrict__ d_params
)
{
    (void)camera_ray;
    const float2 d_xy  = opencv_pinhole_image_plane_bwd(params, state, d_image_point, d_params);
    d_camera_ray.x    += d_xy.x * state.inv_z;
    d_camera_ray.y    += d_xy.y * state.inv_z;
    d_camera_ray.z    += -(state.x * d_xy.x + state.y * d_xy.y) * state.inv_z;
}

template<typename IntrinsicUndistort>
inline __device__ float3 opencv_pinhole_backproject(
    float2 image_point, const OpenCVPinholeParams &params, OpenCVPinholeBackprojectState &__restrict__ state
)
{
    static_assert(
        IntrinsicUndistort::kId == OpenCVPinholeUndistortStrategyId::EarlyExit
        || IntrinsicUndistort::kId == OpenCVPinholeUndistortStrategyId::FixedTen
    );
    const float2 distorted_xy
        = make_float2((image_point.x - params.cx) / params.fx, (image_point.y - params.cy) / params.fy);
    state.xy = IntrinsicUndistort::solve(distorted_xy, params);
    return normalize3(make_float3(state.xy.x, state.xy.y, 1.0f));
}

inline __device__ float3 opencv_pinhole_backproject_output(const OpenCVPinholeBackprojectState &state)
{
    return normalize3(make_float3(state.xy.x, state.xy.y, 1.0f));
}

template<typename LensParams>
__device__ __forceinline__ float2 opencv_pinhole_backproject_lens_bwd(
    const LensParams &params,
    float xs,
    float ys,
    float r2,
    float icD,
    float den,
    float3 d_camera_ray,
    OpenCVPinholeParamGrads &__restrict__ d_params
)
{
    const float3 d_ray_pre = normalize3_bwd(make_float3(xs, ys, 1.0f), d_camera_ray);
    const float2 d_xy_star = make_float2(d_ray_pre.x, d_ray_pre.y);

    float jacobian[2][2];
    compute_dF_dxy(xs, ys, r2, icD, den, params, jacobian);
    const float2 d_distorted_xy = solve_2x2_transposed(jacobian, d_xy_star);

    float2 unused_d_xy = make_float2(0.0f, 0.0f);
    const float d_icD  = -(d_distorted_xy.x * xs + d_distorted_xy.y * ys);
    compute_distortion_bwd(
        xs,
        ys,
        r2,
        icD,
        den,
        params,
        make_float4(d_icD, -d_distorted_xy.x, -d_distorted_xy.y, 0.0f),
        unused_d_xy,
        d_params.distortion
    );

    return d_distorted_xy;
}

__device__ __forceinline__ float2 opencv_pinhole_backproject_calibration_bwd(
    float2 image_point,
    float fx,
    float fy,
    float cx,
    float cy,
    float2 d_distorted_xy,
    OpenCVPinholeParamGrads &__restrict__ d_params
)
{
    d_params.cx += -d_distorted_xy.x / fx;
    d_params.cy += -d_distorted_xy.y / fy;
    d_params.fx += -d_distorted_xy.x * (image_point.x - cx) / (fx * fx);
    d_params.fy += -d_distorted_xy.y * (image_point.y - cy) / (fy * fy);
    return make_float2(d_distorted_xy.x / fx, d_distorted_xy.y / fy);
}

__device__ __forceinline__ float2 opencv_pinhole_backproject_bwd(
    const OpenCVPinholeParams &params,
    float2 image_point,
    float xs,
    float ys,
    float r2,
    float icD,
    float den,
    float3 d_camera_ray,
    OpenCVPinholeParamGrads &__restrict__ d_params
)
{
    const float2 d_distorted_xy
        = opencv_pinhole_backproject_lens_bwd(params, xs, ys, r2, icD, den, d_camera_ray, d_params);
    return opencv_pinhole_backproject_calibration_bwd(
        image_point, params.fx, params.fy, params.cx, params.cy, d_distorted_xy, d_params
    );
}

template<int BlockThreads>
inline __device__ void reduce_opencv_pinhole_grads(
    const OpenCVPinholeParamGrads &local, const OpenCVPinholeIntrinsicGradOutputs &outputs
)
{
    const float focal_x     = block_sum<BlockThreads>(local.fx);
    const float focal_y     = block_sum<BlockThreads>(local.fy);
    const float principal_x = block_sum<BlockThreads>(local.cx);
    const float principal_y = block_sum<BlockThreads>(local.cy);
    float radial[6];
    float tangential[2];
    float thin_prism[4];
#pragma unroll
    for(int i = 0; i < 6; ++i)
    {
        radial[i] = block_sum<BlockThreads>(local.distortion.k[i]);
    }
#pragma unroll
    for(int i = 0; i < 2; ++i)
    {
        tangential[i] = block_sum<BlockThreads>(local.distortion.p[i]);
    }
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        thin_prism[i] = block_sum<BlockThreads>(local.distortion.s[i]);
    }

    if(threadIdx.x == 0)
    {
        if(outputs.focal_length != nullptr)
        {
            atomicAdd(&outputs.focal_length[0], focal_x);
            atomicAdd(&outputs.focal_length[1], focal_y);
        }
        if(outputs.principal_point != nullptr)
        {
            atomicAdd(&outputs.principal_point[0], principal_x);
            atomicAdd(&outputs.principal_point[1], principal_y);
        }
        if(outputs.radial_coeffs != nullptr)
        {
#pragma unroll
            for(int i = 0; i < 6; ++i)
            {
                atomicAdd(&outputs.radial_coeffs[i], radial[i]);
            }
        }
        if(outputs.tangential_coeffs != nullptr)
        {
#pragma unroll
            for(int i = 0; i < 2; ++i)
            {
                atomicAdd(&outputs.tangential_coeffs[i], tangential[i]);
            }
        }
        if(outputs.thin_prism_coeffs != nullptr)
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                atomicAdd(&outputs.thin_prism_coeffs[i], thin_prism[i]);
            }
        }
    }
}

template<typename IntrinsicUndistort>
struct OpenCVPinholeProjectionPolicy
{
    using IntrinsicUndistortStrategy = IntrinsicUndistort;
    using KernelParameters           = OpenCVPinholeProjection_KernelParameters;
    using Params                     = OpenCVPinholeParams;
    using ProjectState               = OpenCVPinholeProjectState;
    using BackprojectState           = OpenCVPinholeBackprojectState;
    using ParamGrads                 = OpenCVPinholeParamGrads;
    using IntrinsicGradOutputs       = OpenCVPinholeIntrinsicGradOutputs;

    template<DistortionOpFamily Op>
    using ScratchIO = OpenCVPinholeScratchIO<Op>;

    static constexpr DistortionSensor kScratchSensor     = DistortionSensor::OpenCVPinhole;
    static constexpr bool kGatePosePointBeforeDistortion = true;

    static_assert(
        IntrinsicUndistort::kId == OpenCVPinholeUndistortStrategyId::EarlyExit
        || IntrinsicUndistort::kId == OpenCVPinholeUndistortStrategyId::FixedTen
    );

    static __device__ Params load(const KernelParameters &projection)
    {
        return load_opencv_pinhole_params(projection);
    }

    static __device__ float2
        project(float3 camera_ray, const Params &params, ProjectState &__restrict__ state, bool &__restrict__ valid)
    {
        return opencv_pinhole_project(camera_ray, params, state, valid);
    }

    static __device__ void project_bwd(
        float3 camera_ray,
        const Params &params,
        const ProjectState &state,
        float2 d_image_point,
        float3 &__restrict__ d_camera_ray,
        ParamGrads &__restrict__ d_params
    )
    {
        if(!state.projection_forward)
        {
            return;
        }
        opencv_pinhole_project_bwd(camera_ray, params, state, d_image_point, d_camera_ray, d_params);
    }

    static __device__ float3 backproject(float2 image_point, const Params &params, BackprojectState &__restrict__ state)
    {
        return opencv_pinhole_backproject<IntrinsicUndistort>(image_point, params, state);
    }

    static __device__ float3 backproject_output(const BackprojectState &state)
    {
        return opencv_pinhole_backproject_output(state);
    }

    static __device__ void finalize_backproject_state_for_scratch(
        const Params &params, BackprojectState &__restrict__ state
    )
    {
        const DistortionResult distortion = compute_distortion(state.xy, params);
        state.r2                          = distortion.r2;
        state.icD                         = distortion.icD;
        state.den                         = distortion.den;
    }

    static __device__ void backproject_bwd(
        float2 image_point,
        const Params &params,
        const BackprojectState &state,
        float3 d_camera_ray,
        float2 &__restrict__ d_image_point,
        ParamGrads &__restrict__ d_params
    )
    {
        const float2 d_input = opencv_pinhole_backproject_bwd(
            params, image_point, state.xy.x, state.xy.y, state.r2, state.icD, state.den, d_camera_ray, d_params
        );
        d_image_point.x += d_input.x;
        d_image_point.y += d_input.y;
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
        const OpenCVPinholeLensParams lens_params = load_opencv_pinhole_lens_params(projection);
        const float2 d_distorted_xy               = opencv_pinhole_backproject_lens_bwd(
            lens_params, state.xy.x, state.xy.y, state.r2, state.icD, state.den, d_camera_ray, d_params
        );
        const float2 d_input = opencv_pinhole_backproject_calibration_bwd(
            image_point,
            safe_nonzero(projection.focal_length[0]),
            safe_nonzero(projection.focal_length[1]),
            projection.principal_point[0],
            projection.principal_point[1],
            d_distorted_xy,
            d_params
        );
        d_image_point.x += d_input.x;
        d_image_point.y += d_input.y;
    }

    static __device__ bool final_project_valid(float2, const Params &, bool projection_valid)
    {
        return projection_valid;
    }

    static __device__ void set_rejected_pose_project_state(ProjectState &__restrict__ state)
    {
        state = {};
    }

    static __device__ void prepare_pose_project_state_for_backward(
        ProjectState &__restrict__ state, float3 projected_ray
    )
    {
        prepare_opencv_pinhole_project_state_for_backward(state, projected_ray);
    }

    static __device__ bool pose_project_backward_enabled(const ProjectState &state)
    {
        return state.pose_point_forward;
    }

    template<int BlockThreads>
    static __device__ void reduce_intrinsic_grads(const ParamGrads &local, const IntrinsicGradOutputs &outputs)
    {
        reduce_opencv_pinhole_grads<BlockThreads>(local, outputs);
    }
};

using PinholeEarlyExitPolicy = OpenCVPinholeProjectionPolicy<EarlyExitIntrinsicUndistort>;
using PinholeFixedTenPolicy  = OpenCVPinholeProjectionPolicy<FixedTenIntrinsicUndistort>;

static_assert(PinholeEarlyExitPolicy::IntrinsicUndistortStrategy::kId == OpenCVPinholeUndistortStrategyId::EarlyExit);
static_assert(PinholeFixedTenPolicy::IntrinsicUndistortStrategy::kId == OpenCVPinholeUndistortStrategyId::FixedTen);

template<typename Scratch>
constexpr bool kOpenCVPinholeNoExternalScratch
    = std::is_same_v<typename Scratch::PolicyTag, NoExternalDistortionPolicyTag>;

template<typename Scratch>
constexpr bool kOpenCVPinholeBivariateScratch
    = std::is_same_v<typename Scratch::PolicyTag, BivariateWindshieldPolicyTag>;

template<typename Scratch>
constexpr __device__ void validate_opencv_pinhole_scratch_policy()
{
    static_assert(kOpenCVPinholeNoExternalScratch<Scratch> || kOpenCVPinholeBivariateScratch<Scratch>);
}

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::CameraRaysToImagePoints>
{
    template<typename Scratch>
    static __device__ void validate()
    {
        validate_opencv_pinhole_scratch_policy<Scratch>();
        static_assert(!Scratch::kIsUndistort);
        static_assert(Scratch::kInverseStashOffset == -1);
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            static_assert(Scratch::kScratchStride == 6);
        }
        else
        {
            static_assert(Scratch::kScratchStride == 10);
        }
    }

    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const OpenCVPinholeProjectState &state)
    {
        validate<Scratch>();
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            scratch[off + 0] = state.x;
            scratch[off + 1] = state.y;
            scratch[off + 2] = state.inv_z;
            scratch[off + 3] = state.r2;
            scratch[off + 4] = state.icD;
            scratch[off + 5] = state.den;
        }
        else
        {
            scratch[off + 0] = state.projection_input.x;
            scratch[off + 1] = state.projection_input.y;
            scratch[off + 2] = state.projection_input.z;
            scratch[off + 3] = state.x;
            scratch[off + 4] = state.y;
            scratch[off + 5] = state.inv_z;
            scratch[off + 6] = state.r2;
            scratch[off + 7] = state.icD;
            scratch[off + 8] = state.den;
            scratch[off + 9] = state.projection_forward ? 1.0f : 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, OpenCVPinholeProjectState &__restrict__ state
    )
    {
        validate<Scratch>();
        state = {};
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            state.x                  = scratch[off + 0];
            state.y                  = scratch[off + 1];
            state.inv_z              = scratch[off + 2];
            state.r2                 = scratch[off + 3];
            state.icD                = scratch[off + 4];
            state.den                = scratch[off + 5];
            state.projection_forward = state.inv_z > 0.0f;
        }
        else
        {
            state.x                  = scratch[off + 3];
            state.y                  = scratch[off + 4];
            state.inv_z              = scratch[off + 5];
            state.r2                 = scratch[off + 6];
            state.icD                = scratch[off + 7];
            state.den                = scratch[off + 8];
            state.projection_forward = scratch[off + 9] != 0.0f;
        }
    }
};

struct OpenCVPinholeBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        validate_opencv_pinhole_scratch_policy<Scratch>();
        static_assert(Scratch::kIsUndistort);
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            static_assert(Scratch::kScratchStride == 5);
            static_assert(Scratch::kInverseStashOffset == -1);
        }
        else
        {
            static_assert(Scratch::kScratchStride == 9);
            static_assert(Scratch::kInverseStashOffset == 5);
        }
    }

    template<typename Scratch>
    static __device__ void save_state(float *scratch, int64_t off, const OpenCVPinholeBackprojectState &state)
    {
        validate<Scratch>();
        scratch[off + 0] = state.xy.x;
        scratch[off + 1] = state.xy.y;
        scratch[off + 2] = state.r2;
        scratch[off + 3] = state.icD;
        scratch[off + 4] = state.den;
    }

    template<typename Scratch>
    static __device__ void load_state(
        const float *scratch, int64_t off, OpenCVPinholeBackprojectState &__restrict__ state
    )
    {
        validate<Scratch>();
        state.xy  = make_float2(scratch[off + 0], scratch[off + 1]);
        state.r2  = scratch[off + 2];
        state.icD = scratch[off + 3];
        state.den = scratch[off + 4];
    }
};

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::ImagePointsToCameraRays> : OpenCVPinholeBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const OpenCVPinholeBackprojectState &state)
    {
        save_state<Scratch>(scratch, off, state);
        if constexpr(kOpenCVPinholeBivariateScratch<Scratch>)
        {
            scratch[off + 8] = 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, OpenCVPinholeBackprojectState &__restrict__ state
    )
    {
        load_state<Scratch>(scratch, off, state);
    }
};

struct OpenCVPinholePoseProjectScratchIO
{
    template<typename Scratch, int ExpectedStride>
    static __device__ void validate()
    {
        validate_opencv_pinhole_scratch_policy<Scratch>();
        static_assert(!Scratch::kIsUndistort);
        static_assert(Scratch::kScratchStride == ExpectedStride);
        static_assert(Scratch::kInverseStashOffset == -1);
    }

    template<typename Scratch, int ExpectedStride>
    static __device__ void save_state(
        float *scratch, int64_t off, float3 p_rel, float3 camera_point, const OpenCVPinholeProjectState &state
    )
    {
        validate<Scratch, ExpectedStride>();
        scratch[off + 0] = p_rel.x;
        scratch[off + 1] = p_rel.y;
        scratch[off + 2] = p_rel.z;
        scratch[off + 3] = camera_point.x;
        scratch[off + 4] = camera_point.y;
        scratch[off + 5] = camera_point.z;
        scratch[off + 6] = state.r2;
        scratch[off + 7] = state.icD;
        scratch[off + 8] = state.den;
    }

    template<typename Scratch, int ExpectedStride>
    static __device__ void load_state(
        const float *scratch,
        int64_t off,
        float3 &__restrict__ p_rel,
        float3 &__restrict__ camera_point,
        OpenCVPinholeProjectState &__restrict__ state
    )
    {
        validate<Scratch, ExpectedStride>();
        p_rel                    = make_float3(scratch[off + 0], scratch[off + 1], scratch[off + 2]);
        camera_point             = make_float3(scratch[off + 3], scratch[off + 4], scratch[off + 5]);
        state                    = {};
        state.r2                 = scratch[off + 6];
        state.icD                = scratch[off + 7];
        state.den                = scratch[off + 8];
        state.pose_point_forward = camera_point.z > 0.0f;
    }
};

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::ProjectWorldPointsMeanPose> : OpenCVPinholePoseProjectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        OpenCVPinholePoseProjectScratchIO::validate<Scratch, 9>();
    }

    template<typename Scratch>
    static __device__ void save_forward(
        float *scratch, int64_t off, float3 p_rel, float3 camera_point, const OpenCVPinholeProjectState &state
    )
    {
        save_state<Scratch, 9>(scratch, off, p_rel, camera_point, state);
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch,
        int64_t off,
        float3 &__restrict__ p_rel,
        float3 &__restrict__ camera_point,
        OpenCVPinholeProjectState &__restrict__ state
    )
    {
        load_state<Scratch, 9>(scratch, off, p_rel, camera_point, state);
    }
};

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::ProjectWorldPointsShutterPose> : OpenCVPinholePoseProjectScratchIO
{
    template<typename Scratch>
    static __device__ void validate()
    {
        OpenCVPinholePoseProjectScratchIO::validate<Scratch, 10>();
    }

    template<typename Scratch>
    static __device__ void save_forward(
        float *scratch,
        int64_t off,
        float3 p_rel,
        float3 camera_point,
        const OpenCVPinholeProjectState &state,
        float alpha
    )
    {
        save_state<Scratch, 10>(scratch, off, p_rel, camera_point, state);
        scratch[off + 9] = alpha;
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch,
        int64_t off,
        float3 &__restrict__ p_rel,
        float3 &__restrict__ camera_point,
        OpenCVPinholeProjectState &__restrict__ state,
        float &__restrict__ alpha
    )
    {
        load_state<Scratch, 10>(scratch, off, p_rel, camera_point, state);
        alpha = scratch[off + 9];
    }
};

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::ImagePointsToWorldRaysStaticPose> : OpenCVPinholeBackprojectScratchIO
{
    template<typename Scratch>
    static __device__ void save_forward(float *scratch, int64_t off, const OpenCVPinholeBackprojectState &state)
    {
        save_state<Scratch>(scratch, off, state);
        if constexpr(kOpenCVPinholeBivariateScratch<Scratch>)
        {
            scratch[off + 8] = 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, OpenCVPinholeBackprojectState &__restrict__ state
    )
    {
        load_state<Scratch>(scratch, off, state);
    }
};

template<>
struct OpenCVPinholeScratchIO<DistortionOpFamily::ImagePointsToWorldRaysShutterPose>
{
    template<typename Scratch>
    static __device__ void validate()
    {
        validate_opencv_pinhole_scratch_policy<Scratch>();
        static_assert(Scratch::kIsUndistort);
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            static_assert(Scratch::kScratchStride == 9);
            static_assert(Scratch::kInverseStashOffset == -1);
        }
        else
        {
            static_assert(Scratch::kScratchStride == 12);
            static_assert(Scratch::kInverseStashOffset == 5);
        }
    }

    template<typename Scratch>
    static __device__ void save_forward(
        float *scratch, int64_t off, const OpenCVPinholeBackprojectState &state, float alpha
    )
    {
        validate<Scratch>();
        scratch[off + 0] = state.xy.x;
        scratch[off + 1] = state.xy.y;
        scratch[off + 2] = state.r2;
        scratch[off + 3] = state.icD;
        scratch[off + 4] = state.den;
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            scratch[off + 5] = alpha;
            scratch[off + 6] = 0.0f;
            scratch[off + 7] = 1.0f;
            scratch[off + 8] = 0.0f;
        }
        else
        {
            scratch[off + 8]  = alpha;
            scratch[off + 9]  = 0.0f;
            scratch[off + 10] = 1.0f;
            scratch[off + 11] = 0.0f;
        }
    }

    template<typename Scratch>
    static __device__ void load_backward(
        const float *scratch, int64_t off, OpenCVPinholeBackprojectState &__restrict__ state, float &__restrict__ alpha
    )
    {
        validate<Scratch>();
        state.xy  = make_float2(scratch[off + 0], scratch[off + 1]);
        state.r2  = scratch[off + 2];
        state.icD = scratch[off + 3];
        state.den = scratch[off + 4];
        if constexpr(kOpenCVPinholeNoExternalScratch<Scratch>)
        {
            alpha = scratch[off + 5];
        }
        else
        {
            alpha = scratch[off + 8];
        }
    }
};
