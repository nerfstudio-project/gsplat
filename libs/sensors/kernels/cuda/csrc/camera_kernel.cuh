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
// Provides thin private adapters that delegate quaternion math to libs/geometry
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
constexpr float kMinRadialDist = 0.8f;
constexpr float kMaxRadialDist = 1.2f;

// ===========================================================================
// Shared structs and type aliases
// ===========================================================================

using CameraParamsValue = OpenCVPinholeProjection_KernelParameters;

struct ProjectionEval {
    float2 image_point;
    bool valid;
};

struct OpenCVPinholeParams {
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

struct DistortionResult {
    float icD;
    float delta_x;
    float delta_y;
    float r2;
    float den;
};

struct DistortionParamGrads {
    float k[6] = {};
    float p[2] = {};
    float s[4] = {};
};

// ===========================================================================
// Scalar utilities
// ===========================================================================

// Returns `value` if |value| >= kDenominatorEpsilon, otherwise returns
// ±kDenominatorEpsilon with the original sign. Used wherever a float appears
// in a denominator to prevent division-by-zero without branching on a predicate.
__device__ __forceinline__ float safe_nonzero(float value) {
    if (fabsf(value) >= kDenominatorEpsilon) {
        return value;
    }
    return value < 0.0f ? -kDenominatorEpsilon : kDenominatorEpsilon;
}

// ===========================================================================
// Parameter loaders
// ===========================================================================

// Identity pass-through: returns the raw KernelParameters struct unchanged.
// Exists so call sites can use the generic CameraParamsValue alias without
// knowing the concrete type.
__device__ __forceinline__ CameraParamsValue load_camera_params(
    OpenCVPinholeProjection_KernelParameters projection) {
    return projection;
}

// Unpacks the flat KernelParameters into OpenCVPinholeParams and applies
// safe_nonzero to fx/fy so downstream reciprocals are always finite.
__device__ __forceinline__ OpenCVPinholeParams load_opencv_pinhole_params(
    OpenCVPinholeProjection_KernelParameters projection) {
    OpenCVPinholeParams p;
    p.fx = safe_nonzero(projection.focal_length[0]);
    p.fy = safe_nonzero(projection.focal_length[1]);
    p.cx = projection.principal_point[0];
    p.cy = projection.principal_point[1];
    for (int i = 0; i < 6; ++i) {
        p.k[i] = projection.radial_coeffs[i];
    }
    p.p[0] = projection.tangential_coeffs[0];
    p.p[1] = projection.tangential_coeffs[1];
    for (int i = 0; i < 4; ++i) {
        p.s[i] = projection.thin_prism_coeffs[i];
    }
    p.res_x = static_cast<float>(projection.width);
    p.res_y = static_cast<float>(projection.height);
    return p;
}

// ===========================================================================
// Row-major 3x3 matrix-vector helpers
// ===========================================================================

// M * v where rows are r0, r1, r2 (column-vector convention).
__device__ __forceinline__ float3 mat_vec(float3 r0, float3 r1, float3 r2, float3 v) {
    return make_float3(dot3(r0, v), dot3(r1, v), dot3(r2, v));
}

// M^T * v — transposed multiply used to apply the inverse rotation (R^T = R^-1 for SO(3)).
__device__ __forceinline__ float3 mat_t_vec(float3 r0, float3 r1, float3 r2, float3 v) {
    return make_float3(
        r0.x * v.x + r1.x * v.y + r2.x * v.z,
        r0.y * v.x + r1.y * v.y + r2.y * v.z,
        r0.z * v.x + r1.z * v.y + r2.z * v.z);
}

// ===========================================================================
// Contiguous-buffer read/write helpers
// ===========================================================================

// The read_*/write_* helpers below reinterpret a contiguous float* buffer as
// float3*/float4* and index by element. Callers must pass a pointer obtained
// from tensor.data_ptr<float>() on a contiguous tensor; PyTorch guarantees
// 16-byte alignment, which is sufficient for both float3 (alignof 4) and
// float4 (alignof 16). Do not call these on offset/strided pointers.
__device__ __forceinline__ float3 read_vec3(const float* data, int64_t idx) {
    return reinterpret_cast<const float3*>(data)[idx];
}

__device__ __forceinline__ float4 read_quat(const float* data, int64_t idx) {
    return reinterpret_cast<const float4*>(data)[idx];
}

// ===========================================================================
// Quaternion component-order swizzles
// ===========================================================================

// Storage convention in gsplat tensors is wxyz; libs/geometry uses xyzw.
// These two helpers convert at the call boundary so the rest of the code
// stays in the libs/geometry xyzw convention.
__device__ __forceinline__ float4 wxyz_to_xyzw(float4 q) {
    return make_float4(q.y, q.z, q.w, q.x);
}

__device__ __forceinline__ float4 xyzw_to_wxyz(float4 q) {
    return make_float4(q.w, q.x, q.y, q.z);
}

// ===========================================================================
// Quaternion adapters (delegate to libs/geometry)
// ===========================================================================

// Private call-boundary glue for libs/geometry quaternion primitives. These
// helpers keep sensorlib callsites float3/float4-shaped while delegating the
// actual quaternion math to libs/geometry/kernels/cuda/csrc/quaternion.cuh.

// Rotates v by q (xyzw order). Wraps quat_rotate_vector_fwd_impl<float>
// from libs/geometry; inputs and outputs stay in float3/float4 shapes.
__device__ __forceinline__ float3 quat_rotate_xyzw_geom(float4 q, float3 v) {
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    quat_rotate_vector_fwd_impl<float>(q.x, q.y, q.z, q.w, v.x, v.y, v.z, &ox, &oy, &oz);
    return make_float3(ox, oy, oz);
}

// Inverse-rotates v by q (xyzw order). Negates the vector part to form the
// conjugate quaternion then delegates to quat_rotate_xyzw_geom.
__device__ __forceinline__ float3 quat_inverse_rotate_xyzw_geom(float4 q, float3 v) {
    return quat_rotate_xyzw_geom(make_float4(-q.x, -q.y, -q.z, q.w), v);
}

// Fills r0/r1/r2 with the rows of the rotation matrix for q (xyzw order).
// Wraps quat_to_matrix_fwd_write<float> from libs/geometry.
__device__ __forceinline__ void quat_to_matrix_xyzw_geom(
    float4 q,
    float3& r0,
    float3& r1,
    float3& r2) {
    float r[9];
    quat_to_matrix_fwd_write<float>(q.x, q.y, q.z, q.w, r);
    r0 = make_float3(r[0], r[1], r[2]);
    r1 = make_float3(r[3], r[4], r[5]);
    r2 = make_float3(r[6], r[7], r[8]);
}

// Same as quat_to_matrix_xyzw_geom but accepts the tensor-storage wxyz order.
// The wxyz↔xyzw swizzle is applied at this call boundary via wxyz_to_xyzw.
__device__ __forceinline__ void quat_to_matrix_wxyz_geom(
    float4 q_wxyz,
    float3& r0,
    float3& r1,
    float3& r2) {
    quat_to_matrix_xyzw_geom(wxyz_to_xyzw(q_wxyz), r0, r1, r2);
}

// Normalizes q (xyzw order) to unit length. Wraps quat_normalize_safe_fwd_write<float>
// from libs/geometry; safe against the zero quaternion.
__device__ __forceinline__ float4 normalize_quat_geom(float4 q) {
    float ox = 0.0f;
    float oy = 0.0f;
    float oz = 0.0f;
    float ow = 0.0f;
    quat_normalize_safe_fwd_write<float>(q.x, q.y, q.z, q.w, &ox, &oy, &oz, &ow);
    return make_float4(ox, oy, oz, ow);
}

// Backward (VJP) call-boundary glue for libs/geometry quaternion primitives.
// Same shape as the forward `_geom` helpers above, but accumulating into
// caller-supplied float4&/float3& grad slots.
__device__ __forceinline__ void quat_rotate_bwd_xyzw_geom(
    float4 q,
    float3 v,
    float3 d_v_out,
    float4& d_q,
    float3& d_v) {
    float gqx = 0.0f;
    float gqy = 0.0f;
    float gqz = 0.0f;
    float gqw = 0.0f;
    float gvx = 0.0f;
    float gvy = 0.0f;
    float gvz = 0.0f;
    quat_rotate_vector_bwd_impl<float>(
        q.x,
        q.y,
        q.z,
        q.w,
        v.x,
        v.y,
        v.z,
        d_v_out.x,
        d_v_out.y,
        d_v_out.z,
        &gqx,
        &gqy,
        &gqz,
        &gqw,
        &gvx,
        &gvy,
        &gvz);
    d_q.x += gqx;
    d_q.y += gqy;
    d_q.z += gqz;
    d_q.w += gqw;
    d_v.x += gvx;
    d_v.y += gvy;
    d_v.z += gvz;
}

__device__ __forceinline__ void quat_inverse_rotate_bwd_xyzw_geom(
    float4 q,
    float3 v,
    float3 d_v_out,
    float4& d_q,
    float3& d_v) {
    float gqx = 0.0f;
    float gqy = 0.0f;
    float gqz = 0.0f;
    float gqw = 0.0f;
    float gvx = 0.0f;
    float gvy = 0.0f;
    float gvz = 0.0f;
    quat_rotate_vector_bwd_impl<float>(
        -q.x,
        -q.y,
        -q.z,
        q.w,
        v.x,
        v.y,
        v.z,
        d_v_out.x,
        d_v_out.y,
        d_v_out.z,
        &gqx,
        &gqy,
        &gqz,
        &gqw,
        &gvx,
        &gvy,
        &gvz);
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
// use in the libs/geometry adapter layer; the swizzle is applied here.
__device__ __forceinline__ float4 read_quat_xyzw_from_wxyz(const float* data, int64_t idx) {
    return wxyz_to_xyzw(read_quat(data, idx));
}

__device__ __forceinline__ void write_vec3(float* data, int64_t idx, float3 v) {
    reinterpret_cast<float3*>(data)[idx] = v;
}

__device__ __forceinline__ void write_quat(float* data, int64_t idx, float4 q) {
    reinterpret_cast<float4*>(data)[idx] = q;
}

// Converts q from xyzw to wxyz order and writes it to the tensor buffer.
__device__ __forceinline__ void write_quat_wxyz_from_xyzw(float* data, int64_t idx, float4 q) {
    write_quat(data, idx, xyzw_to_wxyz(q));
}

// ===========================================================================
// float3 interpolation
// ===========================================================================

__device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) {
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t);
}

// ===========================================================================
// OpenCV pinhole distortion helpers
// ===========================================================================

// Computes the OpenCV pinhole distortion at normalised image-plane coordinates xy.
// Returns the combined radial factor icD = num/den, tangential/thin-prism
// offsets (delta_x, delta_y), squared radius r2, and the denominator den for
// reuse in the backward. xy is in normalised camera-space (x/z, y/z).
__device__ __forceinline__ DistortionResult compute_distortion(
    float2 xy,
    const OpenCVPinholeParams& p) {
    float x = xy.x;
    float y = xy.y;
    float r2 = x * x + y * y;
    float xy_prod = x * y;
    float a1 = 2.0f * xy_prod;
    float a2 = r2 + 2.0f * x * x;
    float a3 = r2 + 2.0f * y * y;
    float num = 1.0f + r2 * (p.k[0] + r2 * (p.k[1] + r2 * p.k[2]));
    float den = safe_nonzero(1.0f + r2 * (p.k[3] + r2 * (p.k[4] + r2 * p.k[5])));
    DistortionResult result;
    result.icD = num * __frcp_rn(den);
    result.delta_x = p.p[0] * a1 + p.p[1] * a2 + r2 * (p.s[0] + r2 * p.s[1]);
    result.delta_y = p.p[0] * a3 + p.p[1] * a1 + r2 * (p.s[2] + r2 * p.s[3]);
    result.r2 = r2;
    result.den = den;
    return result;
}

// Returns only the denominator polynomial den(r2) = 1 + k4*r2 + k5*r2^2 + k6*r2^3,
// clamped away from zero. Used in project_camera_ray_opencv to recompute den
// after calling compute_distortion, avoiding a second full distortion evaluation.
__device__ __forceinline__ float distortion_den(float r2, const OpenCVPinholeParams& p) {
    return safe_nonzero(1.0f + r2 * (p.k[3] + r2 * (p.k[4] + r2 * p.k[5])));
}

// Fixed-point iterative undistort for the OpenCV pinhole model. Converges in
// up to 10 iterations with a 1e-8 L∞ early-exit. xy0 is the distorted
// normalised-plane coordinate; returns the converged undistorted xy.
__device__ __forceinline__ float2 iterative_undistort(
    float2 xy0,
    const OpenCVPinholeParams& p) {
    float2 xy = xy0;
    for (int i = 0; i < 10; ++i) {
        DistortionResult d = compute_distortion(xy, p);
        float2 next = make_float2((xy0.x - d.delta_x) / d.icD, (xy0.y - d.delta_y) / d.icD);
        float max_delta = fmaxf(fabsf(next.x - xy.x), fabsf(next.y - xy.y));
        xy = next;
        if (max_delta < 1.0e-8f) {
            break;
        }
    }
    return xy;
}

// Returns true iff image_point falls inside [0, res_x) x [0, res_y).
__device__ __forceinline__ bool is_in_bounds(float2 image_point, const OpenCVPinholeParams& p) {
    return image_point.x >= 0.0f && image_point.x < p.res_x && image_point.y >= 0.0f && image_point.y < p.res_y;
}

// Projects a camera-space ray through the OpenCV pinhole model and writes
// intermediate values (x, y, inv_z, r2, icD, den) to scratch[scratch_offset..+5]
// for reuse in the backward pass. scratch may be nullptr if no backward is needed.
// Points with ray.z <= 0 or icD outside [kMinRadialDist, kMaxRadialDist] are
// clamped to the sensor boundary and marked invalid.
__device__ __forceinline__ ProjectionEval project_camera_ray_opencv(
    const OpenCVPinholeParams& p,
    float3 ray,
    float* scratch,
    int64_t scratch_offset) {
    float x = 0.0f;
    float y = 0.0f;
    float inv_z = 0.0f;
    float r2 = 0.0f;
    float icD = 0.0f;
    float den = 0.0f;
    float2 image_point = make_float2(0.0f, 0.0f);
    bool valid = false;

    if (ray.z > 0.0f) {
        inv_z = 1.0f / ray.z;
        x = ray.x * inv_z;
        y = ray.y * inv_z;
        DistortionResult d = compute_distortion(make_float2(x, y), p);
        r2 = d.r2;
        icD = d.icD;
        den = distortion_den(r2, p);
        if (icD < kMinRadialDist || icD > kMaxRadialDist) {
            float res_len = sqrtf(p.res_x * p.res_x + p.res_y * p.res_y);
            float inv_sqrt_r2 = (r2 > 0.0f) ? rsqrtf(r2) : 0.0f;
            image_point.x = x * res_len * inv_sqrt_r2 + p.cx;
            image_point.y = y * res_len * inv_sqrt_r2 + p.cy;
        } else {
            float uv_x = x * icD + d.delta_x;
            float uv_y = y * icD + d.delta_y;
            image_point.x = uv_x * p.fx + p.cx;
            image_point.y = uv_y * p.fy + p.cy;
        }
        valid = is_in_bounds(image_point, p);
    }

    if (scratch != nullptr) {
        scratch[scratch_offset + 0] = x;
        scratch[scratch_offset + 1] = y;
        scratch[scratch_offset + 2] = inv_z;
        scratch[scratch_offset + 3] = r2;
        scratch[scratch_offset + 4] = icD;
        scratch[scratch_offset + 5] = den;
    }

    return {image_point, valid};
}

// ===========================================================================
// CameraParamsValue-based projection helpers (used by project_camera_point path)
// ===========================================================================

// Returns the combined radial scale factor icD = num(r2)/den(r2) for the
// OpenCV rational model. r4 and r6 are passed in pre-computed to avoid
// redundant multiplications at callsites that already have them.
__device__ __forceinline__ float radial_factor(
    const CameraParamsValue& params,
    float r2,
    float r4,
    float r6) {
    float num = 1.0f + params.radial_coeffs[0] * r2 + params.radial_coeffs[1] * r4 + params.radial_coeffs[2] * r6;
    float den = safe_nonzero(1.0f + params.radial_coeffs[3] * r2 + params.radial_coeffs[4] * r4 + params.radial_coeffs[5] * r6);
    return num * __frcp_rn(den);
}

// Applies the full OpenCV distortion model (radial + tangential + thin prism)
// to normalised image-plane coordinates xy. Returns the distorted xy.
__device__ __forceinline__ float2 distort_xy(const CameraParamsValue& params, float2 xy) {
    float x = xy.x;
    float y = xy.y;
    float r2 = x * x + y * y;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float radial = radial_factor(params, r2, r4, r6);
    float xy_prod = x * y;
    float dx = 2.0f * params.tangential_coeffs[0] * xy_prod
        + params.tangential_coeffs[1] * (r2 + 2.0f * x * x)
        + params.thin_prism_coeffs[0] * r2
        + params.thin_prism_coeffs[1] * r4;
    float dy = params.tangential_coeffs[0] * (r2 + 2.0f * y * y)
        + 2.0f * params.tangential_coeffs[1] * xy_prod
        + params.thin_prism_coeffs[2] * r2
        + params.thin_prism_coeffs[3] * r4;
    return make_float2(x * radial + dx, y * radial + dy);
}

// Projects a camera-space 3D point to an image coordinate using the OpenCV
// pinhole model. camera_point.z must be > 0 for a valid result.
// validity checks finite-ness of the output and image-plane bounds.
__device__ __forceinline__ ProjectionEval project_camera_point(
    const CameraParamsValue& params,
    float3 camera_point) {
    float inv_z = 1.0f / camera_point.z;
    float2 xy = make_float2(camera_point.x * inv_z, camera_point.y * inv_z);
    float2 distorted = distort_xy(params, xy);
    float2 image_point = make_float2(
        params.focal_length[0] * distorted.x + params.principal_point[0],
        params.focal_length[1] * distorted.y + params.principal_point[1]);
    bool finite = isfinite(image_point.x) && isfinite(image_point.y) && isfinite(camera_point.z);
    bool valid = camera_point.z > 0.0f
        && finite
        && image_point.x >= 0.0f
        && image_point.x < static_cast<float>(params.width)
        && image_point.y >= 0.0f
        && image_point.y < static_cast<float>(params.height);
    return {image_point, valid};
}

// Fixed-point iterative undistort using the CameraParamsValue variant.
// Converges in up to 10 iterations with a 1e-7 L∞ early-exit.
// xy0 is the distorted normalised-plane coordinate; returns the converged undistorted xy.
__device__ __forceinline__ float2 undistort_xy(
    const CameraParamsValue& params,
    float2 xy0) {
    float2 xy = xy0;
    constexpr float kUndistortTolerance = 1.0e-7f;
    for (int i = 0; i < 10; ++i) {
        float x = xy.x;
        float y = xy.y;
        float r2 = x * x + y * y;
        float r4 = r2 * r2;
        float r6 = r4 * r2;
        float radial = radial_factor(params, r2, r4, r6);
        float2 distorted = distort_xy(params, xy);
        float2 delta = make_float2(distorted.x - x * radial, distorted.y - y * radial);
        float inv_radial = __frcp_rn(radial);
        float2 next_xy = make_float2((xy0.x - delta.x) * inv_radial, (xy0.y - delta.y) * inv_radial);
        float max_delta = fmaxf(fabsf(next_xy.x - xy.x), fabsf(next_xy.y - xy.y));
        xy = next_xy;
        if (max_delta < kUndistortTolerance) {
            break;
        }
    }
    return xy;
}

// Lifts an image-plane pixel to a unit camera-space ray. Reverses the pinhole
// projection by subtracting the principal point, dividing by focal lengths, then
// iteratively undistorting, and finally normalising to unit length.
__device__ __forceinline__ float3 camera_ray_from_image_point(
    const CameraParamsValue& params,
    float2 image_point) {
    float2 xy0 = make_float2(
        (image_point.x - params.principal_point[0]) / safe_nonzero(params.focal_length[0]),
        (image_point.y - params.principal_point[1]) / safe_nonzero(params.focal_length[1]));
    float2 xy = undistort_xy(params, xy0);
    return normalize3(make_float3(xy.x, xy.y, 1.0f));
}

// ===========================================================================
// Rolling-shutter timing helpers
// ===========================================================================

// Converts a per-pixel relative scan time in [0, 1] to an absolute timestamp
// in microseconds by linearly interpolating between start and end. Uses double
// precision internally to avoid rounding error on large timestamps.
__device__ __forceinline__ int64_t timestamp_from_relative_time(
    float relative_time,
    int64_t start_timestamp_us,
    int64_t end_timestamp_us) {
    double start = static_cast<double>(start_timestamp_us);
    double duration = static_cast<double>(end_timestamp_us - start_timestamp_us);
    return static_cast<int64_t>(start + static_cast<double>(relative_time) * duration);
}

// Returns the relative scan time in [0, 1] for image_point given the shutter
// direction. GLOBAL shutter always returns 0. For rolling shutters the scanline
// (or column) index is floored/ceiled to the nearest integer scan boundary
// before dividing by (dim - 1).
__device__ __forceinline__ float compute_relative_frame_time_opencv(
    float2 image_point,
    int64_t width,
    int64_t height,
    int64_t shutter_type) {
    float safe_w = static_cast<float>(std::max(width, static_cast<int64_t>(2)) - 1);
    float safe_h = static_cast<float>(std::max(height, static_cast<int64_t>(2)) - 1);
    const auto st = static_cast<gsplat_sensors::ShutterType>(shutter_type);
    if (st == gsplat_sensors::ShutterType::ROLLING_TOP_TO_BOTTOM) {
        return floorf(image_point.y) / safe_h;
    }
    if (st == gsplat_sensors::ShutterType::ROLLING_BOTTOM_TO_TOP) {
        return (static_cast<float>(height) - ceilf(image_point.y)) / safe_h;
    }
    if (st == gsplat_sensors::ShutterType::ROLLING_LEFT_TO_RIGHT) {
        return floorf(image_point.x) / safe_w;
    }
    if (st == gsplat_sensors::ShutterType::ROLLING_RIGHT_TO_LEFT) {
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
__device__ __forceinline__ void compute_distortion_bwd(
    float x,
    float y,
    float r2,
    float icD,
    float den,
    const OpenCVPinholeParams& params,
    float4 d_out,
    float2& d_xy,
    DistortionParamGrads& d_params) {
    if (fabsf(den) < kDenominatorEpsilon) {
        return;
    }
    float d_icD = d_out.x;
    float d_delta_x = d_out.y;
    float d_delta_y = d_out.z;
    float d_r2_out = d_out.w;
    float r2_2 = r2 * r2;
    float r2_3 = r2_2 * r2;
    float inv_den = __frcp_rn(den);
    float dnum_dr2 = params.k[0] + r2 * (2.0f * params.k[1] + r2 * 3.0f * params.k[2]);
    float dden_dr2 = params.k[3] + r2 * (2.0f * params.k[4] + r2 * 3.0f * params.k[5]);
    float dicD_dr2 = (dnum_dr2 - icD * dden_dr2) * inv_den;

    d_params.k[0] += d_icD * (r2 * inv_den);
    d_params.k[1] += d_icD * (r2_2 * inv_den);
    d_params.k[2] += d_icD * (r2_3 * inv_den);
    d_params.k[3] += d_icD * (-icD * r2 * inv_den);
    d_params.k[4] += d_icD * (-icD * r2_2 * inv_den);
    d_params.k[5] += d_icD * (-icD * r2_3 * inv_den);

    float d_r2_accum = d_icD * dicD_dr2 + d_r2_out;
    float two_x = 2.0f * x;
    float two_y = 2.0f * y;
    float two_xy = 2.0f * x * y;
    float a2 = r2 + 2.0f * x * x;
    float a3 = r2 + 2.0f * y * y;

    d_params.p[0] += d_delta_x * two_xy + d_delta_y * a3;
    d_params.p[1] += d_delta_x * a2 + d_delta_y * two_xy;
    d_params.s[0] += d_delta_x * r2;
    d_params.s[1] += d_delta_x * r2_2;
    d_params.s[2] += d_delta_y * r2;
    d_params.s[3] += d_delta_y * r2_2;

    float d_x = 0.0f;
    float d_y = 0.0f;
    d_x += d_delta_x * (params.p[0] * two_y + params.p[1] * (4.0f * x));
    d_y += d_delta_x * (params.p[0] * two_x);
    d_r2_accum += d_delta_x * (params.p[1] + params.s[0] + 2.0f * r2 * params.s[1]);
    d_x += d_delta_y * (params.p[1] * two_y);
    d_y += d_delta_y * (params.p[0] * (4.0f * y) + params.p[1] * two_x);
    d_r2_accum += d_delta_y * (params.p[0] + params.s[2] + 2.0f * r2 * params.s[3]);
    d_x += d_r2_accum * two_x;
    d_y += d_r2_accum * two_y;
    d_xy.x += d_x;
    d_xy.y += d_y;
}

// Fills M[2][2] with the 2x2 Jacobian dF/dxy of the distortion fixed-point
// F(xy) = icD*xy + delta used by the iterative undistort. The IFT backward
// solves M^T * d_xy_dist = d_xy_undist using solve_2x2_transposed.
// Zeroes M and returns early if |den| < kDenominatorEpsilon (degenerate).
__device__ __forceinline__ void compute_dF_dxy(
    float x,
    float y,
    float r2,
    float icD,
    float den,
    const OpenCVPinholeParams& params,
    float M[2][2]) {
    if (fabsf(den) < kDenominatorEpsilon) {
        M[0][0] = 0.0f;
        M[0][1] = 0.0f;
        M[1][0] = 0.0f;
        M[1][1] = 0.0f;
        return;
    }
    float inv_den = __frcp_rn(den);
    float dnum_dr2 = params.k[0] + r2 * (2.0f * params.k[1] + r2 * 3.0f * params.k[2]);
    float dden_dr2 = params.k[3] + r2 * (2.0f * params.k[4] + r2 * 3.0f * params.k[5]);
    float dicD_dr2 = (dnum_dr2 - icD * dden_dr2) * inv_den;
    float dicD_dx = dicD_dr2 * (2.0f * x);
    float dicD_dy = dicD_dr2 * (2.0f * y);
    float d_deltaX_dx = params.p[0] * (2.0f * y) + params.p[1] * (6.0f * x) + (params.s[0] + 2.0f * r2 * params.s[1]) * (2.0f * x);
    float d_deltaX_dy = params.p[0] * (2.0f * x) + params.p[1] * (2.0f * y) + (params.s[0] + 2.0f * r2 * params.s[1]) * (2.0f * y);
    float d_deltaY_dx = params.p[0] * (2.0f * x) + params.p[1] * (2.0f * y) + (params.s[2] + 2.0f * r2 * params.s[3]) * (2.0f * x);
    float d_deltaY_dy = params.p[0] * (6.0f * y) + params.p[1] * (2.0f * x) + (params.s[2] + 2.0f * r2 * params.s[3]) * (2.0f * y);
    M[0][0] = icD + x * dicD_dx + d_deltaX_dx;
    M[0][1] = x * dicD_dy + d_deltaX_dy;
    M[1][0] = y * dicD_dx + d_deltaY_dx;
    M[1][1] = icD + y * dicD_dy + d_deltaY_dy;
}

// Solves M^T * x = b for x. Returns (0, 0) if |det(M)| < 1e-8 (ill-conditioned
// IFT Jacobian); returning zero gradient is safer than propagating a 1e+20-scale
// result that would pass isfinite() and poison Adam moments.
__device__ __forceinline__ float2 solve_2x2_transposed(float M[2][2], float2 b) {
    float det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (fabsf(det) < 1.0e-8f) {
        return make_float2(0.0f, 0.0f);
    }
    float inv_det = 1.0f / det;
    return make_float2(
        inv_det * (M[1][1] * b.x - M[1][0] * b.y),
        inv_det * (-M[0][1] * b.x + M[0][0] * b.y));
}
