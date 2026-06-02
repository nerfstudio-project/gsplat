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

// Device-side helpers for the bivariate windshield external distortion model.
//
// The model maps a camera-space ray to a distorted unit direction by:
//   1. Normalising the ray and computing horizontal angle phi  = asin(nx)
//      and vertical angle theta = asin(ny).
//   2. Evaluating two bivariate polynomials — h_poly(phi, theta) → adj_phi
//      and v_poly(phi, theta) → adj_theta — that capture the windshield glass
//      refraction as a function of angle-of-incidence.
//   3. Reconstructing a unit direction from (sin(adj_phi), sin(adj_theta), z).
//
// The coefficient tensor is always packed as:
//   [h_poly(6), v_poly(15), h_poly_inv(6), v_poly_inv(15)]  (42 floats total).
// Which half is the "distort" polynomial and which is the "undistort" polynomial
// depends on `reference_polynomial` (see bivariate_coeff_base).
//
// This header is included by the bivariate-windshield camera kernel TU only;
// it must not be pulled into host-side translation units.

#pragma once

#include "external_distortion_params.h"
#include "math.cuh"

#include <cuda_runtime.h>

// No-op tag: constructed from a NoExternalDistortion_KernelParameters (which
// carries no data) so the templated kernel body compiles for the no-distortion
// path without special-casing.
struct NoExternalDistortion_Parameters {
    NoExternalDistortion_Parameters(const NoExternalDistortion_KernelParameters&, int) {}
};

// ===========================================================================
// Compile-time sizes for the bivariate windshield polynomials
// ===========================================================================

// Number of triangular-basis coefficients for a degree-2 bivariate poly
// (terms: 1, x, x², y, xy, y²).
constexpr int BIVARIATE_H_POLY_TERMS = 6;
// Number of triangular-basis coefficients for a degree-4 bivariate poly
// (terms for degrees 0–4, i.e. (4+1)(4+2)/2 = 15).
constexpr int BIVARIATE_V_POLY_TERMS = 15;
// Total differentiable params for one direction pair (h + v).
constexpr int BIVARIATE_NUM_DIFF_PARAMS = BIVARIATE_H_POLY_TERMS + BIVARIATE_V_POLY_TERMS;

// Per-thread register copy of the active polynomial pair (forward or inverse),
// loaded from the flat distortion_coeffs pointer by load_bivariate_windshield_params.
struct BivariateWindshieldParams {
    float h_poly[BIVARIATE_H_POLY_TERMS]; // horizontal: degree ≤ 2
    float v_poly[BIVARIATE_V_POLY_TERMS]; // vertical:   degree ≤ 4
    uint32_t h_poly_degree;
    uint32_t v_poly_degree;
};

// Accumulator for coefficient gradients produced by apply_bivariate_distortion_bwd.
// Laid out identically to BivariateWindshieldParams so the bwd helper can write
// directly into the active poly slice before the caller scatters via atomicAdd.
struct BivariateParamGrads {
    float h_poly[BIVARIATE_H_POLY_TERMS];
    float v_poly[BIVARIATE_V_POLY_TERMS];
};

// ===========================================================================
// Coefficient-slice selection
// ===========================================================================

// Return the float-index offset into distortion_coeffs for the active
// polynomial pair given the caller's direction.
//
// The flat buffer has layout:
//   [fwd_h(6), fwd_v(15), inv_h(6), inv_v(15)]
//
// reference_polynomial == ReferencePolynomial.FORWARD (0):
//   distort   → offset 0                     (fwd pair)
//   undistort → offset BIVARIATE_NUM_DIFF_PARAMS (inv pair)
//
// reference_polynomial == ReferencePolynomial.BACKWARD (1):
//   distort   → offset BIVARIATE_NUM_DIFF_PARAMS (inv pair treated as fwd)
//   undistort → offset 0                          (fwd pair treated as inv)
__device__ __forceinline__ uint32_t bivariate_coeff_base(
    int reference_polynomial,
    bool is_undistort) {
    if (is_undistort) {
        return reference_polynomial == 0 ? BIVARIATE_NUM_DIFF_PARAMS : 0u;
    }
    return reference_polynomial == 0 ? 0u : BIVARIATE_NUM_DIFF_PARAMS;
}

// Unpack the active polynomial pair from the flat distortion_coeffs pointer
// into a BivariateWindshieldParams register struct.
// is_undistort — true when the caller is performing the inverse (undistort)
//   pass; selects the complementary coefficient slice via bivariate_coeff_base.
__device__ __forceinline__ BivariateWindshieldParams load_bivariate_windshield_params(
    BivariateWindshieldDistortion_KernelParameters distortion,
    bool is_undistort) {
    uint32_t base = bivariate_coeff_base(distortion.reference_polynomial, is_undistort);
    BivariateWindshieldParams p;
    for (int i = 0; i < BIVARIATE_H_POLY_TERMS; ++i) {
        p.h_poly[i] = distortion.distortion_coeffs[base + i];
    }
    for (int i = 0; i < BIVARIATE_V_POLY_TERMS; ++i) {
        p.v_poly[i] = distortion.distortion_coeffs[base + BIVARIATE_H_POLY_TERMS + i];
    }
    p.h_poly_degree = distortion.h_poly_degree;
    p.v_poly_degree = distortion.v_poly_degree;
    return p;
}

// ===========================================================================
// Bivariate polynomial evaluation (forward)
// ===========================================================================

// Evaluate the bivariate polynomial P(x, y) = sum_{i+j <= order} c_{ij} x^i y^j
// using Horner's scheme along the x-axis rows of the triangular expansion,
// specialised for order in {0, 1, 2, 3, ≥4}.
// c    — coefficient array in y-row-major triangular order, e.g. for order = 2:
//        y^0 row: c[0]=1, c[1]=x, c[2]=x²
//        y^1 row: c[3]=y, c[4]=xy
//        y^2 row: c[5]=y²
// order — polynomial degree; h_poly uses h_poly_degree, v_poly uses v_poly_degree.
// The fallback branch (order ≥ 4) hard-codes degree-4 (15 terms = BIVARIATE_V_POLY_TERMS).
//
// Caller contract: there is no runtime guard against order overrunning the
// caller-supplied buffer. h_poly buffers hold BIVARIATE_H_POLY_TERMS = 6
// coefficients and MUST be passed with order ≤ 2; v_poly buffers hold
// BIVARIATE_V_POLY_TERMS = 15 coefficients and MUST be passed with order ≤ 4.
// Passing h_poly with order > 2 reads out-of-bounds memory.
__device__ __forceinline__ float eval_poly_2d(
    float x,
    float y,
    const float* __restrict__ c,
    uint32_t order) {
    if (order == 0u) {
        return c[0];
    }
    if (order == 1u) {
        float y_coeff0 = c[0] + x * c[1];
        float y_coeff1 = c[2];
        return y_coeff0 + y * y_coeff1;
    }
    if (order == 2u) {
        float y_coeff0 = c[0] + x * (c[1] + x * c[2]);
        float y_coeff1 = c[3] + x * c[4];
        float y_coeff2 = c[5];
        return y_coeff0 + y * (y_coeff1 + y * y_coeff2);
    }
    if (order == 3u) {
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

// Partial derivative dP/dx of the bivariate polynomial evaluated at (x, y).
// Specialised for the same order branches as eval_poly_2d.
// Used by eval_poly_2d_bwd to propagate gradients to x (phi or theta).
__device__ __forceinline__ float eval_poly_2d_dfdx(
    float x,
    float y,
    const float* __restrict__ c,
    uint32_t order) {
    if (order == 0u) {
        return 0.0f;
    }
    if (order == 1u) {
        return c[1];
    }
    if (order == 2u) {
        float dc0 = c[1] + x * (2.0f * c[2]);
        float dc1 = c[4];
        return dc0 + y * dc1;
    }
    if (order == 3u) {
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

// Partial derivative dP/dy of the bivariate polynomial evaluated at (x, y).
// Specialised for the same order branches as eval_poly_2d.
// Used by eval_poly_2d_bwd to propagate gradients to y (the other angle).
__device__ __forceinline__ float eval_poly_2d_dfdy(
    float x,
    float y,
    const float* __restrict__ c,
    uint32_t order) {
    if (order == 0u) {
        return 0.0f;
    }
    if (order == 1u) {
        return c[2];
    }
    if (order == 2u) {
        float yc1 = c[3] + x * c[4];
        return yc1 + y * (2.0f * c[5]);
    }
    if (order == 3u) {
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

// Accumulate gradients through eval_poly_2d into d_x, d_y, and d_c[].
//
// d_result — upstream gradient of the scalar output P(x, y).
// d_x, d_y — accumulated (+=) gradients w.r.t. x and y.
// d_c      — accumulated (+=) gradients w.r.t. each coefficient c[i]; array
//            must be at least (order+1)(order+2)/2 elements, clamped by MaxTerms.
//
// MaxTerms is either BIVARIATE_H_POLY_TERMS (6) or BIVARIATE_V_POLY_TERMS (15)
// depending on which polynomial is being differentiated.  The template param
// enforces a compile-time upper bound so the accumulation array fits in registers.
template <uint32_t MaxTerms>
__device__ __forceinline__ void eval_poly_2d_bwd(
    float x,
    float y,
    const float* __restrict__ c,
    uint32_t order,
    float d_result,
    float& d_x,
    float& d_y,
    float (&d_c)[MaxTerms]) {
    d_x += d_result * eval_poly_2d_dfdx(x, y, c, order);
    d_y += d_result * eval_poly_2d_dfdy(x, y, c, order);

    float px2 = x * x;
    float py2 = y * y;
    if (order == 0u) {
        d_c[0] += d_result;
        return;
    }
    if (order == 1u) {
        d_c[0] += d_result;
        d_c[1] += d_result * x;
        d_c[2] += d_result * y;
        return;
    }
    if (order == 2u) {
        d_c[0] += d_result;
        d_c[1] += d_result * x;
        d_c[2] += d_result * px2;
        d_c[3] += d_result * y;
        d_c[4] += d_result * x * y;
        d_c[5] += d_result * py2;
        return;
    }
    if (order == 3u) {
        float px3 = px2 * x;
        float py3 = py2 * y;
        d_c[0] += d_result;
        d_c[1] += d_result * x;
        d_c[2] += d_result * px2;
        d_c[3] += d_result * px3;
        d_c[4] += d_result * y;
        d_c[5] += d_result * x * y;
        d_c[6] += d_result * px2 * y;
        d_c[7] += d_result * py2;
        d_c[8] += d_result * x * py2;
        d_c[9] += d_result * py3;
        return;
    }
    float px3 = px2 * x;
    float px4 = px3 * x;
    float py3 = py2 * y;
    float py4 = py3 * y;
    d_c[0] += d_result;
    d_c[1] += d_result * x;
    d_c[2] += d_result * px2;
    d_c[3] += d_result * px3;
    d_c[4] += d_result * px4;
    d_c[5] += d_result * y;
    d_c[6] += d_result * x * y;
    d_c[7] += d_result * px2 * y;
    d_c[8] += d_result * px3 * y;
    d_c[9] += d_result * py2;
    d_c[10] += d_result * x * py2;
    d_c[11] += d_result * px2 * py2;
    d_c[12] += d_result * py3;
    d_c[13] += d_result * x * py3;
    d_c[14] += d_result * py4;
}

// ===========================================================================
// Forward and backward distortion application
// ===========================================================================

// Apply the bivariate windshield distortion to a camera-space ray.
//
// camera_ray — unnormalised direction in camera space (x right, y down, z forward).
// p          — active polynomial pair (loaded by load_bivariate_windshield_params).
//
// Returns a unit direction in the distorted camera space:
//   phi      = asin(nx),  theta    = asin(ny)          (incident angles)
//   adj_phi  = h_poly(phi, theta),  adj_theta = v_poly(phi, theta)
//   out.x    = sin(adj_phi),        out.y     = sin(adj_theta)
//   out.z    = sqrt(clamp(1 - out.x² - out.y², 0, 1)) * sign(nz)
// The z sign from the normalised input ray is preserved to handle rear-facing rays.
__device__ __forceinline__ float3 apply_bivariate_distortion(
    float3 camera_ray,
    const BivariateWindshieldParams& p) {
    float inv_len = rsqrtf(camera_ray.x * camera_ray.x + camera_ray.y * camera_ray.y + camera_ray.z * camera_ray.z);
    float3 ray_norm = make_float3(
        camera_ray.x * inv_len,
        camera_ray.y * inv_len,
        camera_ray.z * inv_len);
    float phi = asinf(ray_norm.x);
    float theta = asinf(ray_norm.y);
    float adj_phi = eval_poly_2d(phi, theta, p.h_poly, p.h_poly_degree);
    float adj_theta = eval_poly_2d(phi, theta, p.v_poly, p.v_poly_degree);
    float x = sinf(adj_phi);
    float y = sinf(adj_theta);
    float xy_norm = x * x + y * y;
    float clamped = fminf(fmaxf(1.0f - xy_norm, 0.0f), 1.0f);
    float z_sign = ray_norm.z >= 0.0f ? 1.0f : -1.0f;
    float z = sqrtf(clamped) * z_sign;
    return make_float3(x, y, z);
}

// Backward pass of apply_bivariate_distortion.
//
// Inputs (must match the forward call exactly):
//   camera_ray — original unnormalised input ray.
//   p          — same BivariateWindshieldParams used in the forward pass.
//   d_out      — upstream gradient of the distorted unit direction (float3).
//
// Outputs (accumulated with +=):
//   d_camera_ray — gradient w.r.t. camera_ray; uses normalize3_bwd from math.cuh.
//   d_p          — gradient w.r.t. h_poly[] and v_poly[] coefficients; caller
//                  scatters these into grad_distortion_coeffs via atomicAdd.
//
// Degenerate rays (len_sq == 0 or non-finite) are silently skipped so a single
// bad pixel cannot corrupt the global coefficient gradient accumulator.
// The z output gradient is gated on `clamp_open` to handle the boundary of the
// unit-sphere constraint gracefully.
__device__ __forceinline__ void apply_bivariate_distortion_bwd(
    float3 camera_ray,
    const BivariateWindshieldParams& p,
    float3 d_out,
    float3& d_camera_ray,
    BivariateParamGrads& d_p) {
    float len_sq = camera_ray.x * camera_ray.x + camera_ray.y * camera_ray.y + camera_ray.z * camera_ray.z;
    // Degenerate / non-finite ray: zero-out the local coeff grads and skip,
    // so a single bad row cannot poison atomicAdd into grad_distortion_coeffs.
    if (!(len_sq > 0.0f)) {
        return;
    }
    float inv_len = rsqrtf(len_sq);
    float3 ray_norm = make_float3(
        camera_ray.x * inv_len,
        camera_ray.y * inv_len,
        camera_ray.z * inv_len);
    float phi = asinf(ray_norm.x);
    float theta = asinf(ray_norm.y);
    float adj_phi = eval_poly_2d(phi, theta, p.h_poly, p.h_poly_degree);
    float adj_theta = eval_poly_2d(phi, theta, p.v_poly, p.v_poly_degree);
    float x_out = sinf(adj_phi);
    float y_out = sinf(adj_theta);
    float xy_norm = x_out * x_out + y_out * y_out;
    float clamp_arg = 1.0f - xy_norm;
    bool clamp_open = clamp_arg > 0.0f && clamp_arg < 1.0f;
    float clamped = fminf(fmaxf(clamp_arg, 0.0f), 1.0f);
    float z_sign = ray_norm.z >= 0.0f ? 1.0f : -1.0f;

    float d_x = d_out.x;
    float d_y = d_out.y;
    if (clamp_open) {
        float sqrt_clamped = sqrtf(clamped);
        float d_clamped = 0.5f * z_sign * d_out.z / fmaxf(sqrt_clamped, 1.0e-30f);
        float d_xy_norm = -d_clamped;
        d_x += d_xy_norm * 2.0f * x_out;
        d_y += d_xy_norm * 2.0f * y_out;
    }
    float d_adj_phi = d_x * cosf(adj_phi);
    float d_adj_theta = d_y * cosf(adj_theta);

    float d_phi = 0.0f;
    float d_theta = 0.0f;
    eval_poly_2d_bwd<BIVARIATE_V_POLY_TERMS>(
        phi, theta, p.v_poly, p.v_poly_degree, d_adj_theta, d_phi, d_theta, d_p.v_poly);
    eval_poly_2d_bwd<BIVARIATE_H_POLY_TERMS>(
        phi, theta, p.h_poly, p.h_poly_degree, d_adj_phi, d_phi, d_theta, d_p.h_poly);

    float denom_x = sqrtf(fmaxf(1.0f - ray_norm.x * ray_norm.x, 1.0e-30f));
    float denom_y = sqrtf(fmaxf(1.0f - ray_norm.y * ray_norm.y, 1.0e-30f));
    float3 d_ray_norm = make_float3(d_phi / denom_x, d_theta / denom_y, 0.0f);
    float3 d_ray = normalize3_bwd(camera_ray, d_ray_norm);
    d_camera_ray.x += d_ray.x;
    d_camera_ray.y += d_ray.y;
    d_camera_ray.z += d_ray.z;
}
