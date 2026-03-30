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

// Device-side pose / SE3 / trajectory math. Paired with pose.cu (kernels,
// launches, exports). Includes quaternion.cuh for shared quaternion device
// helpers without a separate third header.
#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "quaternion.cuh"

// -----------------------------------------------------------------------------
// se3pose_from_matrix with normalize-safe quaternion recovery.
// Shared quaternion primitives: quaternion.cuh (QuatNormEps, quat_normalize_safe_*_write, quat_to_matrix_fwd_write).
// Row-major 4x4 flat: upper 3x3 at 0,1,2,4,5,6,8,9,10; translation at 3,7,11
// -----------------------------------------------------------------------------

// Shepperd: row-major R (r00..r22) → quaternion (xyzw) before optional safe normalize; *branch_out ∈ {0,1,2,3} for VJP.
template <typename scalar_t>
__device__ void shepperd_from_matrix_fwd(
    scalar_t r00,
    scalar_t r01,
    scalar_t r02,
    scalar_t r10,
    scalar_t r11,
    scalar_t r12,
    scalar_t r20,
    scalar_t r21,
    scalar_t r22,
    scalar_t* __restrict__ qx,
    scalar_t* __restrict__ qy,
    scalar_t* __restrict__ qz,
    scalar_t* __restrict__ qw,
    int* __restrict__ branch_out) {
    const scalar_t trace = r00 + r11 + r22;
    int br = 3;
    if (trace > r00 && trace > r11 && trace > r22) {
        br = 3;
    } else if (r00 > r11 && r00 > r22) {
        br = 0;
    } else if (r11 > r22) {
        br = 1;
    } else {
        br = 2;
    }
    *branch_out = br;

    if (br == 3) {
        const scalar_t f = scalar_t(1) + trace;
        const scalar_t s = sqrt(f) * scalar_t(2);
        const scalar_t inv_s = scalar_t(1) / s;
        *qx = (r21 - r12) * inv_s;
        *qy = (r02 - r20) * inv_s;
        *qz = (r10 - r01) * inv_s;
        *qw = scalar_t(0.25) * s;
    } else if (br == 0) {
        const scalar_t f = scalar_t(1) + r00 - r11 - r22;
        const scalar_t s = sqrt(f) * scalar_t(2);
        const scalar_t inv_s = scalar_t(1) / s;
        *qx = scalar_t(0.25) * s;
        *qy = (r01 + r10) * inv_s;
        *qz = (r02 + r20) * inv_s;
        *qw = (r21 - r12) * inv_s;
    } else if (br == 1) {
        const scalar_t f = scalar_t(1) + r11 - r00 - r22;
        const scalar_t s = sqrt(f) * scalar_t(2);
        const scalar_t inv_s = scalar_t(1) / s;
        *qx = (r01 + r10) * inv_s;
        *qy = scalar_t(0.25) * s;
        *qz = (r12 + r21) * inv_s;
        *qw = (r02 - r20) * inv_s;
    } else {
        const scalar_t f = scalar_t(1) + r22 - r00 - r11;
        const scalar_t s = sqrt(f) * scalar_t(2);
        const scalar_t inv_s = scalar_t(1) / s;
        *qx = (r02 + r20) * inv_s;
        *qy = (r12 + r21) * inv_s;
        *qz = scalar_t(0.25) * s;
        *qw = (r10 - r01) * inv_s;
    }
}

// Shepperd VJP: (gqx,gqy,gqz,gqw)=∂L/∂q̂ and fixed `br` from forward → g[9] row-major ∂L/∂R (accumulate into g).
template <typename scalar_t>
__device__ void shepperd_from_matrix_bwd(
    int br,
    scalar_t r00,
    scalar_t r01,
    scalar_t r02,
    scalar_t r10,
    scalar_t r11,
    scalar_t r12,
    scalar_t r20,
    scalar_t r21,
    scalar_t r22,
    scalar_t gqx,
    scalar_t gqy,
    scalar_t gqz,
    scalar_t gqw,
    scalar_t* __restrict__ g) {
    for (int k = 0; k < 9; ++k) {
        g[k] = scalar_t(0);
    }
    if (br == 3) {
        const scalar_t trace = r00 + r11 + r22;
        const scalar_t f = scalar_t(1) + trace;
        const scalar_t s2 = sqrt(f);
        const scalar_t inv_s = scalar_t(1) / (scalar_t(2) * s2);
        const scalar_t A = r21 - r12;
        const scalar_t B = r02 - r20;
        const scalar_t C = r10 - r01;
        const scalar_t d_inv_s_df = -scalar_t(1) / (scalar_t(4) * f * s2);
        const scalar_t d_qw_df = scalar_t(1) / (scalar_t(4) * s2);
        const scalar_t dL_d_inv_s = A * gqx + B * gqy + C * gqz;
        const scalar_t dL_df = dL_d_inv_s * d_inv_s_df + gqw * d_qw_df;
        g[0] += dL_df;
        g[4] += dL_df;
        g[8] += dL_df;
        g[7] += inv_s * gqx;
        g[5] -= inv_s * gqx;
        g[2] += inv_s * gqy;
        g[6] -= inv_s * gqy;
        g[3] += inv_s * gqz;
        g[1] -= inv_s * gqz;
    } else if (br == 0) {
        const scalar_t f = scalar_t(1) + r00 - r11 - r22;
        const scalar_t s2 = sqrt(f);
        const scalar_t inv_s = scalar_t(1) / (scalar_t(2) * s2);
        const scalar_t D = r01 + r10;
        const scalar_t E = r02 + r20;
        const scalar_t Fv = r21 - r12;
        const scalar_t d_inv_s_df = -scalar_t(1) / (scalar_t(4) * f * s2);
        const scalar_t d_qx_df = scalar_t(1) / (scalar_t(4) * s2);
        const scalar_t dL_d_inv_s = D * gqy + E * gqz + Fv * gqw;
        const scalar_t dL_df = dL_d_inv_s * d_inv_s_df + gqx * d_qx_df;
        g[0] += dL_df;
        g[4] -= dL_df;
        g[8] -= dL_df;
        g[1] += inv_s * gqy;
        g[3] += inv_s * gqy;
        g[2] += inv_s * gqz;
        g[6] += inv_s * gqz;
        g[7] += inv_s * gqw;
        g[5] -= inv_s * gqw;
    } else if (br == 1) {
        const scalar_t f = scalar_t(1) + r11 - r00 - r22;
        const scalar_t s2 = sqrt(f);
        const scalar_t inv_s = scalar_t(1) / (scalar_t(2) * s2);
        const scalar_t D = r01 + r10;
        const scalar_t E = r12 + r21;
        const scalar_t Fv = r02 - r20;
        const scalar_t d_inv_s_df = -scalar_t(1) / (scalar_t(4) * f * s2);
        const scalar_t d_qy_df = scalar_t(1) / (scalar_t(4) * s2);
        const scalar_t dL_d_inv_s = D * gqx + E * gqz + Fv * gqw;
        const scalar_t dL_df = dL_d_inv_s * d_inv_s_df + gqy * d_qy_df;
        g[0] -= dL_df;
        g[4] += dL_df;
        g[8] -= dL_df;
        g[1] += inv_s * gqx;
        g[3] += inv_s * gqx;
        g[5] += inv_s * gqz;
        g[7] += inv_s * gqz;
        g[2] += inv_s * gqw;
        g[6] -= inv_s * gqw;
    } else {
        const scalar_t f = scalar_t(1) + r22 - r00 - r11;
        const scalar_t s2 = sqrt(f);
        const scalar_t inv_s = scalar_t(1) / (scalar_t(2) * s2);
        const scalar_t D = r02 + r20;
        const scalar_t E = r12 + r21;
        const scalar_t Fv = r10 - r01;
        const scalar_t d_inv_s_df = -scalar_t(1) / (scalar_t(4) * f * s2);
        const scalar_t d_qz_df = scalar_t(1) / (scalar_t(4) * s2);
        const scalar_t dL_d_inv_s = D * gqx + E * gqy + Fv * gqw;
        const scalar_t dL_df = dL_d_inv_s * d_inv_s_df + gqz * d_qz_df;
        g[0] -= dL_df;
        g[4] -= dL_df;
        g[8] += dL_df;
        g[2] += inv_s * gqx;
        g[6] += inv_s * gqx;
        g[5] += inv_s * gqy;
        g[7] += inv_s * gqy;
        g[3] += inv_s * gqw;
        g[1] -= inv_s * gqw;
    }
}

namespace trajectory_cuda {

// Float32 trajectory: uses quaternion.cuh templates (cross3, quat_rotate_vector_*, quat_multiply_impl, quat_slerp_clamp_dot).

// SLERP q1→q2 at parameter ti∈[0,1]; xyzw; hemisphere flip; if dot>0.9995 use normalized lerp. Outputs *ox..*ow.
__device__ void quat_slerp_pair_fwd_f(
    float x1,
    float y1,
    float z1,
    float w1,
    float x2,
    float y2,
    float z2,
    float w2,
    float ti,
    float* ox,
    float* oy,
    float* oz,
    float* ow) {
    const float dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const float s = dot < 0.0f ? -1.0f : 1.0f;
    const float sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const float c_raw = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const float c = quat_slerp_clamp_dot<float>(c_raw);
    const float small_th = quat_slerp_small_angle_dot_threshold<float>();

    if (c > small_th) {
        const float om = 1.0f - ti;
        const float rx = om * x1 + ti * sx;
        const float ry = om * y1 + ti * sy;
        const float rz = om * z1 + ti * sz;
        const float rw = om * w1 + ti * sw;
        const float norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
        const float inv_n = 1.0f / sqrtf(norm_sq);
        *ox = rx * inv_n;
        *oy = ry * inv_n;
        *oz = rz * inv_n;
        *ow = rw * inv_n;
        return;
    }

    const float theta = acosf(c);
    const float sin_theta = sinf(theta);
    const float w1s = sinf((1.0f - ti) * theta) / sin_theta;
    const float w2s = sinf(ti * theta) / sin_theta;
    *ox = w1s * x1 + w2s * sx;
    *oy = w1s * y1 + w2s * sy;
    *oz = w1s * z1 + w2s * sz;
    *ow = w1s * w1 + w2s * sw;
}

// VJP for quat_slerp_pair_fwd_f: forward outputs (rx,ry,rz,rw) and upstream (gx..gw) → *gq1*, *gq2*, *grad_t.
__device__ void quat_slerp_pair_bwd_f(
    float x1,
    float y1,
    float z1,
    float w1,
    float x2,
    float y2,
    float z2,
    float w2,
    float ti,
    float rx,
    float ry,
    float rz,
    float rw,
    float gx,
    float gy,
    float gz,
    float gw,
    float* gq1x,
    float* gq1y,
    float* gq1z,
    float* gq1w,
    float* gq2x,
    float* gq2y,
    float* gq2z,
    float* gq2w,
    float* grad_t) {
    const float dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const float s = dot < 0.0f ? -1.0f : 1.0f;
    const float sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const float c_raw = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const float c = quat_slerp_clamp_dot<float>(c_raw);
    const bool interior_c = (c_raw > -1.0f) && (c_raw < 1.0f);
    const float c_mask = interior_c ? 1.0f : 0.0f;
    const float small_th = quat_slerp_small_angle_dot_threshold<float>();

    if (c > small_th) {
        const float om = 1.0f - ti;
        const float rrx = om * x1 + ti * sx;
        const float rry = om * y1 + ti * sy;
        const float rrz = om * z1 + ti * sz;
        const float rrw = om * w1 + ti * sw;
        const float norm_sq = rrx * rrx + rry * rry + rrz * rrz + rrw * rrw;
        const float r_norm = sqrtf(norm_sq);
        const float yx = rx, yy = ry, yz = rz, yw = rw;
        const float ydotg = yx * gx + yy * gy + yz * gz + yw * gw;
        const float scale = 1.0f / r_norm;
        const float grx = (gx - yx * ydotg) * scale;
        const float gry = (gy - yy * ydotg) * scale;
        const float grz = (gz - yz * ydotg) * scale;
        const float grw = (gw - yw * ydotg) * scale;
        *gq1x = om * grx;
        *gq1y = om * gry;
        *gq1z = om * grz;
        *gq1w = om * grw;
        *gq2x = s * ti * grx;
        *gq2y = s * ti * gry;
        *gq2z = s * ti * grz;
        *gq2w = s * ti * grw;
        const float dq2m_dq1x = sx - x1, dq2m_dq1y = sy - y1, dq2m_dq1z = sz - z1, dq2m_dq1w = sw - w1;
        *grad_t = grx * dq2m_dq1x + gry * dq2m_dq1y + grz * dq2m_dq1z + grw * dq2m_dq1w;
        return;
    }

    const float theta = acosf(c);
    const float sin_theta = sinf(theta);
    const float w1s = sinf((1.0f - ti) * theta) / sin_theta;
    const float w2s = sinf(ti * theta) / sin_theta;
    const float G1 = gx * x1 + gy * y1 + gz * z1 + gw * w1;
    const float G2 = gx * sx + gy * sy + gz * sz + gw * sw;
    const float den = sin_theta * sin_theta;
    const float dw1_dtheta =
        ((1.0f - ti) * cosf((1.0f - ti) * theta) * sin_theta - sinf((1.0f - ti) * theta) * c) / den;
    const float dw2_dtheta =
        (ti * cosf(ti * theta) * sin_theta - sinf(ti * theta) * c) / den;
    const float dw1_dc = sin_theta > 1e-20f ? (-dw1_dtheta / sin_theta) * c_mask : 0.0f;
    const float dw2_dc = sin_theta > 1e-20f ? (-dw2_dtheta / sin_theta) * c_mask : 0.0f;
    const float K = G1 * dw1_dc + G2 * dw2_dc;
    *gq1x = w1s * gx + K * sx;
    *gq1y = w1s * gy + K * sy;
    *gq1z = w1s * gz + K * sz;
    *gq1w = w1s * gw + K * sw;
    const float gq2ex = w2s * gx + K * x1;
    const float gq2ey = w2s * gy + K * y1;
    const float gq2ez = w2s * gz + K * z1;
    const float gq2ew = w2s * gw + K * w1;
    *gq2x = s * gq2ex;
    *gq2y = s * gq2ey;
    *gq2z = s * gq2ez;
    *gq2w = s * gq2ew;
    const float dw1_dt = -theta * cosf((1.0f - ti) * theta) / sin_theta;
    const float dw2_dt = theta * cosf(ti * theta) / sin_theta;
    *grad_t = G1 * dw1_dt + G2 * dw2_dt;
}

// Time-interpolation chain rule: grad_alpha = ∂L/∂α with α=(qt-t_min)/(t_max-t_min), d=t_max-t_min.
// Outputs *g_t0,*g_t1,*g_qt for ∂L/∂time0, ∂L/∂time1, ∂L/∂query_time (handles t0>t1 swap).
__device__ void trajectory_alpha_time_grads_f(
    float t0s,
    float t1s,
    float qt,
    float d,
    float grad_alpha,
    float* g_t0,
    float* g_t1,
    float* g_qt) {
    if (d <= 0.0f) {
        *g_t0 = *g_t1 = *g_qt = 0.0f;
        return;
    }
    const float inv_d = 1.0f / d;
    *g_qt = grad_alpha * inv_d;
    if (t0s <= t1s) {
        const float d2 = d * d;
        *g_t0 = grad_alpha * (qt - t1s) / d2;
        *g_t1 = -grad_alpha * (qt - t0s) / d2;
    } else {
        const float d2 = d * d;
        *g_t1 = grad_alpha * (qt - t0s) / d2;
        *g_t0 = -grad_alpha * (qt - t1s) / d2;
    }
}

// -----------------------------------------------------------------------------
// Float32 trajectory: two-keyframe SLERP + helpers; OOB flags when query time leaves the span.
// -----------------------------------------------------------------------------

// Row i: interpolate pose between keyframes 0/1 at query_time. trans* (N,3), rot* xyzw (N,4), time* indexed by strides st0/st1; query_time stride sqt.
// result_point (N,3) = R(qi)(point) + ti; result_oob[i]∈{0,1} if qt outside [min(t0,t1), max(t0,t1)].
__device__ void trajectory_transform_point_2poses_fwd_device(int64_t i, int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ trans0,
    const float* __restrict__ rot0,
    const float* __restrict__ time0,
    const float* __restrict__ trans1,
    const float* __restrict__ rot1,
    const float* __restrict__ time1,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    float* __restrict__ result_point,
    float* __restrict__ result_oob) {
    const int64_t o3 = i * 3;
    const int64_t o4 = i * 4;
    const float t0s = time0[i * st0];
    const float t1s = time1[i * st1];
    const float qt = query_time[i * sqt];
    const float t_min = fminf(t0s, t1s);
    const float t_max = fmaxf(t0s, t1s);
    const bool oob = (qt < t_min) || (qt > t_max);
    const float d = t_max - t_min;
    const float alpha = d > 0.0f ? (qt - t_min) / d : 0.0f;

    float tx0 = trans0[o3 + 0], ty0 = trans0[o3 + 1], tz0 = trans0[o3 + 2];
    float tx1 = trans1[o3 + 0], ty1 = trans1[o3 + 1], tz1 = trans1[o3 + 2];
    float qx0 = rot0[o4 + 0], qy0 = rot0[o4 + 1], qz0 = rot0[o4 + 2], qw0 = rot0[o4 + 3];
    float qx1 = rot1[o4 + 0], qy1 = rot1[o4 + 1], qz1 = rot1[o4 + 2], qw1 = rot1[o4 + 3];
    float px = point[o3 + 0], py = point[o3 + 1], pz = point[o3 + 2];

    float tix, tiy, tiz;
    float qix, qiy, qiz, qiw;
    if (t0s <= t1s) {
        const float om = 1.0f - alpha;
        tix = om * tx0 + alpha * tx1;
        tiy = om * ty0 + alpha * ty1;
        tiz = om * tz0 + alpha * tz1;
        quat_slerp_pair_fwd_f(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1, alpha, &qix, &qiy, &qiz, &qiw);
    } else {
        const float om = 1.0f - alpha;
        tix = om * tx1 + alpha * tx0;
        tiy = om * ty1 + alpha * ty0;
        tiz = om * tz1 + alpha * tz0;
        quat_slerp_pair_fwd_f(qx1, qy1, qz1, qw1, qx0, qy0, qz0, qw0, alpha, &qix, &qiy, &qiz, &qiw);
    }
    float rx, ry, rz;
    quat_rotate_vector_fwd_impl<float>(qix, qiy, qiz, qiw, px, py, pz, &rx, &ry, &rz);
    result_point[o3 + 0] = rx + tix;
    result_point[o3 + 1] = ry + tiy;
    result_point[o3 + 2] = rz + tiz;
    result_oob[i] = oob ? 1.0f : 0.0f;
}

// VJP for 2-pose point transform: grad_result_point (N,3) → grads for trans0/1, rot0/1 (xyzw), time strides, point, query_time.
__device__ void trajectory_transform_point_2poses_bwd_device(int64_t i, int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ trans0,
    const float* __restrict__ rot0,
    const float* __restrict__ time0,
    const float* __restrict__ trans1,
    const float* __restrict__ rot1,
    const float* __restrict__ time1,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    const float* __restrict__ grad_result_point,
    float* __restrict__ grad_trans0,
    float* __restrict__ grad_rot0,
    float* __restrict__ grad_time0,
    float* __restrict__ grad_trans1,
    float* __restrict__ grad_rot1,
    float* __restrict__ grad_time1,
    float* __restrict__ grad_point,
    float* __restrict__ grad_query_time) {
    const int64_t o3 = i * 3;
    const int64_t o4 = i * 4;
    const float t0s = time0[i * st0];
    const float t1s = time1[i * st1];
    const float qt = query_time[i * sqt];
    const float t_min = fminf(t0s, t1s);
    const float t_max = fmaxf(t0s, t1s);
    const float d = t_max - t_min;
    const float alpha = d > 0.0f ? (qt - t_min) / d : 0.0f;

    float tx0 = trans0[o3 + 0], ty0 = trans0[o3 + 1], tz0 = trans0[o3 + 2];
    float tx1 = trans1[o3 + 0], ty1 = trans1[o3 + 1], tz1 = trans1[o3 + 2];
    float qx0 = rot0[o4 + 0], qy0 = rot0[o4 + 1], qz0 = rot0[o4 + 2], qw0 = rot0[o4 + 3];
    float qx1 = rot1[o4 + 0], qy1 = rot1[o4 + 1], qz1 = rot1[o4 + 2], qw1 = rot1[o4 + 3];
    float px = point[o3 + 0], py = point[o3 + 1], pz = point[o3 + 2];

    float qix, qiy, qiz, qiw;
    if (t0s <= t1s) {
        quat_slerp_pair_fwd_f(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1, alpha, &qix, &qiy, &qiz, &qiw);
    } else {
        quat_slerp_pair_fwd_f(qx1, qy1, qz1, qw1, qx0, qy0, qz0, qw0, alpha, &qix, &qiy, &qiz, &qiw);
    }

    const float gtx = grad_result_point[o3 + 0];
    const float gty = grad_result_point[o3 + 1];
    const float gtz = grad_result_point[o3 + 2];

    float gqix, gqiy, gqiz, gqiw;
    float gpx, gpy, gpz;
    quat_rotate_vector_bwd_impl<float>(
        qix,
        qiy,
        qiz,
        qiw,
        px,
        py,
        pz,
        gtx,
        gty,
        gtz,
        &gqix,
        &gqiy,
        &gqiz,
        &gqiw,
        &gpx,
        &gpy,
        &gpz);

    grad_point[o3 + 0] = gpx;
    grad_point[o3 + 1] = gpy;
    grad_point[o3 + 2] = gpz;

    float grad_alpha = 0.0f;

    if (t0s <= t1s) {
        grad_trans0[o3 + 0] = (1.0f - alpha) * gtx;
        grad_trans0[o3 + 1] = (1.0f - alpha) * gty;
        grad_trans0[o3 + 2] = (1.0f - alpha) * gtz;
        grad_trans1[o3 + 0] = alpha * gtx;
        grad_trans1[o3 + 1] = alpha * gty;
        grad_trans1[o3 + 2] = alpha * gtz;
        grad_alpha += gtx * (tx1 - tx0) + gty * (ty1 - ty0) + gtz * (tz1 - tz0);
    } else {
        grad_trans1[o3 + 0] = (1.0f - alpha) * gtx;
        grad_trans1[o3 + 1] = (1.0f - alpha) * gty;
        grad_trans1[o3 + 2] = (1.0f - alpha) * gtz;
        grad_trans0[o3 + 0] = alpha * gtx;
        grad_trans0[o3 + 1] = alpha * gty;
        grad_trans0[o3 + 2] = alpha * gtz;
        grad_alpha += gtx * (tx0 - tx1) + gty * (ty0 - ty1) + gtz * (tz0 - tz1);
    }

    float gq1x, gq1y, gq1z, gq1w, gq2x, gq2y, gq2z, gq2w, ga_slerp = 0.0f;
    if (t0s <= t1s) {
        quat_slerp_pair_bwd_f(
            qx0,
            qy0,
            qz0,
            qw0,
            qx1,
            qy1,
            qz1,
            qw1,
            alpha,
            qix,
            qiy,
            qiz,
            qiw,
            gqix,
            gqiy,
            gqiz,
            gqiw,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w,
            &gq2x,
            &gq2y,
            &gq2z,
            &gq2w,
            &ga_slerp);
        grad_rot0[o4 + 0] = gq1x;
        grad_rot0[o4 + 1] = gq1y;
        grad_rot0[o4 + 2] = gq1z;
        grad_rot0[o4 + 3] = gq1w;
        grad_rot1[o4 + 0] = gq2x;
        grad_rot1[o4 + 1] = gq2y;
        grad_rot1[o4 + 2] = gq2z;
        grad_rot1[o4 + 3] = gq2w;
    } else {
        quat_slerp_pair_bwd_f(
            qx1,
            qy1,
            qz1,
            qw1,
            qx0,
            qy0,
            qz0,
            qw0,
            alpha,
            qix,
            qiy,
            qiz,
            qiw,
            gqix,
            gqiy,
            gqiz,
            gqiw,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w,
            &gq2x,
            &gq2y,
            &gq2z,
            &gq2w,
            &ga_slerp);
        grad_rot1[o4 + 0] = gq1x;
        grad_rot1[o4 + 1] = gq1y;
        grad_rot1[o4 + 2] = gq1z;
        grad_rot1[o4 + 3] = gq1w;
        grad_rot0[o4 + 0] = gq2x;
        grad_rot0[o4 + 1] = gq2y;
        grad_rot0[o4 + 2] = gq2z;
        grad_rot0[o4 + 3] = gq2w;
    }
    grad_alpha += ga_slerp;

    float g_t0 = 0.0f, g_t1 = 0.0f, g_qt = 0.0f;
    trajectory_alpha_time_grads_f(t0s, t1s, qt, d, grad_alpha, &g_t0, &g_t1, &g_qt);
    grad_time0[i * st0] = g_t0;
    grad_time1[i * st1] = g_t1;
    grad_query_time[i * sqt] = g_qt;
}

// Row i: SLERP rot0/rot1 at query_time (same α and OOB semantics as 2-pose point); result_quat (N,4) xyzw.
__device__ void trajectory_get_rotation_2poses_fwd_device(int64_t i, int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ time0,
    const float* __restrict__ time1,
    const float* __restrict__ query_time,
    const float* __restrict__ rot0,
    const float* __restrict__ rot1,
    float* __restrict__ result_quat,
    float* __restrict__ result_oob) {
    const int64_t o4 = i * 4;
    const float t0s = time0[i * st0];
    const float t1s = time1[i * st1];
    const float qt = query_time[i * sqt];
    const float t_min = fminf(t0s, t1s);
    const float t_max = fmaxf(t0s, t1s);
    const bool oob = (qt < t_min) || (qt > t_max);
    const float d = t_max - t_min;
    const float alpha = d > 0.0f ? (qt - t_min) / d : 0.0f;
    float qx0 = rot0[o4 + 0], qy0 = rot0[o4 + 1], qz0 = rot0[o4 + 2], qw0 = rot0[o4 + 3];
    float qx1 = rot1[o4 + 0], qy1 = rot1[o4 + 1], qz1 = rot1[o4 + 2], qw1 = rot1[o4 + 3];
    float ox, oy, oz, ow;
    if (t0s <= t1s) {
        quat_slerp_pair_fwd_f(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1, alpha, &ox, &oy, &oz, &ow);
    } else {
        quat_slerp_pair_fwd_f(qx1, qy1, qz1, qw1, qx0, qy0, qz0, qw0, alpha, &ox, &oy, &oz, &ow);
    }
    result_quat[o4 + 0] = ox;
    result_quat[o4 + 1] = oy;
    result_quat[o4 + 2] = oz;
    result_quat[o4 + 3] = ow;
    result_oob[i] = oob ? 1.0f : 0.0f;
}

// VJP: grad_result_quat (N,4) → grad_rot0, grad_rot1; time grads via trajectory_alpha_time_grads_f on SLERP ∂L/∂α.
__device__ void trajectory_get_rotation_2poses_bwd_device(int64_t i, int64_t n,
    int64_t st0,
    int64_t st1,
    int64_t sqt,
    const float* __restrict__ time0,
    const float* __restrict__ time1,
    const float* __restrict__ query_time,
    const float* __restrict__ rot0,
    const float* __restrict__ rot1,
    const float* __restrict__ grad_result_quat,
    float* __restrict__ grad_rot0,
    float* __restrict__ grad_rot1,
    float* __restrict__ grad_time0,
    float* __restrict__ grad_time1,
    float* __restrict__ grad_query_time) {
    const int64_t o4 = i * 4;
    const float t0s = time0[i * st0];
    const float t1s = time1[i * st1];
    const float qt = query_time[i * sqt];
    const float t_min = fminf(t0s, t1s);
    const float t_max = fmaxf(t0s, t1s);
    const float d = t_max - t_min;
    const float alpha = d > 0.0f ? (qt - t_min) / d : 0.0f;

    float qx0 = rot0[o4 + 0], qy0 = rot0[o4 + 1], qz0 = rot0[o4 + 2], qw0 = rot0[o4 + 3];
    float qx1 = rot1[o4 + 0], qy1 = rot1[o4 + 1], qz1 = rot1[o4 + 2], qw1 = rot1[o4 + 3];

    float qix, qiy, qiz, qiw;
    if (t0s <= t1s) {
        quat_slerp_pair_fwd_f(qx0, qy0, qz0, qw0, qx1, qy1, qz1, qw1, alpha, &qix, &qiy, &qiz, &qiw);
    } else {
        quat_slerp_pair_fwd_f(qx1, qy1, qz1, qw1, qx0, qy0, qz0, qw0, alpha, &qix, &qiy, &qiz, &qiw);
    }

    const float gx = grad_result_quat[o4 + 0];
    const float gy = grad_result_quat[o4 + 1];
    const float gz = grad_result_quat[o4 + 2];
    const float gw = grad_result_quat[o4 + 3];

    float gq1x, gq1y, gq1z, gq1w, gq2x, gq2y, gq2z, gq2w, ga_slerp = 0.0f;
    if (t0s <= t1s) {
        quat_slerp_pair_bwd_f(
            qx0,
            qy0,
            qz0,
            qw0,
            qx1,
            qy1,
            qz1,
            qw1,
            alpha,
            qix,
            qiy,
            qiz,
            qiw,
            gx,
            gy,
            gz,
            gw,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w,
            &gq2x,
            &gq2y,
            &gq2z,
            &gq2w,
            &ga_slerp);
        grad_rot0[o4 + 0] = gq1x;
        grad_rot0[o4 + 1] = gq1y;
        grad_rot0[o4 + 2] = gq1z;
        grad_rot0[o4 + 3] = gq1w;
        grad_rot1[o4 + 0] = gq2x;
        grad_rot1[o4 + 1] = gq2y;
        grad_rot1[o4 + 2] = gq2z;
        grad_rot1[o4 + 3] = gq2w;
    } else {
        quat_slerp_pair_bwd_f(
            qx1,
            qy1,
            qz1,
            qw1,
            qx0,
            qy0,
            qz0,
            qw0,
            alpha,
            qix,
            qiy,
            qiz,
            qiw,
            gx,
            gy,
            gz,
            gw,
            &gq1x,
            &gq1y,
            &gq1z,
            &gq1w,
            &gq2x,
            &gq2y,
            &gq2z,
            &gq2w,
            &ga_slerp);
        grad_rot1[o4 + 0] = gq1x;
        grad_rot1[o4 + 1] = gq1y;
        grad_rot1[o4 + 2] = gq1z;
        grad_rot1[o4 + 3] = gq1w;
        grad_rot0[o4 + 0] = gq2x;
        grad_rot0[o4 + 1] = gq2y;
        grad_rot0[o4 + 2] = gq2z;
        grad_rot0[o4 + 3] = gq2w;
    }

    float g_t0 = 0.0f, g_t1 = 0.0f, g_qt = 0.0f;
    trajectory_alpha_time_grads_f(t0s, t1s, qt, d, ga_slerp, &g_t0, &g_t1, &g_qt);
    grad_time0[i * st0] = g_t0;
    grad_time1[i * st1] = g_t1;
    grad_query_time[i * sqt] = g_qt;
}

// -----------------------------------------------------------------------------
// Single-keyframe trajectory (float32): OOB = (query_time != keyframe time), transform still applied.
// -----------------------------------------------------------------------------

// Row i: result = R(rot_i) point_i + trans_i; result_oob[i]=1 if query_time[i*sqt] != time[i*st] (exact float compare).
__device__ void trajectory_transform_point_1pose_fwd_device(int64_t i, int64_t n,
    int64_t st,
    int64_t sqt,
    const float* __restrict__ trans,
    const float* __restrict__ rot,
    const float* __restrict__ time,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    float* __restrict__ result_point,
    float* __restrict__ result_oob) {
    const int64_t o3 = i * 3;
    const int64_t o4 = i * 4;
    const float tt = time[i * st];
    const float qt = query_time[i * sqt];
    const bool oob = (qt != tt);

    const float qx = rot[o4 + 0], qy = rot[o4 + 1], qz = rot[o4 + 2], qw = rot[o4 + 3];
    const float px = point[o3 + 0], py = point[o3 + 1], pz = point[o3 + 2];
    const float tx = trans[o3 + 0], ty = trans[o3 + 1], tz = trans[o3 + 2];
    float rx = 0, ry = 0, rz = 0;
    quat_rotate_vector_fwd_impl<float>(qx, qy, qz, qw, px, py, pz, &rx, &ry, &rz);
    result_point[o3 + 0] = rx + tx;
    result_point[o3 + 1] = ry + ty;
    result_point[o3 + 2] = rz + tz;
    result_oob[i] = oob ? 1.0f : 0.0f;
}

// VJP: grad_result_point → grad_trans, grad_rot (xyzw), grad_point; grad_time/grad_query_time not populated by device body (kernel may ignore).
__device__ void trajectory_transform_point_1pose_bwd_device(int64_t i, int64_t n,
    const float* __restrict__ trans,
    const float* __restrict__ rot,
    const float* __restrict__ time,
    const float* __restrict__ point,
    const float* __restrict__ query_time,
    const float* __restrict__ grad_result_point,
    float* __restrict__ grad_trans,
    float* __restrict__ grad_rot,
    float* __restrict__ grad_time,
    float* __restrict__ grad_point,
    float* __restrict__ grad_query_time) {
    const int64_t o3 = i * 3;
    const int64_t o4 = i * 4;
    const float qx = rot[o4 + 0], qy = rot[o4 + 1], qz = rot[o4 + 2], qw = rot[o4 + 3];
    const float px = point[o3 + 0], py = point[o3 + 1], pz = point[o3 + 2];

    const float gtx = grad_result_point[o3 + 0];
    const float gty = grad_result_point[o3 + 1];
    const float gtz = grad_result_point[o3 + 2];

    float gqix, gqiy, gqiz, gqiw;
    float gpx, gpy, gpz;
    quat_rotate_vector_bwd_impl<float>(
        qx,
        qy,
        qz,
        qw,
        px,
        py,
        pz,
        gtx,
        gty,
        gtz,
        &gqix,
        &gqiy,
        &gqiz,
        &gqiw,
        &gpx,
        &gpy,
        &gpz);

    grad_trans[o3 + 0] = gtx;
    grad_trans[o3 + 1] = gty;
    grad_trans[o3 + 2] = gtz;
    grad_rot[o4 + 0] = gqix;
    grad_rot[o4 + 1] = gqiy;
    grad_rot[o4 + 2] = gqiz;
    grad_rot[o4 + 3] = gqiw;
    grad_point[o3 + 0] = gpx;
    grad_point[o3 + 1] = gpy;
    grad_point[o3 + 2] = gpz;
}

// Row i: 7-D pose (tx,ty,tz, qx,qy,qz,qw) → new_q = fr⊗iq, new_t = scale * (R(fr)*it + fxyz); frame quat fr is xyzw.
__device__ void frame_transform_poses_tquat_fwd_device(int64_t i, int64_t n,
    float frx,
    float fry,
    float frz,
    float frw,
    float ftx,
    float fty,
    float ftz,
    float scale,
    const float* __restrict__ input_poses,
    float* __restrict__ output_poses) {
    const int64_t o7 = i * 7;
    const float itx = input_poses[o7 + 0], ity = input_poses[o7 + 1], itz = input_poses[o7 + 2];
    const float iqx = input_poses[o7 + 3], iqy = input_poses[o7 + 4], iqz = input_poses[o7 + 5], iqw = input_poses[o7 + 6];
    float nqx = 0, nqy = 0, nqz = 0, nqw = 0;
    quat_multiply_impl<float>(frx, fry, frz, frw, iqx, iqy, iqz, iqw, &nqx, &nqy, &nqz, &nqw);
    float rtx = 0, rty = 0, rtz = 0;
    quat_rotate_vector_fwd_impl<float>(frx, fry, frz, frw, itx, ity, itz, &rtx, &rty, &rtz);
    const float ntx = scale * (rtx + ftx);
    const float nty = scale * (rty + fty);
    const float ntz = scale * (rtz + ftz);
    output_poses[o7 + 0] = ntx;
    output_poses[o7 + 1] = nty;
    output_poses[o7 + 2] = ntz;
    output_poses[o7 + 3] = nqx;
    output_poses[o7 + 4] = nqy;
    output_poses[o7 + 5] = nqz;
    output_poses[o7 + 6] = nqw;
}

} // namespace trajectory_cuda

// Batch row i: out = R(q_i) p_i + t_i. translation/point (N,3), rotation (N,4) xyzw, contiguous row-major.
template <typename scalar_t>
__device__ void se3pose_transform_point_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ out) {
    const int64_t ot = i * 3;
    const int64_t oq = i * 4;
    const scalar_t qx = rotation[oq + 0], qy = rotation[oq + 1], qz = rotation[oq + 2], qw = rotation[oq + 3];
    const scalar_t px = point[ot + 0], py = point[ot + 1], pz = point[ot + 2];
    const scalar_t tx = translation[ot + 0], ty = translation[ot + 1], tz = translation[ot + 2];
    scalar_t rx = 0, ry = 0, rz = 0;
    quat_rotate_vector_fwd_impl(qx, qy, qz, qw, px, py, pz, &rx, &ry, &rz);
    out[ot + 0] = rx + tx;
    out[ot + 1] = ry + ty;
    out[ot + 2] = rz + tz;
}

// Batch row i: out = R(q_i) d_i (3-vector); rotation xyzw (N,4), direction/out (N,3).
template <typename scalar_t>
__device__ void se3pose_transform_direction_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ direction,
    scalar_t* __restrict__ out) {
    const int64_t ov = i * 3;
    const int64_t oq = i * 4;
    const scalar_t qx = rotation[oq + 0], qy = rotation[oq + 1], qz = rotation[oq + 2], qw = rotation[oq + 3];
    const scalar_t dx = direction[ov + 0], dy = direction[ov + 1], dz = direction[ov + 2];
    quat_rotate_vector_fwd_impl(qx, qy, qz, qw, dx, dy, dz, out + ov, out + ov + 1, out + ov + 2);
}

// Batch row i: out = R(q)^T (p - t) = rotate(q*, p-t) with conjugate q*=(-x,-y,-z,w) in xyzw storage.
template <typename scalar_t>
__device__ void se3pose_inverse_transform_point_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ point,
    scalar_t* __restrict__ out) {
    const int64_t ov = i * 3;
    const int64_t oq = i * 4;
    const scalar_t qx = rotation[oq + 0], qy = rotation[oq + 1], qz = rotation[oq + 2], qw = rotation[oq + 3];
    const scalar_t px = point[ov + 0], py = point[ov + 1], pz = point[ov + 2];
    const scalar_t tx = translation[ov + 0], ty = translation[ov + 1], tz = translation[ov + 2];
    const scalar_t vx = px - tx, vy = py - ty, vz = pz - tz;
    quat_rotate_vector_fwd_impl(-qx, -qy, -qz, qw, vx, vy, vz, out + ov, out + ov + 1, out + ov + 2);
}

// Batch row i: out = R(q)^T d (same conjugate trick as inverse point, without translation).
template <typename scalar_t>
__device__ void se3pose_inverse_transform_direction_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ rotation,
    const scalar_t* __restrict__ direction,
    scalar_t* __restrict__ out) {
    const int64_t ov = i * 3;
    const int64_t oq = i * 4;
    const scalar_t qx = rotation[oq + 0], qy = rotation[oq + 1], qz = rotation[oq + 2], qw = rotation[oq + 3];
    const scalar_t dx = direction[ov + 0], dy = direction[ov + 1], dz = direction[ov + 2];
    quat_rotate_vector_fwd_impl(-qx, -qy, -qz, qw, dx, dy, dz, out + ov, out + ov + 1, out + ov + 2);
}

// Batch row i: out16[i*16..+15] = 4×4 row-major homogeneous [R|t; 0 0 0 1]; R9 row-major 3×3, t length-3.
template <typename scalar_t>
__device__ void se3pose_to_matrix_pack_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ R9,
    scalar_t* __restrict__ out16) {
    const int64_t ot = i * 3;
    const int64_t or9 = i * 9;
    const int64_t o16 = i * 16;
    const scalar_t tx = translation[ot + 0], ty = translation[ot + 1], tz = translation[ot + 2];
    const scalar_t r0 = R9[or9 + 0], r1 = R9[or9 + 1], r2 = R9[or9 + 2];
    const scalar_t r3 = R9[or9 + 3], r4 = R9[or9 + 4], r5 = R9[or9 + 5];
    const scalar_t r6 = R9[or9 + 6], r7 = R9[or9 + 7], r8 = R9[or9 + 8];
    out16[o16 + 0] = r0;
    out16[o16 + 1] = r1;
    out16[o16 + 2] = r2;
    out16[o16 + 3] = tx;
    out16[o16 + 4] = r3;
    out16[o16 + 5] = r4;
    out16[o16 + 6] = r5;
    out16[o16 + 7] = ty;
    out16[o16 + 8] = r6;
    out16[o16 + 9] = r7;
    out16[o16 + 10] = r8;
    out16[o16 + 11] = tz;
    out16[o16 + 12] = scalar_t(0);
    out16[o16 + 13] = scalar_t(0);
    out16[o16 + 14] = scalar_t(0);
    out16[o16 + 15] = scalar_t(1);
}

// Batch row i: inverse SE(3) as row-major 4×4 [R^T | -R^T t; 0 0 0 1]. rotation buffer xyzw if wxyz_format==0 else (w,x,y,z).
template <typename scalar_t>
__device__ void se3pose_to_inverse_matrix_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ translation,
    const scalar_t* __restrict__ rotation,
    scalar_t* __restrict__ out16,
    const int wxyz_format) {
    const int64_t ot = i * 3;
    const int64_t oq = i * 4;
    scalar_t qx, qy, qz, qw;
    if (wxyz_format != 0) {
        qw = rotation[oq + 0];
        qx = rotation[oq + 1];
        qy = rotation[oq + 2];
        qz = rotation[oq + 3];
    } else {
        qx = rotation[oq + 0];
        qy = rotation[oq + 1];
        qz = rotation[oq + 2];
        qw = rotation[oq + 3];
    }
    const scalar_t tx = translation[ot + 0], ty = translation[ot + 1], tz = translation[ot + 2];
    scalar_t r[9];
    quat_to_matrix_fwd_write<scalar_t>(qx, qy, qz, qw, r);
    const scalar_t ux =
        -(r[0] * tx + r[3] * ty + r[6] * tz);
    const scalar_t uy =
        -(r[1] * tx + r[4] * ty + r[7] * tz);
    const scalar_t uz =
        -(r[2] * tx + r[5] * ty + r[8] * tz);
    const int64_t o16 = i * 16;
    out16[o16 + 0] = r[0];
    out16[o16 + 1] = r[3];
    out16[o16 + 2] = r[6];
    out16[o16 + 3] = ux;
    out16[o16 + 4] = r[1];
    out16[o16 + 5] = r[4];
    out16[o16 + 6] = r[7];
    out16[o16 + 7] = uy;
    out16[o16 + 8] = r[2];
    out16[o16 + 9] = r[5];
    out16[o16 + 10] = r[8];
    out16[o16 + 11] = uz;
    out16[o16 + 12] = scalar_t(0);
    out16[o16 + 13] = scalar_t(0);
    out16[o16 + 14] = scalar_t(0);
    out16[o16 + 15] = scalar_t(1);
}

// Batch row i: m16 row-major 4×4 → translation (3) from last column of upper 3×4; rotation xyzw via Shepperd + safe normalize.
template <typename scalar_t>
__device__ void se3pose_from_matrix_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ m16,
    scalar_t* __restrict__ translation,
    scalar_t* __restrict__ rotation) {
    const int64_t o = i * 16;
    const scalar_t tx = m16[o + 3], ty = m16[o + 7], tz = m16[o + 11];
    const int64_t ot = i * 3;
    translation[ot + 0] = tx;
    translation[ot + 1] = ty;
    translation[ot + 2] = tz;
    const scalar_t r00 = m16[o + 0], r01 = m16[o + 1], r02 = m16[o + 2];
    const scalar_t r10 = m16[o + 4], r11 = m16[o + 5], r12 = m16[o + 6];
    const scalar_t r20 = m16[o + 8], r21 = m16[o + 9], r22 = m16[o + 10];
    scalar_t qx = 0, qy = 0, qz = 0, qw = 0;
    int br = 0;
    shepperd_from_matrix_fwd<scalar_t>(
        r00, r01, r02, r10, r11, r12, r20, r21, r22, &qx, &qy, &qz, &qw, &br);
    (void)br;
    const int64_t oq = i * 4;
    quat_normalize_safe_fwd_write<scalar_t>(qx, qy, qz, qw, rotation + oq, rotation + oq + 1, rotation + oq + 2, rotation + oq + 3);
}

// VJP: grad_translation (3), grad_rotation (xyzw per row) → grad_m16 row-major 4×4 (zeros then accumulates t and R blocks).
template <typename scalar_t>
__device__ void se3pose_from_matrix_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ m16,
    const scalar_t* __restrict__ grad_translation,
    const scalar_t* __restrict__ grad_rotation,
    scalar_t* __restrict__ grad_m16) {
    const int64_t o = i * 16;
    for (int k = 0; k < 16; ++k) {
        grad_m16[o + k] = scalar_t(0);
    }
    const int64_t ot = i * 3;
    const int64_t oq = i * 4;
    grad_m16[o + 3] = grad_translation[ot + 0];
    grad_m16[o + 7] = grad_translation[ot + 1];
    grad_m16[o + 11] = grad_translation[ot + 2];
    const scalar_t r00 = m16[o + 0], r01 = m16[o + 1], r02 = m16[o + 2];
    const scalar_t r10 = m16[o + 4], r11 = m16[o + 5], r12 = m16[o + 6];
    const scalar_t r20 = m16[o + 8], r21 = m16[o + 9], r22 = m16[o + 10];
    scalar_t qx = 0, qy = 0, qz = 0, qw = 0;
    int br = 0;
    shepperd_from_matrix_fwd<scalar_t>(
        r00, r01, r02, r10, r11, r12, r20, r21, r22, &qx, &qy, &qz, &qw, &br);
    scalar_t gqx = 0, gqy = 0, gqz = 0, gqw = 0;
    quat_normalize_safe_bwd_write<scalar_t>(
        qx,
        qy,
        qz,
        qw,
        grad_rotation[oq + 0],
        grad_rotation[oq + 1],
        grad_rotation[oq + 2],
        grad_rotation[oq + 3],
        &gqx,
        &gqy,
        &gqz,
        &gqw);
    scalar_t gR[9];
    shepperd_from_matrix_bwd<scalar_t>(br, r00, r01, r02, r10, r11, r12, r20, r21, r22, gqx, gqy, gqz, gqw, gR);
    grad_m16[o + 0] += gR[0];
    grad_m16[o + 1] += gR[1];
    grad_m16[o + 2] += gR[2];
    grad_m16[o + 4] += gR[3];
    grad_m16[o + 5] += gR[4];
    grad_m16[o + 6] += gR[5];
    grad_m16[o + 8] += gR[6];
    grad_m16[o + 9] += gR[7];
    grad_m16[o + 10] += gR[8];
}

