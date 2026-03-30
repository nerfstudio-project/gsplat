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

// Device-side quaternion math for geometry CUDA. Paired with quaternion.cu:
// this header holds __device__ templates/helpers (scalar *_write / *_impl and
// per-row *_fwd_device / *_bwd_device delegating to them); the .cu file holds
// __global__ kernels, launches, and exported *_cuda symbols.
#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

// 3D cross product a × b.
// Inputs: components of a and b. Outputs: *ox,*oy,*oz = result (same scalar type as inputs).
template <typename scalar_t>
__forceinline__ __device__ void cross3(
    scalar_t ax,
    scalar_t ay,
    scalar_t az,
    scalar_t bx,
    scalar_t by,
    scalar_t bz,
    scalar_t* ox,
    scalar_t* oy,
    scalar_t* oz) {
    *ox = ay * bz - az * by;
    *oy = az * bx - ax * bz;
    *oz = ax * by - ay * bx;
}

// Rotate 3-vector v by unit quaternion q: v' = q ⊗ v ⊗ q* (pure quaternion v = (vx,vy,vz,0)), Hamilton xyzw.
// Inputs: q components (qx,qy,qz,qw), v components. Outputs: rotated v in *ox,*oy,*oz.
template <typename scalar_t>
__forceinline__ __device__ void quat_rotate_vector_fwd_impl(
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t qw,
    scalar_t vx,
    scalar_t vy,
    scalar_t vz,
    scalar_t* ox,
    scalar_t* oy,
    scalar_t* oz) {
    scalar_t uvx = 0, uvy = 0, uvz = 0;
    cross3(qx, qy, qz, vx, vy, vz, &uvx, &uvy, &uvz);
    scalar_t uuvx = 0, uuvy = 0, uuvz = 0;
    cross3(qx, qy, qz, uvx, uvy, uvz, &uuvx, &uuvy, &uuvz);
    *ox = vx + scalar_t(2) * (qw * uvx + uuvx);
    *oy = vy + scalar_t(2) * (qw * uvy + uuvy);
    *oz = vz + scalar_t(2) * (qw * uvz + uuvz);
}

// VJP: (gx,gy,gz)=∂L/∂(R v) → *gqx..*gqw, *gvx..*gvz; same cross-product path as quat_rotate_vector_fwd_impl.
template <typename scalar_t>
__forceinline__ __device__ void quat_rotate_vector_bwd_impl(
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t qw,
    scalar_t vx,
    scalar_t vy,
    scalar_t vz,
    scalar_t gx,
    scalar_t gy,
    scalar_t gz,
    scalar_t* gqx,
    scalar_t* gqy,
    scalar_t* gqz,
    scalar_t* gqw,
    scalar_t* gvx,
    scalar_t* gvy,
    scalar_t* gvz) {
    scalar_t uvx = 0, uvy = 0, uvz = 0;
    cross3(qx, qy, qz, vx, vy, vz, &uvx, &uvy, &uvz);
    const scalar_t gux = scalar_t(2) * gx, guy = scalar_t(2) * gy, guz = scalar_t(2) * gz;
    scalar_t gvux = 0, gvuy = 0, gvuz = 0;
    cross3(gux, guy, guz, qx, qy, qz, &gvux, &gvuy, &gvuz);
    gvux += scalar_t(2) * qw * gx;
    gvuy += scalar_t(2) * qw * gy;
    gvuz += scalar_t(2) * qw * gz;
    scalar_t gqxv = 0, gqyv = 0, gqzv = 0;
    cross3(uvx, uvy, uvz, gux, guy, guz, &gqxv, &gqyv, &gqzv);
    scalar_t t1x = 0, t1y = 0, t1z = 0;
    cross3(vx, vy, vz, gvux, gvuy, gvuz, &t1x, &t1y, &t1z);
    gqxv += t1x;
    gqyv += t1y;
    gqzv += t1z;
    const scalar_t gqww = scalar_t(2) * (gx * uvx + gy * uvy + gz * uvz);
    scalar_t dvx = 0, dvy = 0, dvz = 0;
    cross3(gvux, gvuy, gvuz, qx, qy, qz, &dvx, &dvy, &dvz);
    *gqx = gqxv;
    *gqy = gqyv;
    *gqz = gqzv;
    *gqw = gqww;
    *gvx = gx + dvx;
    *gvy = gy + dvy;
    *gvz = gz + dvz;
}

// Hamilton product q_out = q1 ⊗ q2, both in xyzw order (vector part then scalar).
// Outputs: *ox,*oy,*oz,*ow = result quaternion.
template <typename scalar_t>
__forceinline__ __device__ void quat_multiply_impl(
    scalar_t x1,
    scalar_t y1,
    scalar_t z1,
    scalar_t w1,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t w2,
    scalar_t* ox,
    scalar_t* oy,
    scalar_t* oz,
    scalar_t* ow) {
    // Hamilton product in xyzw convention.
    const scalar_t cx = y1 * z2 - z1 * y2;
    const scalar_t cy = z1 * x2 - x1 * z2;
    const scalar_t cz = x1 * y2 - y1 * x2;
    *ox = w1 * x2 + w2 * x1 + cx;
    *oy = w1 * y2 + w2 * y1 + cy;
    *oz = w1 * z2 + w2 * z1 + cz;
    *ow = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
}

// -----------------------------------------------------------------------------
// quat_to_matrix using the native CUDA normalize-and-convert path.
// -----------------------------------------------------------------------------

// Squared-norm threshold: if ‖q‖² < value, treat q as zero and use identity (0,0,0,1) before R(q).
template <typename scalar_t>
struct QuatNormEps {};

template <>
struct QuatNormEps<float> {
    static constexpr float value = 1e-7f;
};

template <>
struct QuatNormEps<double> {
    static constexpr double value = 1e-12;
};

// Normalize q (or identity if near-zero), then write rotation matrix R(q) in row-major 3×3 (9 floats).
// Layout: out9[k] = R_rowmajor with R * column_vector = rotated vector; indices [0..2] row0, [3..5] row1, [6..8] row2.
template <typename scalar_t>
__forceinline__ __device__ void quat_to_matrix_fwd_write(
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t qw,
    scalar_t* __restrict__ out9) {
    const scalar_t norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    const scalar_t eps = QuatNormEps<scalar_t>::value;
    scalar_t x, y, z, w;
    if (fabs(static_cast<double>(norm_sq)) < static_cast<double>(eps)) {
        x = scalar_t(0);
        y = scalar_t(0);
        z = scalar_t(0);
        w = scalar_t(1);
    } else {
        const scalar_t inv_s = scalar_t(1) / sqrt(norm_sq);
        x = qx * inv_s;
        y = qy * inv_s;
        z = qz * inv_s;
        w = qw * inv_s;
    }
    const scalar_t x2 = x * x, y2 = y * y, z2 = z * z;
    const scalar_t xy = x * y, xz = x * z, xw = x * w;
    const scalar_t yz = y * z, yw = y * w, zw = z * w;
    out9[0] = scalar_t(1) - scalar_t(2) * (y2 + z2);
    out9[1] = scalar_t(2) * (xy - zw);
    out9[2] = scalar_t(2) * (xz + yw);
    out9[3] = scalar_t(2) * (xy + zw);
    out9[4] = scalar_t(1) - scalar_t(2) * (x2 + z2);
    out9[5] = scalar_t(2) * (yz - xw);
    out9[6] = scalar_t(2) * (xz - yw);
    out9[7] = scalar_t(2) * (yz + xw);
    out9[8] = scalar_t(1) - scalar_t(2) * (x2 + y2);
}

// Single quaternion: safe normalize to *ox..*ow (near-zero raw q → (0,0,0,1)); same eps as QuatNormEps.
template <typename scalar_t>
__forceinline__ __device__ void quat_normalize_safe_fwd_write(
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t qw,
    scalar_t* __restrict__ ox,
    scalar_t* __restrict__ oy,
    scalar_t* __restrict__ oz,
    scalar_t* __restrict__ ow) {
    const scalar_t norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    const scalar_t eps = QuatNormEps<scalar_t>::value;
    if (fabs(static_cast<double>(norm_sq)) < static_cast<double>(eps)) {
        *ox = scalar_t(0);
        *oy = scalar_t(0);
        *oz = scalar_t(0);
        *ow = scalar_t(1);
    } else {
        const scalar_t inv_s = scalar_t(1) / sqrt(norm_sq);
        *ox = qx * inv_s;
        *oy = qy * inv_s;
        *oz = qz * inv_s;
        *ow = qw * inv_s;
    }
}

// VJP through quat_normalize_safe_fwd_write: (gx,gy,gz,gw)=∂L/∂q̂_safe → (*gqx..*gqw)=∂L/∂(raw q).
template <typename scalar_t>
__forceinline__ __device__ void quat_normalize_safe_bwd_write(
    scalar_t qx,
    scalar_t qy,
    scalar_t qz,
    scalar_t qw,
    scalar_t gx,
    scalar_t gy,
    scalar_t gz,
    scalar_t gw,
    scalar_t* __restrict__ gqx,
    scalar_t* __restrict__ gqy,
    scalar_t* __restrict__ gqz,
    scalar_t* __restrict__ gqw) {
    const scalar_t norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    const scalar_t eps = QuatNormEps<scalar_t>::value;
    if (fabs(static_cast<double>(norm_sq)) < static_cast<double>(eps)) {
        *gqx = scalar_t(0);
        *gqy = scalar_t(0);
        *gqz = scalar_t(0);
        *gqw = scalar_t(0);
        return;
    }
    const scalar_t inv_s = scalar_t(1) / sqrt(norm_sq);
    const scalar_t q_dot_g = qx * gx + qy * gy + qz * gz + qw * gw;
    const scalar_t inv_s3 = inv_s * inv_s * inv_s;
    *gqx = gx * inv_s - qx * q_dot_g * inv_s3;
    *gqy = gy * inv_s - qy * q_dot_g * inv_s3;
    *gqz = gz * inv_s - qz * q_dot_g * inv_s3;
    *gqw = gw * inv_s - qw * q_dot_g * inv_s3;
}

// -----------------------------------------------------------------------------
// quat_slerp_batched with hemisphere handling and lerp fallback.
// Hemisphere flip, clamp(dot), small-angle lerp fallback (dot > 0.9995), per-row t (N,1).
// -----------------------------------------------------------------------------

// Clamp dot product to [-1, 1] before acos in SLERP (avoids NaNs from slight overshoot).
template <typename scalar_t>
__forceinline__ __device__ scalar_t quat_slerp_clamp_dot(scalar_t x) {
    if (x < scalar_t(-1)) {
        return scalar_t(-1);
    }
    if (x > scalar_t(1)) {
        return scalar_t(1);
    }
    return x;
}

// Dot threshold above which SLERP uses normalized linear blend instead of sin/acos path (~0.9995).
template <typename scalar_t>
__forceinline__ __device__ scalar_t quat_slerp_small_angle_dot_threshold() {
    return scalar_t(0.9995);
}

// -----------------------------------------------------------------------------
// quat_angular_distance:
// 2 * acos(clamp(abs(dot(normalize(q1), normalize(q2))), 0, 1))
// -----------------------------------------------------------------------------

// Epsilon for angular distance backward: suppress grad when 1−c² is tiny (float vs double).
template <typename scalar_t>
__forceinline__ __device__ scalar_t quat_angular_dist_acos_eps() {
    return sizeof(scalar_t) <= 4 ? scalar_t(1e-7f) : scalar_t(1e-12);
}

static constexpr double kManifoldQuatEpsD = 1e-12;
static constexpr double kManifoldSmallAngleSq = 1e-6; // theta^2 small-angle threshold

// Double-precision Hamilton product (internal to manifold_interp). xyzw components in/out.
// Outputs: *ox,*oy,*oz,*ow.
static __device__ void quat_mul_d(
    double x1,
    double y1,
    double z1,
    double w1,
    double x2,
    double y2,
    double z2,
    double w2,
    double* ox,
    double* oy,
    double* oz,
    double* ow) {
    const double cx = y1 * z2 - z1 * y2;
    const double cy = z1 * x2 - x1 * z2;
    const double cz = x1 * y2 - y1 * x2;
    *ox = w1 * x2 + w2 * x1 + cx;
    *oy = w1 * y2 + w2 * y1 + cy;
    *oz = w1 * z2 + w2 * z1 + cz;
    *ow = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
}

// VJP for quat_mul_d: (gx,gy,gz,gw) = ∂L/∂q_out → gradients w.r.t. q1 (*g1*) and q2 (*g2*), xyzw.
static __device__ void quat_mul_bwd_d(
    double x1,
    double y1,
    double z1,
    double w1,
    double x2,
    double y2,
    double z2,
    double w2,
    double gx,
    double gy,
    double gz,
    double gw,
    double* g1x,
    double* g1y,
    double* g1z,
    double* g1w,
    double* g2x,
    double* g2y,
    double* g2z,
    double* g2w) {
    *g1x = w2 * gx - z2 * gy + y2 * gz - x2 * gw;
    *g1y = z2 * gx + w2 * gy - x2 * gz - y2 * gw;
    *g1z = -y2 * gx + x2 * gy + w2 * gz - z2 * gw;
    *g1w = x2 * gx + y2 * gy + z2 * gz + w2 * gw;
    *g2x = w1 * gx + z1 * gy - y1 * gz - x1 * gw;
    *g2y = -z1 * gx + w1 * gy + x1 * gz - y1 * gw;
    *g2z = y1 * gx - x1 * gy + w1 * gz - z1 * gw;
    *g2w = x1 * gx + y1 * gy + z1 * gz + w1 * gw;
}

// so(3) log: quaternion (xyzw), normalized if needed, → rotation vector ω (axis * angle, double).
// Outputs: *ox,*oy,*oz = ω; uses hemisphere fix (w≥0) and small-angle series when ‖v‖≈0.
static __device__ void so3_log_fwd_d(
    double qx,
    double qy,
    double qz,
    double qw,
    double* ox,
    double* oy,
    double* oz) {
    double dx = qx, dy = qy, dz = qz, dw = qw;
    double norm_sq = dx * dx + dy * dy + dz * dz + dw * dw;
    if (fabs(norm_sq - 1.0) > kManifoldQuatEpsD) {
        double inv_n = 1.0 / sqrt(norm_sq);
        dx *= inv_n;
        dy *= inv_n;
        dz *= inv_n;
        dw *= inv_n;
    }
    double w_abs = fabs(dw);
    double vx = dx, vy = dy, vz = dz;
    if (dw < 0.0) {
        vx = -vx;
        vy = -vy;
        vz = -vz;
    }
    double v_norm_sq = vx * vx + vy * vy + vz * vz;
    if (v_norm_sq < kManifoldQuatEpsD) {
        *ox = vx * (2.0 - v_norm_sq / 3.0 + 2.0 * v_norm_sq * v_norm_sq / 45.0);
        *oy = vy * (2.0 - v_norm_sq / 3.0 + 2.0 * v_norm_sq * v_norm_sq / 45.0);
        *oz = vz * (2.0 - v_norm_sq / 3.0 + 2.0 * v_norm_sq * v_norm_sq / 45.0);
        return;
    }
    double v_norm = sqrt(v_norm_sq);
    double theta = 2.0 * atan2(v_norm, w_abs);
    double s = theta / v_norm;
    *ox = vx * s;
    *oy = vy * s;
    *oz = vz * s;
}

// VJP for so3_log_fwd_d: upstream (gox,goy,goz)=∂L/∂ω → (*gzx..*gzw)=∂L/∂q on input quaternion (zx,zy,zz,zw).
static __device__ void so3_log_bwd_d(
    double zx,
    double zy,
    double zz,
    double zw,
    double gox,
    double goy,
    double goz,
    double* gzx,
    double* gzy,
    double* gzz,
    double* gzw) {
    *gzx = 0.0;
    *gzy = 0.0;
    *gzz = 0.0;
    *gzw = 0.0;

    double dx = zx, dy = zy, dz = zz, dw = zw;
    double norm_sq = dx * dx + dy * dy + dz * dz + dw * dw;
    bool did_norm = fabs(norm_sq - 1.0) > kManifoldQuatEpsD;
    double inv_r = 1.0;
    double r = sqrt(norm_sq);
    if (did_norm) {
        inv_r = 1.0 / r;
        dx *= inv_r;
        dy *= inv_r;
        dz *= inv_r;
        dw *= inv_r;
    }

    double w_abs = fabs(dw);
    double dw_sign = (dw >= 0.0) ? 1.0 : -1.0;
    double vx = dx, vy = dy, vz = dz;
    bool flip = dw < 0.0;
    if (flip) {
        vx = -vx;
        vy = -vy;
        vz = -vz;
    }
    double v_norm_sq = vx * vx + vy * vy + vz * vz;

    double gvx = 0.0, gvy = 0.0, gvz = 0.0;
    double gw_abs = 0.0;

    if (v_norm_sq < kManifoldQuatEpsD) {
        const double k = 2.0 - v_norm_sq / 3.0 + 2.0 * v_norm_sq * v_norm_sq / 45.0;
        const double dk_dr = -1.0 / 3.0 + 4.0 * v_norm_sq / 45.0;
        const double v_dot_g = vx * gox + vy * goy + vz * goz;
        gvx = k * gox + vx * dk_dr * 2.0 * v_dot_g;
        gvy = k * goy + vy * dk_dr * 2.0 * v_dot_g;
        gvz = k * goz + vz * dk_dr * 2.0 * v_dot_g;
    } else {
        const double v_norm = sqrt(v_norm_sq);
        const double n2 = v_norm_sq + w_abs * w_abs;
        const double theta = 2.0 * atan2(v_norm, w_abs);
        const double s = theta / v_norm;
        const double dtheta_dn = 2.0 * w_abs / n2;
        const double dtheta_dw = -2.0 * v_norm / n2;
        const double dsdn = dtheta_dn / v_norm - theta / (v_norm * v_norm);
        const double dsdw = dtheta_dw / v_norm;
        const double v_dot_g = vx * gox + vy * goy + vz * goz;
        gvx = s * gox + (vx / v_norm) * dsdn * v_dot_g;
        gvy = s * goy + (vy / v_norm) * dsdn * v_dot_g;
        gvz = s * goz + (vz / v_norm) * dsdn * v_dot_g;
        gw_abs = dsdw * v_dot_g;
    }

    double gdx = flip ? -gvx : gvx;
    double gdy = flip ? -gvy : gvy;
    double gdz = flip ? -gvz : gvz;
    double gdw = gw_abs * dw_sign;

    if (did_norm) {
        const double dot_d_g = dx * gdx + dy * gdy + dz * gdz + dw * gdw;
        const double sx = (gdx - dx * dot_d_g) * inv_r;
        const double sy = (gdy - dy * dot_d_g) * inv_r;
        const double sz = (gdz - dz * dot_d_g) * inv_r;
        const double sw = (gdw - dw * dot_d_g) * inv_r;
        *gzx = sx;
        *gzy = sy;
        *gzz = sz;
        *gzw = sw;
    } else {
        *gzx = gdx;
        *gzy = gdy;
        *gzz = gdz;
        *gzw = gdw;
    }
}

// so(3) exp: rotation vector ω → unit quaternion (xyzw); small-angle Taylor when ‖ω‖² is tiny.
// Outputs: *qx,*qy,*qz,*qw.
static __device__ void so3_exp_fwd_d(double wx, double wy, double wz, double* qx, double* qy, double* qz, double* qw) {
    const double theta_sq = wx * wx + wy * wy + wz * wz;
    if (theta_sq < kManifoldSmallAngleSq) {
        const double theta_sq_over_4 = theta_sq * 0.25;
        const double c0 = 0.5 - theta_sq_over_4 / 12.0 + theta_sq * theta_sq / 2880.0;
        const double wpart =
            1.0 - theta_sq_over_4 * (1.0 - theta_sq / 24.0 + theta_sq * theta_sq / 720.0);
        *qx = wx * c0;
        *qy = wy * c0;
        *qz = wz * c0;
        *qw = wpart;
        return;
    }
    const double theta = sqrt(theta_sq);
    const double half_theta = theta * 0.5;
    const double sin_ht = sin(half_theta);
    const double cos_ht = cos(half_theta);
    const double inv_theta = 1.0 / theta;
    const double scale = sin_ht * inv_theta;
    *qx = wx * scale;
    *qy = wy * scale;
    *qz = wz * scale;
    *qw = cos_ht;
}

// VJP for so3_exp_fwd_d: (gqx,gqy,gqz,gqw)=∂L/∂q → (*gwx,*gwy,*gwz)=∂L/∂ω at (wx,wy,wz).
static __device__ void so3_exp_bwd_d(
    double wx,
    double wy,
    double wz,
    double gqx,
    double gqy,
    double gqz,
    double gqw,
    double* gwx,
    double* gwy,
    double* gwz) {
    const double theta_sq = wx * wx + wy * wy + wz * wz;
    if (theta_sq < kManifoldSmallAngleSq) {
        const double r = theta_sq;
        const double p = 0.5 - r / 48.0 + r * r / 2880.0;
        const double dp_dr = -1.0 / 48.0 + r / 1440.0;
        const double theta_sq_over_4 = r * 0.25;
        const double inner = 1.0 - r / 24.0 + r * r / 720.0;
        const double dh_dr =
            0.25 * inner + theta_sq_over_4 * (-1.0 / 24.0 + r / 360.0);
        const double v_dot_g = wx * gqx + wy * gqy + wz * gqz;
        const double coeff = 2.0 * (dp_dr * v_dot_g - dh_dr * gqw);
        *gwx = p * gqx + coeff * wx;
        *gwy = p * gqy + coeff * wy;
        *gwz = p * gqz + coeff * wz;
        return;
    }
    const double theta = sqrt(theta_sq);
    const double half_theta = theta * 0.5;
    const double sin_ht = sin(half_theta);
    const double cos_ht = cos(half_theta);
    const double inv_theta = 1.0 / theta;
    const double a = sin_ht * inv_theta;
    const double a_prime = (0.5 * cos_ht * theta - sin_ht) / (theta * theta);
    const double v_dot_g = wx * gqx + wy * gqy + wz * gqz;
    const double fac_n = (a_prime / theta) * v_dot_g;
    const double fac_w = (-0.5 * sin_ht / theta) * gqw;
    *gwx = a * gqx + wx * (fac_n + fac_w);
    *gwy = a * gqy + wy * (fac_n + fac_w);
    *gwz = a * gqz + wz * (fac_n + fac_w);
}

// SO(3) geodesic interpolation in double: q_out = q1 ⊗ exp(t · log(q1⁻¹ ⊗ q2)), unit quats xyzw.
// t ∈ ℝ typically [0,1]. Outputs: *ox,*oy,*oz,*ow.
static __device__ void manifold_interp_fwd_d(
    double x1,
    double y1,
    double z1,
    double w1,
    double x2,
    double y2,
    double z2,
    double w2,
    double t,
    double* ox,
    double* oy,
    double* oz,
    double* ow) {
    const double c1x = -x1, c1y = -y1, c1z = -z1, c1w = w1;
    double zx, zy, zz, zw;
    quat_mul_d(c1x, c1y, c1z, c1w, x2, y2, z2, w2, &zx, &zy, &zz, &zw);
    double omx, omy, omz;
    so3_log_fwd_d(zx, zy, zz, zw, &omx, &omy, &omz);
    const double ux = t * omx, uy = t * omy, uz = t * omz;
    double ex, ey, ez, ew;
    so3_exp_fwd_d(ux, uy, uz, &ex, &ey, &ez, &ew);
    quat_mul_d(x1, y1, z1, w1, ex, ey, ez, ew, ox, oy, oz, ow);
}

// Batch row i: out_i = q1_i ⊗ q2_i (Hamilton, xyzw).
// Buffers are length 4*n contiguous per row: base offset o = i*4. n is batch size (unused except for API symmetry).
template <typename scalar_t>
__device__ void quat_multiply_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    quat_multiply_impl(x1, y1, z1, w1, x2, y2, z2, w2, out + o, out + o + 1, out + o + 2, out + o + 3);
}

// VJP for quat_multiply_fwd_device: grad_out = ∂L/∂out (xyzw per row) → grad_q1, grad_q2 same layout.
template <typename scalar_t>
__device__ void quat_multiply_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t gx = grad_out[o + 0], gy = grad_out[o + 1], gz = grad_out[o + 2], gw = grad_out[o + 3];

    // grad_q1 = J1^T * grad_out (q2 fixed)
    grad_q1[o + 0] = w2 * gx - z2 * gy + y2 * gz - x2 * gw;
    grad_q1[o + 1] = z2 * gx + w2 * gy - x2 * gz - y2 * gw;
    grad_q1[o + 2] = -y2 * gx + x2 * gy + w2 * gz - z2 * gw;
    grad_q1[o + 3] = x2 * gx + y2 * gy + z2 * gz + w2 * gw;

    // grad_q2 = J2^T * grad_out (q2 variable, q1 fixed); J2 is left-multiply by q1
    grad_q2[o + 0] = w1 * gx + z1 * gy - y1 * gz - x1 * gw;
    grad_q2[o + 1] = -z1 * gx + w1 * gy + x1 * gz - y1 * gw;
    grad_q2[o + 2] = y1 * gx - x1 * gy + w1 * gz - z1 * gw;
    grad_q2[o + 3] = x1 * gx + y1 * gy + z1 * gz + w1 * gw;
}

// Batch row i: out_i = R(q_i) v_i. quat: (N,4) xyzw flat; vec/out: (N,3) row-major (o = i*3 / i*4).
template <typename scalar_t>
__device__ void quat_rotate_vector_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ vec,
    scalar_t* __restrict__ out) {
    const int64_t oq = i * 4;
    const int64_t ov = i * 3;
    const scalar_t qx = quat[oq + 0], qy = quat[oq + 1], qz = quat[oq + 2], qw = quat[oq + 3];
    const scalar_t vx = vec[ov + 0], vy = vec[ov + 1], vz = vec[ov + 2];
    quat_rotate_vector_fwd_impl(qx, qy, qz, qw, vx, vy, vz, out + ov, out + ov + 1, out + ov + 2);
}

// VJP: grad_out = ∂L/∂(R(q)v) (3-vector per row) → grad_quat (4) and grad_vec (3) for row i.
template <typename scalar_t>
__device__ void quat_rotate_vector_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ vec,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat,
    scalar_t* __restrict__ grad_vec) {
    const int64_t oq = i * 4;
    const int64_t ov = i * 3;
    const scalar_t qx = quat[oq + 0], qy = quat[oq + 1], qz = quat[oq + 2], qw = quat[oq + 3];
    const scalar_t vx = vec[ov + 0], vy = vec[ov + 1], vz = vec[ov + 2];
    const scalar_t gx = grad_out[ov + 0], gy = grad_out[ov + 1], gz = grad_out[ov + 2];
    scalar_t gqx = 0, gqy = 0, gqz = 0, gqw = 0, gvx = 0, gvy = 0, gvz = 0;
    quat_rotate_vector_bwd_impl(
        qx,
        qy,
        qz,
        qw,
        vx,
        vy,
        vz,
        gx,
        gy,
        gz,
        &gqx,
        &gqy,
        &gqz,
        &gqw,
        &gvx,
        &gvy,
        &gvz);
    grad_quat[oq + 0] = gqx;
    grad_quat[oq + 1] = gqy;
    grad_quat[oq + 2] = gqz;
    grad_quat[oq + 3] = gqw;
    grad_vec[ov + 0] = gvx;
    grad_vec[ov + 1] = gvy;
    grad_vec[ov + 2] = gvz;
}

// Batch row i: quat (xyzw) → R 3×3 row-major 9 floats at out_flat[i*9 .. i*9+8] (normalize-safe).
template <typename scalar_t>
__device__ void quat_to_matrix_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out_flat) {
    const int64_t o = i * 4;
    quat_to_matrix_fwd_write(
        quat[o + 0], quat[o + 1], quat[o + 2], quat[o + 3], out_flat + i * 9);
}

// VJP: grad_flat row i = ∂L/∂R (9 row-major) → grad_quat row i (xyzw), chain through safe normalize.
template <typename scalar_t>
__device__ void quat_to_matrix_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ grad_flat,
    scalar_t* __restrict__ grad_quat) {
    const int64_t o = i * 4;
    const scalar_t qx = quat[o + 0], qy = quat[o + 1], qz = quat[o + 2], qw = quat[o + 3];
    const scalar_t norm_sq = qx * qx + qy * qy + qz * qz + qw * qw;
    const scalar_t eps = QuatNormEps<scalar_t>::value;
    if (fabs(static_cast<double>(norm_sq)) < static_cast<double>(eps)) {
        grad_quat[o + 0] = scalar_t(0);
        grad_quat[o + 1] = scalar_t(0);
        grad_quat[o + 2] = scalar_t(0);
        grad_quat[o + 3] = scalar_t(0);
        return;
    }
    const scalar_t inv_s = scalar_t(1) / sqrt(norm_sq);
    const scalar_t x = qx * inv_s;
    const scalar_t y = qy * inv_s;
    const scalar_t z = qz * inv_s;
    const scalar_t w = qw * inv_s;
    const scalar_t* __restrict__ g = grad_flat + i * 9;
    const scalar_t g00 = g[0], g01 = g[1], g02 = g[2];
    const scalar_t g10 = g[3], g11 = g[4], g12 = g[5];
    const scalar_t g20 = g[6], g21 = g[7], g22 = g[8];
    // grad w.r.t. normalized quaternion (convertNormalizedQuatToMatrix)
    const scalar_t gx = g01 * scalar_t(2) * y + g02 * scalar_t(2) * z + g10 * scalar_t(2) * y +
        g11 * scalar_t(-4) * x + g12 * scalar_t(-2) * w + g20 * scalar_t(2) * z + g21 * scalar_t(2) * w +
        g22 * scalar_t(-4) * x;
    // M11 = 1 - 2(x^2 + z^2) has ∂M11/∂y = 0 (no g11 term in gy).
    const scalar_t gy = g00 * scalar_t(-4) * y + g01 * scalar_t(2) * x + g02 * scalar_t(2) * w +
        g10 * scalar_t(2) * x + g12 * scalar_t(2) * z + g20 * scalar_t(-2) * w + g21 * scalar_t(2) * z +
        g22 * scalar_t(-4) * y;
    const scalar_t gz = g00 * scalar_t(-4) * z + g01 * scalar_t(-2) * w + g02 * scalar_t(2) * x +
        g10 * scalar_t(2) * w + g11 * scalar_t(-4) * z + g12 * scalar_t(2) * y + g20 * scalar_t(2) * x +
        g21 * scalar_t(2) * y;
    const scalar_t gw = g01 * scalar_t(-2) * z + g02 * scalar_t(2) * y + g10 * scalar_t(2) * z +
        g12 * scalar_t(-2) * x + g20 * scalar_t(-2) * y + g21 * scalar_t(2) * x;
    const scalar_t q_dot_gq = qx * gx + qy * gy + qz * gz + qw * gw;
    const scalar_t inv_s3 = inv_s * inv_s * inv_s;
    grad_quat[o + 0] = gx * inv_s - qx * q_dot_gq * inv_s3;
    grad_quat[o + 1] = gy * inv_s - qy * q_dot_gq * inv_s3;
    grad_quat[o + 2] = gz * inv_s - qz * q_dot_gq * inv_s3;
    grad_quat[o + 3] = gw * inv_s - qw * q_dot_gq * inv_s3;
}

// Batch row i: out = quat/‖quat‖, or (0,0,0,1) if ‖quat‖² below dtype epsilon (see QuatNormEps).
template <typename scalar_t>
__device__ void quat_normalize_safe_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    quat_normalize_safe_fwd_write(
        quat[o + 0],
        quat[o + 1],
        quat[o + 2],
        quat[o + 3],
        out + o,
        out + o + 1,
        out + o + 2,
        out + o + 3);
}

// VJP through quat_normalize_safe_fwd_device: grad_out (xyzw) → grad_quat w.r.t. raw input quaternion.
template <typename scalar_t>
__device__ void quat_normalize_safe_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat) {
    const int64_t o = i * 4;
    quat_normalize_safe_bwd_write(
        quat[o + 0],
        quat[o + 1],
        quat[o + 2],
        quat[o + 3],
        grad_out[o + 0],
        grad_out[o + 1],
        grad_out[o + 2],
        grad_out[o + 3],
        grad_quat + o,
        grad_quat + o + 1,
        grad_quat + o + 2,
        grad_quat + o + 3);
}

// Batch row i: out = q* = (-x,-y,-z,w) for q=(x,y,z,w) xyzw.
template <typename scalar_t>
__device__ void quat_conjugate_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ quat,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    out[o + 0] = -quat[o + 0];
    out[o + 1] = -quat[o + 1];
    out[o + 2] = -quat[o + 2];
    out[o + 3] = quat[o + 3];
}

// VJP: grad_out = ∂L/∂q* → grad_quat = ∂L/∂q (imaginary components negated, real unchanged).
template <typename scalar_t>
__device__ void quat_conjugate_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_quat) {
    const int64_t o = i * 4;
    const scalar_t gx = grad_out[o + 0], gy = grad_out[o + 1], gz = grad_out[o + 2], gw = grad_out[o + 3];
    grad_quat[o + 0] = -gx;
    grad_quat[o + 1] = -gy;
    grad_quat[o + 2] = -gz;
    grad_quat[o + 3] = gw;
}

// Batch row i: axis (3-vector at i*3) and angle[i] (scalar) → quaternion xyzw at i*4.
// Convention: q = (axis*sin(θ/2), cos(θ/2)); caller should ensure axis scale matches intended θ semantics.
template <typename scalar_t>
__device__ void quat_from_axis_angle_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ axis,
    const scalar_t* __restrict__ angle,
    scalar_t* __restrict__ quat) {
    const int64_t oa = i * 3;
    const scalar_t ang = angle[i];
    const scalar_t half = ang * scalar_t(0.5);
    const scalar_t s = sin(half);
    const scalar_t c = cos(half);
    quat[i * 4 + 0] = axis[oa + 0] * s;
    quat[i * 4 + 1] = axis[oa + 1] * s;
    quat[i * 4 + 2] = axis[oa + 2] * s;
    quat[i * 4 + 3] = c;
}

// VJP: grad_quat (xyzw) → grad_axis (3) and grad_angle (scalar per row).
template <typename scalar_t>
__device__ void quat_from_axis_angle_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ axis,
    const scalar_t* __restrict__ angle,
    const scalar_t* __restrict__ grad_quat,
    scalar_t* __restrict__ grad_axis,
    scalar_t* __restrict__ grad_angle) {
    const int64_t oa = i * 3;
    const int64_t oq = i * 4;
    const scalar_t ang = angle[i];
    const scalar_t half = ang * scalar_t(0.5);
    const scalar_t s = sin(half);
    const scalar_t c = cos(half);

    const scalar_t gx = grad_quat[oq + 0], gy = grad_quat[oq + 1], gz = grad_quat[oq + 2], gw = grad_quat[oq + 3];
    const scalar_t ax = axis[oa + 0], ay = axis[oa + 1], az = axis[oa + 2];

    grad_axis[oa + 0] = gx * s;
    grad_axis[oa + 1] = gy * s;
    grad_axis[oa + 2] = gz * s;

    const scalar_t dot_ag = gx * ax + gy * ay + gz * az;
    grad_angle[i] = scalar_t(0.5) * (c * dot_ag - s * gw);
}

// Batch row i: normalized lerp — flip q2 hemisphere if dot(q1,q2)<0, then normalize (1-t)q1+t·q2 (not geodesic).
// Scalar t shared across batch in this device API (same t every row).
template <typename scalar_t>
__device__ void quat_lerp_fwd_device(int64_t i, int64_t n,
    scalar_t t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    scalar_t sx = x2, sy = y2, sz = z2, sw = w2;
    if (dot < scalar_t(0)) {
        sx = -x2;
        sy = -y2;
        sz = -z2;
        sw = -w2;
    }
    const scalar_t om = scalar_t(1) - t;
    const scalar_t rx = om * x1 + t * sx;
    const scalar_t ry = om * y1 + t * sy;
    const scalar_t rz = om * z1 + t * sz;
    const scalar_t rw = om * w1 + t * sw;
    const scalar_t norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
    const scalar_t inv_n = scalar_t(1) / sqrt(norm_sq);
    out[o + 0] = rx * inv_n;
    out[o + 1] = ry * inv_n;
    out[o + 2] = rz * inv_n;
    out[o + 3] = rw * inv_n;
}

// VJP for quat_lerp_fwd_device: needs forward `result` quaternion per row; grad_out xyzw → grad_q1, grad_q2.
template <typename scalar_t>
__device__ void quat_lerp_bwd_device(int64_t i, int64_t n,
    scalar_t t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ result,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const scalar_t s = dot < scalar_t(0) ? scalar_t(-1) : scalar_t(1);
    const scalar_t sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const scalar_t om = scalar_t(1) - t;
    const scalar_t rx = om * x1 + t * sx;
    const scalar_t ry = om * y1 + t * sy;
    const scalar_t rz = om * z1 + t * sz;
    const scalar_t rw = om * w1 + t * sw;
    const scalar_t norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
    const scalar_t r_norm = sqrt(norm_sq);

    const scalar_t yx = result[o + 0], yy = result[o + 1], yz = result[o + 2], yw = result[o + 3];
    const scalar_t gx = grad_out[o + 0], gy = grad_out[o + 1], gz = grad_out[o + 2], gw = grad_out[o + 3];
    const scalar_t ydotg = yx * gx + yy * gy + yz * gz + yw * gw;
    const scalar_t scale = scalar_t(1) / r_norm;
    const scalar_t grx = (gx - yx * ydotg) * scale;
    const scalar_t gry = (gy - yy * ydotg) * scale;
    const scalar_t grz = (gz - yz * ydotg) * scale;
    const scalar_t grw = (gw - yw * ydotg) * scale;

    grad_q1[o + 0] = om * grx;
    grad_q1[o + 1] = om * gry;
    grad_q1[o + 2] = om * grz;
    grad_q1[o + 3] = om * grw;

    grad_q2[o + 0] = s * t * grx;
    grad_q2[o + 1] = s * t * gry;
    grad_q2[o + 2] = s * t * grz;
    grad_q2[o + 3] = s * t * grw;
}

// Batch row i: SLERP(q1_i, q2_i, t[i]); t from t[i] (length-n scalar buffer). Hemisphere flip; if dot≈1, nlerp fallback.
// q1,q2,out: (N,4) xyzw flat.
template <typename scalar_t>
__device__ void quat_slerp_batched_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ t,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    const scalar_t ti = t[i];
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    scalar_t s = dot < scalar_t(0) ? scalar_t(-1) : scalar_t(1);
    const scalar_t sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const scalar_t c_raw = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const scalar_t c = quat_slerp_clamp_dot(c_raw);

    if (c > quat_slerp_small_angle_dot_threshold<scalar_t>()) {
        const scalar_t om = scalar_t(1) - ti;
        const scalar_t rx = om * x1 + ti * sx;
        const scalar_t ry = om * y1 + ti * sy;
        const scalar_t rz = om * z1 + ti * sz;
        const scalar_t rw = om * w1 + ti * sw;
        const scalar_t norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
        const scalar_t inv_n = scalar_t(1) / sqrt(norm_sq);
        out[o + 0] = rx * inv_n;
        out[o + 1] = ry * inv_n;
        out[o + 2] = rz * inv_n;
        out[o + 3] = rw * inv_n;
        return;
    }

    const scalar_t theta = acos(c);
    const scalar_t sin_theta = sin(theta);
    const scalar_t w1s = sin((scalar_t(1) - ti) * theta) / sin_theta;
    const scalar_t w2s = sin(ti * theta) / sin_theta;
    out[o + 0] = w1s * x1 + w2s * sx;
    out[o + 1] = w1s * y1 + w2s * sy;
    out[o + 2] = w1s * z1 + w2s * sz;
    out[o + 3] = w1s * w1 + w2s * sw;
}

// VJP for quat_slerp_batched_fwd_device: grad_out (xyzw), forward `result`; outputs grad_q1, grad_q2, grad_t[i].
template <typename scalar_t>
__device__ void quat_slerp_batched_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ result,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2,
    scalar_t* __restrict__ grad_t) {
    const int64_t o = i * 4;
    const scalar_t ti = t[i];
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t gx = grad_out[o + 0], gy = grad_out[o + 1], gz = grad_out[o + 2], gw = grad_out[o + 3];

    const scalar_t dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const scalar_t s = dot < scalar_t(0) ? scalar_t(-1) : scalar_t(1);
    const scalar_t sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const scalar_t c_raw = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const scalar_t c = quat_slerp_clamp_dot(c_raw);
    const bool interior_c = (c_raw > scalar_t(-1)) && (c_raw < scalar_t(1));
    const scalar_t c_mask = interior_c ? scalar_t(1) : scalar_t(0);

    if (c > quat_slerp_small_angle_dot_threshold<scalar_t>()) {
        const scalar_t om = scalar_t(1) - ti;
        const scalar_t rx = om * x1 + ti * sx;
        const scalar_t ry = om * y1 + ti * sy;
        const scalar_t rz = om * z1 + ti * sz;
        const scalar_t rw = om * w1 + ti * sw;
        const scalar_t norm_sq = rx * rx + ry * ry + rz * rz + rw * rw;
        const scalar_t r_norm = sqrt(norm_sq);

        const scalar_t yx = result[o + 0], yy = result[o + 1], yz = result[o + 2], yw = result[o + 3];
        const scalar_t ydotg = yx * gx + yy * gy + yz * gz + yw * gw;
        const scalar_t scale = scalar_t(1) / r_norm;
        const scalar_t grx = (gx - yx * ydotg) * scale;
        const scalar_t gry = (gy - yy * ydotg) * scale;
        const scalar_t grz = (gz - yz * ydotg) * scale;
        const scalar_t grw = (gw - yw * ydotg) * scale;

        grad_q1[o + 0] = om * grx;
        grad_q1[o + 1] = om * gry;
        grad_q1[o + 2] = om * grz;
        grad_q1[o + 3] = om * grw;

        grad_q2[o + 0] = s * ti * grx;
        grad_q2[o + 1] = s * ti * gry;
        grad_q2[o + 2] = s * ti * grz;
        grad_q2[o + 3] = s * ti * grw;

        const scalar_t dq2m_dq1x = sx - x1, dq2m_dq1y = sy - y1, dq2m_dq1z = sz - z1, dq2m_dq1w = sw - w1;
        grad_t[i] = grx * dq2m_dq1x + gry * dq2m_dq1y + grz * dq2m_dq1z + grw * dq2m_dq1w;
        return;
    }

    const scalar_t theta = acos(c);
    const scalar_t sin_theta = sin(theta);
    const scalar_t w1s = sin((scalar_t(1) - ti) * theta) / sin_theta;
    const scalar_t w2s = sin(ti * theta) / sin_theta;

    const scalar_t G1 = gx * x1 + gy * y1 + gz * z1 + gw * w1;
    const scalar_t G2 = gx * sx + gy * sy + gz * sz + gw * sw;

    const scalar_t den = sin_theta * sin_theta;
    const scalar_t dw1_dtheta =
        ((scalar_t(1) - ti) * cos((scalar_t(1) - ti) * theta) * sin_theta -
         sin((scalar_t(1) - ti) * theta) * c) /
        den;
    const scalar_t dw2_dtheta =
        (ti * cos(ti * theta) * sin_theta - sin(ti * theta) * c) / den;
    const scalar_t dw1_dc = sin_theta > scalar_t(1e-20) ? (-dw1_dtheta / sin_theta) * c_mask : scalar_t(0);
    const scalar_t dw2_dc = sin_theta > scalar_t(1e-20) ? (-dw2_dtheta / sin_theta) * c_mask : scalar_t(0);

    const scalar_t K = G1 * dw1_dc + G2 * dw2_dc;

    grad_q1[o + 0] = w1s * gx + K * sx;
    grad_q1[o + 1] = w1s * gy + K * sy;
    grad_q1[o + 2] = w1s * gz + K * sz;
    grad_q1[o + 3] = w1s * gw + K * sw;

    const scalar_t gq2ex = w2s * gx + K * x1;
    const scalar_t gq2ey = w2s * gy + K * y1;
    const scalar_t gq2ez = w2s * gz + K * z1;
    const scalar_t gq2ew = w2s * gw + K * w1;

    grad_q2[o + 0] = s * gq2ex;
    grad_q2[o + 1] = s * gq2ey;
    grad_q2[o + 2] = s * gq2ez;
    grad_q2[o + 3] = s * gq2ew;

    const scalar_t dw1_dt = -theta * cos((scalar_t(1) - ti) * theta) / sin_theta;
    const scalar_t dw2_dt = theta * cos(ti * theta) / sin_theta;
    grad_t[i] = G1 * dw1_dt + G2 * dw2_dt;
}

// Batch row i: angular distance 2·acos(|⟨q̂1,q̂2⟩|) radians after normalizing both quaternions (xyzw); dist_out[i] scalar.
template <typename scalar_t>
__device__ void quat_angular_distance_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ dist_out) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t n1_sq = x1 * x1 + y1 * y1 + z1 * z1 + w1 * w1;
    const scalar_t n2_sq = x2 * x2 + y2 * y2 + z2 * z2 + w2 * w2;
    const scalar_t inv_s1 = scalar_t(1) / sqrt(n1_sq);
    const scalar_t inv_s2 = scalar_t(1) / sqrt(n2_sq);
    const scalar_t u1x = x1 * inv_s1, u1y = y1 * inv_s1, u1z = z1 * inv_s1, u1w = w1 * inv_s1;
    const scalar_t u2x = x2 * inv_s2, u2y = y2 * inv_s2, u2z = z2 * inv_s2, u2w = w2 * inv_s2;
    const scalar_t d = u1x * u2x + u1y * u2y + u1z * u2z + u1w * u2w;
    scalar_t a = d >= scalar_t(0) ? d : -d;
    if (a < scalar_t(0)) {
        a = scalar_t(0);
    }
    if (a > scalar_t(1)) {
        a = scalar_t(1);
    }
    dist_out[i] = scalar_t(2) * acos(a);
}

// VJP: grad_dist[i] = ∂L/∂(angular distance) → grad_q1, grad_q2 (xyzw per row); handles |dot| and normalize.
template <typename scalar_t>
__device__ void quat_angular_distance_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_dist,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2) {
    const int64_t o = i * 4;
    const scalar_t x1 = q1[o + 0], y1 = q1[o + 1], z1 = q1[o + 2], w1 = q1[o + 3];
    const scalar_t x2 = q2[o + 0], y2 = q2[o + 1], z2 = q2[o + 2], w2 = q2[o + 3];
    const scalar_t n1_sq = x1 * x1 + y1 * y1 + z1 * z1 + w1 * w1;
    const scalar_t n2_sq = x2 * x2 + y2 * y2 + z2 * z2 + w2 * w2;
    const scalar_t inv_s1 = scalar_t(1) / sqrt(n1_sq);
    const scalar_t inv_s2 = scalar_t(1) / sqrt(n2_sq);
    const scalar_t u1x = x1 * inv_s1, u1y = y1 * inv_s1, u1z = z1 * inv_s1, u1w = w1 * inv_s1;
    const scalar_t u2x = x2 * inv_s2, u2y = y2 * inv_s2, u2z = z2 * inv_s2, u2w = w2 * inv_s2;
    const scalar_t d = u1x * u2x + u1y * u2y + u1z * u2z + u1w * u2w;
    scalar_t a = d >= scalar_t(0) ? d : -d;
    if (a < scalar_t(0)) {
        a = scalar_t(0);
    }
    if (a > scalar_t(1)) {
        a = scalar_t(1);
    }
    const scalar_t c = a;
    const scalar_t g_up = grad_dist[i];

    const scalar_t dc_da = (a < scalar_t(1)) ? scalar_t(1) : scalar_t(0);
    const scalar_t one_m_c2 = scalar_t(1) - c * c;
    const scalar_t eps = quat_angular_dist_acos_eps<scalar_t>();
    scalar_t grad_c = scalar_t(0);
    if (one_m_c2 > eps && dc_da != scalar_t(0)) {
        grad_c = -scalar_t(2) * g_up / sqrt(one_m_c2);
    }
    const scalar_t grad_a = grad_c * dc_da;
    scalar_t da_dd = scalar_t(0);
    if (d > scalar_t(0)) {
        da_dd = scalar_t(1);
    } else if (d < scalar_t(0)) {
        da_dd = scalar_t(-1);
    }
    const scalar_t grad_d = grad_a * da_dd;

    const scalar_t gnx1 = grad_d * u2x;
    const scalar_t gny1 = grad_d * u2y;
    const scalar_t gnz1 = grad_d * u2z;
    const scalar_t gnw1 = grad_d * u2w;
    const scalar_t gnx2 = grad_d * u1x;
    const scalar_t gny2 = grad_d * u1y;
    const scalar_t gnz2 = grad_d * u1z;
    const scalar_t gnw2 = grad_d * u1w;

    const scalar_t gdot1 = gnx1 * u1x + gny1 * u1y + gnz1 * u1z + gnw1 * u1w;
    const scalar_t gdot2 = gnx2 * u2x + gny2 * u2y + gnz2 * u2z + gnw2 * u2w;

    if (!(n1_sq > eps)) {
        grad_q1[o + 0] = scalar_t(0);
        grad_q1[o + 1] = scalar_t(0);
        grad_q1[o + 2] = scalar_t(0);
        grad_q1[o + 3] = scalar_t(0);
    } else {
        const scalar_t is1 = inv_s1;
        grad_q1[o + 0] = (gnx1 - u1x * gdot1) * is1;
        grad_q1[o + 1] = (gny1 - u1y * gdot1) * is1;
        grad_q1[o + 2] = (gnz1 - u1z * gdot1) * is1;
        grad_q1[o + 3] = (gnw1 - u1w * gdot1) * is1;
    }
    if (!(n2_sq > eps)) {
        grad_q2[o + 0] = scalar_t(0);
        grad_q2[o + 1] = scalar_t(0);
        grad_q2[o + 2] = scalar_t(0);
        grad_q2[o + 3] = scalar_t(0);
    } else {
        const scalar_t is2 = inv_s2;
        grad_q2[o + 0] = (gnx2 - u2x * gdot2) * is2;
        grad_q2[o + 1] = (gny2 - u2y * gdot2) * is2;
        grad_q2[o + 2] = (gnz2 - u2z * gdot2) * is2;
        grad_q2[o + 3] = (gnw2 - u2w * gdot2) * is2;
    }
}

// Batch row i: SO(3) geodesic q_out = q1 ⊗ exp(t[i]·log(q1⁻¹q2)); internal double precision, output scalar_t xyzw.
template <typename scalar_t>
__device__ void quat_manifold_interp_fwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    scalar_t* __restrict__ out) {
    const int64_t o = i * 4;
    const double td = static_cast<double>(t[i]);
    const double x1 = static_cast<double>(q1[o + 0]);
    const double y1 = static_cast<double>(q1[o + 1]);
    const double z1 = static_cast<double>(q1[o + 2]);
    const double w1 = static_cast<double>(q1[o + 3]);
    const double x2 = static_cast<double>(q2[o + 0]);
    const double y2 = static_cast<double>(q2[o + 1]);
    const double z2 = static_cast<double>(q2[o + 2]);
    const double w2 = static_cast<double>(q2[o + 3]);
    double rx, ry, rz, rw;
    manifold_interp_fwd_d(x1, y1, z1, w1, x2, y2, z2, w2, td, &rx, &ry, &rz, &rw);
    out[o + 0] = static_cast<scalar_t>(rx);
    out[o + 1] = static_cast<scalar_t>(ry);
    out[o + 2] = static_cast<scalar_t>(rz);
    out[o + 3] = static_cast<scalar_t>(rw);
}

// VJP for quat_manifold_interp_fwd_device: grad_out xyzw → grad_q1, grad_q2, grad_t[i].
template <typename scalar_t>
__device__ void quat_manifold_interp_bwd_device(int64_t i, int64_t n,
    const scalar_t* __restrict__ t,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ grad_out,
    scalar_t* __restrict__ grad_q1,
    scalar_t* __restrict__ grad_q2,
    scalar_t* __restrict__ grad_t) {
    const int64_t o = i * 4;
    const double x1 = static_cast<double>(q1[o + 0]);
    const double y1 = static_cast<double>(q1[o + 1]);
    const double z1 = static_cast<double>(q1[o + 2]);
    const double w1 = static_cast<double>(q1[o + 3]);
    const double x2 = static_cast<double>(q2[o + 0]);
    const double y2 = static_cast<double>(q2[o + 1]);
    const double z2 = static_cast<double>(q2[o + 2]);
    const double w2 = static_cast<double>(q2[o + 3]);
    const double td = static_cast<double>(t[i]);

    const double c1x = -x1, c1y = -y1, c1z = -z1, c1w = w1;
    double zx, zy, zz, zw;
    quat_mul_d(c1x, c1y, c1z, c1w, x2, y2, z2, w2, &zx, &zy, &zz, &zw);
    double omx, omy, omz;
    so3_log_fwd_d(zx, zy, zz, zw, &omx, &omy, &omz);
    const double ux = td * omx, uy = td * omy, uz = td * omz;
    double ex, ey, ez, ew;
    so3_exp_fwd_d(ux, uy, uz, &ex, &ey, &ez, &ew);

    const double gx = static_cast<double>(grad_out[o + 0]);
    const double gy = static_cast<double>(grad_out[o + 1]);
    const double gz = static_cast<double>(grad_out[o + 2]);
    const double gw = static_cast<double>(grad_out[o + 3]);

    double gq1x_m, gq1y_m, gq1z_m, gq1w_m, gex, gey, gez, gew;
    quat_mul_bwd_d(x1, y1, z1, w1, ex, ey, ez, ew, gx, gy, gz, gw, &gq1x_m, &gq1y_m, &gq1z_m, &gq1w_m, &gex, &gey, &gez, &gew);

    double gux, guy, guz;
    so3_exp_bwd_d(ux, uy, uz, gex, gey, gez, gew, &gux, &guy, &guz);
    grad_t[i] = static_cast<scalar_t>(gux * omx + guy * omy + guz * omz);
    const double gomx = td * gux, gomy = td * guy, gomz = td * guz;

    double gzx, gzy, gzz, gzw;
    so3_log_bwd_d(zx, zy, zz, zw, gomx, gomy, gomz, &gzx, &gzy, &gzz, &gzw);

    double gcx, gcy, gcz, gcw, gq2x, gq2y, gq2z, gq2w;
    quat_mul_bwd_d(c1x, c1y, c1z, c1w, x2, y2, z2, w2, gzx, gzy, gzz, gzw, &gcx, &gcy, &gcz, &gcw, &gq2x, &gq2y, &gq2z, &gq2w);

    const double gq1x_c = -gcx;
    const double gq1y_c = -gcy;
    const double gq1z_c = -gcz;
    const double gq1w_c = gcw;

    grad_q1[o + 0] = static_cast<scalar_t>(gq1x_m + gq1x_c);
    grad_q1[o + 1] = static_cast<scalar_t>(gq1y_m + gq1y_c);
    grad_q1[o + 2] = static_cast<scalar_t>(gq1z_m + gq1z_c);
    grad_q1[o + 3] = static_cast<scalar_t>(gq1w_m + gq1w_c);
    grad_q2[o + 0] = static_cast<scalar_t>(gq2x);
    grad_q2[o + 1] = static_cast<scalar_t>(gq2y);
    grad_q2[o + 2] = static_cast<scalar_t>(gq2z);
    grad_q2[o + 3] = static_cast<scalar_t>(gq2w);
}

