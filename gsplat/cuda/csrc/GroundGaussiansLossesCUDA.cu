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

#include "Config.h"

#if GSPLAT_BUILD_LOSSES

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAException.h>
#    include <c10/cuda/CUDAStream.h>

#    include <cmath>
#    include <cstdint>
#    include <limits>
#    include <type_traits>

#    include "KernelUtils.cuh"

namespace gsplat
{
// Per-bin statistics layout: sum/sum-of-squares of y, roll, pitch plus a count.
enum
{
    kAccumY      = 0,
    kAccumY2     = 1,
    kAccumRoll   = 2,
    kAccumRoll2  = 3,
    kAccumPitch  = 4,
    kAccumPitch2 = 5,
    kCount       = 6,
    kFieldsCount = 7
};

// Warp aggregation requires complete CUDA warps (32 lanes) and HIP wavefronts
// (32 or 64 lanes) in every block.
constexpr uint32_t kAccumulationBlockSize = 256;

// Sparse warps retain per-hit atomics because seven reductions cost more than
// the contention they remove.
constexpr uint32_t kWarpAggregationMinHits = 4;

// ---------------------------------------------------------------------------
// Scalar / small-vector device helpers
// ---------------------------------------------------------------------------

// Unbiased sample standard deviation from running sums (count > 1 assumed).
template<typename scalar_t>
__device__ __forceinline__ scalar_t std_dev(scalar_t sum, scalar_t sum2, scalar_t count)
{
    const scalar_t mean              = sum / count;
    const scalar_t variance          = (sum2 / count) - (mean * mean);
    const scalar_t unbiased_variance = variance * count / (count - scalar_t(1));
    return sqrt(unbiased_variance > scalar_t(0) ? unbiased_variance : scalar_t(0));
}

// d(std_dev)/d(value) for the sample `value` that contributed to (sum, sum2).
template<typename scalar_t>
__device__ __forceinline__ scalar_t
    std_dev_grad(scalar_t value, scalar_t sum, scalar_t count, scalar_t std, scalar_t v_std)
{
    const scalar_t mean  = sum / count;
    const scalar_t denom = (count - scalar_t(1)) * std;
    if(denom < std::numeric_limits<scalar_t>::epsilon())
    {
        return scalar_t(0);
    }
    return v_std * (scalar_t(1) / denom) * (value - mean);
}

template<typename scalar_t>
__device__ __forceinline__ void quat_normalize(const scalar_t q[4], scalar_t out[4])
{
    const scalar_t inv = rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    out[0]             = q[0] * inv;
    out[1]             = q[1] * inv;
    out[2]             = q[2] * inv;
    out[3]             = q[3] * inv;
}

template<typename scalar_t>
__device__ __forceinline__ void quat_normalize_bwd(
    const scalar_t q[4], const scalar_t qn[4], const scalar_t v_qn[4], scalar_t v_q[4]
)
{
    const scalar_t inv = rsqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
#    pragma unroll
    for(int i = 0; i < 4; i++)
    {
        v_q[i] = scalar_t(0);
#    pragma unroll
        for(int j = 0; j < 4; j++)
        {
            scalar_t neg = qn[i] * qn[j];
            if(i == j)
            {
                neg -= scalar_t(1);
            }
            v_q[i] -= v_qn[j] * neg * inv;
        }
    }
}

// Quaternion (x, y, z, w) to SO3 rotation matrix.
template<typename scalar_t>
__device__ __forceinline__ void quat_to_so3(const scalar_t q[4], scalar_t R[3][3])
{
    scalar_t n[4];
    quat_normalize(q, n);
    R[0][0] = scalar_t(1) - scalar_t(2) * (n[1] * n[1] + n[2] * n[2]);
    R[0][1] = scalar_t(2) * (n[0] * n[1] - n[2] * n[3]);
    R[0][2] = scalar_t(2) * (n[0] * n[2] + n[1] * n[3]);
    R[1][0] = scalar_t(2) * (n[0] * n[1] + n[2] * n[3]);
    R[1][1] = scalar_t(1) - scalar_t(2) * (n[0] * n[0] + n[2] * n[2]);
    R[1][2] = scalar_t(2) * (n[1] * n[2] - n[0] * n[3]);
    R[2][0] = scalar_t(2) * (n[0] * n[2] - n[1] * n[3]);
    R[2][1] = scalar_t(2) * (n[1] * n[2] + n[0] * n[3]);
    R[2][2] = scalar_t(1) - scalar_t(2) * (n[0] * n[0] + n[1] * n[1]);
}

template<typename scalar_t>
__device__ __forceinline__ void quat_to_so3_bwd(const scalar_t q[4], const scalar_t v_R[3][3], scalar_t v_q[4])
{
    scalar_t n[4];
    quat_normalize(q, n);

    scalar_t v_n[4];
    v_n[0] = scalar_t(2) * n[1] * v_R[0][1]
           + scalar_t(2) * n[2] * v_R[0][2]
           + scalar_t(2) * n[1] * v_R[1][0]
           - scalar_t(4) * n[0] * v_R[1][1]
           - scalar_t(2) * n[3] * v_R[1][2]
           + scalar_t(2) * n[2] * v_R[2][0]
           + scalar_t(2) * n[3] * v_R[2][1]
           - scalar_t(4) * n[0] * v_R[2][2];
    v_n[1] = -scalar_t(4) * n[1] * v_R[0][0]
           + scalar_t(2) * n[0] * v_R[0][1]
           + scalar_t(2) * n[3] * v_R[0][2]
           + scalar_t(2) * n[0] * v_R[1][0]
           + scalar_t(2) * n[2] * v_R[1][2]
           - scalar_t(2) * n[3] * v_R[2][0]
           + scalar_t(2) * n[2] * v_R[2][1]
           - scalar_t(4) * n[1] * v_R[2][2];
    v_n[2] = -scalar_t(4) * n[2] * v_R[0][0]
           - scalar_t(2) * n[3] * v_R[0][1]
           + scalar_t(2) * n[0] * v_R[0][2]
           + scalar_t(2) * n[3] * v_R[1][0]
           - scalar_t(4) * n[2] * v_R[1][1]
           + scalar_t(2) * n[1] * v_R[1][2]
           + scalar_t(2) * n[0] * v_R[2][0]
           + scalar_t(2) * n[1] * v_R[2][1];
    v_n[3] = -scalar_t(2) * n[2] * v_R[0][1]
           + scalar_t(2) * n[1] * v_R[0][2]
           + scalar_t(2) * n[2] * v_R[1][0]
           - scalar_t(2) * n[0] * v_R[1][2]
           - scalar_t(2) * n[1] * v_R[2][0]
           + scalar_t(2) * n[0] * v_R[2][1];

    quat_normalize_bwd(q, n, v_n, v_q);
}

// Build the camera-from-world SE3 (4x4) from a [t, q] 7-vector and its inverse.
template<typename scalar_t>
__device__ __forceinline__ void tquat_to_se3(const scalar_t tq[7], scalar_t T[4][4])
{
    const scalar_t q[4] = {tq[3], tq[4], tq[5], tq[6]};
    scalar_t R[3][3];
    quat_to_so3(q, R);
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            T[i][j] = R[i][j];
        }
        T[i][3] = tq[i];
    }
    T[3][0] = T[3][1] = T[3][2] = scalar_t(0);
    T[3][3]                     = scalar_t(1);
}

template<typename scalar_t>
__device__ __forceinline__ void se3_inverse(const scalar_t T[4][4], scalar_t T_inv[4][4])
{
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            T_inv[i][j] = T[j][i];
        }
    }
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
        T_inv[i][3] = -(T_inv[i][0] * T[0][3] + T_inv[i][1] * T[1][3] + T_inv[i][2] * T[2][3]);
    }
    T_inv[3][0] = T_inv[3][1] = T_inv[3][2] = scalar_t(0);
    T_inv[3][3]                             = scalar_t(1);
}

template<typename scalar_t>
__device__ __forceinline__ void se3_apply(const scalar_t T[4][4], const scalar_t p[3], scalar_t out[3])
{
    out[0] = T[0][0] * p[0] + T[0][1] * p[1] + T[0][2] * p[2] + T[0][3];
    out[1] = T[1][0] * p[0] + T[1][1] * p[1] + T[1][2] * p[2] + T[1][3];
    out[2] = T[2][0] * p[0] + T[2][1] * p[1] + T[2][2] * p[2] + T[2][3];
}

template<typename scalar_t>
__device__ __forceinline__ void se3_apply_bwd(const scalar_t T[4][4], const scalar_t v_out[3], scalar_t v_p[3])
{
    v_p[0] = T[0][0] * v_out[0] + T[1][0] * v_out[1] + T[2][0] * v_out[2];
    v_p[1] = T[0][1] * v_out[0] + T[1][1] * v_out[1] + T[2][1] * v_out[2];
    v_p[2] = T[0][2] * v_out[0] + T[1][2] * v_out[1] + T[2][2] * v_out[2];
}

// C = A^T B (transpose_a) or A B, restricted to the leading 3x3 block.
template<bool transpose_a, typename scalar_t>
__device__ __forceinline__ void mat3_mul(const scalar_t A[4][4], const scalar_t B[3][3], scalar_t C[3][3])
{
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = (transpose_a ? A[0][i] : A[i][0]) * B[0][j]
                    + (transpose_a ? A[1][i] : A[i][1]) * B[1][j]
                    + (transpose_a ? A[2][i] : A[i][2]) * B[2][j];
        }
    }
}

template<bool transpose_a, typename scalar_t>
__device__ __forceinline__ void mat3_mul_3x3(const scalar_t A[3][3], const scalar_t B[4][4], scalar_t C[3][3])
{
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            C[i][j] = (transpose_a ? A[0][i] : A[i][0]) * B[0][j]
                    + (transpose_a ? A[1][i] : A[i][1]) * B[1][j]
                    + (transpose_a ? A[2][i] : A[i][2]) * B[2][j];
        }
    }
}

// Gradient of (A op B) w.r.t. the right operand B, A being the constant 4x4
// camera matrix (used transposed). dC -> dB with B a 3x3 block.
template<bool transpose_a, typename scalar_t>
__device__ __forceinline__ void mat3_mul_bwd_to_b(const scalar_t A[4][4], const scalar_t v_C[3][3], scalar_t v_B[3][3])
{
#    pragma unroll
    for(int k = 0; k < 3; k++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            v_B[k][j] = scalar_t(0);
#    pragma unroll
            for(int i = 0; i < 3; i++)
            {
                v_B[k][j] += (transpose_a ? A[k][i] : A[i][k]) * v_C[i][j];
            }
        }
    }
}

// Gradient of (A op B) w.r.t. the left operand A, B being the constant 4x4
// camera matrix. dC -> dA with A a 3x3 block.
template<bool transpose_a, typename scalar_t>
__device__ __forceinline__ void mat3_mul_bwd_to_a_rhs44(
    const scalar_t B[4][4], const scalar_t v_C[3][3], scalar_t v_A[3][3]
)
{
#    pragma unroll
    for(int k = 0; k < 3; k++)
    {
#    pragma unroll
        for(int i = 0; i < 3; i++)
        {
            scalar_t acc = scalar_t(0);
#    pragma unroll
            for(int j = 0; j < 3; j++)
            {
                acc += B[k][j] * v_C[i][j];
            }
            if(transpose_a)
            {
                v_A[k][i] = acc;
            }
            else
            {
                v_A[i][k] = acc;
            }
        }
    }
}

template<typename scalar_t>
__device__ __forceinline__ void so3_to_quat(const scalar_t R[3][3], scalar_t qn[4])
{
    const scalar_t t0    = R[0][0];
    const scalar_t t1    = R[1][1];
    const scalar_t t2    = R[2][2];
    const scalar_t trace = t0 + t1 + t2;

    scalar_t decision[4] = {t0, t1, t2, trace};
    int choice           = 0;
    scalar_t max_val     = decision[0];
#    pragma unroll
    for(int i = 1; i < 4; i++)
    {
        if(decision[i] > max_val)
        {
            max_val = decision[i];
            choice  = i;
        }
    }

    scalar_t q[4];
    if(choice != 3)
    {
        const int i = choice;
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        q[i]        = scalar_t(1) + R[i][i] - R[j][j] - R[k][k];
        q[j]        = R[j][i] + R[i][j];
        q[k]        = R[k][i] + R[i][k];
        q[3]        = R[k][j] - R[j][k];
    }
    else
    {
        q[0] = R[2][1] - R[1][2];
        q[1] = R[0][2] - R[2][0];
        q[2] = R[1][0] - R[0][1];
        q[3] = scalar_t(1) + trace;
    }
    quat_normalize(q, qn);
}

template<typename scalar_t>
__device__ __forceinline__ void so3_to_quat_bwd(
    const scalar_t R[3][3], const scalar_t qn[4], const scalar_t v_qn[4], scalar_t v_R[3][3]
)
{
#    pragma unroll
    for(int i = 0; i < 3; i++)
    {
#    pragma unroll
        for(int j = 0; j < 3; j++)
        {
            v_R[i][j] = scalar_t(0);
        }
    }

    const scalar_t t0    = R[0][0];
    const scalar_t t1    = R[1][1];
    const scalar_t t2    = R[2][2];
    const scalar_t trace = t0 + t1 + t2;
    scalar_t decision[4] = {t0, t1, t2, trace};
    int choice           = 0;
    scalar_t max_val     = decision[0];
#    pragma unroll
    for(int i = 1; i < 4; i++)
    {
        if(decision[i] > max_val)
        {
            max_val = decision[i];
            choice  = i;
        }
    }

    scalar_t q[4];
    if(choice != 3)
    {
        const int i = choice;
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        q[i]        = scalar_t(1) + R[i][i] - R[j][j] - R[k][k];
        q[j]        = R[j][i] + R[i][j];
        q[k]        = R[k][i] + R[i][k];
        q[3]        = R[k][j] - R[j][k];
    }
    else
    {
        q[0] = R[2][1] - R[1][2];
        q[1] = R[0][2] - R[2][0];
        q[2] = R[1][0] - R[0][1];
        q[3] = scalar_t(1) + trace;
    }

    scalar_t v_q[4];
    quat_normalize_bwd(q, qn, v_qn, v_q);

    if(choice != 3)
    {
        const int i = choice;
        const int j = (i + 1) % 3;
        const int k = (j + 1) % 3;
        v_R[i][i]   = v_q[i];
        v_R[j][j]   = -v_q[i];
        v_R[k][k]   = -v_q[i];
        v_R[i][j]   = v_q[j];
        v_R[j][i]   = v_q[j];
        v_R[i][k]   = v_q[k];
        v_R[k][i]   = v_q[k];
        v_R[j][k]   = -v_q[3];
        v_R[k][j]   = v_q[3];
    }
    else
    {
        v_R[0][0] = v_q[3];
        v_R[1][1] = v_q[3];
        v_R[2][2] = v_q[3];
        v_R[0][1] = -v_q[2];
        v_R[1][0] = v_q[2];
        v_R[0][2] = v_q[1];
        v_R[2][0] = -v_q[1];
        v_R[1][2] = -v_q[0];
        v_R[2][1] = v_q[0];
    }
}

// Rotate a world quaternion into the camera frame: R_cam = T^T R_world T,
// then convert back to a quaternion.
template<typename scalar_t>
__device__ __forceinline__ void transform_quat(
    const scalar_t T_cam_world[4][4], const scalar_t q_world[4], scalar_t q_cam[4]
)
{
    scalar_t R_world[3][3], tmp[3][3], R_cam[3][3];
    quat_to_so3(q_world, R_world);
    mat3_mul<true>(T_cam_world, R_world, tmp);    // tmp = T^T R_world
    mat3_mul_3x3<false>(tmp, T_cam_world, R_cam); // R_cam = tmp T
    so3_to_quat(R_cam, q_cam);
}

template<typename scalar_t>
__device__ __forceinline__ void transform_quat_bwd(
    const scalar_t T_cam_world[4][4], const scalar_t q_world[4], const scalar_t v_q_cam[4], scalar_t v_q_world[4]
)
{
    scalar_t R_world[3][3], tmp[3][3], R_cam[3][3];
    quat_to_so3(q_world, R_world);
    mat3_mul<true>(T_cam_world, R_world, tmp);
    mat3_mul_3x3<false>(tmp, T_cam_world, R_cam);

    scalar_t q_cam[4];
    so3_to_quat(R_cam, q_cam);

    scalar_t v_R_cam[3][3];
    so3_to_quat_bwd(R_cam, q_cam, v_q_cam, v_R_cam);

    // R_cam = tmp @ T (tmp on the left, T constant): grad to tmp.
    scalar_t v_tmp[3][3];
    mat3_mul_bwd_to_a_rhs44<false>(T_cam_world, v_R_cam, v_tmp);
    // tmp = T^T @ R_world (T^T constant on the left): grad to R_world.
    scalar_t v_R_world[3][3];
    mat3_mul_bwd_to_b<true>(T_cam_world, v_tmp, v_R_world);

    quat_to_so3_bwd(q_world, v_R_world, v_q_world);
}

// Roll/pitch (rotations around z/x) from a camera-frame quaternion (x, y, z, w).
template<typename scalar_t>
__device__ __forceinline__ void quat_to_roll_pitch(const scalar_t q[4], scalar_t &roll, scalar_t &pitch)
{
    // roll uses (x, y, z, w); pitch uses (z, y, x, w) — see fallback reference.
    const scalar_t sin_r = scalar_t(2) * (q[3] * q[2] + q[0] * q[1]);
    const scalar_t cos_r = scalar_t(1) - scalar_t(2) * (q[1] * q[1] + q[2] * q[2]);
    roll                 = atan2(sin_r, cos_r);

    const scalar_t sin_p = scalar_t(2) * (q[3] * q[0] + q[2] * q[1]);
    const scalar_t cos_p = scalar_t(1) - scalar_t(2) * (q[1] * q[1] + q[0] * q[0]);
    pitch                = atan2(sin_p, cos_p);
}

template<typename scalar_t>
__device__ __forceinline__ void atan2_bwd(scalar_t s, scalar_t c, scalar_t v_angle, scalar_t &v_s, scalar_t &v_c)
{
    scalar_t denom = s * s + c * c;
    if(denom < std::numeric_limits<scalar_t>::epsilon())
    {
        denom = std::numeric_limits<scalar_t>::epsilon();
    }
    const scalar_t g = v_angle / denom;
    v_s              = g * c;
    v_c              = -g * s;
}

template<typename scalar_t>
__device__ __forceinline__ void quat_to_roll_pitch_bwd(
    const scalar_t q[4], scalar_t v_roll, scalar_t v_pitch, scalar_t v_q[4]
)
{
#    pragma unroll
    for(int i = 0; i < 4; i++)
    {
        v_q[i] = scalar_t(0);
    }

    // roll: sin = 2(w z + x y), cos = 1 - 2(y^2 + z^2)
    {
        const scalar_t s = scalar_t(2) * (q[3] * q[2] + q[0] * q[1]);
        const scalar_t c = scalar_t(1) - scalar_t(2) * (q[1] * q[1] + q[2] * q[2]);
        scalar_t v_s, v_c;
        atan2_bwd(s, c, v_roll, v_s, v_c);
        const scalar_t v_s2  = scalar_t(2) * v_s;
        const scalar_t v_c4  = scalar_t(4) * v_c;
        v_q[0]              += v_s2 * q[1];               // x
        v_q[1]              += v_s2 * q[0] - v_c4 * q[1]; // y
        v_q[2]              += v_s2 * q[3] - v_c4 * q[2]; // z
        v_q[3]              += v_s2 * q[2];               // w
    }
    // pitch: sin = 2(w x + z y), cos = 1 - 2(y^2 + x^2)
    {
        const scalar_t s = scalar_t(2) * (q[3] * q[0] + q[2] * q[1]);
        const scalar_t c = scalar_t(1) - scalar_t(2) * (q[1] * q[1] + q[0] * q[0]);
        scalar_t v_s, v_c;
        atan2_bwd(s, c, v_pitch, v_s, v_c);
        const scalar_t v_s2  = scalar_t(2) * v_s;
        const scalar_t v_c4  = scalar_t(4) * v_c;
        v_q[2]              += v_s2 * q[1];               // z
        v_q[1]              += v_s2 * q[2] - v_c4 * q[1]; // y
        v_q[0]              += v_s2 * q[3] - v_c4 * q[0]; // x
        v_q[3]              += v_s2 * q[0];               // w
    }
}

// Map a single point's world position + rotation into camera-frame
// y, z, roll, pitch. Returns the cached camera matrices for the backward.
template<typename scalar_t>
struct PointTransform
{
    scalar_t T_cam_world[4][4];
    scalar_t T_world_cam[4][4];
    scalar_t q_world[4];
    scalar_t pos_world[3];
    scalar_t pos_cam[3];
    scalar_t q_cam[4];
    scalar_t y, z, roll, pitch;

    __device__ __forceinline__ void load(
        const scalar_t *__restrict__ positions,
        const scalar_t *__restrict__ rotations,
        const scalar_t *__restrict__ cam_tquat,
        int64_t idx
    )
    {
        const int64_t i3 = idx * 3;
        const int64_t i4 = idx * 4;
        pos_world[0]     = positions[i3 + 0];
        pos_world[1]     = positions[i3 + 1];
        pos_world[2]     = positions[i3 + 2];
        q_world[0]       = rotations[i4 + 0];
        q_world[1]       = rotations[i4 + 1];
        q_world[2]       = rotations[i4 + 2];
        q_world[3]       = rotations[i4 + 3];

        scalar_t tq[7];
#    pragma unroll
        for(int k = 0; k < 7; k++)
        {
            tq[k] = cam_tquat[k];
        }
        tquat_to_se3(tq, T_cam_world);
        se3_inverse(T_cam_world, T_world_cam);

        se3_apply(T_world_cam, pos_world, pos_cam);
        transform_quat(T_cam_world, q_world, q_cam);
        quat_to_roll_pitch(q_cam, roll, pitch);

        y = pos_cam[1];
        z = pos_cam[2];
    }

    // Scatter gradients on (y, roll, pitch) back to world position + rotation.
    __device__ __forceinline__ void backward(
        scalar_t v_y, scalar_t v_roll, scalar_t v_pitch, scalar_t v_pos_world[3], scalar_t v_rot_world[4]
    )
    {
        scalar_t v_q_cam[4];
        quat_to_roll_pitch_bwd(q_cam, v_roll, v_pitch, v_q_cam);
        transform_quat_bwd(T_cam_world, q_world, v_q_cam, v_rot_world);

        const scalar_t v_pos_cam[3] = {scalar_t(0), v_y, scalar_t(0)};
        se3_apply_bwd(T_world_cam, v_pos_cam, v_pos_world);
    }
};

template<typename scalar_t>
__device__ __forceinline__ void accumulate_bin_per_hit(
    scalar_t *__restrict__ shared_stats_at, scalar_t y, scalar_t roll, scalar_t pitch
)
{
    atomicAdd(&shared_stats_at[kAccumY], y);
    atomicAdd(&shared_stats_at[kAccumY2], y * y);
    atomicAdd(&shared_stats_at[kAccumRoll], roll);
    atomicAdd(&shared_stats_at[kAccumRoll2], roll * roll);
    atomicAdd(&shared_stats_at[kAccumPitch], pitch);
    atomicAdd(&shared_stats_at[kAccumPitch2], pitch * pitch);
    atomicAdd(&shared_stats_at[kCount], scalar_t(1));
}

// Every lane in a complete warp/wave must call this convergently. Threads past
// N remain in the collective and pass hit=false.
template<uint32_t CTA_SIZE, uint32_t MIN_HITS, typename scalar_t>
__device__ __forceinline__ void accumulate_bin_warp_aggregated(
    scalar_t *__restrict__ shared_stats_at, bool hit, scalar_t y, scalar_t roll, scalar_t pitch
)
{
    static_assert(CTA_SIZE % 64 == 0);
    static_assert(MIN_HITS > 0 && MIN_HITS <= 64);

    const auto active_mask = __activemask();
    const auto hit_mask    = __ballot_sync(active_mask, hit);
    if(hit_mask == 0)
    {
        return;
    }

    const uint32_t hit_count = warp_mask_popcount(hit_mask);
    if(hit_count < MIN_HITS)
    {
        if(hit)
        {
            accumulate_bin_per_hit(shared_stats_at, y, roll, pitch);
        }
        return;
    }

    const bool leader = threadIdx.x % warpSize == 0;
    scalar_t sum      = warp_sum(hit ? y : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumY], sum);
    }
    sum = warp_sum(hit ? y * y : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumY2], sum);
    }
    sum = warp_sum(hit ? roll : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumRoll], sum);
    }
    sum = warp_sum(hit ? roll * roll : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumRoll2], sum);
    }
    sum = warp_sum(hit ? pitch : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumPitch], sum);
    }
    sum = warp_sum(hit ? pitch * pitch : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kAccumPitch2], sum);
    }
    sum = warp_sum(hit ? scalar_t(1) : scalar_t(0), active_mask);
    if(leader)
    {
        atomicAdd(&shared_stats_at[kCount], sum);
    }
}

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

// Pass 1: accumulate per-bin statistics. Block-local shared-memory reduction
// keeps global atomics down to one flush per (bin, field).
template<uint32_t CTA_SIZE, typename scalar_t>
__global__ void ground_gaussians_accumulate_kernel(
    const int64_t N,
    const int n_bins,
    const scalar_t *__restrict__ positions,
    const scalar_t *__restrict__ rotations,
    const scalar_t *__restrict__ cam_tquat,
    const scalar_t *__restrict__ random_values,
    const scalar_t min_bias,
    const scalar_t range_bias,
    const scalar_t grid_len,
    scalar_t *__restrict__ stats // [n_bins, kFieldsCount]
)
{
    extern __shared__ char shared_raw[];
    scalar_t *shared_stats = reinterpret_cast<scalar_t *>(shared_raw);

    const int local        = threadIdx.x;
    const int total_fields = n_bins * kFieldsCount;
    for(int off = local; off < total_fields; off += blockDim.x)
    {
        shared_stats[off] = scalar_t(0);
    }
    __syncthreads();

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Seven 64-bit shuffle reductions cost more than the shared atomics they
    // replace, so warp aggregation is intentionally limited to float32.
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        const bool valid = idx < N;
        PointTransform<scalar_t> pt;
        if(valid)
        {
            pt.load(positions, rotations, cam_tquat, idx);
        }

        for(int b = 0; b < n_bins; b++)
        {
            const scalar_t bin_min    = random_values[b] * range_bias + min_bias;
            const scalar_t bin_max    = bin_min + grid_len;
            scalar_t *shared_stats_at = shared_stats + b * kFieldsCount;

            bool hit   = false;
            scalar_t y = scalar_t(0), roll = scalar_t(0), pitch = scalar_t(0);
            if(valid)
            {
                hit   = bin_min <= pt.z && pt.z < bin_max;
                y     = pt.y;
                roll  = pt.roll;
                pitch = pt.pitch;
            }
            accumulate_bin_warp_aggregated<CTA_SIZE, kWarpAggregationMinHits>(shared_stats_at, hit, y, roll, pitch);
        }
    }
    else if(idx < N)
    {
        PointTransform<scalar_t> pt;
        pt.load(positions, rotations, cam_tquat, idx);

        for(int b = 0; b < n_bins; b++)
        {
            const scalar_t bin_min = random_values[b] * range_bias + min_bias;
            const scalar_t bin_max = bin_min + grid_len;
            if(bin_min <= pt.z && pt.z < bin_max)
            {
                scalar_t *shared_stats_at = shared_stats + b * kFieldsCount;
                accumulate_bin_per_hit(shared_stats_at, pt.y, pt.roll, pt.pitch);
            }
        }
    }
    __syncthreads();

    for(int off = local; off < total_fields; off += blockDim.x)
    {
        const scalar_t v = shared_stats[off];
        if(v != scalar_t(0))
        {
            atomicAdd(&stats[off], v);
        }
    }
}

// Pass 2: turn per-bin statistics into the scalar loss, normalized by n_bins.
template<typename scalar_t>
__global__ void ground_gaussians_reduce_kernel(
    const int n_bins, const scalar_t *__restrict__ stats, const scalar_t rotation_lambda, scalar_t *__restrict__ loss
)
{
    extern __shared__ char shared_raw[];
    scalar_t *partial = reinterpret_cast<scalar_t *>(shared_raw);

    const int b       = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t bin_loss = scalar_t(0);
    if(b < n_bins)
    {
        const scalar_t *s    = stats + b * kFieldsCount;
        const scalar_t count = s[kCount];
        if(count > scalar_t(1))
        {
            const scalar_t std_y     = std_dev(s[kAccumY], s[kAccumY2], count);
            const scalar_t std_roll  = std_dev(s[kAccumRoll], s[kAccumRoll2], count);
            const scalar_t std_pitch = std_dev(s[kAccumPitch], s[kAccumPitch2], count);
            bin_loss                 = std_y + rotation_lambda * (std_roll + std_pitch);
        }
    }

    partial[threadIdx.x] = bin_loss;
    __syncthreads();
    for(int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            partial[threadIdx.x] += partial[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0)
    {
        atomicAdd(loss, partial[0] / static_cast<scalar_t>(n_bins));
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
// ---------------------------------------------------------------------------
template<typename scalar_t>
__global__ void ground_gaussians_bwd_kernel(
    const int64_t N,
    const int n_bins,
    const scalar_t *__restrict__ positions,
    const scalar_t *__restrict__ rotations,
    const scalar_t *__restrict__ cam_tquat,
    const scalar_t *__restrict__ random_values,
    const scalar_t *__restrict__ stats,
    const scalar_t *__restrict__ v_loss,
    const scalar_t min_bias,
    const scalar_t range_bias,
    const scalar_t grid_len,
    const scalar_t rotation_lambda,
    scalar_t *__restrict__ v_positions,
    scalar_t *__restrict__ v_rotations
)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= N)
    {
        return;
    }

    const scalar_t v_out = v_loss[0];

    PointTransform<scalar_t> pt;
    pt.load(positions, rotations, cam_tquat, idx);

    scalar_t v_pos_accum[3] = {scalar_t(0), scalar_t(0), scalar_t(0)};
    scalar_t v_rot_accum[4] = {scalar_t(0), scalar_t(0), scalar_t(0), scalar_t(0)};

    for(int b = 0; b < n_bins; b++)
    {
        const scalar_t bin_min = random_values[b] * range_bias + min_bias;
        const scalar_t bin_max = bin_min + grid_len;
        if(!(bin_min <= pt.z && pt.z < bin_max))
        {
            continue;
        }

        const scalar_t *s    = stats + b * kFieldsCount;
        const scalar_t count = s[kCount];
        if(count <= scalar_t(1))
        {
            continue;
        }

        const scalar_t std_y     = std_dev(s[kAccumY], s[kAccumY2], count);
        const scalar_t std_roll  = std_dev(s[kAccumRoll], s[kAccumRoll2], count);
        const scalar_t std_pitch = std_dev(s[kAccumPitch], s[kAccumPitch2], count);

        const scalar_t v_y     = std_dev_grad(pt.y, s[kAccumY], count, std_y, v_out);
        const scalar_t v_roll  = std_dev_grad(pt.roll, s[kAccumRoll], count, std_roll, v_out);
        const scalar_t v_pitch = std_dev_grad(pt.pitch, s[kAccumPitch], count, std_pitch, v_out);

        scalar_t v_pos_world[3], v_rot_world[4];
        const scalar_t inv_bins = scalar_t(1) / static_cast<scalar_t>(n_bins);
        pt.backward(
            v_y * inv_bins,
            rotation_lambda * v_roll * inv_bins,
            rotation_lambda * v_pitch * inv_bins,
            v_pos_world,
            v_rot_world
        );

#    pragma unroll
        for(int i = 0; i < 3; i++)
        {
            v_pos_accum[i] += v_pos_world[i];
        }
#    pragma unroll
        for(int i = 0; i < 4; i++)
        {
            v_rot_accum[i] += v_rot_world[i];
        }
    }

    const int64_t i3    = idx * 3;
    const int64_t i4    = idx * 4;
    v_positions[i3 + 0] = v_pos_accum[0];
    v_positions[i3 + 1] = v_pos_accum[1];
    v_positions[i3 + 2] = v_pos_accum[2];
    v_rotations[i4 + 0] = v_rot_accum[0];
    v_rotations[i4 + 1] = v_rot_accum[1];
    v_rotations[i4 + 2] = v_rot_accum[2];
    v_rotations[i4 + 3] = v_rot_accum[3];
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

void launch_ground_gaussians_fwd_kernel(
    const at::Tensor &positions,
    const at::Tensor &rotations,
    const at::Tensor &cam_tquat,
    const at::Tensor &random_values,
    float min_bias,
    float range_bias,
    float grid_len,
    float rotation_lambda,
    at::Tensor &stats,
    at::Tensor &loss
)
{
    const int64_t N  = positions.size(0);
    const int n_bins = static_cast<int>(random_values.size(0));
    if(n_bins == 0)
    {
        return;
    }

    constexpr uint32_t threads = kAccumulationBlockSize;
    const auto stream          = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        positions.scalar_type(),
        "ground_gaussians_fwd",
        [&]()
        {
            const size_t accum_shmem = static_cast<size_t>(n_bins) * kFieldsCount * sizeof(scalar_t);
            // The accumulate kernel needs n_bins * kFieldsCount * sizeof(scalar_t) bytes of
            // dynamic shared memory per block; cap it against the device limit so an
            // oversized random_values fails with a clear message rather than an opaque
            // launch error.
            int cur_device           = 0;
            cudaGetDevice(&cur_device);
            int max_shmem_per_block = 0;
            cudaDeviceGetAttribute(&max_shmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, cur_device);
            const size_t max_shmem = static_cast<size_t>(max_shmem_per_block);
            TORCH_CHECK(
                accum_shmem <= max_shmem,
                "ground_gaussians_fwd: random_values.shape[0] (",
                n_bins,
                ") needs ",
                accum_shmem,
                " bytes of per-block shared memory, exceeding the device limit of ",
                max_shmem,
                " (max ",
                max_shmem / (kFieldsCount * sizeof(scalar_t)),
                " bins for this dtype)"
            );
            if(N > 0)
            {
                const dim3 grid(static_cast<uint32_t>((N + threads - 1) / threads));
                ground_gaussians_accumulate_kernel<threads, scalar_t><<<grid, threads, accum_shmem, stream>>>(
                    N,
                    n_bins,
                    positions.data_ptr<scalar_t>(),
                    rotations.data_ptr<scalar_t>(),
                    cam_tquat.data_ptr<scalar_t>(),
                    random_values.data_ptr<scalar_t>(),
                    static_cast<scalar_t>(min_bias),
                    static_cast<scalar_t>(range_bias),
                    static_cast<scalar_t>(grid_len),
                    stats.data_ptr<scalar_t>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }

            const dim3 reduce_grid(static_cast<uint32_t>((n_bins + threads - 1) / threads));
            const size_t reduce_shmem = static_cast<size_t>(threads) * sizeof(scalar_t);
            ground_gaussians_reduce_kernel<scalar_t><<<reduce_grid, threads, reduce_shmem, stream>>>(
                n_bins, stats.data_ptr<scalar_t>(), static_cast<scalar_t>(rotation_lambda), loss.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}

void launch_ground_gaussians_bwd_kernel(
    const at::Tensor &positions,
    const at::Tensor &rotations,
    const at::Tensor &cam_tquat,
    const at::Tensor &random_values,
    const at::Tensor &stats,
    const at::Tensor &v_loss,
    float min_bias,
    float range_bias,
    float grid_len,
    float rotation_lambda,
    at::Tensor &v_positions,
    at::Tensor &v_rotations
)
{
    const int64_t N  = positions.size(0);
    const int n_bins = static_cast<int>(random_values.size(0));
    if(N == 0 || n_bins == 0)
    {
        return;
    }

    const int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(static_cast<uint32_t>((N + threads - 1) / threads));

    AT_DISPATCH_FLOATING_TYPES(
        positions.scalar_type(),
        "ground_gaussians_bwd",
        [&]()
        {
            ground_gaussians_bwd_kernel<scalar_t><<<grid, threads, 0, stream>>>(
                N,
                n_bins,
                positions.data_ptr<scalar_t>(),
                rotations.data_ptr<scalar_t>(),
                cam_tquat.data_ptr<scalar_t>(),
                random_values.data_ptr<scalar_t>(),
                stats.data_ptr<scalar_t>(),
                v_loss.data_ptr<scalar_t>(),
                static_cast<scalar_t>(min_bias),
                static_cast<scalar_t>(range_bias),
                static_cast<scalar_t>(grid_len),
                static_cast<scalar_t>(rotation_lambda),
                v_positions.data_ptr<scalar_t>(),
                v_rotations.data_ptr<scalar_t>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    );
}
} // namespace gsplat

#endif
