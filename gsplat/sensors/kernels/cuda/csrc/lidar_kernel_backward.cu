/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Backward CUDA kernels for the RowOffsetStructuredSpinningLidar projection.
//
// Closed-form per-thread VJPs for the ray<->angle conversions, a scatter
// (atomicAdd) into the three angle tables for elements->angles, the SLERP-VJP
// pose-grad reduction for generate, and the frozen-alpha IFT VJP for inverse.

#include "lidar_kernel.cuh"

#include <c10/cuda/CUDAException.h>

using gsplat_sensors::kLidarThreads;
using gsplat_sensors::slerp_pair_fwd;

namespace
{
dim3 grid_for_count(int64_t count)
{
    return dim3(static_cast<unsigned int>((count + kLidarThreads - 1) / kLidarThreads));
}

// det_guard_eps — dtype-dependent ill-conditioning threshold for the inverse
// projection.  Returning ZERO gradient below this beats a 1/fmax blow-up that
// passes isfinite() and poisons Adam.  float32 -> 1e-8, float64 -> 1e-12.
template<typename T>
constexpr __device__ __forceinline__ T det_guard_eps();

template<>
constexpr __device__ __forceinline__ float det_guard_eps<float>()
{
    return 1.0e-8f;
}

template<>
constexpr __device__ __forceinline__ double det_guard_eps<double>()
{
    return 1.0e-12;
}

// VJP for slerp_pair_fwd: forward outputs (rx,ry,rz,rw) and upstream
// (gx..gw) -> gq1*, gq2* (xyzw).  ti is non-differentiable here (derived from
// the column index), so the slerp time grad is computed and discarded by the
// caller; this helper does not emit it.
template<typename T>
__device__ __forceinline__ void slerp_pair_bwd(
    T x1,
    T y1,
    T z1,
    T w1,
    T x2,
    T y2,
    T z2,
    T w2,
    T ti,
    T rx,
    T ry,
    T rz,
    T rw,
    T gx,
    T gy,
    T gz,
    T gw,
    T *gq1x,
    T *gq1y,
    T *gq1z,
    T *gq1w,
    T *gq2x,
    T *gq2y,
    T *gq2z,
    T *gq2w
)
{
    const T dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const T s   = dot < T(0) ? T(-1) : T(1);
    const T sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const T c_raw         = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const T c             = quat_slerp_clamp_dot<T>(c_raw);
    const bool interior_c = (c_raw > T(-1)) && (c_raw < T(1));
    const T c_mask        = interior_c ? T(1) : T(0);

    if(c > quat_slerp_small_angle_dot_threshold<T>())
    {
        const T om     = T(1) - ti;
        const T rrx    = om * x1 + ti * sx;
        const T rry    = om * y1 + ti * sy;
        const T rrz    = om * z1 + ti * sz;
        const T rrw    = om * w1 + ti * sw;
        const T r_norm = sqrt(rrx * rrx + rry * rry + rrz * rrz + rrw * rrw);
        const T ydotg  = rx * gx + ry * gy + rz * gz + rw * gw;
        const T scale  = T(1) / r_norm;
        const T grx    = (gx - rx * ydotg) * scale;
        const T gry    = (gy - ry * ydotg) * scale;
        const T grz    = (gz - rz * ydotg) * scale;
        const T grw    = (gw - rw * ydotg) * scale;
        *gq1x          = om * grx;
        *gq1y          = om * gry;
        *gq1z          = om * grz;
        *gq1w          = om * grw;
        *gq2x          = s * ti * grx;
        *gq2y          = s * ti * gry;
        *gq2z          = s * ti * grz;
        *gq2w          = s * ti * grw;
        return;
    }

    const T theta      = acos(c);
    const T sin_theta  = sin(theta);
    const T w1s        = sin((T(1) - ti) * theta) / sin_theta;
    const T w2s        = sin(ti * theta) / sin_theta;
    const T G1         = gx * x1 + gy * y1 + gz * z1 + gw * w1;
    const T G2         = gx * sx + gy * sy + gz * sz + gw * sw;
    const T den        = sin_theta * sin_theta;
    const T dw1_dtheta = ((T(1) - ti) * cos((T(1) - ti) * theta) * sin_theta - sin((T(1) - ti) * theta) * c) / den;
    const T dw2_dtheta = (ti * cos(ti * theta) * sin_theta - sin(ti * theta) * c) / den;
    const T dw1_dc     = sin_theta > T(1e-20) ? (-dw1_dtheta / sin_theta) * c_mask : T(0);
    const T dw2_dc     = sin_theta > T(1e-20) ? (-dw2_dtheta / sin_theta) * c_mask : T(0);
    const T K          = G1 * dw1_dc + G2 * dw2_dc;
    *gq1x              = w1s * gx + K * sx;
    *gq1y              = w1s * gy + K * sy;
    *gq1z              = w1s * gz + K * sz;
    *gq1w              = w1s * gw + K * sw;
    *gq2x              = s * (w2s * gx + K * x1);
    *gq2y              = s * (w2s * gy + K * y1);
    *gq2z              = s * (w2s * gz + K * z1);
    *gq2w              = s * (w2s * gw + K * w1);
}

// Fused block reduction for the 14 control-pose grad components
// (t0xyz, t1xyz, q0xyzw, q1xyzw).  One warp-shuffle pass per slot, a single
// __syncthreads, then the writer thread (warp 0) holds the 14 column totals.
// Replaces 14 sequential lidar_block_sum calls (28 __syncthreads).  Per-slot
// summation order matches lidar_block_sum's warp-xor tree; the cross-warp sum
// is the same set of per-warp partials.  Results land in out[0..13] on
// threadIdx.x == 0; out is undefined on other threads.  Uniform path only.
template<typename T, int BlockThreads>
__device__ __forceinline__ void lidar_block_reduce_pose(const T (&values)[14], T (&out)[14])
{
    static_assert(BlockThreads % 32 == 0, "BlockThreads must be a multiple of warp size");
    constexpr int kNumWarps = BlockThreads / 32;
    constexpr int kSlots    = 14;
    __shared__ T warp_sums[kNumWarps][kSlots];

    const unsigned int mask = 0xFFFFFFFFu;
    const int lane          = threadIdx.x & 31;
    const int warp          = threadIdx.x >> 5;
#pragma unroll
    for(int i = 0; i < kSlots; ++i)
    {
        T v = values[i];
        for(int offset = 16; offset > 0; offset >>= 1)
        {
            v += __shfl_xor_sync(mask, v, offset);
        }
        if(lane == 0)
        {
            warp_sums[warp][i] = v;
        }
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
#pragma unroll
        for(int i = 0; i < kSlots; ++i)
        {
            T total = T(0);
#pragma unroll
            for(int w = 0; w < kNumWarps; ++w)
            {
                total += warp_sums[w][i];
            }
            out[i] = total;
        }
    }
}

// VJP of ray -> angles.  elev = atan2(z, r), r = hypot(x, y), az = atan2(y, x).
//   d(elev)/d{x,y,z} = {-z*x/(rho2*r), -z*y/(rho2*r), r/rho2}, rho2 = x^2+y^2+z^2
//   d(az)/d{x,y,z}   = {-y/r2, x/r2, 0},                        r2  = x^2+y^2
template<typename T>
__global__ void sensor_rays_to_sensor_angles_backward_kernel(
    int64_t count,
    const T *__restrict__ sensor_rays,
    const T *__restrict__ grad_sensor_angles,
    T *__restrict__ grad_sensor_rays
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    T x      = sensor_rays[idx * 3 + 0];
    T y      = sensor_rays[idx * 3 + 1];
    T z      = sensor_rays[idx * 3 + 2];
    T g_elev = grad_sensor_angles[idx * 2 + 0];
    T g_az   = grad_sensor_angles[idx * 2 + 1];

    T r2   = x * x + y * y;
    T rho2 = r2 + z * z;
    T r    = sqrt(r2);

    // Guard the singular axis (ray along +/-Z): r==0 leaves azimuth undefined
    // and the elevation derivative finite (r/rho2 -> 0); return zero on the
    // ill-defined azimuth component rather than propagating a NaN.
    T grad_x = T(0);
    T grad_y = T(0);
    T grad_z = T(0);
    if(rho2 > T(0))
    {
        grad_z += g_elev * (r / rho2);
        if(r > T(0))
        {
            T inv   = T(1) / (rho2 * r);
            grad_x += g_elev * (-z * x * inv);
            grad_y += g_elev * (-z * y * inv);
            grad_x += g_az * (-y / r2);
            grad_y += g_az * (x / r2);
        }
    }
    grad_sensor_rays[idx * 3 + 0] = grad_x;
    grad_sensor_rays[idx * 3 + 1] = grad_y;
    grad_sensor_rays[idx * 3 + 2] = grad_z;
}

// VJP of angles -> ray.  rx = ce*ca, ry = ce*sa, rz = se.
template<typename T>
__global__ void sensor_angles_to_sensor_rays_backward_kernel(
    int64_t count,
    const T *__restrict__ sensor_angles,
    const T *__restrict__ grad_sensor_rays,
    T *__restrict__ grad_sensor_angles
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    T elevation = sensor_angles[idx * 2 + 0];
    T azimuth   = sensor_angles[idx * 2 + 1];
    T ce, se, ca, sa;
    sincos_t<T>(elevation, se, ce);
    sincos_t<T>(azimuth, sa, ca);

    T gx = grad_sensor_rays[idx * 3 + 0];
    T gy = grad_sensor_rays[idx * 3 + 1];
    T gz = grad_sensor_rays[idx * 3 + 2];

    T grad_elev = gx * (-se * ca) + gy * (-se * sa) + gz * (ce);
    T grad_az   = gx * (-ce * sa) + gy * (ce * ca);

    grad_sensor_angles[idx * 2 + 0] = grad_elev;
    grad_sensor_angles[idx * 2 + 1] = grad_az;
}

// Max rows the shared-memory privatization path supports.  The shipped
// Waymo/Pandaset/Hesai sensors are all <= 128 rows; larger models fall back to
// the direct-global scatter.
constexpr int kLidarMaxSmemRows = 128;

// Scatter VJP of elements -> angles.  Each thread accumulates into the single
// (row, col) slots it read.  normalize_angle has unit derivative a.e., so the
// azimuth gradient passes straight through to both the column azimuth and the
// per-row offset.  OOB elements contribute nothing.  atomicAdd accumulates
// across all threads sharing a row/col, so shared indices sum (not overwrite).
//
// The two per-ROW tables (row elevations, row azimuth offsets) draw heavy
// L2-atomic contention: every element scatters into one of n_rows <= 128 slots,
// so thousands of atomics funnel into a handful of addresses.  When n_rows fits
// kLidarMaxSmemRows, those two accumulations are privatized in shared memory
// (block-local atomics, far cheaper than L2) and flushed to global once per
// occupied row per block on a uniform path (no early return before the
// barriers).  The per-COLUMN table (n_columns slots) is far less contended and
// stays on the direct-global path.
template<typename T>
__global__ void elements_to_sensor_angles_backward_kernel(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t *__restrict__ elements,
    const T *__restrict__ grad_sensor_angles,
    T *__restrict__ grad_row_elevations_rad,
    T *__restrict__ grad_column_azimuths_rad,
    T *__restrict__ grad_row_azimuth_offsets_rad
)
{
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool use_smem = (n_rows <= kLidarMaxSmemRows);

    if(!use_smem)
    {
        // Direct-global fallback for models with more rows than the smem budget.
        if(idx >= count)
        {
            return;
        }
        int32_t row = elements[idx * 2 + 0];
        int32_t col = elements[idx * 2 + 1];
        if(row < 0 || row >= n_rows || col < 0 || col >= n_columns)
        {
            return;
        }
        T g_elev = grad_sensor_angles[idx * 2 + 0];
        T g_az   = grad_sensor_angles[idx * 2 + 1];
        if(grad_row_elevations_rad != nullptr)
        {
            atomicAdd(&grad_row_elevations_rad[row], g_elev);
        }
        if(grad_column_azimuths_rad != nullptr)
        {
            atomicAdd(&grad_column_azimuths_rad[col], g_az);
        }
        if(has_row_offsets != 0 && grad_row_azimuth_offsets_rad != nullptr)
        {
            atomicAdd(&grad_row_azimuth_offsets_rad[row], g_az);
        }
        return;
    }

    // Shared-memory-privatized row accumulation.  Zero, accumulate, flush all
    // happen on a uniform path so every thread reaches both __syncthreads().
    __shared__ T smem_row_elev[kLidarMaxSmemRows];
    __shared__ T smem_row_off[kLidarMaxSmemRows];
    for(int i = threadIdx.x; i < n_rows; i += blockDim.x)
    {
        smem_row_elev[i] = T(0);
        smem_row_off[i]  = T(0);
    }
    __syncthreads();

    if(idx < count)
    {
        int32_t row = elements[idx * 2 + 0];
        int32_t col = elements[idx * 2 + 1];
        if(row >= 0 && row < n_rows && col >= 0 && col < n_columns)
        {
            T g_elev = grad_sensor_angles[idx * 2 + 0];
            T g_az   = grad_sensor_angles[idx * 2 + 1];
            if(grad_row_elevations_rad != nullptr)
            {
                atomicAdd(&smem_row_elev[row], g_elev);
            }
            if(grad_column_azimuths_rad != nullptr)
            {
                atomicAdd(&grad_column_azimuths_rad[col], g_az);
            }
            if(has_row_offsets != 0 && grad_row_azimuth_offsets_rad != nullptr)
            {
                atomicAdd(&smem_row_off[row], g_az);
            }
        }
    }
    __syncthreads();

    for(int i = threadIdx.x; i < n_rows; i += blockDim.x)
    {
        if(grad_row_elevations_rad != nullptr)
        {
            atomicAdd(&grad_row_elevations_rad[i], smem_row_elev[i]);
        }
        if(has_row_offsets != 0 && grad_row_azimuth_offsets_rad != nullptr)
        {
            atomicAdd(&grad_row_azimuth_offsets_rad[i], smem_row_off[i]);
        }
    }
}

// Backward for generate_spinning_lidar_rays.  Composes:
//   grad_world_rays[3:6] -> R(q_interp) sensor_ray VJP -> grad_q_interp (xyzw)
//                                                       + grad_sensor_ray
//   grad_sensor_ray -> spherical_to_cartesian VJP -> grad_elev, grad_az
//                       (scattered per (row, col) into the 3 angle tables)
//   grad_q_interp -> slerp_pair VJP -> grad_q0/q1 (xyzw, hemisphere folded in)
//   grad_world_rays[0:3] (origin) -> lerp3 -> grad_t0 = (1-a) g, grad_t1 = a g.
// Control-pose grads are shared across all threads, so each component is
// block-reduced then atomicAdded once per block.  The slerp time grad is
// non-differentiable here (alpha from column index) and is never emitted.
// angle-table grads scatter per index with direct atomicAdd.  The block_sum
// reductions run on a uniform path (no early return) so no thread skips a
// __syncthreads().
template<typename T>
__global__ void __launch_bounds__(kLidarThreads, 5) generate_spinning_lidar_rays_backward_kernel(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    const int32_t *__restrict__ elements,
    const T *__restrict__ row_elevations_rad,
    const T *__restrict__ column_azimuths_rad,
    const T *__restrict__ row_azimuth_offsets_rad,
    const T *__restrict__ control_translations,
    const T *__restrict__ control_rotations,
    const T *__restrict__ grad_world_rays,
    T *__restrict__ grad_row_elevations_rad,
    T *__restrict__ grad_column_azimuths_rad,
    T *__restrict__ grad_row_azimuth_offsets_rad,
    T *__restrict__ grad_control_translations,
    T *__restrict__ grad_control_rotations
)
{
    int64_t idx       = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = idx < count;

    // Per-thread accumulators for the shared control-pose grads (xyzw rotation).
    T g_t0x = T(0), g_t0y = T(0), g_t0z = T(0);
    T g_t1x = T(0), g_t1y = T(0), g_t1z = T(0);
    T g_q0x = T(0), g_q0y = T(0), g_q0z = T(0), g_q0w = T(0);
    T g_q1x = T(0), g_q1y = T(0), g_q1z = T(0), g_q1w = T(0);

    if(active)
    {
        int32_t row;
        int32_t col;
        if(does_generate_elements != 0)
        {
            row = static_cast<int32_t>(idx / n_columns);
            col = static_cast<int32_t>(idx - static_cast<int64_t>(row) * n_columns);
        }
        else
        {
            row = elements[idx * 2 + 0];
            col = elements[idx * 2 + 1];
        }
        const bool valid = (row >= 0 && row < n_rows && col >= 0 && col < n_columns);

        if(valid)
        {
            T elevation = row_elevations_rad[row];
            T azimuth   = column_azimuths_rad[col];
            if(has_row_offsets != 0)
            {
                azimuth += row_azimuth_offsets_rad[row];
                azimuth  = gsplat_sensors::normalize_angle<T>(azimuth);
            }

            T relative_time = (n_columns > 1) ? static_cast<T>(col) / static_cast<T>(n_columns - 1) : T(0);
            T alpha         = relative_time < T(0) ? T(0) : (relative_time > T(1) ? T(1) : relative_time);

            const T q0x = control_rotations[1], q0y = control_rotations[2], q0z = control_rotations[3],
                    q0w = control_rotations[0];
            const T q1x = control_rotations[5], q1y = control_rotations[6], q1z = control_rotations[7],
                    q1w = control_rotations[4];

            T qix, qiy, qiz, qiw;
            slerp_pair_fwd<T>(q0x, q0y, q0z, q0w, q1x, q1y, q1z, q1w, alpha, &qix, &qiy, &qiz, &qiw);
            T srx, sry, srz;
            spherical_to_cartesian<T>(elevation, azimuth, srx, sry, srz);

            const T g_ox = grad_world_rays[idx * 6 + 0];
            const T g_oy = grad_world_rays[idx * 6 + 1];
            const T g_oz = grad_world_rays[idx * 6 + 2];
            const T g_dx = grad_world_rays[idx * 6 + 3];
            const T g_dy = grad_world_rays[idx * 6 + 4];
            const T g_dz = grad_world_rays[idx * 6 + 5];

            // world_dir = R(q_interp) sensor_ray.
            T gq_ix, gq_iy, gq_iz, gq_iw;
            T g_srx, g_sry, g_srz;
            quat_rotate_vector_bwd_impl<T>(
                qix,
                qiy,
                qiz,
                qiw,
                srx,
                sry,
                srz,
                g_dx,
                g_dy,
                g_dz,
                &gq_ix,
                &gq_iy,
                &gq_iz,
                &gq_iw,
                &g_srx,
                &g_sry,
                &g_srz
            );

            // sensor_ray = spherical_to_cartesian(elev, az).
            //   rx = ce*ca, ry = ce*sa, rz = se.
            const T ce = cos(elevation), se = sin(elevation);
            const T ca = cos(azimuth), sa = sin(azimuth);
            const T g_elev = g_srx * (-se * ca) + g_sry * (-se * sa) + g_srz * (ce);
            const T g_az   = g_srx * (-ce * sa) + g_sry * (ce * ca);

            if(grad_row_elevations_rad != nullptr)
            {
                atomicAdd(&grad_row_elevations_rad[row], g_elev);
            }
            if(grad_column_azimuths_rad != nullptr)
            {
                atomicAdd(&grad_column_azimuths_rad[col], g_az);
            }
            if(has_row_offsets != 0 && grad_row_azimuth_offsets_rad != nullptr)
            {
                atomicAdd(&grad_row_azimuth_offsets_rad[row], g_az);
            }

            // q_interp = slerp(q0, q1, alpha).
            slerp_pair_bwd<T>(
                q0x,
                q0y,
                q0z,
                q0w,
                q1x,
                q1y,
                q1z,
                q1w,
                alpha,
                qix,
                qiy,
                qiz,
                qiw,
                gq_ix,
                gq_iy,
                gq_iz,
                gq_iw,
                &g_q0x,
                &g_q0y,
                &g_q0z,
                &g_q0w,
                &g_q1x,
                &g_q1y,
                &g_q1z,
                &g_q1w
            );

            // origin = lerp3(t0, t1, alpha).
            const T om = T(1) - alpha;
            g_t0x      = om * g_ox;
            g_t0y      = om * g_oy;
            g_t0z      = om * g_oz;
            g_t1x      = alpha * g_ox;
            g_t1y      = alpha * g_oy;
            g_t1z      = alpha * g_oz;
        }
    }

    // Block-reduce the shared control-pose grads (uniform path, no early return).
    const T pose_vals[14]
        = {g_t0x, g_t0y, g_t0z, g_t1x, g_t1y, g_t1z, g_q0x, g_q0y, g_q0z, g_q0w, g_q1x, g_q1y, g_q1z, g_q1w};
    T r[14];
    lidar_block_reduce_pose<T, kLidarThreads>(pose_vals, r);

    if(threadIdx.x == 0)
    {
        if(grad_control_translations != nullptr)
        {
            atomicAdd(&grad_control_translations[0], r[0]);
            atomicAdd(&grad_control_translations[1], r[1]);
            atomicAdd(&grad_control_translations[2], r[2]);
            atomicAdd(&grad_control_translations[3], r[3]);
            atomicAdd(&grad_control_translations[4], r[4]);
            atomicAdd(&grad_control_translations[5], r[5]);
        }
        if(grad_control_rotations != nullptr)
        {
            // xyzw register grads -> wxyz storage (w into slot 0).
            atomicAdd(&grad_control_rotations[0], r[9]);
            atomicAdd(&grad_control_rotations[1], r[6]);
            atomicAdd(&grad_control_rotations[2], r[7]);
            atomicAdd(&grad_control_rotations[3], r[8]);
            atomicAdd(&grad_control_rotations[4], r[13]);
            atomicAdd(&grad_control_rotations[5], r[10]);
            atomicAdd(&grad_control_rotations[6], r[11]);
            atomicAdd(&grad_control_rotations[7], r[12]);
        }
    }
}

// Backward for inverse_project_spinning_lidar (IFT path).  Re-loads the
// converged interpolation alpha from scratch, then takes ONE differentiable step
// from the converged pose: s = R(q(alpha))^T (p - t(alpha)); ray = s/|s|;
// angles = cart2sph(ray).  The LINEAR relative_time map is piecewise-constant,
// so d(relative_time)/d(angles) = 0 and the implicit-function correction term
// vanishes -- the frozen-alpha direct VJP is the exact gradient.  The
// cart2sph singular axis is det-guarded with a dtype-dependent eps; below it the
// azimuth gradient is zeroed (never 1/fmax).  Angle-table grads are zero (no
// differentiable path through the piecewise-constant column lookup), so only
// world_points + the two control poses receive gradient.  Control-pose grads are
// block-reduced then atomicAdded once per block on a uniform path (no early
// return) so no thread skips a __syncthreads().
template<typename T>
__global__ void __launch_bounds__(kLidarThreads, 5) inverse_project_spinning_lidar_backward_kernel(
    int64_t count,
    const T *__restrict__ world_points,
    const T *__restrict__ control_translations, // (2, 3)
    const T *__restrict__ control_rotations,    // (2, 4) wxyz
    const bool *__restrict__ valid_flags,
    const T *__restrict__ grad_sensor_angles,
    const T *__restrict__ scratch,
    T *__restrict__ grad_world_points,
    T *__restrict__ grad_control_translations,
    T *__restrict__ grad_control_rotations
)
{
    int64_t idx       = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = idx < count;

    T g_t0x = T(0), g_t0y = T(0), g_t0z = T(0);
    T g_t1x = T(0), g_t1y = T(0), g_t1z = T(0);
    T g_q0x = T(0), g_q0y = T(0), g_q0z = T(0), g_q0w = T(0);
    T g_q1x = T(0), g_q1y = T(0), g_q1z = T(0), g_q1w = T(0);

    if(active)
    {
        T g_px = T(0), g_py = T(0), g_pz = T(0);
        if(valid_flags[idx])
        {
            const T alpha = scratch[idx];
            const T px    = world_points[idx * 3 + 0];
            const T py    = world_points[idx * 3 + 1];
            const T pz    = world_points[idx * 3 + 2];

            const T t0x = control_translations[0], t0y = control_translations[1], t0z = control_translations[2];
            const T t1x = control_translations[3], t1y = control_translations[4], t1z = control_translations[5];
            const T q0x = control_rotations[1], q0y = control_rotations[2], q0z = control_rotations[3],
                    q0w = control_rotations[0];
            const T q1x = control_rotations[5], q1y = control_rotations[6], q1z = control_rotations[7],
                    q1w = control_rotations[4];

            const T om      = T(1) - alpha;
            const T trans_x = om * t0x + alpha * t1x;
            const T trans_y = om * t0y + alpha * t1y;
            const T trans_z = om * t0z + alpha * t1z;
            T qix, qiy, qiz, qiw;
            slerp_pair_fwd<T>(q0x, q0y, q0z, q0w, q1x, q1y, q1z, q1w, alpha, &qix, &qiy, &qiz, &qiw);

            // s = R(q)^T (p - t), then ray = s / |s|.
            T sx, sy, sz;
            se3_inverse_transform_point<T>(qix, qiy, qiz, qiw, trans_x, trans_y, trans_z, px, py, pz, &sx, &sy, &sz);
            const T s2         = fmax(sx * sx + sy * sy + sz * sz, gsplat_sensors::normalize_normsq_floor<T>());
            const T s_norm     = sqrt(s2);
            const T inv_s_norm = T(1) / s_norm;
            const T rx = sx * inv_s_norm, ry = sy * inv_s_norm, rz = sz * inv_s_norm;

            // cart2sph VJP: elev = atan2(z, hypot(x,y)); az = atan2(y, x).
            const T g_elev = grad_sensor_angles[idx * 2 + 0];
            const T g_az   = grad_sensor_angles[idx * 2 + 1];
            const T r2     = rx * rx + ry * ry;
            const T rho2   = r2 + rz * rz;
            const T r      = sqrt(r2);
            const T eps    = det_guard_eps<T>();
            T g_rx = T(0), g_ry = T(0), g_rz = T(0);
            if(rho2 > eps)
            {
                g_rz += g_elev * (r / rho2);
                if(r2 > eps)
                {
                    const T inv  = T(1) / (rho2 * r);
                    g_rx        += g_elev * (-rz * rx * inv);
                    g_ry        += g_elev * (-rz * ry * inv);
                    g_rx        += g_az * (-ry / r2);
                    g_ry        += g_az * (rx / r2);
                }
            }

            // normalize VJP: g_s = (g_r - r (r . g_r)) / |s|.
            const T rdotg = rx * g_rx + ry * g_ry + rz * g_rz;
            const T g_sx  = (g_rx - rx * rdotg) * inv_s_norm;
            const T g_sy  = (g_ry - ry * rdotg) * inv_s_norm;
            const T g_sz  = (g_rz - rz * rdotg) * inv_s_norm;

            // s = rotate(q_conj, v), v = p - t.  q_conj = (-qx,-qy,-qz,qw).
            const T vx = px - trans_x, vy = py - trans_y, vz = pz - trans_z;
            T g_cqx, g_cqy, g_cqz, g_cqw, g_vx, g_vy, g_vz;
            quat_rotate_vector_bwd_impl<T>(
                -qix, -qiy, -qiz, qiw, vx, vy, vz, g_sx, g_sy, g_sz, &g_cqx, &g_cqy, &g_cqz, &g_cqw, &g_vx, &g_vy, &g_vz
            );
            // Undo the conjugate sign on the rotation grad.
            const T g_qix = -g_cqx, g_qiy = -g_cqy, g_qiz = -g_cqz, g_qiw = g_cqw;

            // v = p - t.
            g_px         = g_vx;
            g_py         = g_vy;
            g_pz         = g_vz;
            const T g_tx = -g_vx, g_ty = -g_vy, g_tz = -g_vz;

            // t = lerp(t0, t1, alpha).
            g_t0x = om * g_tx;
            g_t0y = om * g_ty;
            g_t0z = om * g_tz;
            g_t1x = alpha * g_tx;
            g_t1y = alpha * g_ty;
            g_t1z = alpha * g_tz;

            // q = slerp(q0, q1, alpha).
            slerp_pair_bwd<T>(
                q0x,
                q0y,
                q0z,
                q0w,
                q1x,
                q1y,
                q1z,
                q1w,
                alpha,
                qix,
                qiy,
                qiz,
                qiw,
                g_qix,
                g_qiy,
                g_qiz,
                g_qiw,
                &g_q0x,
                &g_q0y,
                &g_q0z,
                &g_q0w,
                &g_q1x,
                &g_q1y,
                &g_q1z,
                &g_q1w
            );
        }
        if(grad_world_points != nullptr)
        {
            grad_world_points[idx * 3 + 0] = g_px;
            grad_world_points[idx * 3 + 1] = g_py;
            grad_world_points[idx * 3 + 2] = g_pz;
        }
    }

    const T pose_vals[14]
        = {g_t0x, g_t0y, g_t0z, g_t1x, g_t1y, g_t1z, g_q0x, g_q0y, g_q0z, g_q0w, g_q1x, g_q1y, g_q1z, g_q1w};
    T r[14];
    lidar_block_reduce_pose<T, kLidarThreads>(pose_vals, r);

    if(threadIdx.x == 0)
    {
        if(grad_control_translations != nullptr)
        {
            atomicAdd(&grad_control_translations[0], r[0]);
            atomicAdd(&grad_control_translations[1], r[1]);
            atomicAdd(&grad_control_translations[2], r[2]);
            atomicAdd(&grad_control_translations[3], r[3]);
            atomicAdd(&grad_control_translations[4], r[4]);
            atomicAdd(&grad_control_translations[5], r[5]);
        }
        if(grad_control_rotations != nullptr)
        {
            // xyzw register grads -> wxyz storage (w into slot 0).
            atomicAdd(&grad_control_rotations[0], r[9]);
            atomicAdd(&grad_control_rotations[1], r[6]);
            atomicAdd(&grad_control_rotations[2], r[7]);
            atomicAdd(&grad_control_rotations[3], r[8]);
            atomicAdd(&grad_control_rotations[4], r[13]);
            atomicAdd(&grad_control_rotations[5], r[10]);
            atomicAdd(&grad_control_rotations[6], r[11]);
            atomicAdd(&grad_control_rotations[7], r[12]);
        }
    }
}
} // namespace

// ===========================================================================
// Launchers
// ===========================================================================

template<typename T>
void sensor_rays_to_sensor_angles_backward_launch(
    int64_t count, const T *sensor_rays, const T *grad_sensor_angles, T *grad_sensor_rays, cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    sensor_rays_to_sensor_angles_backward_kernel<T>
        <<<grid_for_count(count), kLidarThreads, 0, stream>>>(count, sensor_rays, grad_sensor_angles, grad_sensor_rays);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T>
void sensor_angles_to_sensor_rays_backward_launch(
    int64_t count, const T *sensor_angles, const T *grad_sensor_rays, T *grad_sensor_angles, cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    sensor_angles_to_sensor_rays_backward_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count, sensor_angles, grad_sensor_rays, grad_sensor_angles
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename T>
void elements_to_sensor_angles_backward_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    const int32_t *elements,
    const T *grad_sensor_angles,
    T *grad_row_elevations_rad,
    T *grad_column_azimuths_rad,
    T *grad_row_azimuth_offsets_rad,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    elements_to_sensor_angles_backward_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        n_rows,
        n_columns,
        has_row_offsets,
        elements,
        grad_sensor_angles,
        grad_row_elevations_rad,
        grad_column_azimuths_rad,
        grad_row_azimuth_offsets_rad
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void sensor_rays_to_sensor_angles_backward_launch<float>(
    int64_t, const float *, const float *, float *, cudaStream_t
);
template void sensor_rays_to_sensor_angles_backward_launch<double>(
    int64_t, const double *, const double *, double *, cudaStream_t
);
template void sensor_angles_to_sensor_rays_backward_launch<float>(
    int64_t, const float *, const float *, float *, cudaStream_t
);
template void sensor_angles_to_sensor_rays_backward_launch<double>(
    int64_t, const double *, const double *, double *, cudaStream_t
);
template void elements_to_sensor_angles_backward_launch<float>(
    int64_t, int64_t, int64_t, int, const int32_t *, const float *, float *, float *, float *, cudaStream_t
);
template void elements_to_sensor_angles_backward_launch<double>(
    int64_t, int64_t, int64_t, int, const int32_t *, const double *, double *, double *, double *, cudaStream_t
);

template<typename T>
void generate_spinning_lidar_rays_backward_launch(
    int64_t count,
    int64_t n_rows,
    int64_t n_columns,
    int has_row_offsets,
    int does_generate_elements,
    const int32_t *elements,
    const T *row_elevations_rad,
    const T *column_azimuths_rad,
    const T *row_azimuth_offsets_rad,
    const T *control_translations,
    const T *control_rotations,
    const T *grad_world_rays,
    T *grad_row_elevations_rad,
    T *grad_column_azimuths_rad,
    T *grad_row_azimuth_offsets_rad,
    T *grad_control_translations,
    T *grad_control_rotations,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    generate_spinning_lidar_rays_backward_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        n_rows,
        n_columns,
        has_row_offsets,
        does_generate_elements,
        elements,
        row_elevations_rad,
        column_azimuths_rad,
        row_azimuth_offsets_rad,
        control_translations,
        control_rotations,
        grad_world_rays,
        grad_row_elevations_rad,
        grad_column_azimuths_rad,
        grad_row_azimuth_offsets_rad,
        grad_control_translations,
        grad_control_rotations
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void generate_spinning_lidar_rays_backward_launch<float>(
    int64_t,
    int64_t,
    int64_t,
    int,
    int,
    const int32_t *,
    const float *,
    const float *,
    const float *,
    const float *,
    const float *,
    const float *,
    float *,
    float *,
    float *,
    float *,
    float *,
    cudaStream_t
);
template void generate_spinning_lidar_rays_backward_launch<double>(
    int64_t,
    int64_t,
    int64_t,
    int,
    int,
    const int32_t *,
    const double *,
    const double *,
    const double *,
    const double *,
    const double *,
    const double *,
    double *,
    double *,
    double *,
    double *,
    double *,
    cudaStream_t
);

template<typename T>
void inverse_project_spinning_lidar_backward_launch(
    int64_t count,
    const T *world_points,
    const T *control_translations,
    const T *control_rotations,
    const bool *valid_flags,
    const T *grad_sensor_angles,
    const T *scratch,
    T *grad_world_points,
    T *grad_control_translations,
    T *grad_control_rotations,
    cudaStream_t stream
)
{
    if(count <= 0)
    {
        return;
    }
    inverse_project_spinning_lidar_backward_kernel<T><<<grid_for_count(count), kLidarThreads, 0, stream>>>(
        count,
        world_points,
        control_translations,
        control_rotations,
        valid_flags,
        grad_sensor_angles,
        scratch,
        grad_world_points,
        grad_control_translations,
        grad_control_rotations
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void inverse_project_spinning_lidar_backward_launch<float>(
    int64_t,
    const float *,
    const float *,
    const float *,
    const bool *,
    const float *,
    const float *,
    float *,
    float *,
    float *,
    cudaStream_t
);
template void inverse_project_spinning_lidar_backward_launch<double>(
    int64_t,
    const double *,
    const double *,
    const double *,
    const bool *,
    const double *,
    const double *,
    double *,
    double *,
    double *,
    cudaStream_t
);
