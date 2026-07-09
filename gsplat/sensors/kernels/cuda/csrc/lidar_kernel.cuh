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

// Device-side math helpers for the spinning-LiDAR CUDA kernels.
//
// Tier-2 nvcc-only header.  Pulls in the Tier-1 POD parameter pack
// (lidar_params.h), which also defines the SpinningDirection enum.
//
// Defines the device-side math the LiDAR kernels share: the SLERP
// forward/backward (VJP) used by the rolling shutter, and the FOV /
// nearest-row / nearest-column inverse-projection helpers.  The geometry
// helpers (resolved via build.py extra_include_paths) supply the quaternion
// rotate/conjugate math (quaternion.cuh), the scalar SE3 inverse transform
// (pose.cuh), and the spherical<->cartesian angle conversions
// (coordinate_conversions.cuh).

#pragma once

#include "lidar_params.h"

#include "coordinate_conversions.cuh"
#include "pose.cuh"
#include "quaternion.cuh"

#include <cuda_runtime.h>

#include <stdint.h>

namespace gsplat_sensors
{
// kThreads — 1-D block size shared by all LiDAR launchers, mirroring the
// camera kernels.
constexpr int kLidarThreads = 256;

// ===========================================================================
// Sensor-frame angle math (dtype-templated)
// ===========================================================================
//
// sensor_angles are stored [elevation, azimuth].  Sensor frame is +X forward,
// +Y left, +Z up.  These helpers are dtype-polymorphic (templated on
// T = float/double) so fp64 gradcheck is exact; the public binding
// force-float32 layer lives in Python.

template<typename T>
constexpr __device__ __forceinline__ T pi_const();

template<>
constexpr __device__ __forceinline__ float pi_const<float>()
{
    return 3.14159265358979323846f;
}

template<>
constexpr __device__ __forceinline__ double pi_const<double>()
{
    return 3.14159265358979323846;
}

// normalize_angle — fmod-based wrap to [-PI, PI).
// Derivative is 1 almost everywhere, so the backward path treats it as
// identity on the azimuth gradient.
template<typename T>
__device__ __forceinline__ T normalize_angle(T angle)
{
    const T pi     = pi_const<T>();
    const T two_pi = T(2) * pi;
    angle          = fmod(angle + pi, two_pi);
    if(angle < T(0))
    {
        angle += two_pi;
    }
    return angle - pi;
}

// normalize_normsq_floor — dtype-dependent floor clamped onto |v|^2 before the
// ray-normalize sqrt, preventing a NaN gradient at the zero vector; the clamp
// has no effect for any physically meaningful ray.  float32 -> 1e-20,
// float64 -> 1e-40.
template<typename T>
constexpr __device__ __forceinline__ T normalize_normsq_floor();

template<>
constexpr __device__ __forceinline__ float normalize_normsq_floor<float>()
{
    return 1.0e-20f;
}

template<>
constexpr __device__ __forceinline__ double normalize_normsq_floor<double>()
{
    return 1.0e-40;
}

// ===========================================================================
// Two-pose SLERP (dtype-templated, scalar in/out, xyzw)
// ===========================================================================
//
// The geometry pose.cuh `quat_slerp_pair_fwd_f/bwd_f` helpers are float-only,
// so fp64 gradcheck cannot route through them.  These are the scalar form of
// the dtype-templated `quat_slerp_batched_fwd_device/bwd_device` math
// (quaternion.cuh) — hemisphere flip, clamp(dot), nlerp small-angle fallback —
// transcribed to single-quaternion args so the LiDAR ray generator stays
// float/double polymorphic without re-deriving the SLERP VJP.

// SLERP q1->q2 at ti, xyzw, hemisphere flip, nlerp fallback when dot>0.9995.
template<typename T>
__device__ __forceinline__ void slerp_pair_fwd(
    T x1, T y1, T z1, T w1, T x2, T y2, T z2, T w2, T ti, T *ox, T *oy, T *oz, T *ow
)
{
    const T dot = x1 * x2 + y1 * y2 + z1 * z2 + w1 * w2;
    const T s   = dot < T(0) ? T(-1) : T(1);
    const T sx = s * x2, sy = s * y2, sz = s * z2, sw = s * w2;
    const T c_raw = x1 * sx + y1 * sy + z1 * sz + w1 * sw;
    const T c     = quat_slerp_clamp_dot<T>(c_raw);

    if(c > quat_slerp_small_angle_dot_threshold<T>())
    {
        const T om    = T(1) - ti;
        const T rx    = om * x1 + ti * sx;
        const T ry    = om * y1 + ti * sy;
        const T rz    = om * z1 + ti * sz;
        const T rw    = om * w1 + ti * sw;
        const T inv_n = T(1) / sqrt(rx * rx + ry * ry + rz * rz + rw * rw);
        *ox           = rx * inv_n;
        *oy           = ry * inv_n;
        *oz           = rz * inv_n;
        *ow           = rw * inv_n;
        return;
    }

    const T theta     = acos(c);
    const T sin_theta = sin(theta);
    const T w1s       = sin((T(1) - ti) * theta) / sin_theta;
    const T w2s       = sin(ti * theta) / sin_theta;
    *ox               = w1s * x1 + w2s * sx;
    *oy               = w1s * y1 + w2s * sy;
    *oz               = w1s * z1 + w2s * sz;
    *ow               = w1s * w1 + w2s * sw;
}
} // namespace gsplat_sensors
