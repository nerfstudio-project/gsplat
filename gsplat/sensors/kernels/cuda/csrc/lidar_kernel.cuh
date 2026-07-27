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
// Defines LiDAR-specific device math: FOV checks, nearest-row/column lookup,
// and inverse-projection helpers. Shared rotation, pose interpolation, and
// ray/angle helpers come from geometry CUDA headers.

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
} // namespace gsplat_sensors
