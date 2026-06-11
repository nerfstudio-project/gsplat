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

// Spherical<->cartesian angle conversions for sensor-frame ray math.
//
// Sensor frame is +X forward, +Y left, +Z up; angles stored [elevation, azimuth].
//   angles -> unit ray : (cos_e*cos(az), cos_e*sin(az), sin(elev))
//   ray   -> angles    : elev = atan2(z, hypot(x,y)); az = atan2(y, x)
// dtype-templated (T = float/double) so fp64 gradcheck is exact.

#pragma once

#include <cuda_runtime.h>

namespace gsplat_geometry
{
// sincos_t — one sincos call per angle (float -> sincosf, double -> sincos),
// so the paired sin/cos of each angle issue a single special-function op
// instead of two separate transcendentals.
template<typename T>
__device__ __forceinline__ void sincos_t(T angle, T &s, T &c);

template<>
__device__ __forceinline__ void sincos_t<float>(float angle, float &s, float &c)
{
    sincosf(angle, &s, &c);
}

template<>
__device__ __forceinline__ void sincos_t<double>(double angle, double &s, double &c)
{
    sincos(angle, &s, &c);
}

// spherical_to_cartesian — angles (elevation, azimuth) -> unit ray.
template<typename T>
__device__ __forceinline__ void spherical_to_cartesian(T elevation, T azimuth, T &out_x, T &out_y, T &out_z)
{
    T sin_e, cos_e, sin_a, cos_a;
    sincos_t<T>(elevation, sin_e, cos_e);
    sincos_t<T>(azimuth, sin_a, cos_a);
    out_x = cos_e * cos_a;
    out_y = cos_e * sin_a;
    out_z = sin_e;
}

// cartesian_to_spherical — ray (x, y, z) -> (elevation, azimuth).
template<typename T>
__device__ __forceinline__ void cartesian_to_spherical(T x, T y, T z, T &out_elevation, T &out_azimuth)
{
    T xy_norm     = sqrt(x * x + y * y);
    out_elevation = atan2(z, xy_norm);
    out_azimuth   = atan2(y, x);
}
} // namespace gsplat_geometry
