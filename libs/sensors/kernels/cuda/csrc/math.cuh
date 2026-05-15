/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal float3 math utilities for the OpenCV pinhole camera CUDA kernels.
// Provides add3, sub3, scale3, dot3, normalize3, and the normalize3 backward
// (Jacobian of y = v / |v| w.r.t. v). Kept separate from camera_kernel.cuh so
// math.cuh can be included in translation units that do not pull in the full
// camera parameter headers.

#pragma once

#include <cuda_runtime.h>

// ===========================================================================
// float3 arithmetic
// ===========================================================================

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 scale3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// ===========================================================================
// float3 normalize (forward and backward)
// ===========================================================================

// Clamps |v|^2 to 1e-20 before the sqrt to prevent a NaN gradient at the
// zero vector; the clamp has no effect for any physically meaningful ray.
__device__ __forceinline__ float3 normalize3(float3 v) {
    float norm = sqrtf(fmaxf(dot3(v, v), 1.0e-20f));
    float inv_norm = 1.0f / norm;
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

// Backward of normalize3: dy/dv = (I - y y^T) / |v|, so d_v = (d_y - (y . d_y) * y) / |v|.
// v is the primal input (not the normalized output); d_y is the upstream gradient.
__device__ __forceinline__ float3 normalize3_bwd(float3 v, float3 d_y) {
    float len = sqrtf(fmaxf(dot3(v, v), 1.0e-20f));
    float inv_len = 1.0f / len;
    float3 y = make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
    float dot = y.x * d_y.x + y.y * d_y.y + y.z * d_y.z;
    return make_float3(
        (d_y.x - dot * y.x) * inv_len,
        (d_y.y - dot * y.y) * inv_len,
        (d_y.z - dot * y.z) * inv_len);
}
