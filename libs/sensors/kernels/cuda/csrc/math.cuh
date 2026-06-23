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

// Minimal float3 math utilities for the OpenCV pinhole camera CUDA kernels.
// Provides add3, sub3, scale3, dot3, normalize3, and the normalize3 backward
// (Jacobian of y = v / |v| w.r.t. v). Kept separate from camera_kernel.cuh so
// math.cuh can be included in translation units that do not pull in the full
// camera parameter headers.

#pragma once

#include <cuda_runtime.h>

constexpr float kNormalize3NormSqFloor = 1.0e-20f;
constexpr int kWarpSize = 32;
constexpr unsigned int kFullWarpMask = 0xffffffffu;

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
    float norm = sqrtf(fmaxf(dot3(v, v), kNormalize3NormSqFloor));
    float inv_norm = 1.0f / norm;
    return make_float3(v.x * inv_norm, v.y * inv_norm, v.z * inv_norm);
}

// Backward of normalize3: dy/dv = (I - y y^T) / |v|, so d_v = (d_y - (y . d_y) * y) / |v|.
// v is the primal input (not the normalized output); d_y is the upstream gradient.
__device__ __forceinline__ float3 normalize3_bwd(float3 v, float3 d_y) {
    // Floor matches the forward `normalize3` so the adjoint is consistent on near-zero vectors.
    float len = sqrtf(fmaxf(dot3(v, v), kNormalize3NormSqFloor));
    float inv_len = 1.0f / len;
    float3 y = make_float3(v.x * inv_len, v.y * inv_len, v.z * inv_len);
    float dot = y.x * d_y.x + y.y * d_y.y + y.z * d_y.z;
    return make_float3(
        (d_y.x - dot * y.x) * inv_len,
        (d_y.y - dot * y.y) * inv_len,
        (d_y.z - dot * y.z) * inv_len);
}

// Returns the block sum of `value` on every lane of warp 0
// (threadIdx.x < kWarpSize); lanes in other warps read 0. Current callers
// consume only threadIdx.x == 0.
template <int BlockThreads>
__device__ __forceinline__ float block_sum(float value) {
    static_assert(
        BlockThreads % kWarpSize == 0,
        "BlockThreads must be a multiple of warp size");
    constexpr int kNumWarps = BlockThreads / kWarpSize;
    __shared__ float warp_sums[kNumWarps];

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(kFullWarpMask, value, offset);
    }
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    float total = 0.0f;
    if (warp == 0) {
        total = threadIdx.x < kNumWarps ? warp_sums[lane] : 0.0f;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            total += __shfl_xor_sync(kFullWarpMask, total, offset);
        }
    }
    // Callers invoke block_sum repeatedly with the same shared warp_sums; keep
    // all lanes from the previous call clear before the next call can clobber it.
    __syncthreads();
    return total;
}
