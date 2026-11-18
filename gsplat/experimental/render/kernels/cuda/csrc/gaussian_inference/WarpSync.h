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

#pragma once

#include <cstdint>

// ---------------------------------------------------------------------------
// 32-lane warp primitives.
//
// These kernels are written for NVIDIA's 32-lane warps: ballot results are
// packed into 32-bit shared bitfields (visible[], out_active_mask), a 32x32
// bit-matrix transpose runs across a warp, and a 32-way cooperative binary
// search assumes 32 probing lanes. That 32-bit data layout is a contract shared
// with other kernels and the host, so it must be preserved verbatim.
//
// On AMD GCN the hardware wavefront is 64 lanes wide, so a wavefront spans TWO
// 32-lane logical warps (lanes 0-31 and lanes 32-63). We keep the 32-lane
// semantics by:
//   - shuffling within width-32 subgroups, and
//   - taking a full 64-bit __ballot and extracting the 32-bit half belonging to
//     the caller's logical warp.
// The caller's logical warp is selected by (threadIdx.x & 32): 0 for the low
// half, 32 for the high half. Each logical warp therefore still produces its
// own independent 32-bit mask and its own lane 0, so packing code that keys off
// (threadIdx.x & 31) and writes on lane 0 works unchanged.
//
// On CUDA these forward to the native _sync intrinsics with a full 32-lane mask.
// ---------------------------------------------------------------------------

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)

__device__ __forceinline__ uint32_t WarpBallot32(bool pred)
{
    const unsigned long long b = __ballot(pred);
    const unsigned shift       = (threadIdx.x & 32u); // 0 or 32
    return static_cast<uint32_t>((b >> shift) & 0xFFFFFFFFull);
}

template<typename T>
__device__ __forceinline__ T WarpShfl32(T val, int src_lane)
{
    return __shfl(val, src_lane, 32);
}

template<typename T>
__device__ __forceinline__ T WarpShflXor32(T val, int lane_mask)
{
    return __shfl_xor(val, lane_mask, 32);
}

template<typename T>
__device__ __forceinline__ T WarpShflUp32(T val, unsigned int delta)
{
    return __shfl_up(val, delta, 32);
}

#else // CUDA

__device__ __forceinline__ uint32_t WarpBallot32(bool pred)
{
    return __ballot_sync(~0u, pred);
}

template<typename T>
__device__ __forceinline__ T WarpShfl32(T val, int src_lane)
{
    return __shfl_sync(~0u, val, src_lane, 32);
}

template<typename T>
__device__ __forceinline__ T WarpShflXor32(T val, int lane_mask)
{
    return __shfl_xor_sync(~0u, val, lane_mask, 32);
}

template<typename T>
__device__ __forceinline__ T WarpShflUp32(T val, unsigned int delta)
{
    return __shfl_up_sync(~0u, val, delta, 32);
}

#endif
