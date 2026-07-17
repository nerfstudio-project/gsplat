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

namespace gsplat
{
// Shuffle-direction policies for the shared warp tree reduction. ShflDown
// converges each result into lane 0 — the shape for partials on their way to
// a single atomic flush. ShflXor is the butterfly form, leaving each result
// in every lane — reduce-and-broadcast in one tree, for row statistics that
// all lanes consume afterwards. Both cost the same shuffle count; the choice
// is the consumer's, not a performance trade.
struct ShflDown
{
    template<int Width, typename mask_t, typename T>
    __device__ __forceinline__ T operator()(mask_t full_mask, T value, int offset) const
    {
        return __shfl_down_sync(full_mask, value, offset, Width);
    }
};

struct ShflXor
{
    template<int Width, typename mask_t, typename T>
    __device__ __forceinline__ T operator()(mask_t full_mask, T value, int offset) const
    {
        return __shfl_xor_sync(full_mask, value, offset, Width);
    }
};

// Combine policies. Max keeps the comparison written exactly as the scalar
// loops it replaces (`other > value` picks other), so NaN propagation matches
// the previously unshared code bit for bit.
struct SumCombine
{
    template<typename T>
    __device__ __forceinline__ T operator()(T value, T other) const
    {
        return value + other;
    }
};

struct MaxCombine
{
    template<typename T>
    __device__ __forceinline__ T operator()(T value, T other) const
    {
        return other > value ? other : value;
    }
};

// The classic halving-offset warp/wave tree shared by every reduction below.
// full_mask must contain every lane of the reduction group, including lane
// 0; subset masks are unsupported. Every lane must participate convergently.
// Variadic accumulators reduce in one interleaved tree; each value's fold
// order is identical to reducing it alone.
//
// WaveSize is the LOGICAL reduction width, fixed at compile time so a
// caller's lane layout and its tree provably share one constant (and the
// unroll is guaranteed by the language). The default is 32 because every
// consumer in this codebase lays out its lanes and masks by a literal 32:
// on a 64-lane HIP wave, a WaveSize-32 tree reduces each 32-lane half
// independently (xor/down offsets <= 16 never cross the halves, and HIP
// ignores shuffle masks), which is exactly the grouping those layouts
// assume. Do NOT use the native warpSize here - an adaptive width would
// silently mix two 32-lane groups on wave64 hardware.
template<int WaveSize = 32, typename Shfl, typename Combine, typename mask_t, typename... Ts>
__device__ __forceinline__ void warp_tree_reduce(Shfl shfl, Combine combine, mask_t full_mask, Ts &...values)
{
    static_assert(WaveSize > 0 && (WaveSize & (WaveSize - 1)) == 0, "WaveSize must be a power of two");
    // The mask must be wide enough to name every lane of the reduction group;
    // this couples the mask width to the reduction width (a wave64 group needs
    // a 64-bit mask, not a 32-bit one).
    static_assert(
        sizeof(mask_t) * 8 >= static_cast<unsigned>(WaveSize), "the shuffle mask must have at least WaveSize bits"
    );
#ifndef USE_ROCM
    // CUDA build: the physical warp is always 32 lanes. A non-32 WaveSize would
    // shuffle across nonexistent lanes (an offset >= 32, or a sub-warp shuffle
    // width past the warp, reads the caller's own value) and silently double
    // sums. A ROCm build (-DUSE_ROCM) targets its wavefront and is exempt; the
    // mask-width assert above still holds there.
    static_assert(WaveSize == 32, "non-32 WaveSize on a CUDA build corrupts reductions");
#endif
#pragma unroll
    for(int offset = WaveSize / 2; offset > 0; offset >>= 1)
    {
        // WaveSize is passed as the shuffle width (a compile-time constant, so
        // this is codegen-identical to the default on CUDA) so the shuffle
        // wraps within the logical group rather than the native warp/wave.
        ((values = combine(values, shfl.template operator()<WaveSize>(full_mask, values, offset))), ...);
    }
}

// Warp/wave sum reduction; lane 0 of each WaveSize group receives each sum.
template<int WaveSize = 32, typename mask_t, typename... Ts>
__device__ __forceinline__ void warp_reduce_sum_all(mask_t full_mask, Ts &...values)
{
    warp_tree_reduce<WaveSize>(ShflDown{}, SumCombine{}, full_mask, values...);
}

// Butterfly sum/max: every lane receives each result (no separate broadcast).
template<int WaveSize = 32, typename mask_t, typename... Ts>
__device__ __forceinline__ void warp_reduce_sum_all_lanes(mask_t full_mask, Ts &...values)
{
    warp_tree_reduce<WaveSize>(ShflXor{}, SumCombine{}, full_mask, values...);
}

template<int WaveSize = 32, typename mask_t, typename... Ts>
__device__ __forceinline__ void warp_reduce_max_all_lanes(mask_t full_mask, Ts &...values)
{
    warp_tree_reduce<WaveSize>(ShflXor{}, MaxCombine{}, full_mask, values...);
}

template<typename scalar_t, typename mask_t>
__device__ __forceinline__ scalar_t warp_sum(scalar_t value, mask_t full_mask)
{
    warp_reduce_sum_all(full_mask, value);
    return value;
}

// Identity key projection for lower_bound.
struct IdentityKey
{
    template<typename T>
    __device__ __forceinline__ T operator()(T v) const
    {
        return v;
    }
};

// First index in [0, count) whose projected key is not less than query, over
// data[start .. start + count) (sorted by the projected key). The classic
// lower_bound tree; Key projects a stored element to its comparison key
// (identity by default), so packed-key searches can pass an extractor.
template<typename T, typename Query, typename Key = IdentityKey>
__device__ __forceinline__ int64_t
    lower_bound(const T *__restrict__ data, int64_t start, int64_t count, Query query, Key key = {})
{
    int64_t lo = 0;
    int64_t hi = count;
    while(lo < hi)
    {
        const int64_t mid = lo + ((hi - lo) >> 1);
        if(key(data[start + mid]) < query)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

template<typename mask_t>
__device__ __forceinline__ uint32_t warp_mask_popcount(mask_t mask)
{
    static_assert(sizeof(mask_t) == sizeof(uint32_t) || sizeof(mask_t) == sizeof(uint64_t));
    if constexpr(sizeof(mask_t) == sizeof(uint32_t))
    {
        return static_cast<uint32_t>(__popc(static_cast<unsigned>(mask)));
    }
    else
    {
        return static_cast<uint32_t>(__popcll(static_cast<unsigned long long>(mask)));
    }
}

// CTA-wide barrier. Single-warp CTAs can use the cheaper warp sync; larger CTAs
// need a true block barrier.
template<uint32_t CTA_SIZE_T>
__device__ __forceinline__ void cta_sync()
{
    if constexpr(CTA_SIZE_T <= 32)
    {
        __syncwarp();
    }
    else
    {
        __syncthreads();
    }
}
} // namespace gsplat
