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
// Native warp/wave sum reduction. full_mask must contain every lane in the
// native warp/wave, including lane 0; subset masks are unsupported. Every lane
// must participate convergently, and lane 0 receives each sum.
template<typename mask_t, typename... Ts>
__device__ __forceinline__ void warp_reduce_sum_all(mask_t full_mask, Ts &...values)
{
#pragma unroll
    for(int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        ((values += __shfl_down_sync(full_mask, values, offset)), ...);
    }
}

template<typename scalar_t, typename mask_t>
__device__ __forceinline__ scalar_t warp_sum(scalar_t value, mask_t full_mask)
{
    warp_reduce_sum_all(full_mask, value);
    return value;
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
