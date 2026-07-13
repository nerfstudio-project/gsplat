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
