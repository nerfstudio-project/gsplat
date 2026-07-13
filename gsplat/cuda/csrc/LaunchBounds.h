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
// Per-architecture hardware cap on thread blocks per SM. ptxas rejects
// min_blocks_per_sm > HW cap under --warning-as-error.
//   sm_90, sm_100, sm_120: 32 blocks/SM
//   everything else:       16 blocks/SM (covers Ampere/Ada and anything older)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
inline constexpr uint32_t GSPLAT_ARCH_MAX_BLOCKS_PER_SM = 32;
#else
inline constexpr uint32_t GSPLAT_ARCH_MAX_BLOCKS_PER_SM = 16;
#endif

constexpr uint32_t arch_limited_blocks_per_sm(uint32_t max_blocks)
{
    return (GSPLAT_ARCH_MAX_BLOCKS_PER_SM < max_blocks) ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM : max_blocks;
}
} // namespace gsplat
