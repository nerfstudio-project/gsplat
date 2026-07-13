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
// Number of render-tiles in the x dimension of a macro-tile.
inline constexpr int32_t MACRO_TILE_WIDTH         = 8;
inline constexpr int32_t MACRO_TILE_HEIGHT        = 4;
inline constexpr int32_t RT_GAUSS_BATCH_SIZE      = 128;
inline constexpr int32_t RT_GAUSSIANS_PER_MASK    = 32;
inline constexpr int32_t RT_MASKS_PER_GAUSS_BATCH = RT_GAUSS_BATCH_SIZE / RT_GAUSSIANS_PER_MASK;

static_assert(RT_GAUSSIANS_PER_MASK <= 32, "RT_GAUSSIANS_PER_MASK must fit in a uint32_t bitmask");

static_assert(
    RT_GAUSS_BATCH_SIZE % RT_GAUSSIANS_PER_MASK == 0, "RT_GAUSS_BATCH_SIZE must be a multiple of RT_GAUSSIANS_PER_MASK"
);
} // namespace gsplat
