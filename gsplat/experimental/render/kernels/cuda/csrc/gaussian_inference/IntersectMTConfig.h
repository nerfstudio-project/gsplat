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

namespace higs {

// Fused macro-tile dimensions
static constexpr int32_t FUSED_MACRO_TILE_WIDTH  = 8;
static constexpr int32_t FUSED_MACRO_TILE_HEIGHT = 4;
static constexpr int32_t FUSED_GAUSS_BATCH_SIZE  = 1024;

static constexpr int32_t MT_CHUNK_SIZE = 8192;
static constexpr int32_t MT_META_TILE  = 64;
static constexpr int32_t MT_META_WORDS = MT_META_TILE / 2;
static constexpr int32_t MT_CHUNK_TILE = 65535 / MT_CHUNK_SIZE + 1;

static_assert(MT_META_TILE % 2 == 0, "metadata tile must pack an even number of uint16 entries");
static_assert(MT_CHUNK_SIZE <= 65535, "chunk-local counts must fit in uint16_t");
static_assert((MT_CHUNK_TILE - 1) * MT_CHUNK_SIZE <= 65535, "chunk-local base must fit in uint16_t");

} // namespace higs
