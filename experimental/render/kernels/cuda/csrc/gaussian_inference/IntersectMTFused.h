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

#include <ATen/core/Tensor.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

#include "InferenceTypes.h"

namespace higs {
class MTOffsets;
}

// Fused macro-tile intersection + rasterization pipeline.
//
// execute() runs the macro-tile intersection stages:
//   count → chunk-base scan + offsets → fill → sort
//
// rasterize() runs the fused rasterization kernels:
//   macro_tile_rasterize → macro_tile_post_blend
//
// Uses FUSED_MACRO_TILE_WIDTH/HEIGHT constants for macro-tile dimensions.
class IntersectMTFused : public IIntersectionStage
{
public:
    IntersectMTFused();
    ~IntersectMTFused() override;

    // Runs macro-tile intersection stages (count, offsets, fill, sort).
    void execute(const at::Tensor &means2d, const at::Tensor &depths, const at::Tensor &conics,
                 const at::Tensor &visible, int32_t tile_size, int32_t tile_width, int32_t tile_height,
                 cudaStream_t stream) override;

    // Runs fused rasterize + post-blend kernels using the intersection results from execute().
    // Must be called after execute() on the same frame.
    void rasterize(const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
                   const at::Tensor &backgrounds, uint32_t image_width, uint32_t image_height,
                   at::Tensor &render_colors);

private:
    // Macro-tile intersection buffers
    at::Tensor m_mtGaussCounts;                     // [n_mt] int32 — per-macro-tile gaussian hit counts
    at::Tensor m_mtGaussOffsets;                    // [n_mt+1] int32 — exclusive prefix sum of counts
    at::Tensor m_mtChunkMetaWords;                  // [packed rows] int32 words containing uint16 counts / local bases
    at::Tensor m_mtChunkTileSums;                   // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 chunk-tile sums
    at::Tensor m_mtChunkTileCarry;                  // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 chunk-tile carries
    at::Tensor m_mtDepthKeys;                       // [n_mt_isects] int32 — depth as bit-reinterpreted int32
    at::Tensor m_mtGaussIds;                        // [n_mt_isects] int32 — gaussian indices
    at::Tensor m_mtDepthKeysSorted;                 // [n_mt_isects] int32 — sorted depth keys
    at::Tensor m_mtGaussIdsSorted;                  // [n_mt_isects] int32 — gaussian IDs sorted by depth per macro-tile
    at::Tensor m_mtBatchOffsets;                    // [n_mt+1] int32 — gaussian batch start indices per macro-tile
    int64_t m_mtIsectCapacity    = 0;               // high-water-mark for m_mtDepthKeys/m_mtGaussIds allocation
    int64_t m_mtSortedCapacity   = 0;               // high-water-mark for m_mtDepthKeysSorted/m_mtGaussIdsSorted
    int64_t m_nMacroIsects       = 0;               // total macro-tile intersections (from offsets readback)
    int32_t m_macroTileCols      = 0;               // macro-tile grid width
    int32_t m_nMacroTiles        = 0;               // total macro-tiles (cols × rows)
    int32_t m_numRasterBatches   = 0;               // total CTA batches across all macro-tiles
    int32_t m_tileSize           = 0;               // render-tile size in pixels (8 or 16)
    int32_t m_tileWidth          = 0;               // render-tile grid width
    int32_t m_tileHeight         = 0;               // render-tile grid height
    int32_t m_chunkMetaWordsSize = 0;               // number of packed int32 metadata words
    int32_t m_chunkTileCarrySize = 0;               // number of int32 carry entries
    int32_t m_numChunks          = 0;               // number of gaussian chunks in the current frame
    int32_t m_numChunkTiles      = 0;               // number of chunk tiles in the current frame
    int32_t m_numMtTiles         = 0;               // number of macro-tile metadata tiles in the current frame
    std::unique_ptr<higs::MTOffsets> m_mtOffsets; // persistent mt offset scan helper

    // Cached scratch buffers for launch_mt_chunk_bases (avoids 2 x at::empty per frame)
    at::Tensor m_chunkBlockSums;                    // [n_chunk_blocks * n_mt_tiles * MT_META_TILE] int32
    at::Tensor m_chunkBlockPrefixes;                // [n_chunk_blocks * n_mt_tiles * MT_META_TILE] int32
    int64_t m_chunkBlockScratchCapacity = 0;        // high-water-mark for block_sums/block_prefixes

    // Fused rasterize buffers (written by macro_tile_rasterize_kernel, read by post_blend)
    at::Tensor m_tileBuffer;             // [N] half — fixed-layout partial RGBT per (cta, mt-slot)
    at::Tensor m_activeMask;             // [total_ctas] uint32 — per-CTA 32-bit tile overlap mask
    int64_t m_tileBufferCapacity    = 0; // high-water-mark for m_tileBuffer (in half elements)
    int64_t m_metadataCapacity      = 0; // high-water-mark for m_activeMask

};
