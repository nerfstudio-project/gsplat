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

// IntersectMTFused: fused macro-tile intersection + rasterization pipeline.
// Uses FUSED_MACRO_TILE_WIDTH/HEIGHT for macro-tile dimensions.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

#include "Constants.h"
#include "IntersectCommon.h"
#include "MacroTileIntersect.h"
#include "IntersectMTConfig.h"
#include "IntersectMTFused.h"
#include "MacroTileRasterize.h"

namespace higs {

// ============================================================
// Thin __global__ wrappers using FUSED_MACRO_TILE_WIDTH/HEIGHT
// ============================================================

// Macro-tile binning kernel for fused pipeline.
// Thin wrapper around mt_binning_device_impl using FUSED_MACRO_TILE_WIDTH/HEIGHT.
// See IntersectCommon.h for the fused count -> chunk-base scan -> single-pass fill flow.
template<TileBinMode MODE, int32_t TILE_SIZE>
__global__ void __launch_bounds__(MT_CTA_THREADS, MT_MIN_BLOCKS)
    fused_mt_binning_kernel(const int32_t N,                                 // total gaussian count
                            const float *__restrict__ means2d,               // [N, 2] gaussian 2D positions
                            const uint32_t *__restrict__ visible,            // [ceil(N/32)] visibility bitmask
                            const float *__restrict__ depths,                // [N] gaussian depths (FILL only)
                            const __half *__restrict__ conics,               // [N, 4] half {l0,l1,l2,opacity}
                            const int32_t macro_tile_cols,                   // macro-tile grid width
                            const int32_t macro_tile_rows,                   // macro-tile grid height
                            const int32_t *__restrict__ mt_gauss_offsets,    // [n_mt+1] FILL in: exclusive prefix sum
                            int32_t *__restrict__ mt_depth_keys,             // [n_mt_isects] FILL out: depth as int32
                            int32_t *__restrict__ mt_gauss_ids,              // [n_mt_isects] FILL out: gaussian index
                            int32_t *__restrict__ mt_chunk_meta_words,       // [packed rows] COUNT/FILL metadata
                            const int32_t *__restrict__ mt_chunk_tile_carry, // [chunk_tile, mt] carry prefixes
                            const int32_t n_mt_tiles)                        // number of macro-tile metadata tiles
{
    mt_binning_device_impl<MODE, TILE_SIZE, FUSED_MACRO_TILE_WIDTH, FUSED_MACRO_TILE_HEIGHT>(
        N, means2d, visible, depths, conics, macro_tile_cols, macro_tile_rows, mt_gauss_offsets, mt_depth_keys,
        mt_gauss_ids, mt_chunk_meta_words, mt_chunk_tile_carry, n_mt_tiles);
}

// ============================================================
// Host launch helpers
// ============================================================

static void launch_fused_mt_binning_impl(TileBinMode mode, int32_t N, int32_t tile_size, int32_t macro_tile_cols,
                                         int32_t macro_tile_rows, const float *means2d, const uint32_t *visible,
                                         const float *depths, const __half *conics, const int32_t *mt_gauss_offsets,
                                         int32_t *mt_depth_keys, int32_t *mt_gauss_ids, int32_t *mt_chunk_meta_words,
                                         const int32_t *mt_chunk_tile_carry, int32_t n_mt_tiles)
{
    if (N == 0)
    {
        return;
    }

    const int32_t n_macro_tiles = macro_tile_cols * macro_tile_rows;
    const int32_t n_blocks      = (N + MT_CHUNK_SIZE - 1) / MT_CHUNK_SIZE;
    const bool is_fill          = (mode == TileBinMode::FILL);

    // COUNT: 1 smem array (chunk_gauss_counts), FILL: 2 (+ chunk_base_idx)
    const size_t smem_bytes = (is_fill ? 2 : 1) * n_macro_tiles * sizeof(int32_t);

    auto launch = [&](auto kernel_fn) {
        if (smem_bytes > 48 * 1024 &&
            cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes) != cudaSuccess)
        {
            AT_ERROR("Failed to set shared memory size for fused macro-tile binning kernel (requested ", smem_bytes,
                     " bytes, ", n_macro_tiles, " macro-tiles)");
        }
        kernel_fn<<<n_blocks, MT_CTA_THREADS, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
            N, means2d, visible, depths, conics, macro_tile_cols, macro_tile_rows, mt_gauss_offsets, mt_depth_keys,
            mt_gauss_ids, mt_chunk_meta_words, mt_chunk_tile_carry, n_mt_tiles);
    };

    if (is_fill)
    {
        switch (tile_size)
        {
        case 16:
            launch(fused_mt_binning_kernel<TileBinMode::FILL, 16>);
            break;
        case 8:
            launch(fused_mt_binning_kernel<TileBinMode::FILL, 8>);
            break;
        default:
            TORCH_CHECK(false, "tile_size must be 8 or 16, got ", tile_size);
            break;
        }
    }
    else
    {
        switch (tile_size)
        {
        case 16:
            launch(fused_mt_binning_kernel<TileBinMode::COUNT, 16>);
            break;
        case 8:
            launch(fused_mt_binning_kernel<TileBinMode::COUNT, 8>);
            break;
        default:
            TORCH_CHECK(false, "tile_size must be 8 or 16, got ", tile_size);
            break;
        }
    }
}

} // namespace higs

// ============================================================
// IntersectMTFused implementation
// ============================================================

IntersectMTFused::IntersectMTFused() = default;
IntersectMTFused::~IntersectMTFused() = default;

// Runs macro-tile intersection stages for the fused pipeline.
// Dispatch order:
//   1. mt count       — chunk-local histograms flushed to packed metadata rows
//   2. mt chunk scan  — packed uint16 local bases + chunk-tile carries
//   3. mt offsets     — fused prefix sums: mt_gauss_offsets + mt_batch_offsets
//   4. mt fill        — single-pass write of (depth_key, gauss_id) pairs
//   5. mt sort        — segmented radix sort by depth within each macro-tile
void IntersectMTFused::execute(const at::Tensor &means2d, const at::Tensor &depths, const at::Tensor &conics,
                               const at::Tensor &visible, int32_t tile_size, int32_t tile_width, int32_t tile_height,
                               cudaStream_t stream)
{
    using namespace higs;
    auto ceilDivInt = [](int32_t x, int32_t y) -> int32_t { return (x + y - 1) / y; };

    static constexpr int32_t MTW  = FUSED_MACRO_TILE_WIDTH;
    static constexpr int32_t MTH  = FUSED_MACRO_TILE_HEIGHT;
    m_macroTileCols               = (tile_width + MTW - 1) / MTW;
    const int32_t macro_tile_rows = (tile_height + MTH - 1) / MTH;
    m_nMacroTiles                 = m_macroTileCols * macro_tile_rows;
    m_tileSize                    = tile_size;
    m_tileWidth                   = tile_width;
    m_tileHeight                  = tile_height;

    const int32_t N = means2d.size(-2);
    auto opts_i32   = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

    m_numChunks          = ceilDivInt(N, MT_CHUNK_SIZE);
    m_numMtTiles         = ceilDivInt(m_nMacroTiles, MT_META_TILE);
    m_numChunkTiles      = ceilDivInt(m_numChunks, MT_CHUNK_TILE);
    m_chunkMetaWordsSize = m_numChunkTiles * m_numMtTiles * MT_CHUNK_TILE * MT_META_WORDS;
    m_chunkTileCarrySize = m_numChunkTiles * m_numMtTiles * MT_META_TILE;

    // Ensure macro-tile buffers are allocated for current grid size
    if (!m_mtGaussOffsets.defined() || m_mtGaussOffsets.size(0) != (int64_t)(m_nMacroTiles + 1))
    {
        m_mtGaussCounts   = at::zeros({(int64_t)m_nMacroTiles}, opts_i32);
        m_mtGaussOffsets  = at::zeros({(int64_t)m_nMacroTiles + 1}, opts_i32);
        m_mtBatchOffsets  = at::zeros({(int64_t)m_nMacroTiles + 1}, opts_i32);
        m_mtIsectCapacity = 0;
        static_assert((FUSED_GAUSS_BATCH_SIZE & (FUSED_GAUSS_BATCH_SIZE - 1)) == 0,
                      "FUSED_GAUSS_BATCH_SIZE must be a power of 2");
        m_mtOffsets = std::make_unique<higs::MTOffsets>(m_nMacroTiles, CEIL_LOG2<FUSED_GAUSS_BATCH_SIZE>);
    }

    if (!m_mtChunkMetaWords.defined() || m_mtChunkMetaWords.size(0) != (int64_t)m_chunkMetaWordsSize)
    {
        m_mtChunkMetaWords = at::empty({(int64_t)m_chunkMetaWordsSize}, opts_i32);
    }

    if (!m_mtChunkTileCarry.defined() || m_mtChunkTileCarry.size(0) != (int64_t)m_chunkTileCarrySize)
    {
        m_mtChunkTileCarry = at::empty({(int64_t)m_chunkTileCarrySize}, opts_i32);
    }

    if (!m_mtChunkTileSums.defined() || m_mtChunkTileSums.size(0) != (int64_t)m_chunkTileCarrySize)
    {
        m_mtChunkTileSums = at::empty({(int64_t)m_chunkTileCarrySize}, opts_i32);
    }

    // Stage 1: macro-tile count
    launch_fused_mt_binning_impl(TileBinMode::COUNT, N, tile_size, m_macroTileCols, macro_tile_rows,
                                 means2d.data_ptr<float>(),
                                 reinterpret_cast<const uint32_t *>(visible.data_ptr<int32_t>()), nullptr,
                                 reinterpret_cast<const __half *>(conics.data_ptr<at::Half>()), nullptr, nullptr,
                                 nullptr, m_mtChunkMetaWords.data_ptr<int32_t>(), nullptr, m_numMtTiles);

    // Stage 2: macro-tile chunk-base scan
    launch_mt_chunk_bases(m_mtChunkMetaWords, m_mtChunkTileSums, m_mtChunkTileCarry, m_mtGaussCounts, m_nMacroTiles,
                          m_numChunks,
                          &m_chunkBlockSums, &m_chunkBlockPrefixes, &m_chunkBlockScratchCapacity);

    // Stage 3: macro-tile offsets (gauss + batch prefix sums from counts)
    m_mtOffsets->execute(m_mtGaussCounts, m_mtGaussOffsets, m_mtBatchOffsets);

    m_nMacroIsects = m_mtGaussOffsets[m_nMacroTiles].item<int64_t>();

    if (m_nMacroIsects > 0)
    {
        if (m_nMacroIsects > m_mtIsectCapacity)
        {
            m_mtDepthKeys     = at::empty({m_nMacroIsects}, opts_i32);
            m_mtGaussIds      = at::empty({m_nMacroIsects}, opts_i32);
            m_mtIsectCapacity = m_nMacroIsects;
        }
        if (m_nMacroIsects > m_mtSortedCapacity)
        {
            m_mtDepthKeysSorted = at::empty({m_nMacroIsects}, opts_i32);
            m_mtGaussIdsSorted  = at::empty({m_nMacroIsects}, opts_i32);
            m_mtSortedCapacity  = m_nMacroIsects;
        }

        // Stage 4: Macro-tile fill
        launch_fused_mt_binning_impl(
            TileBinMode::FILL, N, tile_size, m_macroTileCols, macro_tile_rows, means2d.data_ptr<float>(),
            reinterpret_cast<const uint32_t *>(visible.data_ptr<int32_t>()), depths.data_ptr<float>(),
            reinterpret_cast<const __half *>(conics.data_ptr<at::Half>()), m_mtGaussOffsets.data_ptr<int32_t>(),
            m_mtDepthKeys.data_ptr<int32_t>(), m_mtGaussIds.data_ptr<int32_t>(), m_mtChunkMetaWords.data_ptr<int32_t>(),
            m_mtChunkTileCarry.data_ptr<int32_t>(), m_numMtTiles);

        // Stage 5: Macro-tile segmented sort
        launch_mt_segmented_sort(m_nMacroIsects, m_nMacroTiles, m_mtGaussOffsets, m_mtDepthKeys, m_mtGaussIds,
                                 m_mtDepthKeysSorted, m_mtGaussIdsSorted);
        m_numRasterBatches = m_mtBatchOffsets[-1].item<int32_t>();
    }
    else
    {
        m_numRasterBatches = 0;
    }
}

void IntersectMTFused::rasterize(const at::Tensor &means2d, const at::Tensor &conics, const at::Tensor &colors,
                                 const at::Tensor &backgrounds, uint32_t image_width, uint32_t image_height,
                                 at::Tensor &render_colors)
{
    using namespace higs;

    auto opts_i32 = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);

    // per-batch metadata (high-water-mark; first frame or capacity growth only)
    const int32_t num_batches = max(m_numRasterBatches, 1);
    if (num_batches > m_metadataCapacity)
    {
        m_activeMask       = at::zeros({num_batches}, opts_i32);
        m_metadataCapacity = num_batches;
    }

    // worst-case tile buffer: each batch can write up to MACRO_TILE_SIZE tiles
    constexpr int32_t MACRO_TILE_SIZE = FUSED_MACRO_TILE_WIDTH * FUSED_MACRO_TILE_HEIGHT;
    const int64_t max_tiles           = max(static_cast<int64_t>(m_numRasterBatches) * MACRO_TILE_SIZE, static_cast<int64_t>(1));
    const int64_t max_tile_buf_size   = max_tiles * m_tileSize * m_tileSize * 4;
    if (max_tile_buf_size > m_tileBufferCapacity)
    {
        m_tileBuffer         = at::empty({max_tile_buf_size}, at::TensorOptions().dtype(at::kHalf).device(at::kCUDA));
        m_tileBufferCapacity = max_tile_buf_size;
    }

    // Fused macro-tile rasterize kernel (skipped when no intersections)
    if (m_nMacroIsects > 0 && m_numRasterBatches > 0)
    {
        launch_macro_tile_rasterize(m_numRasterBatches, m_nMacroTiles, m_macroTileCols, m_tileSize, m_tileWidth,
                                    m_tileHeight, means2d, conics, colors, m_mtGaussOffsets, m_mtGaussIdsSorted,
                                    m_mtBatchOffsets, m_tileBuffer, m_activeMask);
    }

    // Post-blend: composites partial results → final image (always runs; renders background when no gaussians)
    launch_macro_tile_post_blend(m_tileWidth, m_tileHeight, m_tileSize, m_nMacroTiles, m_macroTileCols,
                                 m_mtBatchOffsets, m_tileBuffer, m_activeMask, backgrounds, image_width, image_height,
                                 render_colors);
}
