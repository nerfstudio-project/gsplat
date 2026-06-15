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

// Shared device code and data structures for macro-tile intersection pipelines.
// Used by both IntersectMT.cu (standard pipeline) and IntersectMTFused.cu (fused pipeline).
// Contains __device__ functions and plain structs — no __global__ kernels.

#pragma once

#include <ATen/core/Tensor.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "Utils.h"

#include "Constants.h"
#include "IntersectMTConfig.h"

namespace higs {

// compile-time log2
template<uint32_t N>
inline constexpr uint32_t FLOOR_LOG2 = N <= 1 ? 0 : 1 + FLOOR_LOG2<N / 2>;

template<uint32_t N>
inline constexpr uint32_t CEIL_LOG2 = N <= 1 ? 0 : FLOOR_LOG2<N - 1> + 1;

// ============================================================
// Shared constants for macro-tile binning kernels
// ============================================================

enum class TileBinMode : uint32_t
{
    COUNT,
    FILL
};

static constexpr int32_t MT_CTA_THREADS = 768; // threads per CTA for macro-tile binning
static constexpr int32_t MT_MIN_BLOCKS  = 2;   // minimum number of blocks for macro-tile binning

// ============================================================
// Device helpers
// ============================================================

struct GaussianConic
{
    float A, B, C, disc, t;
    float2 center, bbox_min, bbox_max, bbox_argmin, bbox_argmax;
};

__device__ inline bool decode_gaussian_conic(const __half *__restrict__ conics, const float *__restrict__ means2d,
                                             int32_t idx, GaussianConic &gc)
{
    __half conics_opac[4];
    AssignAs<uint2>(conics_opac, conics[idx * 4]);
    const float l0      = __half2float(conics_opac[0]);
    const float l1      = __half2float(conics_opac[1]);
    const float l2      = __half2float(conics_opac[2]);
    const float opacity = __half2float(conics_opac[3]);

    gc.A = l0 * l0;
    gc.B = l0 * l1;
    gc.C = l1 * l1 + l2 * l2;

    gc.disc = -(l0 * l2) * (l0 * l2);
    gc.t    = min(MAX_EXTEND * MAX_EXTEND, 2.0f * __logf(opacity * INV_ALPHA_THRESHOLD));

    const float neg_t_over_disc = -gc.t / gc.disc;
    const float x_extent        = sqrtf(neg_t_over_disc * gc.C);
    const float y_extent        = sqrtf(neg_t_over_disc * gc.A);

    AssignAs<float2>(gc.center, means2d[idx * 2]);
    gc.bbox_min = {gc.center.x - x_extent, gc.center.y - y_extent};
    gc.bbox_max = {gc.center.x + x_extent, gc.center.y + y_extent};

    const float Bx_over_C = gc.B * x_extent / gc.C;
    const float By_over_A = gc.B * y_extent / gc.A;
    gc.bbox_argmin        = {gc.center.y + Bx_over_C, gc.center.x + By_over_A};
    gc.bbox_argmax        = {gc.center.y - Bx_over_C, gc.center.x - By_over_A};

    const float dx = gc.bbox_max.x - gc.bbox_min.x;
    const float dy = gc.bbox_max.y - gc.bbox_min.y;
    if (dx <= 0.f || dy <= 0.f)
    {
        return false;
    }
    return true;
}

template<typename EmitFn>
__device__ inline int32_t process_tiles_macro(float A, float B, float C, float disc, float t, float2 p, float2 bbox_min,
                                              float2 bbox_max, float2 bbox_argmin, float2 bbox_argmax, int2 rect_min,
                                              int2 rect_max, int32_t tile_size_x, int32_t tile_size_y,
                                              int32_t grid_cols, bool isY, EmitFn emit)
{
    auto compute_ellipse_intersection = [](float A, float B, float C, float disc, float t, float2 p, bool isY,
                                           float coord) -> float2 {
        const float p_u   = isY ? p.y : p.x;
        const float p_v   = isY ? p.x : p.y;
        const float coeff = isY ? A : C;

        const float h         = coord - p_u;
        const float sqrt_term = sqrtf(disc * h * h + t * coeff);

        return {(-B * h - sqrt_term) / coeff + p_v, (-B * h + sqrt_term) / coeff + p_v};
    };

    const float BLOCK_X = (float)tile_size_x;
    const float BLOCK_Y = (float)tile_size_y;

    float BLOCK_U, BLOCK_V;
    if (isY)
    {
        rect_min    = {rect_min.y, rect_min.x};
        rect_max    = {rect_max.y, rect_max.x};
        bbox_min    = {bbox_min.y, bbox_min.x};
        bbox_max    = {bbox_max.y, bbox_max.x};
        bbox_argmin = {bbox_argmin.y, bbox_argmin.x};
        bbox_argmax = {bbox_argmax.y, bbox_argmax.x};
        BLOCK_U     = BLOCK_Y;
        BLOCK_V     = BLOCK_X;
    }
    else
    {
        BLOCK_U = BLOCK_X;
        BLOCK_V = BLOCK_Y;
    }

    int32_t tiles_count = 0;
    float2 intersect_min_line, intersect_max_line;
    float ellipse_min, ellipse_max;
    float min_line, max_line;

    intersect_max_line = {bbox_max.y, bbox_min.y};

    min_line = rect_min.x * BLOCK_U;
    if (bbox_min.x <= min_line)
    {
        intersect_min_line = compute_ellipse_intersection(A, B, C, disc, t, p, isY, min_line);
    }
    else
    {
        intersect_min_line = intersect_max_line;
    }

#pragma unroll 1
    for (int u = rect_min.x; u < rect_max.x; ++u)
    {
        max_line = min_line + BLOCK_U;
        if (max_line <= bbox_max.x)
        {
            intersect_max_line = compute_ellipse_intersection(A, B, C, disc, t, p, isY, max_line);
        }

        if (min_line <= bbox_argmin.y && bbox_argmin.y < max_line)
        {
            ellipse_min = bbox_min.y;
        }
        else
        {
            ellipse_min = min(intersect_min_line.x, intersect_max_line.x);
        }

        if (min_line <= bbox_argmax.y && bbox_argmax.y < max_line)
        {
            ellipse_max = bbox_max.y;
        }
        else
        {
            ellipse_max = max(intersect_min_line.y, intersect_max_line.y);
        }

        const int min_tile_v = max(rect_min.y, min(rect_max.y, (int)(ellipse_min / BLOCK_V)));
        const int max_tile_v = min(rect_max.y, max(rect_min.y, (int)(ellipse_max / BLOCK_V + 1)));

#pragma unroll 1
        for (int v = min_tile_v; v < max_tile_v; v++)
        {
            const int32_t tile_id = isY ? (u * grid_cols + v) : (v * grid_cols + u);
            emit(tile_id);
        }
        tiles_count += max_tile_v - min_tile_v;

        intersect_min_line = intersect_max_line;
        min_line           = max_line;
    }
    return tiles_count;
}

// Warp-cooperative 32-way binary search: all 32 lanes probe different points
// each round, narrowing by ~33x instead of 2x. Completes in ~2-3 rounds
// for typical macro-tile counts vs ~10-14 scalar iterations.
// Requires all warp threads to participate (callers guarantee this).
__device__ inline int32_t find_macro_tile(const int32_t *offsets, // [n+1] sorted offset array
                                          int32_t n,              // number of segments
                                          int32_t key)            // value to look up
{
    const int32_t lane = threadIdx.x & 31;
    int32_t lo = 0, hi = n;

    // exact a * b / 33 without 64-bit math; b must be in [1, 32].
    // the /33 literals are compile-time constants so the compiler emits
    // multiply-high + shift, not actual division.
    auto div33 = [](int32_t a, int32_t b) -> int32_t {
        const int32_t q = a / 33;
        const int32_t r = a - q * 33;
        return q * b + r * b / 33;
    };

    while (lo < hi)
    {
        const int32_t range   = hi - lo;
        const int32_t probe   = lo + div33(range, lane + 1);
        const bool take_more  = (offsets[probe + 1] <= key);
        const uint32_t ballot = __ballot_sync(~0u, take_more);
        const int32_t n_true  = __popc(ballot);

        // 32 probes create 33 sub-intervals. n_true tells us which one
        // contains the crossover.
        if (n_true == 32)
        {
            lo = lo + div33(range, 32) + 1;
        }
        else if (n_true == 0)
        {
            hi = lo + div33(range, 1);
        }
        else
        {
            hi = lo + div33(range, n_true + 1);
            lo = lo + div33(range, n_true) + 1;
        }
    }
    return lo;
}

// ============================================================
// Macro-tile binning device implementation
// ============================================================
//
// Two-phase pipeline producing per-macro-tile sorted (depth, gaussian_id) lists.
// Hoisted into a __device__ function so each .cu file can wrap it in a thin
// __global__ kernel with its own macro-tile dimensions (MTW × MTH).
//
// Phase 1 — COUNT:
//   Each CTA processes one gaussian chunk. For each gaussian, AccuTile determines
//   which macro-tiles it overlaps and accumulates those hits into a shared-memory
//   histogram, then flushes the chunk histogram to packed uint16 metadata rows.
//
// Between COUNT and FILL, launch_mt_chunk_bases() converts packed uint16 counts
// into packed uint16 chunk-local bases plus int32 chunk-tile carries.
// launch_mt_offsets() then scans the final per-macro-tile counts into
// mt_gauss_offsets / mt_batch_offsets.
//
// Phase 2 — FILL:
//   Loads the precomputed per-(chunk, macro-tile) bases and writes
//   (depth_bits, gauss_id) in a single geometry pass.

template<TileBinMode MODE, int32_t TILE_SIZE, int32_t MTW, int32_t MTH>
__device__ void mt_binning_device_impl(const int32_t N, const float *__restrict__ means2d,
                                       const uint32_t *__restrict__ visible, const float *__restrict__ depths,
                                       const __half *__restrict__ conics, const int32_t macro_tile_cols,
                                       const int32_t macro_tile_rows, const int32_t *__restrict__ mt_gauss_offsets,
                                       int32_t *__restrict__ mt_depth_keys, int32_t *__restrict__ mt_gauss_ids,
                                       int32_t *__restrict__ mt_chunk_meta_words,
                                       const int32_t *__restrict__ mt_chunk_tile_carry, int32_t n_mt_tiles)
{
    const int32_t n_macro_tiles = macro_tile_cols * macro_tile_rows;
    const int32_t chunk_idx     = blockIdx.x;
    extern __shared__ int32_t smem[];
    int32_t *const smem_chunk_gauss_counts = smem;

    if constexpr (MODE == TileBinMode::COUNT)
    {
#pragma unroll 1
        for (int32_t i = threadIdx.x; i < n_macro_tiles; i += blockDim.x)
        {
            smem_chunk_gauss_counts[i] = 0;
        }
        __syncthreads();
    }

    const int32_t chunk_start   = blockIdx.x * MT_CHUNK_SIZE;
    const int32_t chunk_end     = min(chunk_start + MT_CHUNK_SIZE, N);
    const int32_t chunk_tile    = chunk_idx / MT_CHUNK_TILE;
    const int32_t chunk_in_tile = chunk_idx % MT_CHUNK_TILE;

    constexpr int32_t TILE_SIZE_MACRO_X = MTW * TILE_SIZE;
    constexpr int32_t TILE_SIZE_MACRO_Y = MTH * TILE_SIZE;
    constexpr float INV_MTS_X           = 1.0f / (float)TILE_SIZE_MACRO_X;
    constexpr float INV_MTS_Y           = 1.0f / (float)TILE_SIZE_MACRO_Y;

    auto meta_row_word_base = [&](int32_t mt_tile, int32_t row_in_tile) {
        return (chunk_tile * n_mt_tiles + mt_tile) * (MT_CHUNK_TILE * MT_META_WORDS) + row_in_tile * MT_META_WORDS;
    };

    // Common intersection logic: decode gaussian, compute macro-tile overlap, call emit for each hit
    auto for_each_macro_tile_hit = [&](int32_t i, auto emit) {
        if (!(visible[static_cast<uint32_t>(i) >> 5] & (1u << (static_cast<uint32_t>(i) & 0x1fu))))
        {
            return;
        }

        GaussianConic gc;
        if (!decode_gaussian_conic(conics, means2d, i, gc))
        {
            return;
        }

        const int2 rect_min = {max(0, min(macro_tile_cols, (int)(gc.bbox_min.x * INV_MTS_X))),
                               max(0, min(macro_tile_rows, (int)(gc.bbox_min.y * INV_MTS_Y)))};
        const int2 rect_max = {max(0, min(macro_tile_cols, (int)(gc.bbox_max.x * INV_MTS_X + 1.f))),
                               max(0, min(macro_tile_rows, (int)(gc.bbox_max.y * INV_MTS_Y + 1.f)))};

        const int y_span = rect_max.y - rect_min.y;
        const int x_span = rect_max.x - rect_min.x;
        if (y_span * x_span == 0)
        {
            return;
        }

        const bool isY = y_span < x_span;

        process_tiles_macro(gc.A, gc.B, gc.C, gc.disc, gc.t, gc.center, gc.bbox_min, gc.bbox_max, gc.bbox_argmin,
                            gc.bbox_argmax, rect_min, rect_max, TILE_SIZE_MACRO_X, TILE_SIZE_MACRO_Y, macro_tile_cols,
                            isY, emit);
    };

    if constexpr (MODE == TileBinMode::COUNT)
    {
        // Count mode: build a chunk-local histogram and flush it to metadata.
#pragma unroll 1
        for (int32_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x)
        {
            for_each_macro_tile_hit(i, [&](int32_t mt) { atomicAdd(&smem_chunk_gauss_counts[mt], 1); });
        }
        __syncthreads();
#pragma unroll 1
        for (int32_t pair_idx = threadIdx.x; pair_idx < n_mt_tiles * MT_META_WORDS; pair_idx += blockDim.x)
        {
            const int32_t mt_tile      = pair_idx / MT_META_WORDS;
            const int32_t pair_in_tile = pair_idx % MT_META_WORDS;
            const int32_t mt0          = mt_tile * MT_META_TILE + pair_in_tile * 2;
            const int32_t mt1          = mt0 + 1;

            const int32_t count0 = (mt0 < n_macro_tiles) ? smem_chunk_gauss_counts[mt0] : 0;
            const int32_t count1 = (mt1 < n_macro_tiles) ? smem_chunk_gauss_counts[mt1] : 0;

            mt_chunk_meta_words[meta_row_word_base(mt_tile, chunk_in_tile) + pair_in_tile] = count0 | (count1 << 16);
        }
    }
    else
    {
        int32_t *const smem_chunk_base_idx = smem + n_macro_tiles;

#pragma unroll 1
        for (int32_t mt = threadIdx.x; mt < n_macro_tiles; mt += blockDim.x)
        {
            const int32_t mt_tile = mt / MT_META_TILE;
            const int32_t mt_lane = mt % MT_META_TILE;

            const uint32_t packed    = mt_chunk_meta_words[meta_row_word_base(mt_tile, chunk_in_tile) + mt_lane / 2];
            const int32_t local_base = (mt_lane & 1) ? (packed >> 16) : (packed & 0xFFFFu);

            const int32_t carry_idx = (chunk_tile * n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;

            smem_chunk_base_idx[mt]     = mt_gauss_offsets[mt] + mt_chunk_tile_carry[carry_idx] + local_base;
            smem_chunk_gauss_counts[mt] = 0;
        }
        __syncthreads();

        // Fill (depth, gauss_id) pairs to global output
#pragma unroll 1
        for (int32_t i = chunk_start + threadIdx.x; i < chunk_end; i += blockDim.x)
        {
            const int32_t depth_i32 = reinterpret_cast<const int32_t *>(depths)[i];

            for_each_macro_tile_hit(i, [&](int32_t mt) {
                const int32_t local_slot = atomicAdd(&smem_chunk_gauss_counts[mt], 1);
                const int32_t write_pos  = smem_chunk_base_idx[mt] + local_slot;
                mt_depth_keys[write_pos] = depth_i32;
                mt_gauss_ids[write_pos]  = i;
            });
        }
    }
}

// Computes chunk-local bases and chunk-tile carries from packed uint16 count rows.
// Outputs final per-macro-tile gaussian counts for the existing mt offset scan.
void launch_mt_chunk_bases(
    at::Tensor inout_mt_chunk_meta_words, // [packed rows] int32 words containing uint16 counts / local bases
    at::Tensor out_mt_chunk_tile_sums,    // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 tile sums
    at::Tensor out_mt_chunk_tile_carry,   // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 exclusive tile carries
    at::Tensor out_mt_gauss_counts,       // [n_mt] int32 per-macro-tile gaussian counts
    int32_t n_macro_tiles,                // number of macro-tiles
    int32_t n_chunks,                     // number of gaussian chunks
    at::Tensor* inout_block_sums = nullptr,     // optional cached scratch buffer
    at::Tensor* inout_block_prefixes = nullptr, // optional cached scratch buffer
    int64_t* inout_scratch_capacity = nullptr);  // optional high-water-mark tracker

// stateful helper that owns scratch buffers for the persistent mt offset scan.
// construct once per resolution; call execute() each frame.
class MTOffsets
{
public:
    // n_macro_tiles: number of macro-tiles (cols x rows)
    // gauss_batch_log2: log2(gaussians per batch)
    MTOffsets(int32_t n_macro_tiles, int32_t gauss_batch_log2);

    // launches the single persistent kernel on the current CUDA stream.
    void execute(const at::Tensor &in_mt_gauss_counts, at::Tensor &out_mt_gauss_offsets,
                 at::Tensor &out_mt_batch_offsets);

private:
    int32_t m_nMacroTiles;
    int32_t m_gaussBatchLog2;
    int m_numSMs;              // cached SM count for persistent CTA grid sizing
    int32_t m_barrierExpected; // monotonically increasing barrier target, grows by n_ctas each execute()
    at::Tensor m_scratch;      // [n_ctas*2 + 32] int32: barrier + int2 block sums
};

} // namespace higs
