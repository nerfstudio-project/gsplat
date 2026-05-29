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

// Fused macro-tile offset scan: shared implementation for the fused
// macro-tile intersection pipeline.
// Produces mt_gauss_offsets and mt_batch_offsets from raw per-macro-tile
// gaussian counts in a single persistent kernel pass.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_scan.cuh>

#include "IntersectCommon.h"
#include "IntersectMTConfig.h"

namespace higs {

// CTA size for the persistent mt offset scan kernel. must be a multiple of
// warp size. larger values process more elements per CTA, reducing the number
// of grid barriers needed.
static constexpr int32_t MT_OFFSETS_BLOCK_SIZE = 128;
static_assert(MT_OFFSETS_BLOCK_SIZE % 32 == 0, "block size must be a multiple of warp size");

static constexpr int32_t MT_CHUNK_LOCAL_THREADS = MT_CHUNK_TILE * MT_META_WORDS;
static constexpr int32_t MT_CHUNK_CARRY_ROWS    = 8;
static constexpr int32_t MT_CHUNK_CARRY_THREADS = MT_CHUNK_CARRY_ROWS * MT_META_WORDS;
static constexpr int32_t MT_CHUNK_SCAN_STRIDE   = MT_META_WORDS + 4;

struct Int2Add
{
    __device__ int2 operator()(const int2 &a, const int2 &b) const
    {
        return {a.x + b.x, a.y + b.y};
    }
};

// Parameter block for the chunk-base scan that converts packed uint16 counts
// into packed uint16 local bases plus int32 carries and final mt_gauss_counts.
struct MtChunkBasesArgs
{
    int32_t n_macro_tiles;        // number of macro-tiles
    int32_t n_chunks;             // number of gaussian chunks
    int32_t n_mt_tiles;           // ceil(n_macro_tiles / MT_META_TILE)
    int32_t n_chunk_tiles;        // ceil(n_chunks / MT_CHUNK_TILE)
    int32_t *mt_chunk_meta_words; // [packed rows] int32 words containing uint16 counts / local bases
    int32_t *mt_chunk_tile_sums;  // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 tile sums
    int32_t *mt_chunk_tile_carry; // [n_chunk_tiles*n_mt_tiles*MT_META_TILE] int32 exclusive tile carries
    int32_t *mt_gauss_counts;     // [n_mt] int32 per-macro-tile gaussian counts
};

// stage 1: one CTA per metadata tile. the CTA loads the full 8x32 packed-word
// tile to shared memory with row-major coalescing, then 32 logical 8-lane warps
// run independent CUB warp scans across the chunk rows for one metadata column
// each. results are written back row-major and tile sums are emitted for stage 2.
template<int32_t MT_TILE>
__global__ void mt_chunk_local_bases_kernel(MtChunkBasesArgs args)
{
    static_assert(MT_TILE == MT_META_TILE, "kernel launch must match metadata tile width");

    using WarpScanT = cub::WarpScan<uint32_t, MT_CHUNK_TILE>;
    __shared__ typename WarpScanT::TempStorage scan_storage[MT_META_WORDS * 2];
    __shared__ uint32_t smem_words[MT_CHUNK_TILE][MT_CHUNK_SCAN_STRIDE];
    __shared__ uint32_t smem_bases[MT_CHUNK_TILE][MT_CHUNK_SCAN_STRIDE];

    const int32_t tid        = threadIdx.x;
    const int32_t load_row   = tid / MT_META_WORDS;
    const int32_t load_col   = tid % MT_META_WORDS;
    const int32_t mt_tile    = blockIdx.x;
    const int32_t chunk_tile = blockIdx.y;
    uint32_t *const meta_u32 = reinterpret_cast<uint32_t *>(args.mt_chunk_meta_words);

    auto chunk_row_word_base = [=](const int32_t row_in_tile) {
        return ((chunk_tile * args.n_mt_tiles + mt_tile) * MT_CHUNK_TILE + row_in_tile) * MT_META_WORDS;
    };
    auto chunk_tile_carry_idx = [=](const int32_t mt_lane) {
        return (chunk_tile * args.n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };

    const int32_t global_chunk     = chunk_tile * MT_CHUNK_TILE + load_row;
    const int32_t tile_word_base   = chunk_row_word_base(0);
    smem_words[load_row][load_col] = (global_chunk < args.n_chunks) ? meta_u32[tile_word_base + tid] : 0u;
    __syncthreads();

    const int32_t scan_row      = tid % MT_CHUNK_TILE;
    const int32_t scan_col      = tid / MT_CHUNK_TILE;
    const int32_t mt0           = mt_tile * MT_TILE + scan_col * 2;
    const int32_t mt1           = mt0 + 1;
    const uint32_t packed_count = smem_words[scan_row][scan_col];
    const uint32_t count_lo     = packed_count & 0xFFFFu;
    const uint32_t count_hi     = packed_count >> 16;

    uint32_t base_lo = 0;
    uint32_t base_hi = 0;
    WarpScanT(scan_storage[scan_col * 2 + 0]).ExclusiveSum(count_lo, base_lo);
    WarpScanT(scan_storage[scan_col * 2 + 1]).ExclusiveSum(count_hi, base_hi);

    const int32_t scan_chunk       = chunk_tile * MT_CHUNK_TILE + scan_row;
    const uint16_t base_lo_u16     = (scan_chunk < args.n_chunks && mt0 < args.n_macro_tiles)
                                         ? static_cast<uint16_t>(base_lo)
                                         : static_cast<uint16_t>(0);
    const uint16_t base_hi_u16     = (scan_chunk < args.n_chunks && mt1 < args.n_macro_tiles)
                                         ? static_cast<uint16_t>(base_hi)
                                         : static_cast<uint16_t>(0);
    smem_bases[scan_row][scan_col] = static_cast<uint32_t>(base_lo_u16) | (static_cast<uint32_t>(base_hi_u16) << 16);

    if (scan_row == MT_CHUNK_TILE - 1)
    {
        const int32_t tile_base = chunk_tile_carry_idx(0);

        reinterpret_cast<int2 *>(args.mt_chunk_tile_sums + tile_base)[scan_col] = {
            static_cast<int32_t>(base_lo + count_lo), static_cast<int32_t>(base_hi + count_hi)};
    }
    __syncthreads();

    if (global_chunk < args.n_chunks)
    {
        meta_u32[tile_word_base + tid] = smem_bases[load_row][load_col];
    }
}

// stage 2a: one CTA per chunk-block metadata tile. the CTA loads one block of
// chunk-tile sums, runs 32 logical 8-lane warp scans over the chunk dimension,
// writes local carries to the final tensor, and emits one block sum per column.
template<int32_t MT_TILE>
__global__ void mt_chunk_tile_carry_kernel(MtChunkBasesArgs args, int32_t *__restrict__ out_block_sums)
{
    static_assert(MT_TILE == MT_META_TILE, "kernel launch must match metadata tile width");

    using WarpScanT = cub::WarpScan<int2, MT_CHUNK_CARRY_ROWS>;
    __shared__ typename WarpScanT::TempStorage scan_storage[MT_META_WORDS];
    __shared__ int2 smem_sums[MT_CHUNK_CARRY_ROWS][MT_CHUNK_SCAN_STRIDE];
    __shared__ int2 smem_carries[MT_CHUNK_CARRY_ROWS][MT_CHUNK_SCAN_STRIDE];

    const int32_t tid         = threadIdx.x;
    const int32_t load_row    = tid / MT_META_WORDS;
    const int32_t load_col    = tid % MT_META_WORDS;
    const int32_t mt_tile     = blockIdx.x;
    const int32_t chunk_block = blockIdx.y;
    const int32_t chunk_tile  = chunk_block * MT_CHUNK_CARRY_ROWS + load_row;

    auto chunk_tile_carry_idx = [=](const int32_t row_chunk_tile, const int32_t mt_lane) {
        return (row_chunk_tile * args.n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };
    auto block_tile_carry_idx = [=](const int32_t mt_lane) {
        return (chunk_block * args.n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };

    if (chunk_tile < args.n_chunk_tiles)
    {
        const int32_t tile_base = chunk_tile_carry_idx(chunk_tile, 0);
        reinterpret_cast<int2 *>(smem_sums[load_row])[load_col] =
            reinterpret_cast<const int2 *>(args.mt_chunk_tile_sums + tile_base)[load_col];
    }
    else
    {
        reinterpret_cast<int2 *>(smem_sums[load_row])[load_col] = {0, 0};
    }
    __syncthreads();

    const int32_t scan_row = tid % MT_CHUNK_CARRY_ROWS;
    const int32_t pair_col = tid / MT_CHUNK_CARRY_ROWS;
    int2 scan_out;
    WarpScanT(scan_storage[pair_col]).ExclusiveScan(smem_sums[scan_row][pair_col], scan_out, int2{0, 0}, Int2Add());
    smem_carries[scan_row][pair_col] = scan_out;
    __syncthreads();

    if (chunk_tile < args.n_chunk_tiles)
    {
        const int32_t tile_base = chunk_tile_carry_idx(chunk_tile, 0);
        reinterpret_cast<int2 *>(args.mt_chunk_tile_carry + tile_base)[load_col] =
            reinterpret_cast<int2 *>(smem_carries[load_row])[load_col];
    }

    if (scan_row == MT_CHUNK_CARRY_ROWS - 1)
    {
        const int32_t block_base = block_tile_carry_idx(0);
        reinterpret_cast<int2 *>(out_block_sums + block_base)[pair_col] =
            Int2Add{}(smem_carries[scan_row][pair_col], smem_sums[scan_row][pair_col]);
    }
}

// stage 2b: one CTA per mt_tile. each thread scans the much shorter block-sum
// stream for one packed int2 metadata column and writes a block prefix for
// stage 2c. the final accumulated prefix becomes mt_gauss_counts.
template<int32_t MT_TILE>
__global__ void mt_chunk_tile_block_prefix_kernel(const int32_t n_chunk_blocks, const int32_t n_mt_tiles,
                                                  const int32_t n_macro_tiles, const int32_t *__restrict__ block_sums,
                                                  int32_t *__restrict__ block_prefixes,
                                                  int32_t *__restrict__ mt_gauss_counts)
{
    static_assert(MT_TILE == MT_META_TILE, "kernel launch must match metadata tile width");

    const int32_t mt_tile  = blockIdx.x;
    const int32_t pair_col = threadIdx.x;
    const int32_t mt0      = mt_tile * MT_TILE + pair_col * 2;
    const int32_t mt1      = mt0 + 1;

    auto block_tile_carry_idx = [=](const int32_t chunk_block, const int32_t mt_lane) {
        return (chunk_block * n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };

    int2 running = {0, 0};
#pragma unroll 1
    for (int32_t chunk_block = 0; chunk_block < n_chunk_blocks; ++chunk_block)
    {
        const int32_t block_base = block_tile_carry_idx(chunk_block, 0);

        reinterpret_cast<int2 *>(block_prefixes + block_base)[pair_col] = running;

        const int2 sum = reinterpret_cast<const int2 *>(block_sums + block_base)[pair_col];
        running        = Int2Add{}(running, sum);
    }

    if (mt0 < n_macro_tiles)
    {
        mt_gauss_counts[mt0] = running.x;
    }
    if (mt1 < n_macro_tiles)
    {
        mt_gauss_counts[mt1] = running.y;
    }
}

// stage 2c: adds the per-block prefix from stage 2b to all local carries
// produced by stage 2a.
template<int32_t MT_TILE>
__global__ void mt_chunk_tile_add_prefix_kernel(const int32_t n_chunk_tiles, const int32_t n_mt_tiles,
                                                const int32_t *__restrict__ block_prefixes,
                                                int32_t *__restrict__ mt_chunk_tile_carry)
{
    static_assert(MT_TILE == MT_META_TILE, "kernel launch must match metadata tile width");

    const int32_t tid         = threadIdx.x;
    const int32_t load_row    = tid / MT_META_WORDS;
    const int32_t load_col    = tid % MT_META_WORDS;
    const int32_t mt_tile     = blockIdx.x;
    const int32_t chunk_block = blockIdx.y;
    const int32_t chunk_tile  = chunk_block * MT_CHUNK_CARRY_ROWS + load_row;

    auto block_tile_carry_idx = [=](const int32_t block_idx, const int32_t mt_lane) {
        return (block_idx * n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };

    auto chunk_tile_carry_idx = [=](const int32_t row_chunk_tile, const int32_t mt_lane) {
        return (row_chunk_tile * n_mt_tiles + mt_tile) * MT_META_TILE + mt_lane;
    };

    if (chunk_tile >= n_chunk_tiles)
    {
        return;
    }

    const int32_t block_base = block_tile_carry_idx(chunk_block, 0);
    const int2 prefix        = reinterpret_cast<const int2 *>(block_prefixes + block_base)[load_col];
    const int32_t tile_base  = chunk_tile_carry_idx(chunk_tile, 0);
    int2 *const carries      = reinterpret_cast<int2 *>(mt_chunk_tile_carry + tile_base);
    carries[load_col]        = Int2Add{}(carries[load_col], prefix);
}

// parameter block for the persistent single-kernel mt offset scan.
// n_ctas and n_scan_blocks are derived from gridDim.x and n_macro_tiles
// inside the kernel rather than passed explicitly.
struct MtOffsetsArgs
{
    int32_t n_macro_tiles;
    const int32_t *mt_gauss_counts;
    int32_t *mt_gauss_offsets;
    int32_t *mt_batch_offsets;
    int32_t gauss_batch_log2;
    int2 *block_sums_gauss_batch; // [gridDim.x] per-CTA totals scratch
    int32_t *barrier_counter;     // single monotonically increasing atomic counter
    int32_t barrier_base;         // counter value at start of this invocation
};

void launch_mt_chunk_bases(at::Tensor inout_mt_chunk_meta_words, at::Tensor out_mt_chunk_tile_sums,
                           at::Tensor out_mt_chunk_tile_carry, at::Tensor out_mt_gauss_counts, int32_t n_macro_tiles,
                           int32_t n_chunks,
                           at::Tensor* inout_block_sums, at::Tensor* inout_block_prefixes,
                           int64_t* inout_scratch_capacity)
{
    if (n_macro_tiles == 0)
    {
        return;
    }

    out_mt_gauss_counts.zero_();
    if (n_chunks == 0)
    {
        return;
    }

    auto ceilDivInt = [](const int32_t x, const int32_t y) -> int32_t { return (x + y - 1) / y; };

    MtChunkBasesArgs args;
    args.n_macro_tiles       = n_macro_tiles;
    args.n_chunks            = n_chunks;
    args.n_mt_tiles          = ceilDivInt(n_macro_tiles, MT_META_TILE);
    args.n_chunk_tiles       = ceilDivInt(n_chunks, MT_CHUNK_TILE);
    args.mt_chunk_meta_words = inout_mt_chunk_meta_words.data_ptr<int32_t>();
    args.mt_chunk_tile_sums  = out_mt_chunk_tile_sums.data_ptr<int32_t>();
    args.mt_chunk_tile_carry = out_mt_chunk_tile_carry.data_ptr<int32_t>();
    args.mt_gauss_counts     = out_mt_gauss_counts.data_ptr<int32_t>();

    const int32_t n_chunk_blocks = ceilDivInt(args.n_chunk_tiles, MT_CHUNK_CARRY_ROWS);
    const auto stream            = at::cuda::getCurrentCUDAStream();
    const dim3 local_grid(args.n_mt_tiles, args.n_chunk_tiles);
    const dim3 carry_grid(args.n_mt_tiles, n_chunk_blocks);
    auto opts_i32          = at::TensorOptions().dtype(at::kInt).device(at::kCUDA);
    const int64_t required = (int64_t)n_chunk_blocks * args.n_mt_tiles * MT_META_TILE;

    // Use caller-supplied cached scratch buffers when available (avoids 2x at::empty per frame).
    // Falls back to local temporaries when no cache pointers are provided.
    at::Tensor local_block_sums, local_block_prefixes;
    at::Tensor& block_sums     = inout_block_sums     ? *inout_block_sums     : local_block_sums;
    at::Tensor& block_prefixes = inout_block_prefixes ? *inout_block_prefixes : local_block_prefixes;

    int64_t cap = inout_scratch_capacity ? *inout_scratch_capacity : 0;
    if (required > cap) {
        block_sums     = at::empty({required}, opts_i32);
        block_prefixes = at::empty({required}, opts_i32);
        if (inout_scratch_capacity) *inout_scratch_capacity = required;
    }

    mt_chunk_local_bases_kernel<MT_META_TILE><<<local_grid, MT_CHUNK_LOCAL_THREADS, 0, stream>>>(args);
    mt_chunk_tile_carry_kernel<MT_META_TILE>
        <<<carry_grid, MT_CHUNK_CARRY_THREADS, 0, stream>>>(args, block_sums.data_ptr<int32_t>());
    mt_chunk_tile_block_prefix_kernel<MT_META_TILE><<<args.n_mt_tiles, MT_META_WORDS, 0, stream>>>(
        n_chunk_blocks, args.n_mt_tiles, n_macro_tiles, block_sums.data_ptr<int32_t>(),
        block_prefixes.data_ptr<int32_t>(), args.mt_gauss_counts);
    mt_chunk_tile_add_prefix_kernel<MT_META_TILE><<<carry_grid, MT_CHUNK_CARRY_THREADS, 0, stream>>>(
        args.n_chunk_tiles, args.n_mt_tiles, block_prefixes.data_ptr<int32_t>(), args.mt_chunk_tile_carry);
}

// single persistent kernel: phase 1 block-wide scan with carry, grid barrier,
// phase 2 block-wide aggregate by CTA 0, grid barrier, phase 3 propagate.
// launched with min(n_scan_blocks, num_sms) CTAs of MT_OFFSETS_BLOCK_SIZE threads.
__global__ void mt_offsets_kernel(MtOffsetsArgs args)
{
    using BlockScanInt2 = cub::BlockScan<int2, MT_OFFSETS_BLOCK_SIZE>;

    __shared__ typename BlockScanInt2::TempStorage scan_storage;

    const int32_t n_ctas        = gridDim.x;
    const int32_t n_scan_blocks = (args.n_macro_tiles + MT_OFFSETS_BLOCK_SIZE - 1) / MT_OFFSETS_BLOCK_SIZE;
    const int32_t cta_id        = blockIdx.x;
    const int32_t tid           = threadIdx.x;

    // grid barrier using a single monotonically increasing counter. the lambda
    // tracks how many times it's been called; signed subtraction handles int32
    // wraparound because the "window" (n_ctas) is tiny relative to the range.
    int32_t barrier_call_count = 0;
    auto gridBarrier           = [&]() {
        barrier_call_count++;
        const int32_t target = args.barrier_base + n_ctas * barrier_call_count;
        __threadfence();
        if (tid == 0)
        {
            int32_t arrived = atomicAdd(args.barrier_counter, 1) + 1;
            if (arrived - target < 0)
            {
                while (atomicAdd(args.barrier_counter, 0) - target < 0)
                {
                }
            }
        }
        __syncthreads();
    };

    // compute contiguous scan-block range for this CTA
    const int32_t blocks_per_cta = n_scan_blocks / n_ctas;
    const int32_t extra_blocks   = n_scan_blocks % n_ctas;
    const int32_t block_start    = cta_id * blocks_per_cta + min(cta_id, extra_blocks);
    const int32_t block_end      = block_start + blocks_per_cta + (cta_id < extra_blocks ? 1 : 0);

    // phase 1: block-wide scans with carry-over between scan blocks
    int2 carry_gb = {0, 0};

#pragma unroll 1
    for (int32_t block = block_start; block < block_end; ++block)
    {
        const int32_t mt = block * MT_OFFSETS_BLOCK_SIZE + tid;

        int32_t n_gauss   = 0;
        int32_t n_batches = 0;
        if (mt < args.n_macro_tiles)
        {
            n_gauss   = args.mt_gauss_counts[mt];
            n_batches = (n_gauss > 0) ? ((n_gauss - 1) >> args.gauss_batch_log2) + 1 : 0;
        }

        // fused int2 block scan for gauss + batch
        int2 gb_val = {n_gauss, n_batches};
        int2 gb_inclusive;
        int2 gb_aggregate;
        BlockScanInt2(scan_storage).InclusiveScan(gb_val, gb_inclusive, Int2Add{}, gb_aggregate);

        // add carry-in from previous scan blocks
        gb_inclusive = Int2Add{}(gb_inclusive, carry_gb);

        if (mt < args.n_macro_tiles)
        {
            args.mt_gauss_offsets[mt + 1] = gb_inclusive.x;
            args.mt_batch_offsets[mt + 1] = gb_inclusive.y;
        }

        if (block == 0 && tid == 0)
        {
            args.mt_gauss_offsets[0] = 0;
            args.mt_batch_offsets[0] = 0;
        }

        carry_gb = Int2Add{}(carry_gb, gb_aggregate);

        __syncthreads();
    }

    // write per-CTA total to scratch
    if (tid == 0)
    {
        args.block_sums_gauss_batch[cta_id] = carry_gb;
    }

    // single-CTA fast path: no aggregate or propagate needed
    if (n_ctas == 1)
    {
        return;
    }

    gridBarrier();

    // phase 2: CTA 0 computes in-place exclusive prefix over per-CTA totals
    // using block-wide scans in chunks of MT_OFFSETS_BLOCK_SIZE, with carry between chunks
    if (cta_id == 0)
    {
        int2 agg_carry_gb          = {0, 0};
        const int32_t n_agg_chunks = (n_ctas + MT_OFFSETS_BLOCK_SIZE - 1) / MT_OFFSETS_BLOCK_SIZE;

#pragma unroll 1
        for (int32_t chunk = 0; chunk < n_agg_chunks; ++chunk)
        {
            const int32_t idx = chunk * MT_OFFSETS_BLOCK_SIZE + tid;

            int2 gb_total = (idx < n_ctas) ? args.block_sums_gauss_batch[idx] : int2{0, 0};

            int2 gb_exclusive;
            int2 gb_aggregate;
            BlockScanInt2(scan_storage)
                .ExclusiveScan(gb_total, gb_exclusive, int2{0, 0}, Int2Add{}, gb_aggregate);

            gb_exclusive = Int2Add{}(gb_exclusive, agg_carry_gb);

            if (idx < n_ctas)
            {
                args.block_sums_gauss_batch[idx] = gb_exclusive;
            }

            agg_carry_gb = Int2Add{}(agg_carry_gb, gb_aggregate);

            __syncthreads();
        }
    }

    gridBarrier();

    // phase 3: each CTA (except CTA 0) reads its prefix and adds to all
    // output elements in its range. CTA 0's results are already globally correct.
    if (cta_id == 0)
    {
        return;
    }

    const int2 prefix_gb = args.block_sums_gauss_batch[cta_id];

#pragma unroll 1
    for (int32_t block = block_start; block < block_end; ++block)
    {
        const int32_t mt  = block * MT_OFFSETS_BLOCK_SIZE + tid;
        const int32_t pos = mt + 1;

        if (mt < args.n_macro_tiles)
        {
            args.mt_gauss_offsets[pos] += prefix_gb.x;
            args.mt_batch_offsets[pos] += prefix_gb.y;
        }
    }
}

// MTOffsets implementation

MTOffsets::MTOffsets(int32_t n_macro_tiles, int32_t gauss_batch_log2)
    : m_nMacroTiles(n_macro_tiles)
    , m_gaussBatchLog2(gauss_batch_log2)
    , m_numSMs(0)
    , m_barrierExpected(0)
{
    cudaDeviceGetAttribute(&m_numSMs, cudaDevAttrMultiProcessorCount, 0);

    const int32_t n_scan_blocks = (n_macro_tiles + MT_OFFSETS_BLOCK_SIZE - 1) / MT_OFFSETS_BLOCK_SIZE;
    const int32_t n_ctas        = min(n_scan_blocks, m_numSMs);

    // scratch layout: [32] int32 (barrier) + [n_ctas] int2 (gauss+batch sums)
    const int64_t scratch_ints = n_ctas * 2 + 32;
    m_scratch                  = at::zeros({scratch_ints}, at::TensorOptions().dtype(at::kInt).device(at::kCUDA));
}

void MTOffsets::execute(const at::Tensor &in_mt_gauss_counts, at::Tensor &out_mt_gauss_offsets,
                        at::Tensor &out_mt_batch_offsets)
{
    if (m_nMacroTiles == 0)
    {
        return;
    }

    const int32_t n_scan_blocks = (m_nMacroTiles + MT_OFFSETS_BLOCK_SIZE - 1) / MT_OFFSETS_BLOCK_SIZE;
    const int32_t n_ctas        = min(n_scan_blocks, m_numSMs);
    int32_t *scratch_ptr        = m_scratch.data_ptr<int32_t>();

    MtOffsetsArgs args;
    args.n_macro_tiles          = m_nMacroTiles;
    args.mt_gauss_counts        = in_mt_gauss_counts.data_ptr<int32_t>();
    args.mt_gauss_offsets       = out_mt_gauss_offsets.data_ptr<int32_t>();
    args.mt_batch_offsets       = out_mt_batch_offsets.data_ptr<int32_t>();
    args.gauss_batch_log2       = m_gaussBatchLog2;
    args.block_sums_gauss_batch = reinterpret_cast<int2 *>(scratch_ptr + 32);
    args.barrier_counter        = scratch_ptr;
    // barrier counter grows monotonically; signed subtraction in the kernel
    // handles int32 wraparound so no reset is ever needed.

    // TODO: this works quite well and we don't need to reset the counter ever, however the entire solution
    // is quite fragile from thread or stream safety POV. This counter is instance specific so we shouldn't be
    // submitting it to different streams or devices since they will all contend for the same counter.
    args.barrier_base = m_barrierExpected;
    m_barrierExpected += 2 * n_ctas;

    mt_offsets_kernel<<<n_ctas, MT_OFFSETS_BLOCK_SIZE, 0, at::cuda::getCurrentCUDAStream()>>>(args);
}

} // namespace higs
