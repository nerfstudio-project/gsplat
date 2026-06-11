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

// Macro-tile rasterize + post-blend kernels.
//
// Kernel 1 (macro_tile_rasterize_kernel):
//   CTA_WARPS warps per (macro-tile, batch) pair.  Two-phase design
//   that trades the MACRO_TILE_SIZE-tile register accumulators for smem masks.
//
//   Phase 1 — fused load + visibility mask generation:
//     Each warp processes one 32-gaussian mini-batch per iteration.
//     Lane j loads gaussian j from global, writes persistent smem
//     (xy_mt, conic_raw), then computes overlap against all 32
//     render-tiles (4 rows x 8 cols) with the data still in registers.
//     Each lane produces a 32-bit tile_mask (bit t = overlaps tile t).
//     A 5-stage warp_bit_transpose converts the per-gaussian tile masks
//     into per-tile gaussian masks (bit j = gaussian j overlaps this
//     tile), which are written directly to smem_masks — no atomicOr
//     needed since each warp owns its mini-batch slots exclusively.
//     Colors are NOT loaded here — they are deferred to right before
//     phase 2 into the overlap region.
//
//   Transition:
//     Warp 0 ORs mask words across all mini-batches to find any_hit
//     per tile.  Computes active_mask via __ballot_sync, writes output
//     mask, populates the tile work queue.
//
//   Deferred color load (between transition and phase 2):
//     Full CTA re-reads gauss ids and pulls per-gaussian colors from
//     global into the overlap region.  Colors are packed {RG : half2,
//     B : half} to drop the unused 4th slot of the [N,4] half layout.
//
//   Phase 2 — per-tile rasterization:
//     Loops over active tiles with work-stealing warp assignment.  Each
//     warp cooperatively rasterizes one tile using N_PAIRS x 4 half2
//     accumulators.  For each tile, reads pre-computed masks from smem
//     and iterates set bits to fetch gaussian data from smem (xy_mt,
//     conic, color_rg, color_b).
//     Writes partial RGBT to sparse tile_buffer after each tile.
//
// Kernel 2 (macro_tile_post_blend_kernel):
//   1 warp per render tile.  Walks batches of the parent macro-tile in
//   order, reads partial RGBT from sparse buffer (skipping inactive
//   batches via active_mask), composites front-to-back with transmittance
//   early-out, writes final image.

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include "Common.h"
#include "Constants.h"
#include "IntersectCommon.h"
#include "IntersectMTConfig.h"
#include "MacroTileRasterize.h"

namespace higs {

static constexpr int32_t RASTERIZE_CTA_SIZE   = 320;
static constexpr int32_t RASTERIZE_MIN_BLOCKS = 3;
// ring-buffer queue capacity
static constexpr int32_t RASTERIZE_QUEUE_CAPACITY = 128;

static constexpr int32_t BLEND_CTA_SIZE   = 256;
static constexpr int32_t BLEND_MIN_BLOCKS = 4;

static constexpr float LOG2_INV_AT = 7.9943534f; // log2(1/ALPHA_THRESHOLD) = log2(255)
static_assert(ALPHA_THRESHOLD == 1.0f / 255.0f, "LOG2_INV_AT / ALPHA_THRESHOLD mismatch -> update LOG2_INV_AT");
static constexpr float MIN_TRANSPARENCY = 1e-4f;

// 32x32 bit-matrix transpose within a warp via 5 stages of __shfl_xor_sync.
// input:  thread j holds val where bit t means "gaussian j overlaps tile t"
// output: thread t holds val where bit j means "gaussian j overlaps tile t"
__device__ __forceinline__ uint32_t WarpBitTranspose(uint32_t val)
{
    constexpr uint32_t MASKS[5] = {0x0000FFFF, 0x00FF00FF, 0x0F0F0F0F, 0x33333333, 0x55555555};
    const int32_t lane_id       = threadIdx.x & 31;
#pragma unroll
    for (int32_t k = 4; k >= 0; --k)
    {
        const int32_t delta  = 1 << k;
        const uint32_t m     = MASKS[4 - k];
        const uint32_t other = __shfl_xor_sync(~0u, val, delta);
        val = (lane_id & delta) ? (val & ~m) | ((other >> delta) & m) : (val & m) | ((other & m) << delta);
    }
    return val;
};

// generate a 32-bit render-tile mask for a single gaussian against all 32 render-tiles.
__device__ __forceinline__ uint32_t GetMacroTileVisibilityMask(const __half2 &xy_mt, const __half2 (&rast_conic)[2],
                                                               float neg_l1_rcp_C, float t_rast)
{
    // test against all 32 render-tiles (4 rows x 4 column-pairs).
    // row setup and inner loop are pure half2 math.

    // half2 broadcasts for the overlap test (2 columns at a time)
    const __half2 h2_mt_rel_x     = __low2half2(xy_mt);
    const __half2 h2_mt_rel_y     = __high2half2(xy_mt);
    const __half2 h2_c_l0         = __low2half2(rast_conic[0]);
    const __half2 h2_c_l1         = __high2half2(rast_conic[0]);
    const __half2 h2_c_l2         = __low2half2(rast_conic[1]);
    const __half2 h2_neg_l1_rcp_C = __float2half2_rn(neg_l1_rcp_C);
    const __half2 h2_t_rast       = __float2half2_rn(t_rast);

    uint32_t tile_mask_even = 0;
    uint32_t tile_mask_odd  = 0;
#pragma unroll
    for (int32_t r = 0; r < FUSED_MACRO_TILE_HEIGHT; ++r)
    {
        const __half2 h2_ny = __hsub2(h2_mt_rel_y, __float2half2_rn(r + 0.5f - FUSED_MACRO_TILE_HEIGHT * 0.5f));
        const uint32_t ny_inside_mask = __hlt2_mask(__habs2(h2_ny), __float2half2_rn(0.5f));
        const __half2 h2_dy0          = __hsub2(__float2half2_rn(-0.5f), h2_ny);
        const __half2 h2_dy1          = __hadd2(h2_dy0, __float2half2_rn(1.f));
        const __half2 h2_dy_h         = __h2if(__hgeu2_mask(h2_ny, __float2half2_rn(0.0f)), h2_dy1, h2_dy0);
        const __half2 h2_l1_dy        = __hmul2(h2_c_l1, h2_dy_h);
        const __half2 h2_v_h          = __hmul2(h2_c_l2, h2_dy_h);
        const __half2 h2_v_h_sq       = __hmul2(h2_v_h, h2_v_h);

#pragma unroll
        for (int32_t c = 0; c < FUSED_MACRO_TILE_WIDTH; c += 2)
        {
            const int32_t t = r * FUSED_MACRO_TILE_WIDTH + c;

            const __half2 nx = __hsub2(h2_mt_rel_x, __floats2half2_rn(c + 0.5f - FUSED_MACRO_TILE_WIDTH * 0.5f,
                                                                      c + 1.5f - FUSED_MACRO_TILE_WIDTH * 0.5f));

            const __half2 dx0    = __hsub2(__float2half2_rn(-0.5f), nx);
            const __half2 l0_dx0 = __hmul2(h2_c_l0, dx0);
            const __half2 l0_dx1 = __hadd2(l0_dx0, h2_c_l0);

            // horizontal edge test
            const __half2 u_h =
                __hmin2(__hmax2(__float2half2_rn(0.0f), __hadd2(l0_dx0, h2_l1_dy)), __hadd2(l0_dx1, h2_l1_dy));
            const __half2 q_h = __hfma2(u_h, u_h, h2_v_h_sq);

            // vertical edge test
            const __half2 l0_dx = __h2if(__hgeu2_mask(nx, __float2half2_rn(0.0f)), l0_dx1, l0_dx0);
            const __half2 dy_v  = __hmin2(__hmax2(__hmul2(l0_dx, h2_neg_l1_rcp_C), h2_dy0), h2_dy1);
            const __half2 u_v   = __hfma2(h2_c_l1, dy_v, l0_dx);
            const __half2 v_v   = __hmul2(h2_c_l2, dy_v);
            const __half2 q_v   = __hfma2(v_v, v_v, __hmul2(u_v, u_v));

            // center inside = |nx| < 0.5 AND |ny| < 0.5 (both masks in {bit0, bit16} format)
            const uint32_t center_hit = ny_inside_mask & __hlt2_mask(__habs2(nx), __float2half2_rn(0.5f));
            const uint32_t edge_hit   = __hle2_mask(__hmin2(q_h, q_v), h2_t_rast);
            const uint32_t hit        = center_hit | edge_hit;

            // __hxx2_mask: bit 0 = low half, bit 16 = high half → pack to positions t, t+1
            tile_mask_even |= (hit & 1u) << t;
            tile_mask_odd |= (hit >> 31) << (t + 1);
        }
    }
    return tile_mask_even | tile_mask_odd;
};

// kernel arguments passed as a single struct to reduce parameter count.
struct MacroTileRasterizeArgs
{
    int32_t n_macro_tiles;
    int32_t macro_tile_cols;
    int32_t tile_width;
    int32_t tile_height;
    const float *means2d;
    const __half *conics;
    const __half *colors;
    const int32_t *mt_gauss_offsets;
    const int32_t *mt_gauss_ids_sorted;
    const int32_t *mt_batch_offsets;
    __half *tile_buffer;
    uint32_t *out_active_mask;
};

template<int32_t TILE_SIZE>
__global__ void __launch_bounds__(RASTERIZE_CTA_SIZE, RASTERIZE_MIN_BLOCKS)
    macro_tile_rasterize_kernel(MacroTileRasterizeArgs args)
{
    constexpr int32_t MACRO_TILE_SIZE = FUSED_MACRO_TILE_WIDTH * FUSED_MACRO_TILE_HEIGHT;
    constexpr int32_t CTA_WARPS       = RASTERIZE_CTA_SIZE / 32;

    // a single mini-batch corresponds to a 32-bit mask
    constexpr int32_t MINI_BATCH_SIZE  = 32;
    constexpr int32_t MAX_MINI_BATCHES = FUSED_GAUSS_BATCH_SIZE / MINI_BATCH_SIZE;

    static_assert(MACRO_TILE_SIZE == 32, "macro-tile must be exactly one warp");
    static_assert(TILE_SIZE == 8 || TILE_SIZE == 16, "TILE_SIZE must be 8 or 16");
    static_assert(RASTERIZE_CTA_SIZE % 32 == 0, "RASTERIZE_CTA_SIZE must be a multiple of warp size");
    static_assert(CTA_WARPS > 1, "single-warp path removed — CTA_WARPS must be > 1");

    constexpr float INV_TILE_SIZE      = 1.0f / TILE_SIZE;
    // std::numbers::log2e_v<float> would be preferable (C++20) but M_LOG2E
    // is used for CUDA device-code compatibility.
    constexpr float LOG2E              = static_cast<float>(M_LOG2E);
    constexpr float CHOL_SCALE         = 0.84932180028f * TILE_SIZE; // sqrt(LOG2E/2) * TILE_SIZE
    constexpr float MAX_EXTEND_CHOL_SQ = (LOG2E * 0.5f) * MAX_EXTEND * MAX_EXTEND;

    // tile=8 splits a warp into 2 half-warps so each gaussian is broadcast to 16 lanes
    // instead of 32 — halves smem LDS issue rate at the cost of 2x the accumulator
    // register pressure per thread. tile=16 keeps GAUSS_PER_WARP=1 (registers would otherwise spike).
    constexpr int32_t GAUSS_PER_WARP    = (TILE_SIZE == 8) ? 2 : 1;
    constexpr int32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / (32 / GAUSS_PER_WARP);
    constexpr int32_t N_PAIRS           = PIXELS_PER_THREAD / 2;

    // shared memory — single raw block, pointers chained from top
    constexpr int32_t SMEM_MASKS_COUNT = MAX_MINI_BATCHES * MACRO_TILE_SIZE;
    constexpr int32_t SMEM_SIZE =
        FUSED_GAUSS_BATCH_SIZE * (sizeof(__half2) + sizeof(uint2) + sizeof(__half2) + sizeof(__half2)) +
        (MACRO_TILE_SIZE + 2) * sizeof(int32_t) + SMEM_MASKS_COUNT * sizeof(uint32_t) + 1 * sizeof(int32_t);

    static_assert((RASTERIZE_QUEUE_CAPACITY & (RASTERIZE_QUEUE_CAPACITY - 1)) == 0,
                  "QUEUE_CAPACITY must be a power of 2");

    // N_PAIRS x {R,G,B,T} half2 accumulators — each [pp] is a contiguous uint4
    using TileAccum = __half2[N_PAIRS][4];

    // let NVCC know that warp_id is a warm uniform (doh!)
    const int32_t warp_id = __shfl_sync(~0u, threadIdx.x >> 5, 0);
    const int32_t lane_id = threadIdx.x & 31;

    __shared__ alignas(16) char smem_raw[SMEM_SIZE];

    // gaussian position + conic (written in phase 1, read in phase 2)
    __half2 *smem_xy_mt   = reinterpret_cast<__half2 *>(smem_raw);
    uint2 *smem_conic_raw = reinterpret_cast<uint2 *>(smem_xy_mt + FUSED_GAUSS_BATCH_SIZE);

    // tile work queue: [0..31] tile ids, [32] queue size, [33] pull counter
    int32_t *smem_tile_queue = reinterpret_cast<int32_t *>(smem_conic_raw + FUSED_GAUSS_BATCH_SIZE);
    int32_t &smem_queue_size = smem_tile_queue[MACRO_TILE_SIZE];
    int32_t &smem_queue_idx  = smem_tile_queue[MACRO_TILE_SIZE + 1];

    // per-gaussian color packed as 4xfp16 (loaded before phase 2)
    ushort4 *smem_color = reinterpret_cast<ushort4 *>(smem_tile_queue + MACRO_TILE_SIZE + 2);

    // visibility masks (filled during fused load via warp transpose, read in phase 2)
    uint32_t *smem_masks = reinterpret_cast<uint32_t *>(smem_color + FUSED_GAUSS_BATCH_SIZE);

    // cross-warp broadcast scratch:
    int32_t *smem_scratch = reinterpret_cast<int32_t *>(smem_masks + SMEM_MASKS_COUNT);
    int32_t &smem_mt_id   = smem_scratch[0];

    // per-warp gaussian index queue for rasterization (ring buffer, filled across mini-batches).
    // align to 16-bytes so the unrolled rasterization loop can use uint4 loads.
    __shared__ __align__(16) uint16_t smem_warp_queue[CTA_WARPS * RASTERIZE_QUEUE_CAPACITY];

    // pixel coordinates within a tile (same for all tiles, derived from lane_id).
    // for GAUSS_PER_WARP=2 each half-warp owns the entire tile; lane_in_half/is_half_a reduce to
    // lane_id/true respectively for GAUSS_PER_WARP=1, so the GAUSS_PER_WARP=1 codegen is unchanged.
    constexpr uint32_t TILE_X_MASK  = TILE_SIZE - 1;
    constexpr uint32_t TILE_X_SHIFT = (TILE_SIZE == 16) ? 4 : 3;
    const int32_t lane_in_half      = lane_id & ((32 / GAUSS_PER_WARP) - 1);
    const bool is_half_a            = lane_id < (32 / GAUSS_PER_WARP);
    const uint32_t thread_x         = lane_in_half & TILE_X_MASK;
    const uint32_t thread_y         = lane_in_half >> TILE_X_SHIFT;
    // pixel x in tile units relative to macro-tile center (includes intra-tile + centering offset)
    const float px = static_cast<float>(thread_x) * INV_TILE_SIZE + (0.5f * INV_TILE_SIZE - 0.5f) +
                     (0.5f - FUSED_MACRO_TILE_WIDTH * 0.5f);
    // pixel y in tile units relative to macro-tile center (includes intra-tile + centering offset)
    const float py = static_cast<float>(thread_y) * INV_TILE_SIZE + (0.5f * INV_TILE_SIZE - 0.5f) +
                     (0.5f - FUSED_MACRO_TILE_HEIGHT * 0.5f);

    // rasterize a single gaussian into all N_PAIRS row-pair accumulators.
    // smem loads and dx/u0_base are invariant across pairs — only dy-dependent
    // math runs per pair, keeping smem traffic at 1 load per gaussian.
    auto rasterizeGaussian = [&smem_xy_mt, &smem_conic_raw, &smem_color](int32_t idx, const __half2 &offset_x,
                                                                         const __half2(&offset_y)[N_PAIRS],
                                                                         TileAccum &acc) {
        const __half2 xy_mt_j = smem_xy_mt[idx];
        __half2 rc[2];
        AssignAs<uint2>(rc, smem_conic_raw[idx]);
        __half2 col[2];
        AssignAs<uint2>(col, smem_color[idx]);

        const __half2 sigma_thresh = __float2half2_rn(LOG2_INV_AT);
        const __half2 l00          = __low2half2(rc[0]);
        const __half2 l10          = __high2half2(rc[0]);
        const __half2 l11          = __low2half2(rc[1]);
        const __half2 neg_log_opac = __high2half2(rc[1]);
        const __half2 cr           = __half2half2(__low2half(col[0]));
        const __half2 cg           = __half2half2(__high2half(col[0]));
        const __half2 cb           = __low2half2(col[1]);

        const __half2 dx      = __hsub2(__low2half2(xy_mt_j), offset_x);
        const __half2 mean_y  = __high2half2(xy_mt_j);
        const __half2 u0_base = __hmul2(l00, dx);

#pragma unroll
        for (int32_t pp = 0; pp < N_PAIRS; ++pp)
        {
            const __half2 dy = __hsub2(mean_y, offset_y[pp]);
            const __half2 u0 = __hfma2(l10, dy, u0_base);
            const __half2 u1 = __hmul2(l11, dy);
            // fused sigma: u0^2 + u1^2 - log2(opac), so exp2(-sigma) = opac * exp2(-(u0^2+u1^2))
            const __half2 sigma = __hfma2(u1, u1, __hfma2(u0, u0, neg_log_opac));
            // threshold in sigma domain (sigma < log2(1/AT) implies alpha > AT);
            // strict < avoids boundary rounding from exp2 approx; comparison
            // runs parallel with hneg2+exp2, off the critical path
            const uint32_t ok_mask = __hlt2_mask(sigma, sigma_thresh);
            const __half2 alpha    = __h2exp2_approx(__hneg2(sigma));
            const __half2 alpha_ht = __h2if(ok_mask, alpha, __float2half2_rn(0.0f));
            const __half2 vis      = __hmul2(alpha_ht, acc[pp][3]);
            acc[pp][0]             = __hfma2(cr, vis, acc[pp][0]);
            acc[pp][1]             = __hfma2(cg, vis, acc[pp][1]);
            acc[pp][2]             = __hfma2(cb, vis, acc[pp][2]);
            acc[pp][3]             = __hsub2_sat(acc[pp][3], vis);
        }
    };

    // CTA → (mt_id, mt_batch_idx) mapping: plain sequential dispatch.
    int32_t mt_batch_idx = blockIdx.x;
    if (warp_id == 0)
    {
        smem_mt_id = find_macro_tile(args.mt_batch_offsets, args.n_macro_tiles, mt_batch_idx);
    }
    __syncthreads();
    const int32_t mt_id  = smem_mt_id;
    const int32_t mt_col = mt_id % args.macro_tile_cols;
    const int32_t mt_row = mt_id / args.macro_tile_cols;

    const int32_t batch_idx = mt_batch_idx - args.mt_batch_offsets[mt_id];

    // PHASE 1: fused load + overlap + transpose + transition ----

    // macro-tile center in tile units
    const float mt_cx = static_cast<float>(mt_col * FUSED_MACRO_TILE_WIDTH) + FUSED_MACRO_TILE_WIDTH * 0.5f;
    const float mt_cy = static_cast<float>(mt_row * FUSED_MACRO_TILE_HEIGHT) + FUSED_MACRO_TILE_HEIGHT * 0.5f;

    // gaussian range for this batch
    const int32_t mt_start         = args.mt_gauss_offsets[mt_id];
    const int32_t mt_end           = args.mt_gauss_offsets[mt_id + 1];
    const int32_t gauss_start      = mt_start + batch_idx * FUSED_GAUSS_BATCH_SIZE;
    const int32_t gauss_end        = min(gauss_start + FUSED_GAUSS_BATCH_SIZE, mt_end);
    const int32_t n_gauss          = gauss_end - gauss_start;
    const int32_t num_mini_batches = (n_gauss + MINI_BATCH_SIZE - 1) / MINI_BATCH_SIZE;

    // fused load + visibility mask generation: each warp processes one mini-batch
    // per iteration. lane j loads gaussian j and computes overlap against all 32
    // render-tiles, then a warp bit-matrix transpose converts the per-gaussian
    // tile masks into per-tile gaussian masks written directly to smem_masks.
#pragma unroll 1
    for (int32_t mb = warp_id; mb < num_mini_batches; mb += CTA_WARPS)
    {
        const int32_t i = mb * MINI_BATCH_SIZE + lane_id;

        uint32_t tile_mask = 0;
        if (i < n_gauss)
        {
            const int32_t g = args.mt_gauss_ids_sorted[gauss_start + i];

            float2 pos;
            AssignAs<float2>(pos, args.means2d[g * 2]);
            __half2 raw_conic[2];
            AssignAs<uint2>(raw_conic, args.conics[g * 4]);

            // write persistent smem for phase 2
            const float mt_rel_x = pos.x * INV_TILE_SIZE - mt_cx;
            const float mt_rel_y = pos.y * INV_TILE_SIZE - mt_cy;
            // gaussian position in render-tile units relative to macro-tile center
            const __half2 xy_mt = __floats2half2_rn(mt_rel_x, mt_rel_y);
            smem_xy_mt[i]       = xy_mt;

            // cholesky factors in render-tile units, with -log2(opac) fused into the conic
            // slot formerly occupied by raw opacity (eliminates one hmul2 in the rasterization loop)
            const float opacity   = __half2float(__high2half(raw_conic[1]));
            const float log2_opac = __log2f(opacity);
            __half2 rast_conic[2] = {
                __hmul2(raw_conic[0], __floats2half2_rn(CHOL_SCALE, CHOL_SCALE)),
                __floats2half2_rn(__half2float(__low2half(raw_conic[1])) * CHOL_SCALE, -log2_opac)};
            AssignAs<uint2>(smem_conic_raw[i], rast_conic);

            // neg_l1_rcp_C and t_rast need fp32 (division, log2)
            const float c_l1_f = __half2float(__high2half(rast_conic[0]));
            const float c_l2_f = __half2float(__low2half(rast_conic[1]));

            const float C            = c_l1_f * c_l1_f + c_l2_f * c_l2_f;
            const float neg_l1_rcp_C = (C > 0.f) ? (-c_l1_f / C) : 0.f;
            const float t_rast       = min(MAX_EXTEND_CHOL_SQ, log2_opac + LOG2_INV_AT);

            tile_mask = GetMacroTileVisibilityMask(xy_mt, rast_conic, neg_l1_rcp_C, t_rast);
        }

        // warp bit-matrix transpose: per-gaussian tile masks -> per-tile gaussian masks
        const uint32_t gauss_mask                  = WarpBitTranspose(tile_mask);
        smem_masks[mb * MACRO_TILE_SIZE + lane_id] = gauss_mask;
    }

    // zero remaining mask slots (no shuffles, just stores)
#pragma unroll 1
    for (int32_t mb = num_mini_batches + warp_id; mb < MAX_MINI_BATCHES; mb += CTA_WARPS)
    {
        smem_masks[mb * MACRO_TILE_SIZE + lane_id] = 0;
    }

    // deferred color load
#pragma unroll 1
    for (int32_t i = threadIdx.x; i < n_gauss; i += blockDim.x)
    {
        const int32_t g = args.mt_gauss_ids_sorted[gauss_start + i];
        AssignAs<uint2>(smem_color[i], args.colors[g * 4]);
    }

    __syncthreads(); // persistent smem + masks visible to all warps

    // transition: warp 0 reads combined masks from smem, computes active_mask, writes output mask.
    // tile_buffer write positions are fixed (batch_cta_idx * MACRO_TILE_SIZE + t) so no offset
    // allocation is needed for addressing.
    if (warp_id == 0)
    {
        uint32_t any_hit = 0;
#pragma unroll
        for (int32_t mb = 0; mb < MAX_MINI_BATCHES; ++mb)
        {
            any_hit |= smem_masks[mb * MACRO_TILE_SIZE + lane_id];
        }

        const int32_t global_tile_x = mt_col * FUSED_MACRO_TILE_WIDTH + (lane_id % FUSED_MACRO_TILE_WIDTH);
        const int32_t global_tile_y = mt_row * FUSED_MACRO_TILE_HEIGHT + (lane_id / FUSED_MACRO_TILE_WIDTH);
        const uint32_t valid_mask =
            __ballot_sync(~0u, (global_tile_x < args.tile_width) && (global_tile_y < args.tile_height));
        const uint32_t active_mask = __ballot_sync(~0u, any_hit != 0) & valid_mask;

        // populate tile work queue for multi-warp rasterization
        if (active_mask & (1u << lane_id))
        {
            smem_tile_queue[__popc(active_mask & ((1u << lane_id) - 1))] = lane_id;
        }
        // get number of tiles to render
        const int32_t n_active_tiles = __popc(active_mask);
        if (lane_id == 0)
        {
            // write queue metadata
            smem_queue_size = n_active_tiles;
            smem_queue_idx  = CTA_WARPS;
            // write output mask for post-blend kernel
            args.out_active_mask[mt_batch_idx] = active_mask;
        }
    }
    __syncthreads();

    // PHASE 2: per-tile rasterization

    // work-stealing: warps pull tiles from shared queue via atomics
    const int32_t n_active_tiles = smem_queue_size;
    int32_t queue_idx            = warp_id;

#pragma unroll 1
    while (queue_idx < n_active_tiles)
    {
        // rasterize one tile and write its RGBT to tile_buffer.
        // tile_buffer layout is fixed: slot (batch_cta_idx, t) lives at index batch_cta_idx * MACRO_TILE_SIZE + t.
        const int32_t tile_idx = smem_tile_queue[queue_idx];

        // combined pixel offset: render-tile column + intra-tile pixel x, in macro-tile-relative tile units
        const __half2 offset_x = __float2half2_rn(static_cast<float>(tile_idx % FUSED_MACRO_TILE_WIDTH) + px);
        // combined pixel-pair offsets: render-tile row + intra-tile pixel y + per-pair row step
        const __half2 base_y = __float2half2_rn(static_cast<float>(tile_idx / FUSED_MACRO_TILE_WIDTH) + py);

        // distance in rows between even/odd lanes of the accumulator.
        // for GAUSS_PER_WARP=2 each half-warp (16 lanes) covers the entire tile, so the row stride
        // halves and N_PAIRS doubles compared to GAUSS_PER_WARP=1 — total pixel coverage per half is identical.
        constexpr int32_t ROW_STRIDE  = (32 / GAUSS_PER_WARP) / TILE_SIZE;
        constexpr float ROW_PAIR_STEP = ROW_STRIDE * INV_TILE_SIZE;
        __half2 offset_y[N_PAIRS];
#pragma unroll
        for (int32_t pp = 0; pp < N_PAIRS; ++pp)
        {
            offset_y[pp] = __hadd2(base_y, __floats2half2_rn((2 * pp) * ROW_PAIR_STEP, (2 * pp + 1) * ROW_PAIR_STEP));
        }

        constexpr int32_t TILE_SIZE_BYTES = TILE_SIZE * TILE_SIZE * 4 * sizeof(__half);
        constexpr int32_t PAIR_STRIDE     = 32 * sizeof(uint4);

        // seed accumulators to identity.
        // For GAUSS_PER_WARP=2 the drain loop wipes half-B's slot back to identity at the
        // start of every iteration, so half-B never needs a special seed: its
        // identity init here just feeds into the (immediately overwritten) wipe.
        TileAccum acc_rgbt;
#pragma unroll
        for (int32_t pp = 0; pp < N_PAIRS; ++pp)
        {
            acc_rgbt[pp][0] = __float2half2_rn(0.f);
            acc_rgbt[pp][1] = __float2half2_rn(0.f);
            acc_rgbt[pp][2] = __float2half2_rn(0.f);
            acc_rgbt[pp][3] = __float2half2_rn(1.f);
        }

        // ring-buffer rasterization: stream masks across mini-batches into a
        // 64-entry queue, drain in 32-entry batches so the 8x-unrolled core
        // always runs at full ILP.
        constexpr int32_t QUEUE_MASK    = RASTERIZE_QUEUE_CAPACITY - 1;
        constexpr int32_t DRAIN_BATCH   = RASTERIZE_QUEUE_CAPACITY / 2;
        constexpr int32_t UNROLL_FACTOR = GAUSS_PER_WARP == 1 ? 8 : 4;

        auto *warp_queue   = smem_warp_queue + warp_id * RASTERIZE_QUEUE_CAPACITY;
        int32_t write_head = 0;
        int32_t read_head  = 0;
        int32_t mini_batch = 0;
        uint32_t tile_done = 0;

#pragma unroll 1
        while (mini_batch < num_mini_batches || write_head > read_head)
        {
            // fill phase: stream masks into queue until >= DRAIN_BATCH entries or out of masks
#pragma unroll 1
            while (mini_batch < num_mini_batches && write_head - read_head < DRAIN_BATCH)
            {
                const uint32_t mask = smem_masks[mini_batch * MACRO_TILE_SIZE + tile_idx];
                if (mask & (1u << lane_id))
                {
                    const int32_t pos = __popc(mask & ((1u << lane_id) - 1));

                    warp_queue[(write_head + pos) & QUEUE_MASK] = mini_batch * MINI_BATCH_SIZE + lane_id;
                }
                write_head += __popc(mask);
                ++mini_batch;
            }
            __syncwarp();

            // process phase: drain up to DRAIN_BATCH entries
            const int32_t count = min(write_head - read_head, DRAIN_BATCH);
            const int32_t base  = read_head & QUEUE_MASK;
            if constexpr (GAUSS_PER_WARP == 1)
            {
                const int32_t n_unrolled = count & ~(UNROLL_FACTOR - 1);
#pragma unroll 1
                for (int32_t q = 0; q < n_unrolled; q += UNROLL_FACTOR)
                {
                    static_assert(UNROLL_FACTOR == 4 || UNROLL_FACTOR == 8, "UNROLL_FACTOR must be 4 or 8");
                    using VT = std::conditional_t<UNROLL_FACTOR == 4, uint2, uint4>;
                    // let's use batched loads of queue indices to improve performance
                    uint16_t b_idx[UNROLL_FACTOR];
                    AssignAs<VT>(b_idx, warp_queue[base + q]);
#pragma unroll
                    for (int32_t i = 0; i < UNROLL_FACTOR; ++i)
                    {
                        rasterizeGaussian(b_idx[i], offset_x, offset_y, acc_rgbt);
                    }
                }
                // tail
#pragma unroll 1
                for (int32_t q = n_unrolled; q < count; ++q)
                {
                    rasterizeGaussian(warp_queue[base + q], offset_x, offset_y, acc_rgbt);
                }
            }
            else
            {
                // reset half-B's accumulator to identity so the next drain batch's compose
                // adds half-B's fresh contributions on top of pure-identity; half-A carries
                // forward the merged state to serve as the running "previous-merged" seed.
                if (!is_half_a)
                {
#pragma unroll
                    for (int32_t pp = 0; pp < N_PAIRS; ++pp)
                    {
                        acc_rgbt[pp][0] = __float2half2_rn(0.f);
                        acc_rgbt[pp][1] = __float2half2_rn(0.f);
                        acc_rgbt[pp][2] = __float2half2_rn(0.f);
                        acc_rgbt[pp][3] = __float2half2_rn(1.f);
                    }
                }

                // GAUSS_PER_WARP=2 drain: half-A processes a contiguous prefix of the queue, half-B the
                // suffix, so the compose below merges them in front-to-back order. Both halves
                // run the same loop shape with their own (base, count) parameters.
                const int32_t n_unroll_iters = count / (2 * UNROLL_FACTOR);
                const int32_t n_unrolled     = n_unroll_iters * UNROLL_FACTOR;
                const int32_t half_a_total   = (n_unroll_iters > 0) ? n_unrolled : ((count + 1) >> 1);

                const int32_t unrolled_base = is_half_a ? base : (base + half_a_total);
                const int32_t tail_base     = unrolled_base + n_unrolled;
                const int32_t tail_count =
                    is_half_a ? (half_a_total - n_unrolled) : (count - half_a_total - n_unrolled);

#pragma unroll 1
                for (int32_t q = 0; q < n_unrolled; q += UNROLL_FACTOR)
                {
                    static_assert(UNROLL_FACTOR == 4 || UNROLL_FACTOR == 8, "UNROLL_FACTOR must be 4 or 8");
                    using VT = std::conditional_t<UNROLL_FACTOR == 4, uint2, uint4>;
                    // each half-warp broadcasts UNROLL_FACTOR queue indices to its 16 lanes via
                    // a single LDS with two aligned effective addresses (one per half).
                    uint16_t b_idx[UNROLL_FACTOR];
                    AssignAs<VT>(b_idx, warp_queue[unrolled_base + q]);
#pragma unroll
                    for (int32_t i = 0; i < UNROLL_FACTOR; ++i)
                    {
                        rasterizeGaussian(b_idx[i], offset_x, offset_y, acc_rgbt);
                    }
                }
#pragma unroll 1
                for (int32_t q = 0; q < tail_count; ++q)
                {
                    // no QUEUE_MASK wrap needed: same invariant as the unrolled phase
                    // (base in {0, 32}, my_tail_base + q < QUEUE_CAPACITY).
                    rasterizeGaussian(warp_queue[tail_base + q], offset_x, offset_y, acc_rgbt);
                }

                // half-warp merge: exchange accumulator state with the shfl_xor(16) partner,
                // then redundantly compose A-on-top-of-B on both halves so every lane ends
                // up holding the full merged GAUSS_PER_WARP=2 state. this lets the saturation test below
                // (and the write-out's GAUSS_PER_WARP=2->GAUSS_PER_WARP=1 reshape) run without any half-warp masking.
                TileAccum other;
#pragma unroll
                for (int32_t pp = 0; pp < N_PAIRS; ++pp)
                {
#pragma unroll
                    for (int32_t k = 0; k < 4; ++k)
                    {
                        other[pp][k] = __shfl_xor_sync(~0u, acc_rgbt[pp][k], 16);
                    }
                }
#pragma unroll
                for (int32_t pp = 0; pp < N_PAIRS; ++pp)
                {
                    const __half2 a_r = is_half_a ? acc_rgbt[pp][0] : other[pp][0];
                    const __half2 a_g = is_half_a ? acc_rgbt[pp][1] : other[pp][1];
                    const __half2 a_b = is_half_a ? acc_rgbt[pp][2] : other[pp][2];
                    const __half2 a_t = is_half_a ? acc_rgbt[pp][3] : other[pp][3];
                    const __half2 b_r = is_half_a ? other[pp][0] : acc_rgbt[pp][0];
                    const __half2 b_g = is_half_a ? other[pp][1] : acc_rgbt[pp][1];
                    const __half2 b_b = is_half_a ? other[pp][2] : acc_rgbt[pp][2];
                    const __half2 b_t = is_half_a ? other[pp][3] : acc_rgbt[pp][3];
                    acc_rgbt[pp][0]   = __hfma2(a_t, b_r, a_r);
                    acc_rgbt[pp][1]   = __hfma2(a_t, b_g, a_g);
                    acc_rgbt[pp][2]   = __hfma2(a_t, b_b, a_b);
                    acc_rgbt[pp][3]   = __hmul2(a_t, b_t);
                }
            }
            read_head += count;

            // early exit: all pixels in this tile saturated
#pragma unroll
            for (int32_t pp = 0; pp < N_PAIRS; ++pp)
            {
                uint32_t stripe_done = __hlt2_mask(acc_rgbt[pp][3], __float2half2_rn(MIN_TRANSPARENCY));
                if constexpr (N_PAIRS > 1)
                {
                    stripe_done &= (0x10001u << pp);
                }
                tile_done |= stripe_done;
            }
            // done bitmask: bits 0..N_PAIRS-1 = even pixels, bits 16..16+N_PAIRS-1 = odd pixels.
            // for N_PAIRS == 1, use ~0u for all pixels done
            constexpr uint32_t DONE_MASK = N_PAIRS == 1 ? ~0u : (((1u << N_PAIRS) - 1u) * 0x10001);

            if (__ballot_sync(~0u, tile_done == DONE_MASK) == ~0u)
            {
                break;
            }
        }

        // write tile — pair-major layout for coalesced access across the warp
        const int64_t tile_base = static_cast<int64_t>(mt_batch_idx) * (MACRO_TILE_SIZE * TILE_SIZE_BYTES) +
                                  tile_idx * TILE_SIZE_BYTES + lane_id * sizeof(uint4);
        uint8_t *dst_ptr = reinterpret_cast<uint8_t *>(args.tile_buffer) + tile_base;
        if constexpr (GAUSS_PER_WARP == 1)
        {
#pragma unroll
            for (int32_t pp = 0; pp < N_PAIRS; ++pp)
            {
                // let's use streaming stores since we're not going to touch this data anymore
                uint4 val;
                AssignAs<uint4>(val, acc_rgbt[pp]);
                __stcs(reinterpret_cast<uint4 *>(dst_ptr + pp * PAIR_STRIDE), val);
            }
        }
        else
        {
            // GAUSS_PER_WARP=2 write-out: reshape the merged GAUSS_PER_WARP=2 accumulator back to the tile_buffer's
            // GAUSS_PER_WARP=1 (full-warp) layout. lane i and lane i+16 hold identical merged GAUSS_PER_WARP=2 data
            // post-compose, so they extract complementary halves: half-A keeps the lows
            // (lower GAUSS_PER_WARP=1 thread_y rows), half-B keeps the highs (upper thread_y rows).
            constexpr int32_t N_PAIRS_OUT = N_PAIRS / 2;
#pragma unroll
            for (int32_t pp = 0; pp < N_PAIRS_OUT; ++pp)
            {
                __half2 out[4];
#pragma unroll
                for (int32_t k = 0; k < 4; ++k)
                {
                    out[k] = is_half_a ? __lows2half2(acc_rgbt[2 * pp][k], acc_rgbt[2 * pp + 1][k])
                                       : __highs2half2(acc_rgbt[2 * pp][k], acc_rgbt[2 * pp + 1][k]);
                }
                uint4 val;
                AssignAs<uint4>(val, out);
                __stcs(reinterpret_cast<uint4 *>(dst_ptr + pp * PAIR_STRIDE), val);
            }
        }

        // get next tile from queue
        if (lane_id == 0)
        {
            queue_idx = atomicAdd(&smem_queue_idx, 1);
        }
        queue_idx = __shfl_sync(~0u, queue_idx, 0);
    }

}

// post-blend kernel: composites partial RGBT batches into final image.
// multiple warps per CTA, each warp independently processes one render tile.
template<int32_t TILE_SIZE>
__global__ void __launch_bounds__(BLEND_CTA_SIZE, BLEND_MIN_BLOCKS)
    macro_tile_post_blend_kernel(const int32_t n_tiles, const int32_t tile_width, const int32_t macro_tile_cols,
                                 const int32_t *__restrict__ mt_batch_offsets, const __half *__restrict__ tile_buffer,
                                 const uint32_t *__restrict__ in_active_mask, const __half *__restrict__ backgrounds,
                                 const uint32_t image_width, const uint32_t image_height,
                                 __half *__restrict__ render_colors)
{
    static_assert(TILE_SIZE == 8 || TILE_SIZE == 16, "TILE_SIZE must be 8 or 16");
    constexpr int32_t MACRO_TILE_SIZE   = FUSED_MACRO_TILE_WIDTH * FUSED_MACRO_TILE_HEIGHT;
    constexpr int32_t WARPS_PER_CTA     = BLEND_CTA_SIZE / 32;
    constexpr int32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / 32;
    constexpr int32_t N_PAIRS           = PIXELS_PER_THREAD / 2;

    using TileAccum = __half2[N_PAIRS][4];

    const int32_t warp_id       = threadIdx.x >> 5;
    const int32_t lane_id       = threadIdx.x & 31;
    const int32_t flat_tile_idx = blockIdx.x * WARPS_PER_CTA + warp_id;

    if (flat_tile_idx >= n_tiles)
    {
        return;
    }

    const int32_t tile_x = flat_tile_idx % tile_width;
    const int32_t tile_y = flat_tile_idx / tile_width;

    const int32_t mt_col = tile_x / FUSED_MACRO_TILE_WIDTH;
    const int32_t mt_row = tile_y / FUSED_MACRO_TILE_HEIGHT;
    const int32_t mt_id  = mt_row * macro_tile_cols + mt_col;
    const int32_t local_rt =
        (tile_y % FUSED_MACRO_TILE_HEIGHT) * FUSED_MACRO_TILE_WIDTH + (tile_x % FUSED_MACRO_TILE_WIDTH);

    constexpr uint32_t TILE_X_MASK  = TILE_SIZE - 1;
    constexpr uint32_t TILE_X_SHIFT = (TILE_SIZE == 16) ? 4 : 3;

    const int32_t thread_x = lane_id & TILE_X_MASK;
    const int32_t thread_y = lane_id >> TILE_X_SHIFT;
    const int32_t out_x    = tile_x * TILE_SIZE + thread_x;

    // per-pair accumulators: N_PAIRS x {R, G, B, T}
    TileAccum acc_rgbt;
#pragma unroll
    for (int32_t pp = 0; pp < N_PAIRS; ++pp)
    {
        acc_rgbt[pp][0] = __float2half2_rn(0.0f);
        acc_rgbt[pp][1] = __float2half2_rn(0.0f);
        acc_rgbt[pp][2] = __float2half2_rn(0.0f);
        acc_rgbt[pp][3] = __float2half2_rn(1.0f);
    }

    const int32_t batch_start = mt_batch_offsets[mt_id];
    const int32_t batch_end   = mt_batch_offsets[mt_id + 1];

    constexpr int32_t TILE_SIZE_BYTES = TILE_SIZE * TILE_SIZE * 4 * sizeof(__half);
    // pair-major layout: all 32 lanes' data for one pair contiguous, then next pair
    constexpr int32_t PAIR_STRIDE = 32 * sizeof(uint4);

    uint32_t tile_done = 0;
#pragma unroll 1
    for (int32_t batch_idx = batch_start; batch_idx < batch_end; ++batch_idx)
    {
        const uint32_t mask = in_active_mask[batch_idx];
        if (!(mask & (1u << local_rt)))
        {
            continue;
        }

        // fixed layout: tile (batch_idx, local_rt) lives at slot batch_idx * MACRO_TILE_SIZE + local_rt
        const int64_t pos =
            static_cast<int64_t>(batch_idx) * (MACRO_TILE_SIZE * TILE_SIZE_BYTES) + local_rt * TILE_SIZE_BYTES;
        const int64_t tile_base = pos + lane_id * sizeof(uint4);

#pragma unroll
        for (int32_t pp = 0; pp < N_PAIRS; ++pp)
        {
            __half2 batch[4];
            AssignAs<uint4>(batch, reinterpret_cast<const uint8_t *>(tile_buffer)[tile_base + pp * PAIR_STRIDE]);

            acc_rgbt[pp][0] = __hfma2(acc_rgbt[pp][3], batch[0], acc_rgbt[pp][0]);
            acc_rgbt[pp][1] = __hfma2(acc_rgbt[pp][3], batch[1], acc_rgbt[pp][1]);
            acc_rgbt[pp][2] = __hfma2(acc_rgbt[pp][3], batch[2], acc_rgbt[pp][2]);
            acc_rgbt[pp][3] = __hmul2(acc_rgbt[pp][3], batch[3]);

            uint32_t is_done = __hlt2_mask(acc_rgbt[pp][3], __float2half2_rn(MIN_TRANSPARENCY));
            if constexpr (N_PAIRS > 1)
            {
                is_done &= (0x10001u << pp);
            }
            tile_done |= is_done;
        }

        constexpr uint32_t DONE_MASK = N_PAIRS == 1 ? ~0u : (((1u << N_PAIRS) - 1u) * 0x10001);

        // warp-wide early exit: all pixels in this tile are saturated
        if (__ballot_sync(~0u, tile_done == DONE_MASK) == ~0u)
        {
            break;
        }
    }

    if (out_x < image_width)
    {
        __half2 bg[2];
        if (backgrounds != nullptr)
        {
            AssignAs<uint2>(bg, backgrounds[0]);
        }
        else
        {
            bg[0] = __float2half2_rn(0.0f);
            bg[1] = __float2half2_rn(0.0f);
        }

        constexpr uint32_t ROW_STRIDE = 32 / TILE_SIZE;

#pragma unroll
        for (int32_t pp = 0; pp < N_PAIRS; ++pp)
        {
            const __half2 final_r = __hfma2(acc_rgbt[pp][3], __low2half2(bg[0]), acc_rgbt[pp][0]);
            const __half2 final_g = __hfma2(acc_rgbt[pp][3], __high2half2(bg[0]), acc_rgbt[pp][1]);
            const __half2 final_b = __hfma2(acc_rgbt[pp][3], __low2half2(bg[1]), acc_rgbt[pp][2]);

            const uint32_t out_y0 = tile_y * TILE_SIZE + thread_y + (2 * pp) * ROW_STRIDE;
            if (out_y0 < image_height)
            {
                const __half2 out[2] = {__lows2half2(final_r, final_g), __lows2half2(final_b, acc_rgbt[pp][3])};
                AssignAs<uint2>(reinterpret_cast<uint2 *>(render_colors)[out_y0 * image_width + out_x], out);
            }
            const uint32_t out_y1 = out_y0 + ROW_STRIDE;
            if (out_y1 < image_height)
            {
                const __half2 out[2] = {__highs2half2(final_r, final_g), __highs2half2(final_b, acc_rgbt[pp][3])};
                AssignAs<uint2>(reinterpret_cast<uint2 *>(render_colors)[out_y1 * image_width + out_x], out);
            }
        }
    }
}

// host launch wrappers

void launch_macro_tile_rasterize(int32_t total_ctas, int32_t n_macro_tiles, int32_t macro_tile_cols, int32_t tile_size,
                                 int32_t tile_width, int32_t tile_height, const at::Tensor &means2d,
                                 const at::Tensor &conics, const at::Tensor &colors, const at::Tensor &mt_gauss_offsets,
                                 const at::Tensor &mt_gauss_ids_sorted, const at::Tensor &mt_batch_offsets,
                                 at::Tensor &tile_buffer, at::Tensor &active_mask)
{
    TORCH_CHECK(tile_size == 8 || tile_size == 16, "fused kernel supports tile_size 8 or 16, got ", tile_size);

    if (total_ctas == 0)
    {
        return;
    }

    auto stream = at::cuda::getCurrentCUDAStream();

    MacroTileRasterizeArgs kargs;
    kargs.n_macro_tiles       = n_macro_tiles;
    kargs.macro_tile_cols     = macro_tile_cols;
    kargs.tile_width          = tile_width;
    kargs.tile_height         = tile_height;
    kargs.means2d             = means2d.data_ptr<float>();
    kargs.conics              = reinterpret_cast<const __half *>(conics.data_ptr<at::Half>());
    kargs.colors              = reinterpret_cast<const __half *>(colors.data_ptr<at::Half>());
    kargs.mt_gauss_offsets    = mt_gauss_offsets.data_ptr<int32_t>();
    kargs.mt_gauss_ids_sorted = mt_gauss_ids_sorted.data_ptr<int32_t>();
    kargs.mt_batch_offsets    = mt_batch_offsets.data_ptr<int32_t>();
    kargs.tile_buffer         = reinterpret_cast<__half *>(tile_buffer.data_ptr<at::Half>());
    kargs.out_active_mask     = reinterpret_cast<uint32_t *>(active_mask.data_ptr<int32_t>());

    auto launch = [&](auto kernel_fn) { kernel_fn<<<total_ctas, RASTERIZE_CTA_SIZE, 0, stream>>>(kargs); };

    switch (tile_size)
    {
    case 16:
        launch(macro_tile_rasterize_kernel<16>);
        break;
    case 8:
        launch(macro_tile_rasterize_kernel<8>);
        break;
    }
}

void launch_macro_tile_post_blend(int32_t tile_width, int32_t tile_height, int32_t tile_size, int32_t n_macro_tiles,
                                  int32_t macro_tile_cols, const at::Tensor &mt_batch_offsets,
                                  const at::Tensor &tile_buffer, const at::Tensor &active_mask,
                                  const at::Tensor &backgrounds, uint32_t image_width, uint32_t image_height,
                                  at::Tensor &render_colors)
{
    TORCH_CHECK(tile_size == 8 || tile_size == 16, "post-blend supports tile_size 8 or 16, got ", tile_size);

    constexpr int32_t WARPS_PER_CTA = BLEND_CTA_SIZE / 32;

    const int32_t n_tiles = tile_width * tile_height;
    const int32_t grid    = (n_tiles + WARPS_PER_CTA - 1) / WARPS_PER_CTA;

    const __half *bg_ptr =
        backgrounds.defined() ? reinterpret_cast<const __half *>(backgrounds.data_ptr<at::Half>()) : nullptr;

    auto stream = at::cuda::getCurrentCUDAStream();

    auto launch = [&](auto kernel_fn) {
        kernel_fn<<<grid, BLEND_CTA_SIZE, 0, stream>>>(
            n_tiles, tile_width, macro_tile_cols, mt_batch_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const __half *>(tile_buffer.data_ptr<at::Half>()),
            reinterpret_cast<const uint32_t *>(active_mask.data_ptr<int32_t>()), bg_ptr, image_width, image_height,
            reinterpret_cast<__half *>(render_colors.data_ptr<at::Half>()));
    };

    switch (tile_size)
    {
    case 16:
        launch(macro_tile_post_blend_kernel<16>);
        break;
    case 8:
        launch(macro_tile_post_blend_kernel<8>);
        break;
    }
}

} // namespace higs
