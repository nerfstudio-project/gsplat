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

#include "Common.h"
#include "RasterizeSparseAddressing.cuh"
#include "RasterizeToPixels3DGSDevice.cuh"

namespace gsplat
{
// Sparse counterpart of rasterize_contributing_common_kernel
// (RasterizeContributingCommon.cuh). It reuses the same per-op Accumulator
// (init_image / init / transmittance / on_hit / finalize) and the same
// front-to-back traversal; only the addressing differs, exactly as for the
// sparse rasterizer: each block owns one active tile (decoded from
// active_tiles), the gaussian range comes from the compacted 1-D tile_offsets,
// and outputs are packed in original-pixel order ([P, ...]) -- the per-pixel
// slot is looked up through the per-tile bitmask (popcount rank) and pixel_map.
//
// Accumulators index their outputs by the slot passed as `pix_id`; here that is
// the global packed [P] index, so init_image is called with image_id 0 (no
// per-image offset). Inactive in-tile positions stay resident to cooperatively
// load shared memory but never call finalize.
template<uint32_t TILE_SIZE, uint32_t CTA_SIZE, typename Accumulator>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_contributing_common_sparse_kernel(
    const vec2 *__restrict__ means2d,    // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,     // [I, N, 3] or [nnz, 3]
    const float *__restrict__ opacities, // [I, N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // sparse layout
    const int32_t *__restrict__ active_tiles,      // [AT]
    const int32_t *__restrict__ tile_offsets,      // [AT + 1]
    const int32_t *__restrict__ flatten_ids,       // [n_isects]
    const uint64_t *__restrict__ tile_pixel_mask,  // [AT, words]
    const int64_t *__restrict__ tile_pixel_cumsum, // [AT], inclusive
    const int64_t *__restrict__ pixel_map,         // [P]
    const uint32_t words,
    Accumulator accum
)
{
    constexpr uint32_t BATCH_SIZE        = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert(
        (TILE_SIZE & (TILE_SIZE - 1)) == 0, "TILE_SIZE must be a power of 2 (TILE_MASK/TILE_SHIFT rely on this)"
    );
    static_assert(PIXELS_PER_THREAD > 0, "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    // Each block owns one active tile; decode its dense (image, tile) id.
    const uint32_t ord        = blockIdx.x;
    const uint32_t n_tiles    = tile_width * tile_height;
    const int32_t global_tile = active_tiles[ord];
    uint32_t image_id, tile_x, tile_y;
    sparse_decode_tile(global_tile, n_tiles, tile_width, image_id, tile_x, tile_y);
    (void)image_id; // outputs are global [P, ...]; no per-image offset needed

    const uint32_t tid      = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // Outputs are global [P, ...] indexed by pixel_map slot, so no per-image
    // pointer offset.
    accum.init_image(0, image_width, image_height);

    // Exclusive start of this tile's active pixels within pixel_map.
    const int64_t pix_start         = (ord == 0) ? 0 : tile_pixel_cumsum[ord - 1];
    const uint64_t *tile_mask_words = tile_pixel_mask + static_cast<int64_t>(ord) * words;

    const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px       = static_cast<float>(out_x) + 0.5f;

    float py[PIXELS_PER_THREAD];
    int32_t pix_slot[PIXELS_PER_THREAD]; // packed [P] output slot, -1 if inactive
    uint32_t done_mask = 0;
#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        const uint32_t local_row = thread_y + p * ROW_STRIDE;
        py[p]                    = static_cast<float>(tile_y * TILE_SIZE + local_row) + 0.5f;
        pix_slot[p]              = -1;

        // Raster-order index within the tile, matching build_sparse_tile_layout.
        const uint32_t in_tile = local_row * TILE_SIZE + thread_x;
        const int64_t slot     = sparse_pixel_slot(tile_mask_words, in_tile, pix_start, pixel_map);
        if(slot < 0)
        {
            done_mask |= (1u << p);
            continue;
        }
        pix_slot[p] = static_cast<int32_t>(slot);
    }

    const int32_t range_start  = tile_offsets[ord];
    const int32_t range_end    = tile_offsets[ord + 1];
    const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

    extern __shared__ int s[];
    int32_t *id_batch      = reinterpret_cast<int32_t *>(s);                          // [BATCH_SIZE]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]);         // [BATCH_SIZE]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]
    accum.init_shared(&conic_batch[BATCH_SIZE]);

#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        accum.init(p, tid, pix_slot[p]);
    }

#pragma unroll 1
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        const uint32_t batch_start = range_start + BATCH_SIZE * b;
        const uint32_t idx         = batch_start + tid;
        if(idx < static_cast<uint32_t>(range_end))
        {
            const int32_t g       = flatten_ids[idx];
            id_batch[tid]         = g;
            const vec2 xy         = means2d[g];
            xy_opacity_batch[tid] = {xy.x, xy.y, opacities[g]};
            conic_batch[tid]      = conics[g];
        }

        if constexpr(CTA_SIZE <= 32)
        {
            __syncwarp();
        }
        else
        {
            __syncthreads();
        }

        const uint32_t batch_size = min(BATCH_SIZE, static_cast<uint32_t>(range_end - batch_start));
        for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
        {
            const int32_t g        = id_batch[t];
            const int32_t local_id = accum.local_id(g);
            const vec3 conic       = conic_batch[t];
            const vec3 xy_opac     = xy_opacity_batch[t];
            const float opac       = xy_opac.z;
            const float dx         = xy_opac.x - px;

#pragma unroll
            for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
            {
                if(done_mask & (1u << p))
                {
                    continue;
                }

                const float dy          = xy_opac.y - py[p];
                const GaussianWeight gw = eval_gaussian_weight(conic, dx, dy, opac);
                if(!gw.valid)
                {
                    continue;
                }
                const float alpha = gw.alpha;

                const float T      = accum.transmittance(p);
                const float next_T = T * (1.0f - alpha);
                if(next_T <= TRANSMITTANCE_THRESHOLD)
                {
                    done_mask |= (1u << p);
                    continue;
                }

                accum.on_hit(p, tid, pix_slot[p], local_id, alpha, T, next_T);
            }
        }

        if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE)
        {
            break;
        }
    }

#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if(pix_slot[p] >= 0)
        { // active in-tile position
            accum.finalize(p, tid, pix_slot[p]);
        }
    }
}
} // namespace gsplat
