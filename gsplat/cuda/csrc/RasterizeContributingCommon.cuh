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

namespace gsplat
{
template<uint32_t CTA_SIZE>
constexpr int64_t rasterize_contributing_common_shmem_size()
{
    return CTA_SIZE * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));
}

template<uint32_t TILE_SIZE, uint32_t CTA_SIZE, typename Accumulator>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_contributing_common_kernel(
    const uint32_t n_isects,
    const vec2 *__restrict__ means2d,    // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,     // [I, N, 3] or [nnz, 3]
    const float *__restrict__ opacities, // [I, N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
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
    static_assert(
        PIXELS_PER_THREAD > 0,
        "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * "
        "TILE_SIZE"
    );

    const int32_t image_id     = blockIdx.x;
    const uint32_t grid_width  = gridDim.z;
    const uint32_t grid_height = gridDim.y;

    const uint32_t tile_x = blockIdx.z;
    const uint32_t tile_y = blockIdx.y;
    const int32_t tile_id = blockIdx.y * grid_width + blockIdx.z;

    const uint32_t tid      = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    tile_offsets += image_id * grid_height * grid_width;
    accum.init_image(image_id, image_width, image_height);

    const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px       = static_cast<float>(out_x) + 0.5f;

    uint32_t out_y[PIXELS_PER_THREAD];
    float py[PIXELS_PER_THREAD];
    int32_t pix_id[PIXELS_PER_THREAD];
#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        out_y[p]  = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE;
        py[p]     = static_cast<float>(out_y[p]) + 0.5f;
        pix_id[p] = static_cast<int32_t>(out_y[p] * image_width + out_x);
    }

    // Keep OOB threads alive so the block can cooperatively load gaussians and
    // evaluate __syncthreads_count without deadlock.
    uint32_t done_mask = (out_x >= image_width) ? ALL_DONE : 0;
#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        if(out_y[p] >= image_height)
        {
            done_mask |= (1u << p);
        }
    }

    const int32_t range_start  = tile_offsets[tile_id];
    const int32_t range_end    = (image_id == static_cast<int32_t>(gridDim.x) - 1)
                                      && (tile_id == static_cast<int32_t>(grid_width * grid_height) - 1)
                                   ? n_isects
                                   : tile_offsets[tile_id + 1];
    const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

    extern __shared__ int s[];
    int32_t *id_batch      = reinterpret_cast<int32_t *>(s);                          // [BATCH_SIZE]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]);         // [BATCH_SIZE]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]
    accum.init_shared(&conic_batch[BATCH_SIZE]);

#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        accum.init(p, tid, pix_id[p]);
    }

#pragma unroll 1
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        const uint32_t batch_start = range_start + BATCH_SIZE * b;
        const uint32_t idx         = batch_start + tid;
        if(idx < range_end)
        {
            const int32_t g       = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
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

                const float dy    = xy_opac.y - py[p];
                const float sigma = 0.5f * (conic.x * dx * dx + conic.z * dy * dy) + conic.y * dx * dy;
                const float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
                if(sigma < 0.f || alpha < ALPHA_THRESHOLD)
                {
                    continue;
                }

                const float T      = accum.transmittance(p);
                const float next_T = T * (1.0f - alpha);
                if(next_T <= TRANSMITTANCE_THRESHOLD)
                {
                    done_mask |= (1u << p);
                    continue;
                }

                accum.on_hit(p, tid, pix_id[p], local_id, alpha, T, next_T);
            }
        }

        if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE)
        {
            break;
        }
    }

    if(out_x < image_width)
    {
#pragma unroll
        for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            if(out_y[p] < image_height)
            {
                accum.finalize(p, tid, pix_id[p]);
            }
        }
    }
}
} // namespace gsplat
