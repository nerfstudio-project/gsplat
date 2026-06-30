/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#    include <ATen/Dispatch.h>
#    include <ATen/core/Tensor.h>
#    include <c10/cuda/CUDAStream.h>

#    include "Common.h"
#    include "Dispatch.h"
#    include "Rasterization.h"
#    include "RasterizeSparseAddressing.cuh"
#    include "RasterizeToPixels3DGSDevice.cuh"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

////////////////////////////////////////////////////////////////
// Forward (sparse)
////////////////////////////////////////////////////////////////

// Sparse counterpart of rasterize_to_pixels_3dgs_fwd_kernel. The blending math
// is identical (shared in RasterizeToPixels3DGSDevice.cuh); only the addressing
// differs: each block owns one active tile (decoded from active_tiles), the
// gaussian range comes from the compacted 1-D tile_offsets, and outputs are
// written packed in original-pixel order at slots looked up through the
// per-tile bitmask and pixel_map.
template<uint32_t CDIM, uint32_t TILE_SIZE, uint32_t CTA_SIZE>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_to_pixels_sparse_fwd_kernel(
    const vec2 *__restrict__ means2d,      // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,       // [I, N, 3] or [nnz, 3]
    const float *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,   // [I, N] or [nnz]
    const float *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,        // [I, tile_height, tile_width]
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
    // outputs (packed in original pixel order)
    float *__restrict__ render_colors, // [P, CDIM]
    float *__restrict__ render_alphas, // [P, 1]
    int32_t *__restrict__ last_ids     // [P]
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

    const uint32_t tid      = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }

    // Exclusive start of this tile's active pixels within pixel_map (the
    // inclusive cumsum reads cumsum[ord-1]).
    const int64_t pix_start         = (ord == 0) ? 0 : tile_pixel_cumsum[ord - 1];
    const uint64_t *tile_mask_words = tile_pixel_mask + (int64_t)ord * words;

    // Per-thread pixel centers and the compacted output slot for each active
    // pixel this thread owns (-1 marks an inactive in-tile position).
    const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px       = (float)out_x + 0.5f;

    float py[PIXELS_PER_THREAD];
    int64_t out_idx[PIXELS_PER_THREAD];
    uint32_t done_mask = 0;
#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        const uint32_t local_row = thread_y + p * ROW_STRIDE;
        const uint32_t out_y     = tile_y * TILE_SIZE + local_row;
        py[p]                    = (float)out_y + 0.5f;
        out_idx[p]               = -1;

        // Raster-order index within the tile, matching the bitmask packing of
        // build_sparse_tile_layout (in_tile = (row % ts) * ts + (col % ts)).
        const uint32_t in_tile = local_row * TILE_SIZE + thread_x;
        const int64_t slot     = sparse_pixel_slot(tile_mask_words, in_tile, pix_start, pixel_map);
        if(slot < 0)
        {
            done_mask |= (1u << p);
            continue;
        }
        out_idx[p] = slot;
    }

    // Masked-off tile: write background to its active pixels and return, so no
    // active output slot is left uninitialized.
    if(masks != nullptr && !masks[global_tile])
    {
#    pragma unroll
        for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
        {
            if(done_mask & (1u << p))
            {
                continue;
            }
            const int64_t o = out_idx[p];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                render_colors[o * CDIM + k] = backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
            render_alphas[o] = 0.0f;
            last_ids[o]      = 0;
        }
        return;
    }

    // Gaussians intersecting this active tile, from the compacted offsets.
    const int32_t range_start  = tile_offsets[ord];
    const int32_t range_end    = tile_offsets[ord + 1];
    const uint32_t num_batches = (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [BATCH_SIZE]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]);         // [BATCH_SIZE]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]

    float T[PIXELS_PER_THREAD];
#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        T[p] = 1.0f;
    }
    uint32_t cur_idx[PIXELS_PER_THREAD]    = {0u};
    float pix_out[PIXELS_PER_THREAD][CDIM] = {0.f};

#    pragma unroll 1
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        // each thread fetch 1 gaussian from front to back
        const uint32_t batch_start = range_start + BATCH_SIZE * b;
        const uint32_t idx         = batch_start + tid;
        if(idx < (uint32_t)range_end)
        {
            const int32_t g       = flatten_ids[idx];
            id_batch[tid]         = g;
            const vec2 xy         = means2d[g];
            const float opac      = opacities[g];
            xy_opacity_batch[tid] = {xy.x, xy.y, opac};
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

        const uint32_t batch_size = min(BATCH_SIZE, (uint32_t)range_end - batch_start);
        for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
        {
            const vec3 conic   = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const int32_t g    = id_batch[t];
            const float *c_ptr = colors + g * CDIM;
#    pragma unroll
            for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
            {
                if(done_mask & (1u << p))
                {
                    continue;
                }
                if(rasterize_to_pixels_3dgs_blend_fwd<CDIM>(
                       conic, xy_opac, px, py[p], c_ptr, batch_start + t, T[p], pix_out[p], cur_idx[p]
                   ))
                {
                    done_mask |= (1u << p);
                }
            }
        }

        // resync before the next batch; end early if the whole tile is done
        if(__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE)
        {
            break;
        }
    }

#    pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD; ++p)
    {
        const int64_t o = out_idx[p];
        if(o < 0)
        {
            continue; // inactive in-tile position
        }
        render_alphas[o] = 1.0f - T[p];
#    pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            render_colors[o * CDIM + k]
                = backgrounds == nullptr ? pix_out[p][k] : (pix_out[p][k] + T[p] * backgrounds[k]);
        }
        last_ids[o] = static_cast<int32_t>(cur_idx[p]);
    }
}

void launch_rasterize_to_pixels_sparse_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // sparse layout
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    // outputs
    at::Tensor renders, // [P, channels]
    at::Tensor alphas,  // [P, 1]
    at::Tensor last_ids // [P]
)
{
    const uint32_t AT = active_tiles.size(0);
    if(AT == 0)
    {
        return; // no active tiles -> nothing to render
    }
    const uint32_t words = tile_pixel_mask.size(1);

    const dim3 grid = {AT, 1, 1};

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernel = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

        auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>()
        {
            const dim3 threads       = dim3{CTA_SIZE, 1, 1};
            const int64_t shmem_size = CTA_SIZE * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

            if(cudaFuncSetAttribute(
                   rasterize_to_pixels_sparse_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   shmem_size
               )
               != cudaSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size."
                );
            }

            const float *bg_ptr   = backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr;
            const bool *masks_ptr = masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr;

            rasterize_to_pixels_sparse_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    bg_ptr,
                    masks_ptr,
                    image_width,
                    image_height,
                    tile_width,
                    tile_height,
                    active_tiles.const_data_ptr<int32_t>(),
                    tile_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    reinterpret_cast<const uint64_t *>(tile_pixel_mask.const_data_ptr<int64_t>()),
                    tile_pixel_cumsum.const_data_ptr<int64_t>(),
                    pixel_map.const_data_ptr<int64_t>(),
                    words,
                    renders.data_ptr<float>(),
                    alphas.data_ptr<float>(),
                    last_ids.data_ptr<int32_t>()
                );
        };

        // One thread per pixel (CTA_SIZE == tile_size^2, PIXELS_PER_THREAD == 1).
        // Unlike the dense forward -- which packs multiple pixels per thread
        // (e.g. 16x16 tiles on 64 threads) -- the sparse forward does per-pixel
        // setup (bitmask popcount rank + pixel_map gather) that a packed layout
        // would repeat PIXELS_PER_THREAD times per thread at lower occupancy.
        // One thread per pixel measured ~1.6-2.3x faster here, matching the
        // backward's thread mapping.
        if(tile_size == 16)
        {
            launch_variant.template operator()<16, 256>();
        }
        else if(tile_size == 4)
        {
            launch_variant.template operator()<4, 16>();
        }
        else
        {
            AT_ERROR("Unsupported tile_size ", tile_size, "; supported values are {4, 16}.");
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
