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
#    include <ATen/cuda/Atomic.cuh>
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>

#    include "Common.h"
#    include "Dispatch.h"
#    include "Rasterization.h"
#    include "RasterizeSparseAddressing.cuh"
#    include "RasterizeToPixels3DGSDevice.cuh"
#    include "Utils.cuh"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Backward (sparse)
////////////////////////////////////////////////////////////////

// Sparse counterpart of rasterize_to_pixels_3dgs_bwd_kernel. One thread per
// tile pixel (block dims tile_size x tile_size); each block owns one active
// tile decoded from active_tiles. The per-gaussian gradient math is shared with
// the dense path (RasterizeToPixels3DGSDevice.cuh); only addressing differs:
// the gaussian range comes from the compacted 1-D tile_offsets, and the
// forward-output/grad reads index the packed [P, ...] buffers through the
// per-tile bitmask and pixel_map. Inactive in-tile positions stay resident to
// cooperatively load shared memory but contribute no gradient.
template<uint32_t CDIM>
__global__ void rasterize_to_pixels_sparse_bwd_kernel(
    // fwd inputs
    const vec2 *__restrict__ means2d,      // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,       // [I, N, 3] or [nnz, 3]
    const float *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,   // [I, N] or [nnz]
    const float *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,        // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
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
    // fwd outputs
    const float *__restrict__ render_alphas, // [P, 1]
    const int32_t *__restrict__ last_ids,    // [P]
    // grad of outputs
    const float *__restrict__ v_render_colors, // [P, CDIM]
    const float *__restrict__ v_render_alphas, // [P, 1]
    // grad of inputs (scattered)
    vec2 *__restrict__ v_means2d_abs, // [I, N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,     // [I, N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,      // [I, N, 3] or [nnz, 3]
    float *__restrict__ v_colors,     // [I, N, CDIM] or [nnz, CDIM]
    float *__restrict__ v_opacities   // [I, N] or [nnz]
)
{
    auto block                = cg::this_thread_block();
    const uint32_t ord        = block.group_index().x;
    const uint32_t n_tiles    = tile_width * tile_height;
    const int32_t global_tile = active_tiles[ord];
    uint32_t image_id, tile_x, tile_y;
    sparse_decode_tile(global_tile, n_tiles, tile_width, image_id, tile_x, tile_y);

    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    // Masked tile contributes only a constant background -> zero gradient.
    if(masks != nullptr && !masks[global_tile])
    {
        return;
    }

    const uint32_t local_row = block.thread_index().y;
    const uint32_t local_col = block.thread_index().x;
    const float px           = (float)(tile_x * tile_size + local_col) + 0.5f;
    const float py           = (float)(tile_y * tile_size + local_row) + 0.5f;

    // Active-pixel test + compacted output slot (-1 when this in-tile position
    // holds no requested pixel). Active positions are always in-bounds.
    const int64_t pix_start         = (ord == 0) ? 0 : tile_pixel_cumsum[ord - 1];
    const uint64_t *tile_mask_words = tile_pixel_mask + (int64_t)ord * words;
    const uint32_t in_tile          = local_row * tile_size + local_col;
    const int64_t out_idx           = sparse_pixel_slot(tile_mask_words, in_tile, pix_start, pixel_map);
    const bool inside               = out_idx >= 0;

    const int32_t range_start  = tile_offsets[ord];
    const int32_t range_end    = tile_offsets[ord + 1];
    const uint32_t block_size  = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]);
    float *rgbs_batch      = (float *)&conic_batch[block_size]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    const float T_final     = inside ? 1.0f - render_alphas[out_idx] : 0.0f;
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[out_idx] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#    pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = inside ? v_render_colors[out_idx * CDIM + k] : 0.0f;
    }
    const float v_render_a = inside ? v_render_alphas[out_idx] : 0.0f;

    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final   = cg::reduce(warp, bin_final, cg::greater<int>());
    for(uint32_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        const int32_t batch_end  = range_end - 1 - block_size * b;
        const int32_t batch_size = min((int32_t)block_size, batch_end + 1 - range_start);
        const int32_t idx        = batch_end - tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx];
            id_batch[tr]         = g;
            const vec2 xy        = means2d[g];
            const float opac     = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr]      = conics[g];
#    pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel, back to front
        for(uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end - t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if(valid)
            {
                conic                   = conic_batch[t];
                vec3 xy_opac            = xy_opacity_batch[t];
                opac                    = xy_opac.z;
                delta                   = {xy_opac.x - px, xy_opac.y - py};
                const GaussianWeight gw = eval_gaussian_weight(conic, delta.x, delta.y, opac);
                vis                     = gw.vis;
                alpha                   = gw.alpha;
                if(!gw.valid)
                {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if(!warp.any(valid))
            {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local      = {0.f, 0.f, 0.f};
            vec2 v_xy_local         = {0.f, 0.f};
            vec2 v_xy_abs_local     = {0.f, 0.f};
            float v_opacity_local   = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if(valid)
            {
                rasterize_to_pixels_3dgs_blend_bwd<CDIM>(
                    conic,
                    delta,
                    opac,
                    vis,
                    alpha,
                    rgbs_batch + t * CDIM,
                    v_render_c,
                    v_render_a,
                    T_final,
                    backgrounds,
                    v_means2d_abs != nullptr,
                    T,
                    buffer,
                    v_rgb_local,
                    v_conic_local,
                    v_xy_local,
                    v_xy_abs_local,
                    v_opacity_local
                );
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if(v_means2d_abs != nullptr)
            {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

void launch_rasterize_to_pixels_sparse_bwd_kernel(
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
    // forward outputs
    const at::Tensor render_alphas, // [P, 1]
    const at::Tensor last_ids,      // [P]
    // gradients of outputs
    const at::Tensor v_render_colors, // [P, channels]
    const at::Tensor v_render_alphas, // [P, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, channels] or [nnz, channels]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    const uint32_t AT       = active_tiles.size(0);
    const uint32_t n_isects = flatten_ids.size(0);
    if(AT == 0 || n_isects == 0)
    {
        return; // nothing to scatter into the (already zeroed) grad buffers
    }
    const uint32_t words = tile_pixel_mask.size(1);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid    = {AT, 1, 1};

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

        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);

        if(cudaFuncSetAttribute(
               rasterize_to_pixels_sparse_bwd_kernel<CDIM>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }

        rasterize_to_pixels_sparse_bwd_kernel<CDIM><<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
            colors.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().const_data_ptr<float>() : nullptr,
            masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            active_tiles.const_data_ptr<int32_t>(),
            tile_offsets.const_data_ptr<int32_t>(),
            flatten_ids.const_data_ptr<int32_t>(),
            reinterpret_cast<const uint64_t *>(tile_pixel_mask.const_data_ptr<int64_t>()),
            tile_pixel_cumsum.const_data_ptr<int64_t>(),
            pixel_map.const_data_ptr<int64_t>(),
            words,
            render_alphas.const_data_ptr<float>(),
            last_ids.const_data_ptr<int32_t>(),
            v_render_colors.const_data_ptr<float>(),
            v_render_alphas.const_data_ptr<float>(),
            v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>()) : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>()
        );
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
