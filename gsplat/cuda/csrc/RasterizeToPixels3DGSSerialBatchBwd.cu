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
#    include "ATenAtomic.cuh"
#    include <c10/cuda/CUDAStream.h>
#    include <cooperative_groups.h>

#    include "Common.h"
#    include "Rasterization.h"
#    include "RasterizeToPixels3DGSDevice.cuh"
#    include "Utils.cuh"
#    include "Dispatch.h"

namespace gsplat
{
using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

#    if USE_ROCM
// ---------------------------------------------------------------------------
// MI300-class (wave64, gfx9) optimized backward for 64-thread tiles
// (tile_size == 8). One block is exactly one wavefront, so:
//   * the per-gaussian data (id / xy+opacity / conic) is held in registers and
//     broadcast to the wave with __shfl instead of staging it in shared memory
//     (only the per-channel colors still use LDS),
//   * the per-gaussian gradient reductions use DPP row/bcast ops
//     (v_mov_b32_dpp), which fold into the consuming add and need no LDS,
//   * the CDIM-wide v_colors reduction leaves element e in lane e % 64, so the
//     final atomics are wave-coalesced instead of serialized on one lane.
// ---------------------------------------------------------------------------
template<int CTRL>
__device__ __forceinline__ float dpp_mov(float v)
{
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, true));
}

template<int CTRL>
__device__ __forceinline__ int32_t dpp_mov(int32_t v)
{
    return __builtin_amdgcn_mov_dpp(v, CTRL, 0xf, 0xf, true);
}

// DPP controls: ROW_SHR(n) = 0x110 + n, ROW_BCAST15 = 0x142, ROW_BCAST31 = 0x143.
// After the last step lane 63 holds the reduction of the whole wave.
__device__ __forceinline__ float dpp_wave64_sum(float val)
{
    float tmp = val + dpp_mov<0x118>(val);
    tmp       = tmp + dpp_mov<0x114>(tmp);
    tmp       = tmp + dpp_mov<0x112>(tmp);
    tmp       = tmp + dpp_mov<0x111>(tmp);
    tmp       = tmp + dpp_mov<0x142>(tmp);
    tmp       = tmp + dpp_mov<0x143>(tmp);
    return tmp;
}

__device__ __forceinline__ int32_t dpp_wave64_max(int32_t val)
{
    int32_t tmp = max(val, dpp_mov<0x118>(val));
    tmp         = max(tmp, dpp_mov<0x114>(tmp));
    tmp         = max(tmp, dpp_mov<0x112>(tmp));
    tmp         = max(tmp, dpp_mov<0x111>(tmp));
    tmp         = max(tmp, dpp_mov<0x142>(tmp));
    tmp         = max(tmp, dpp_mov<0x143>(tmp));
    return __shfl(tmp, 63);
}

// Wave sum, result broadcast to every lane.
__device__ __forceinline__ void dpp_sum_all(float &val)
{
    val = __shfl(dpp_wave64_sum(val), 63);
}

__device__ __forceinline__ void dpp_sum_all(vec2 &val)
{
    dpp_sum_all(val.x);
    dpp_sum_all(val.y);
}

__device__ __forceinline__ void dpp_sum_all(vec3 &val)
{
    dpp_sum_all(val.x);
    dpp_sum_all(val.y);
    dpp_sum_all(val.z);
}

// Wave sum of each of the `numel` elements of `val`; the reduced element e ends
// up in val[e / 64] of lane (e % 64), ready for a wave-coalesced store.
template<uint32_t numel>
__device__ __forceinline__ void dpp_sum_spread(float (&val)[numel], const uint32_t lane)
{
#        pragma unroll
    for(uint32_t e = 0; e < numel; ++e)
    {
        const float sum = __shfl(dpp_wave64_sum(val[e]), 63);
        if(lane == e % 64)
        {
            val[e / 64] = sum;
        }
    }
}

template<uint32_t CDIM, typename scalar_t>
__launch_bounds__(64) __global__ void rasterize_bs64_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const int64_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,   // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,       // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,    // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    const uint32_t max_batch_size
)
{
    auto block              = cg::this_thread_block();
    auto linear_block_index = block.group_index().x + block_offset;
    dim3 block_index(
        linear_block_index / (tile_width * tile_height),
        (linear_block_index / tile_width) % tile_height,
        linear_block_index % tile_width
    );

    uint32_t image_id = block_index.x;
    uint32_t tile_id  = block_index.y * tile_width + block_index.z;
    uint32_t i        = block_index.y * tile_size + block.thread_index().y;
    uint32_t j        = block_index.z * tile_size + block.thread_index().x;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + max_batch_size - 1) / max_batch_size;

    // Only the per-channel colors live in shared memory: [max_batch_size * CDIM]
    extern __shared__ int s[];
    float *rgbs_batch = (float *)s;

    // this is the T AFTER the last gaussian in this pixel
    float T_final           = 1.0f - render_alphas[pix_id];
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#        pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr              = block.thread_rank();
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);

    const int32_t warp_bin_final = dpp_wave64_max(bin_final);

    // this thread's gaussian of the current batch (broadcast to the wave via __shfl)
    int32_t g_reg = 0;
    vec3 xy_opacity_reg;
    vec3 conic_reg;
    for(int64_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // batch_end_offset is relative to this tile's first intersection.
        const int64_t batch_end_offset = num_intersections - 1 - (int64_t)max_batch_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < max_batch_size ? remaining : max_batch_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(tr < batch_size)
        {
            const int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            g_reg           = g;
            const vec2 xy   = means2d[g];
            const float opac = opacities[g];
            xy_opacity_reg   = {xy.x, xy.y, opac};
            conic_reg        = conics[g];
#        pragma unroll
            for(uint32_t k = 0; k < CDIM; ++k)
            {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
            {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;
            conic.x = __shfl(conic_reg.x, t);
            conic.y = __shfl(conic_reg.y, t);
            conic.z = __shfl(conic_reg.z, t);
            vec3 xy_opac;
            xy_opac.x = __shfl(xy_opacity_reg.x, t);
            xy_opac.y = __shfl(xy_opacity_reg.y, t);
            xy_opac.z = __shfl(xy_opacity_reg.z, t);

            if(valid)
            {
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
                // compute the current T for this gaussian
                float ra         = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                // update v_rgb for this gaussian
                const float fac  = alpha * T;
#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#        pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(opac * vis <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local
                        = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local
                        = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                           v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    if(v_means2d_abs != nullptr)
                    {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#        pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            dpp_sum_spread<CDIM>(v_rgb_local, tr); // element e is left in lane e % 64
            dpp_sum_all(v_conic_local);
            dpp_sum_all(v_xy_local);
            if(v_means2d_abs != nullptr)
            {
                dpp_sum_all(v_xy_abs_local);
            }
            dpp_sum_all(v_opacity_local);

            const int32_t g = __shfl(g_reg, t); // flatten index in [I * N] or [nnz]

            // wave-coalesced: lane l updates channel k + l
            float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#        pragma unroll
            for(uint32_t k = 0; k < CDIM; k += 64)
            {
                if(k + tr < CDIM)
                {
                    unsafeAtomicAdd(v_rgb_ptr + k + tr, v_rgb_local[k / 64]);
                }
            }

            if(tr == 0)
            {
                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                unsafeAtomicAdd(v_conic_ptr, v_conic_local.x);
                unsafeAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                unsafeAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                unsafeAtomicAdd(v_xy_ptr, v_xy_local.x);
                unsafeAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    unsafeAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    unsafeAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                unsafeAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// Use the wave64-optimized kernel for 64-thread tiles when its (color-only)
// shared memory fits in the 64 KB LDS. Fills in the launch configuration.
template<uint32_t CDIM>
inline bool rocm_use_bs64_bwd(const uint32_t tile_size, uint32_t &max_batch_size, int64_t &shmem_size)
{
    if(tile_size * tile_size != 64)
    {
        return false;
    }
    const uint32_t batch = CDIM <= 32 ? 64u : 32u;
    const int64_t bytes  = static_cast<int64_t>(batch) * CDIM * sizeof(float);
    if(bytes > 64 * 1024)
    {
        return false;
    }
    max_batch_size = batch;
    shmem_size     = bytes;
    return true;
}
#    endif

template<uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const int64_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t block_offset,
    const int64_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids,       // [..., image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities // [..., N] or [nnz]
)
{
    auto block              = cg::this_thread_block();
    auto linear_block_index = block.group_index().x + block_offset;
    dim3 block_index(
        linear_block_index / (tile_width * tile_height),
        (linear_block_index / tile_width) % tile_height,
        linear_block_index % tile_width
    );

    uint32_t image_id = block_index.x;
    uint32_t tile_id  = block_index.y * tile_width + block_index.z;
    uint32_t i        = block_index.y * tile_size + block.thread_index().y;
    uint32_t j        = block_index.z * tile_size + block.thread_index().x;

    tile_offsets    += image_id * tile_height * tile_width;
    render_alphas   += image_id * image_height * image_width;
    last_ids        += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if(backgrounds != nullptr)
    {
        backgrounds += image_id * CDIM;
    }
    if(masks != nullptr)
    {
        masks += image_id * tile_height * tile_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if(masks != nullptr && !masks[tile_id])
    {
        return;
    }

    const float px       = (float)j + 0.5f;
    const float py       = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int64_t range_start = tile_offsets[tile_id];
    const int64_t range_end
        = (image_id == I - 1) && (tile_id == tile_width * tile_height - 1) ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size       = block.size();
    const int64_t num_intersections = range_end - range_start;
    const int64_t num_batches       = (num_intersections + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch      = (int32_t *)s;                                            // [block_size]
    vec3 *xy_opacity_batch = reinterpret_cast<vec3 *>(&id_batch[block_size]);         // [block_size]
    vec3 *conic_batch      = reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch      = (float *)&conic_batch[block_size];                       // [block_size * CDIM]
#    if USE_ROCM
    using warp_reduce_float_t = rocprim::warp_reduce<float, 64>;
    auto *warp_storage_base   = (typename warp_reduce_float_t::storage_type *)(rgbs_batch + block_size * CDIM);
#    endif

    // this is the T AFTER the last gaussian in this pixel
    float T_final           = 1.0f - render_alphas[pix_id];
    float T                 = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM]      = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#    pragma unroll
    for(uint32_t k = 0; k < CDIM; ++k)
    {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
#    if USE_ROCM
    cg::thread_block_tile<64> warp = cg::tiled_partition<64>(block);
#    else
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
#    endif
#    if USE_ROCM
    const int32_t warp_bin_final = reduce_max_shuffle(bin_final);
#    else
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());
#    endif
    for(int64_t b = 0; b < num_batches; ++b)
    {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch_end_offset is relative to this tile's first intersection.
        const int64_t batch_end_offset = num_intersections - 1 - block_size * b;
        const int64_t remaining        = batch_end_offset + 1;
        const uint32_t batch_size      = static_cast<uint32_t>(remaining < block_size ? remaining : block_size);
        const int64_t idx              = range_start + batch_end_offset - tr;
        if(idx >= range_start)
        {
            int32_t g            = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
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
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        const int64_t first_t64 = batch_end_offset > warp_bin_final ? batch_end_offset - warp_bin_final : 0;
        const uint32_t first_t  = first_t64 < batch_size ? static_cast<uint32_t>(first_t64) : batch_size;
        for(uint32_t t = first_t; t < batch_size; ++t)
        {
            bool valid = inside;
            if(batch_end_offset - t > bin_final)
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
                // compute the current T for this gaussian
                float ra         = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T               *= ra;
                // update v_rgb for this gaussian
                const float fac  = alpha * T;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) * v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if(backgrounds != nullptr)
                {
                    float accum = 0.f;
#    pragma unroll
                    for(uint32_t k = 0; k < CDIM; ++k)
                    {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if(opac * vis <= MAX_ALPHA)
                {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local
                        = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
                    v_xy_local
                        = {v_sigma * (conic.x * delta.x + conic.y * delta.y),
                           v_sigma * (conic.y * delta.x + conic.z * delta.y)};
                    if(v_means2d_abs != nullptr)
                    {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
#    if USE_ROCM
            rocprim_warpSum<CDIM, 64>(v_rgb_local, warp_storage_base);
            rocprim_warpSum<64>(v_conic_local, warp_storage_base);
            rocprim_warpSum<64>(v_xy_local, warp_storage_base);
            if(v_means2d_abs != nullptr)
            {
                rocprim_warpSum<64>(v_xy_abs_local, warp_storage_base);
            }
            rocprim_warpSum<64>(v_opacity_local, warp_storage_base);
#    else
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if(v_means2d_abs != nullptr)
            {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
#    endif
            if(warp.thread_rank() == 0)
            {
                int32_t g        = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#    if USE_ROCM
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    unsafeAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                unsafeAtomicAdd(v_conic_ptr, v_conic_local.x);
                unsafeAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                unsafeAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                unsafeAtomicAdd(v_xy_ptr, v_xy_local.x);
                unsafeAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    unsafeAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    unsafeAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                unsafeAtomicAdd(v_opacities + g, v_opacity_local);
#    else
#    pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    atomicAdd_system(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                atomicAdd_system(v_conic_ptr, v_conic_local.x);
                atomicAdd_system(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd_system(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                atomicAdd_system(v_xy_ptr, v_xy_local.x);
                atomicAdd_system(v_xy_ptr + 1, v_xy_local.y);

                if(v_means2d_abs != nullptr)
                {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    atomicAdd_system(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd_system(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                atomicAdd_system(v_opacities + g, v_opacity_local);
#    endif
            }
        }
    }
}

void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);                        // number of gaussians
    uint32_t I           = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads     = {tile_size, tile_size, 1};
    uint32_t n_tiles = I * tile_height * tile_width;
    dim3 grid        = {n_tiles, 1, 1};

    if(n_isects == 0)
    {
        // skip the kernel launch if there are no elements
        return;
    }

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

#    if USE_ROCM
        const uint32_t block_size = tile_size * tile_size;
        const uint32_t warps_per_block = (block_size + 63) / 64;
        std::size_t warp_scratch_bytes
            = warps_per_block * sizeof(typename rocprim::warp_reduce<float, 64>::storage_type);
        int64_t shmem_size = tile_size
                           * tile_size
                           * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM)
                           + warp_scratch_bytes;
#    else
        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
#    endif

#    if USE_ROCM
        uint32_t bs64_max_batch_size = 0;
        if(rocm_use_bs64_bwd<CDIM>(tile_size, bs64_max_batch_size, shmem_size))
        {
            hipError_t bs64_err = hipFuncSetAttribute(
                reinterpret_cast<void *>(rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_size)
            );
            if(bs64_err != hipSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size. HIP Error: ",
                    hipGetErrorString(bs64_err)
                );
            }
            rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>
                <<<grid, threads, shmem_size, GET_CURRENT_STREAM()>>>(
                    I,
                    N,
                    n_isects,
                    packed,
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
                    0,
                    tile_offsets.const_data_ptr<int64_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    render_alphas.const_data_ptr<float>(),
                    last_ids.const_data_ptr<int32_t>(),
                    v_render_colors.const_data_ptr<float>(),
                    v_render_alphas.const_data_ptr<float>(),
                    v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                              : nullptr,
                    reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>(),
                    bs64_max_batch_size
                );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
#    endif

#    ifndef USE_ROCM
        if(cudaFuncSetAttribute(
               rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size
           )
           != cudaSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ", shmem_size, " bytes), try lowering tile_size."
            );
        }
#    else
        hipError_t err = hipFuncSetAttribute(
            reinterpret_cast<void *>(rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shmem_size)
        );
        if(err != hipSuccess)
        {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size. HIP Error: ",
                hipGetErrorString(err)
            );
        }
#    endif

        rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>
            <<<grid, threads, shmem_size, GET_CURRENT_STREAM()>>>(
                I,
                N,
                n_isects,
                packed,
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
                0,
                tile_offsets.const_data_ptr<int64_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

void launch_rasterize_to_pixels_3dgs_bwd_kernels(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
)
{
    bool packed = means2d.dim() == 2;

    uint32_t N           = packed ? 0 : means2d.size(-2);
    uint32_t I           = render_alphas.numel() / (image_height * image_width);
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width  = tile_offsets.size(-1);
    int64_t n_isects     = flatten_ids.size(0);
    uint32_t n_tiles     = I * tile_height * tile_width;

    dim3 threads = {tile_size, tile_size, 1};

    if(n_isects == 0)
    {
        return;
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(
        SupportedChannels::contains(channels),
        "Unsupported number of color channels: ",
        channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h)."
    );

    auto launch_kernels = [&]<typename ChannelsT>()
    {
        constexpr uint32_t CDIM = ChannelsT::value;

#    if USE_ROCM
        const uint32_t block_size = tile_size * tile_size;
        const uint32_t warps_per_block = (block_size + 63) / 64;
        std::size_t warp_scratch_bytes
            = warps_per_block * sizeof(typename rocprim::warp_reduce<float, 64>::storage_type);
        int64_t shmem_size = tile_size
                           * tile_size
                           * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM)
                           + warp_scratch_bytes;
#    else
        int64_t shmem_size
            = tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
#    endif
#    if USE_ROCM
        uint32_t bs64_max_batch_size = 0;
        const bool use_bs64          = rocm_use_bs64_bwd<CDIM>(tile_size, bs64_max_batch_size, shmem_size);
#    endif

        for(const auto device_id: c10::irange(c10::cuda::device_count()))
        {
            C10_CUDA_CHECK(cudaSetDevice(device_id));
#    ifndef USE_ROCM
            if(cudaFuncSetAttribute(
                   rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>,
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
#    else
            hipError_t err = hipFuncSetAttribute(
                use_bs64 ? reinterpret_cast<void *>(rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float>)
                         : reinterpret_cast<void *>(rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float>),
                hipFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shmem_size)
            );
            if(err != hipSuccess)
            {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size. HIP Error: ",
                    hipGetErrorString(err)
                );
            }
#    endif
            auto stream = GET_CURRENT_STREAM();

            int64_t block_offset, block_count;
            std::tie(block_offset, block_count) = chunk(n_tiles, device_id);
            if(block_count > 0)
            {
                dim3 grid = {static_cast<uint32_t>(block_count), 1, 1};
#    if USE_ROCM
                if(use_bs64)
                {
                    rasterize_bs64_to_pixels_3dgs_bwd_kernel<CDIM, float><<<grid, threads, shmem_size, stream>>>(
                        I,
                        N,
                        n_isects,
                        packed,
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
                        block_offset,
                        tile_offsets.const_data_ptr<int64_t>(),
                        flatten_ids.const_data_ptr<int32_t>(),
                        render_alphas.const_data_ptr<float>(),
                        last_ids.const_data_ptr<int32_t>(),
                        v_render_colors.const_data_ptr<float>(),
                        v_render_alphas.const_data_ptr<float>(),
                        v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                                  : nullptr,
                        reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                        reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                        v_colors.data_ptr<float>(),
                        v_opacities.data_ptr<float>(),
                        bs64_max_batch_size
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                    continue;
                }
#    endif
                rasterize_to_pixels_3dgs_bwd_kernel<CDIM, float><<<grid, threads, shmem_size, stream>>>(
                    I,
                    N,
                    n_isects,
                    packed,
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
                    block_offset,
                    tile_offsets.const_data_ptr<int64_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    render_alphas.const_data_ptr<float>(),
                    last_ids.const_data_ptr<int32_t>(),
                    v_render_colors.const_data_ptr<float>(),
                    v_render_alphas.const_data_ptr<float>(),
                    v_means2d_abs.has_value() ? reinterpret_cast<vec2 *>(v_means2d_abs.value().data_ptr<float>())
                                              : nullptr,
                    reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
                    reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),
                    v_colors.data_ptr<float>(),
                    v_opacities.data_ptr<float>()
                );
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernels));
    TORCH_CHECK(
        dispatched,
        "dispatch failed: no matching compile-time instantiation for runtime "
        "parameters"
    );

    merge_streams();
}
} // namespace gsplat

#endif
