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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "Rasterization.h"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, uint32_t TILE_SIZE, uint32_t CTA_SIZE>
__global__ void __launch_bounds__(CTA_SIZE) rasterize_to_pixels_3dgs_fwd_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const float *__restrict__ colors,         // [I, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,      // [I, N] or [nnz]
    const float *__restrict__ backgrounds,    // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const int32_t *__restrict__ isect_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ render_colors,        // [I, image_height, image_width, CDIM]
    float *__restrict__ render_alphas,        // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids            // [I, image_height, image_width]
) {
    constexpr uint32_t BATCH_SIZE        = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE; // (TILE, CTA) = (16,64)->4, (8,32)->2, (4,16)->1
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert((TILE_SIZE & (TILE_SIZE - 1)) == 0, "TILE_SIZE must be a power of 2 (TILE_MASK/TILE_SHIFT rely on this)");
    static_assert(PIXELS_PER_THREAD > 0, "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    const int32_t image_id = blockIdx.x;
    const uint32_t grid_width  = gridDim.z;
    const uint32_t grid_height = gridDim.y;

    const uint32_t tile_x = blockIdx.z;
    const uint32_t tile_y = blockIdx.y;
    const int32_t tile_id = blockIdx.y * grid_width + blockIdx.z;

    const uint32_t tid = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK; // X & 0xF(15) == X % 16
    const uint32_t thread_y = tid >> TILE_SHIFT; // X >> 4 == X / 16

    isect_offsets += image_id * grid_height * grid_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * grid_height * grid_width;
    }

    // X can be computed easily. Y needs to process 4 different pixel positions.
    // Also adjust pixel coordinates from top left corner to center.
    const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
    const float px = (float)out_x + 0.5f;

    uint32_t out_y[PIXELS_PER_THREAD];
    float py[PIXELS_PER_THREAD];
    int32_t pix_id[PIXELS_PER_THREAD];
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        out_y[p] = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE;
        py[p] = (float)out_y[p] + 0.5f;
        pix_id[p] = (int32_t)(out_y[p] * image_width + out_x);
    }

    // Evaluate other early exist criteria. We can't directly OOB return
    // because __syncthreads_count evaluates predicates for all threads
    // in the block and will block until all threads have evaluated the
    // predicate.
    uint32_t done_mask = (out_x >= image_width) ? ALL_DONE : 0;
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if(out_y[p] >= image_height) {
            done_mask |= (1u << p);
        }
    }

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False. Check each pixel individually: a
    // single thread can straddle the image boundary when the tile is
    // partially clipped, so some of its pixels may be in-bounds while
    // others are OOB.
    if (masks != nullptr && !masks[tile_id]) {
#pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            if (done_mask & (1u << p)) {
                continue; // this pixel is OOB, skip
            }
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id[p] * CDIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    const int32_t range_start = isect_offsets[tile_id];
    const int32_t range_end =
        (image_id == (int32_t)gridDim.x - 1) &&
        (tile_id == (int32_t)(grid_width * grid_height) - 1)
            ? n_isects
            : isect_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + BATCH_SIZE - 1) / BATCH_SIZE;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [BATCH_SIZE]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[BATCH_SIZE]); // [BATCH_SIZE]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[BATCH_SIZE]); // [BATCH_SIZE]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T[PIXELS_PER_THREAD];
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        T[p] = 1.0f;
    }
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx[PIXELS_PER_THREAD] = {0u};
    // result of the rendering for each pixel
    float pix_out[PIXELS_PER_THREAD][CDIM] = {0.f};

    // unroll 1: keep the outer batch loop rolled, unrolling it would inflate
    // register pressure (and icache pressure for long loops) without helping
    // throughput here. The inner per-pixel and per-CDIM loops below are also
    // unrolled.
#pragma unroll 1
    for (uint32_t b = 0; b < num_batches; ++b) {
        // each thread fetch 1 gaussian from front to back
        const uint32_t batch_start = range_start + BATCH_SIZE * b;
        const uint32_t idx = batch_start + tid; // index of gaussian to load
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tid] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tid] = {xy.x, xy.y, opac};
            conic_batch[tid] = conics[g];
        }

        // wait for other threads to collect the gaussians in batch. A CTA
        // with <= 32 threads is a single warp, so a warp-level sync suffices.
        if constexpr (CTA_SIZE <= 32) {
            __syncwarp();
        }
        else {
            __syncthreads();
        }

        // process gaussians in the current batch
        const uint32_t batch_size = min(BATCH_SIZE, ((uint32_t)range_end - batch_start));
        for (uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const float dx = xy_opac.x - px;

#pragma unroll
            for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                if (done_mask & (1u << p)) {
                    continue;
                }

                const float dy = xy_opac.y - py[p];
                const float sigma = 0.5f * (conic.x * dx * dx +
                                            conic.z * dy * dy) +
                                            conic.y * dx * dy;
                float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    continue;
                }

                const float next_T = T[p] * (1.0f - alpha);
                if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                    done_mask |= (1u << p);
                    continue;
                }

                const int32_t g = id_batch[t];
                const float vis = alpha * T[p];
                const float *c_ptr = colors + g * CDIM;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[p][k] += c_ptr[k] * vis;
                }
                cur_idx[p] = batch_start + t;
                T[p] = next_T;
            }
        }

        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done_mask == ALL_DONE) >= BATCH_SIZE) {
            break;
        }
    }

    if (out_x < image_width) {
#pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            if (out_y[p] < image_height) {
                // Here T is the transmittance AFTER the last gaussian in this pixel.
                // We (should) store double precision as T would be used in backward
                // pass and it can be very small and causing large diff in gradients
                // with float32. However, double precision makes the backward pass 1.5x
                // slower so we stick with float for now.
                render_alphas[pix_id[p]] = 1.0f - T[p];
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    render_colors[pix_id[p] * CDIM + k] =
                        backgrounds == nullptr ? pix_out[p][k]
                                            : (pix_out[p][k] + T[p] * backgrounds[k]);
                }
                last_ids[pix_id[p]] = static_cast<int32_t>(cur_idx[p]);
            }
        }
    }
}

void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., grid_h, grid_w]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor isect_offsets, // [..., grid_h, grid_w]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
) {
    const bool packed = means2d.dim() == 2;

    const uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    const uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    const uint32_t grid_h = isect_offsets.size(-2);
    const uint32_t grid_w = isect_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);

    const dim3 grid = {I, grid_h, grid_w};

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(SupportedChannels::contains(channels),
        "Unsupported number of color channels: ", channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h).");

    auto launch_kernel = [&]<typename ChannelsT>() {
        constexpr uint32_t CDIM = ChannelsT::value;

        auto launch_variant = [&]<uint32_t TILE_SIZE, uint32_t CTA_SIZE>() {
            const dim3 threads = dim3{CTA_SIZE, 1, 1};
            const int64_t shmem_size =
                CTA_SIZE * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

            if (cudaFuncSetAttribute(
                    rasterize_to_pixels_3dgs_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem_size
                ) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes), try lowering tile_size."
                );
            }

            const float *bg_ptr = backgrounds.has_value()
                ? backgrounds.value().const_data_ptr<float>()
                : nullptr;
            const bool *masks_ptr = masks.has_value()
                ? masks.value().const_data_ptr<bool>()
                : nullptr;

            rasterize_to_pixels_3dgs_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    n_isects,
                    packed,
                    reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
                    reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    bg_ptr,
                    masks_ptr,
                    image_width,
                    image_height,
                    isect_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    renders.data_ptr<float>(),
                    alphas.data_ptr<float>(),
                    last_ids.data_ptr<int32_t>()
                );
        };

        // NOTE: Here we need to support tile_size=4 temporarily, because test_basic.py
        // dynamically sets the tile_size=4 when CDIM >=32 to avoid exceeding the
        // maximal shared memory size via the backward kernel's shared memory config.
        // When we also convert the 3DGS backward rasterizer pass, it will require much
        // less shared memory since we're iterating with PPT=4 and will therefore be able
        // to remove tile_size=4 both here and in test_basic.py.
        if (tile_size == 16) {
            launch_variant.template operator()<16, 64>();
        } else if (tile_size == 4) {
            launch_variant.template operator()<4, 16>();
        } else {
            AT_ERROR(
                "Unsupported tile_size ", tile_size,
                "; supported values are {4, 16}."
            );
        }
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

} // namespace gsplat

#endif
