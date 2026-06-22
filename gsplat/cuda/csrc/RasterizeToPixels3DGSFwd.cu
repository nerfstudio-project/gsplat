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
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "Common.h"
#include "GemmRasterUtils.cuh"
#include "Rasterization.h"
#include "MacroUtils.h"


namespace gsplat {

namespace cg = cooperative_groups;

namespace {

constexpr uint32_t kGemmTileSize = gemm_raster::kGemmTileSize;
constexpr uint32_t kGemmBlockSize = gemm_raster::kGemmBlockSize;
constexpr uint32_t kGemmBatchSize = gemm_raster::kGemmBatchSize;
constexpr uint32_t kGemmVecLen = gemm_raster::kGemmVecLen;

enum class RasterizeToPixels3DGSFwdImpl {
    Gemm = 0,
    Legacy = 1
};

// Python passes a small integer selector through the bindings so backend choice
// stays explicit in the public API.
inline RasterizeToPixels3DGSFwdImpl rasterize_to_pixels_3dgs_fwd_impl_from_id(
    const int64_t impl_id
) {
    switch (impl_id) {
    case 0:
        return RasterizeToPixels3DGSFwdImpl::Gemm;
    case 1:
    default:
        return RasterizeToPixels3DGSFwdImpl::Legacy;
    }
}

inline bool is_rasterize_to_pixels_gemm_supported(const uint32_t tile_size) {
    return gemm_raster::is_gemm_raster_supported(tile_size);
}

inline bool should_use_rasterize_to_pixels_gemm(
    const uint32_t tile_size,
    const uint32_t cdim,
    const RasterizeToPixels3DGSFwdImpl python_impl
) {
    // GEMM is only available for the RGB forward path under the current
    // Tensor Core specialization.
    const bool gemm_supported =
        is_rasterize_to_pixels_gemm_supported(tile_size) && (cdim == 3);

    if (python_impl == RasterizeToPixels3DGSFwdImpl::Gemm) {
        return gemm_supported;
    }
    return false;
}

} // namespace

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////
template <uint32_t CDIM, typename scalar_t>
__global__ void transform_coefs(
    const uint32_t N,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    uint2* rgbo_encoded
)
{
    auto idx = cg::this_grid().thread_rank();
    if(idx >= N)
        return;

    // The GEMM raster kernel expects RGB + opacity packed as two half2 lanes.
    // Keep this packing in one place so the main kernel can focus on tile work.
    float4 features = make_float4(
        colors[idx * CDIM], colors[idx * CDIM + 1], colors[idx * CDIM + 2], opacities[idx]
    );

    uint RG = gemm_raster::pack_two_fp32_to_half2_uint(features.x, features.y);
    uint BO = gemm_raster::pack_two_fp32_to_half2_uint(features.z, features.w);
    rgbo_encoded[idx] = make_uint2(RG, BO);
}


template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_gemm_fwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const uint2* __restrict__ rgbo_encoded,
    const scalar_t *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t
        *__restrict__ render_colors, // [I, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids        // [I, image_height, image_width]
) {

    auto block = cg::this_thread_block();
    const uint32_t tr = block.thread_rank();
    const uint32_t local_i = block.thread_index().y;
    const uint32_t local_j = block.thread_index().x;
    const uint32_t warp_id = tr / 32;
    const uint32_t lane_id = tr & 31;

    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + local_i;
    uint32_t j = block.group_index().z * tile_size + local_j;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    const int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;
    bool warp_done = (__ballot_sync(~0, done) == (~0));

    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
        }
        return;
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + kGemmBlockSize - 1) / kGemmBlockSize;

    // Shared memory tensors used by the 16x16 tile GEMM path.
    __shared__ __half pixel_gaussian_matrix[kGemmBlockSize][kGemmVecLen];
    __shared__ __half power_matrix[kGemmBatchSize][kGemmBlockSize + 8];
    __shared__ uint2 cdim_opacity_smem[kGemmBlockSize];

    // Build the per-pixel vector for the quadratic form used by the GEMM
    // kernel: power = -sigma = dot(gaussian_vec, pixel_vec).
    const float dx = 0.5f * (static_cast<float>(tile_size) - 1.0f) -
                     static_cast<float>(local_j);
    const float dy = 0.5f * (static_cast<float>(tile_size) - 1.0f) -
                     static_cast<float>(local_i);
    pixel_gaussian_matrix[tr][0] = __float2half(-0.5f);
    pixel_gaussian_matrix[tr][1] = __float2half(-0.5f * dx * dx);
    pixel_gaussian_matrix[tr][2] = __float2half(dx);
    pixel_gaussian_matrix[tr][3] = __float2half(-0.5f * dy * dy);
    pixel_gaussian_matrix[tr][4] = __float2half(dy);
    pixel_gaussian_matrix[tr][5] = __float2half(dx * dy);
    pixel_gaussian_matrix[tr][6] = __float2half(0.0f);
    pixel_gaussian_matrix[tr][7] = __float2half(0.0f);
    __syncwarp();

    uint32_t vp_reg[4];
    const __half *B_tile_ptr = &pixel_gaussian_matrix[lane_id + warp_id * 32][0];
    gemm_raster::load_matrix_x4(
        vp_reg[0],
        vp_reg[1],
        vp_reg[2],
        vp_reg[3],
        __cvta_generic_to_shared(B_tile_ptr)
    );

    const float tile_center_x =
        block.group_index().z * tile_size + 0.5f * static_cast<float>(tile_size);
    const float tile_center_y =
        block.group_index().y * tile_size + 0.5f * static_cast<float>(tile_size);

    const __half h_zero = __float2half(0.0f);
    const float max_alpha = MAX_ALPHA;
    const float alpha_threshold = ALPHA_THRESHOLD;
    const float transmittance_threshold = TRANSMITTANCE_THRESHOLD;

    float T = 1.0f;
    uint32_t cur_idx = 0;
    float pix_out[CDIM];
    #pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        pix_out[k] = 0.0f;
    }

    const uint32_t gid = lane_id >> 2;
    const uint32_t tid4 = lane_id & 3;
    const uint32_t col0 = tid4 * 2;
    const uint32_t row0 = gid;
    const uint32_t row1 = gid + 8;

    for (uint32_t b = 0; b < num_batches; ++b) {

        int num_done = __syncthreads_count(done);
        if(num_done == kGemmBlockSize)
            break;

        const int32_t batch_start =
            range_start + static_cast<int32_t>(kGemmBlockSize * b);
        const int32_t idx = batch_start + static_cast<int32_t>(tr);
        if (idx < range_end) {
            const int32_t g = flatten_ids[idx];
            const vec2 xy = means2d[g];
            const vec3 conic = conics[g];

            cdim_opacity_smem[tr] = rgbo_encoded[g];
            const float d0x = xy.x - tile_center_x;
            const float d0y = xy.y - tile_center_y;
            __half *dst = pixel_gaussian_matrix[tr];
            dst[0] = __float2half(
                conic.x * d0x * d0x + conic.z * d0y * d0y +
                2.0f * conic.y * d0x * d0y
            );
            dst[1] = __float2half(conic.x);
            dst[2] = __float2half(-(conic.x * d0x + conic.y * d0y));
            dst[3] = __float2half(conic.z);
            dst[4] = __float2half(-(conic.z * d0y + conic.y * d0x));
            dst[5] = __float2half(-conic.y);
            dst[6] = __float2half(0.0f);
            dst[7] = __float2half(0.0f);
        }
        __syncthreads();

        const uint32_t remaining =
            static_cast<uint32_t>(range_end - batch_start);
        const uint32_t batch_size =
            remaining < kGemmBlockSize ? remaining : kGemmBlockSize;

        for (uint32_t m = 0; m < batch_size && !warp_done; m += kGemmBatchSize) {
            const __half *A_tile_ptr = &pixel_gaussian_matrix[lane_id + m][0];

            uint32_t vg_reg[2];
            gemm_raster::load_matrix_x2(
                vg_reg[0],
                vg_reg[1],
                __cvta_generic_to_shared(A_tile_ptr)
            );

            uint32_t rc0 = 0;
            uint32_t rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0,
                rc1,
                vg_reg[0],
                vg_reg[1],
                vp_reg[0],
                rc0,
                rc1
            );
            *(uint32_t *)&power_matrix[row0][warp_id * 32 + col0] = rc0;
            *(uint32_t *)&power_matrix[row1][warp_id * 32 + col0] = rc1;

            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0,
                rc1,
                vg_reg[0],
                vg_reg[1],
                vp_reg[1],
                rc0,
                rc1
            );
            *(uint32_t *)&power_matrix[row0][warp_id * 32 + 8 + col0] = rc0;
            *(uint32_t *)&power_matrix[row1][warp_id * 32 + 8 + col0] = rc1;

            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0,
                rc1,
                vg_reg[0],
                vg_reg[1],
                vp_reg[2],
                rc0,
                rc1
            );
            *(uint32_t *)&power_matrix[row0][warp_id * 32 + 16 + col0] = rc0;
            *(uint32_t *)&power_matrix[row1][warp_id * 32 + 16 + col0] = rc1;

            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0,
                rc1,
                vg_reg[0],
                vg_reg[1],
                vp_reg[3],
                rc0,
                rc1
            );
            *(uint32_t *)&power_matrix[row0][warp_id * 32 + 24 + col0] = rc0;
            *(uint32_t *)&power_matrix[row1][warp_id * 32 + 24 + col0] = rc1;

            const uint32_t chunk =
                min(kGemmBatchSize, static_cast<uint32_t>(batch_size - m));
#pragma unroll
            for (uint32_t t = 0; t < chunk; ++t) {
                __half power_h = power_matrix[t][tr];
                const __half exp_term_h =
                    gemm_raster::fast_exp2_f16(
                        __hmul(power_h, gemm_raster::log2e_half())
                    );
                const float exp_term = __half2float(exp_term_h);

                const uint packed_c0_c1 = cdim_opacity_smem[m + t].x;
                const uint packed_c2_opac = cdim_opacity_smem[m + t].y;
                const half2 c0_c1 = gemm_raster::uint2half2(packed_c0_c1);
                const half2 c2_opac_h2 = gemm_raster::uint2half2(packed_c2_opac);
                const float opac = __half2float(__high2half(c2_opac_h2));

                float alpha = fminf(max_alpha, opac * exp_term);
                if (alpha < alpha_threshold) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);

                const float vis = alpha * T;
                const float c0 = __half2float(__low2half(c0_c1));
                const float c1 = __half2float(__high2half(c0_c1));
                const float c2 = __half2float(__low2half(c2_opac_h2));
                pix_out[0] += c0 * vis;
                pix_out[1] += c1 * vis;
                pix_out[2] += c2 * vis;

                cur_idx = batch_start + m + t;
                T = next_T;
            }
            if (T <= transmittance_threshold) {
                done = true;
            }

            if (__ballot_sync(~0, done) == (~0)) {
                warp_done = true;
            }
        }
    }

    if (inside) {
        render_alphas[pix_id] = static_cast<scalar_t>(1.0f - T);
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            const float color_f = backgrounds == nullptr
                ? pix_out[k]
                : (pix_out[k] + T * backgrounds[k]);
            render_colors[pix_id * CDIM + k] = static_cast<scalar_t>(color_f);
        }
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_legacy_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [I, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    scalar_t
        *__restrict__ render_colors, // [I, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids        // [I, image_height, image_width]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id * CDIM + k] =
                    backgrounds == nullptr ? 0.0f : backgrounds[k];
            }
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(MAX_ALPHA, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const int64_t rasterize_fwd_impl,
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? means2d.size(0) : means2d.size(-2); // number of gaussians per image
    // `flatten_ids` indexes the flattened Gaussian row space: [nnz] in packed
    // mode and [I * N] in unpacked mode. Size the encoded color/opacity buffer
    // by the flattened row count so both paths index the same address space.
    const uint32_t num_gemmsplats = means2d.numel() / 2;
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    const auto python_impl =
        rasterize_to_pixels_3dgs_fwd_impl_from_id(rasterize_fwd_impl);
    const bool use_gemm =
        should_use_rasterize_to_pixels_gemm(tile_size, CDIM, python_impl);

    if (use_gemm) {
        dim3 gemm_threads = {kGemmTileSize, kGemmTileSize, 1};
        at::Tensor rgbo_encoded = at::empty(
            {static_cast<int64_t>(num_gemmsplats), 2},
            colors.options().dtype(at::kInt)
        );
        // Encode the per-Gaussian RGB/opacity payload once before rasterization.
        // This avoids re-packing inside each image tile.
        transform_coefs<CDIM, float>
            <<< (num_gemmsplats + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream() >>>(
                num_gemmsplats,
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                reinterpret_cast<uint2 *>(rgbo_encoded.data_ptr<int32_t>())
            );
        // Keep inputs in compact GEMM form, but accumulate transmittance and
        // output colors in float inside the kernel to reduce parity error.
        rasterize_to_pixels_gemm_fwd_kernel<CDIM, float>
            <<<grid, gemm_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                I,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
                reinterpret_cast<uint2 *>(rgbo_encoded.data_ptr<int32_t>()),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                last_ids.data_ptr<int32_t>()
            );
        return;
    }

    int64_t shmem_size =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_3dgs_fwd_legacy_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_3dgs_fwd_legacy_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<const vec2 *>(means2d.const_data_ptr<float>()),
            reinterpret_cast<const vec3 *>(conics.const_data_ptr<float>()),
            colors.const_data_ptr<float>(),
            opacities.const_data_ptr<float>(),
            backgrounds.has_value()
                ? backgrounds.value().const_data_ptr<float>()
                : nullptr,
            masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.const_data_ptr<int32_t>(),
            flatten_ids.const_data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>()
        );
}

// Explicit instantiations matching the dispatch in Rasterization.cpp.
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const int64_t rasterize_fwd_impl,                                      \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor last_ids                                                    \
    );

GSPLAT_FOR_EACH(__INS__, GSPLAT_NUM_CHANNELS)
#undef __INS__

} // namespace gsplat

#endif
