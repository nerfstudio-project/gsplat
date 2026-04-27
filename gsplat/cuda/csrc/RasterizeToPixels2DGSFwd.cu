/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#if GSPLAT_BUILD_2DGS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cstdlib>
#include <cstring>

#include "Common.h"
#include "GemmRasterUtils.cuh"
#include "Rasterization.h"
#include "MacroUtils.h"

namespace gsplat {

namespace cg = cooperative_groups;

constexpr uint32_t kGemmTileSize = gemm_raster::kGemmTileSize;
constexpr uint32_t kGemmBlockSize = gemm_raster::kGemmBlockSize;
constexpr uint32_t kGemmBatchSize = gemm_raster::kGemmBatchSize;
constexpr uint32_t kGemmVecLen = gemm_raster::kGemmVecLen;
constexpr float kGemmDenomMin = 1e-6f;

inline bool is_rasterize_to_pixels_2dgs_fwd_gemm_supported(
    const uint32_t tile_size
) {
    return gemm_raster::is_gemm_raster_supported(tile_size);
}

enum class RasterizeToPixels2DGSFwdImpl {
    Gemm = 0,
    Legacy = 1
};

// Python passes a small integer selector through the bindings so backend choice
// stays explicit in the public API.
inline RasterizeToPixels2DGSFwdImpl rasterize_to_pixels_2dgs_fwd_impl_from_id(
    const int64_t impl_id
) {
    switch (impl_id) {
    case 0:
        return RasterizeToPixels2DGSFwdImpl::Gemm;
    case 1:
    default:
        return RasterizeToPixels2DGSFwdImpl::Legacy;
    }
}

inline bool should_use_rasterize_to_pixels_2dgs_fwd_gemm(
    const uint32_t tile_size,
    const RasterizeToPixels2DGSFwdImpl python_impl
) {
    const bool gemm_supported =
        is_rasterize_to_pixels_2dgs_fwd_gemm_supported(tile_size);
    if (python_impl == RasterizeToPixels2DGSFwdImpl::Gemm) {
        return gemm_supported;
    }
    return false;
}

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_fwd_gemm_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    const vec2
        *__restrict__ means2d, // Projected Gaussian means. [..., N, 2] if
                               // packed is False, [nnz, 2] if packed is True.
    const scalar_t
        *__restrict__ ray_transforms, // transformation matrices that transforms
                                      // xy-planes in pixel spaces into splat
                                      // coordinates. [..., N, 3, 3] if packed is
                                      // False, [nnz, channels] if packed is
                                      // True. This is (KWH)^{-1} in the paper
                                      // (takes screen [x,y] and map to [u,v])
    const scalar_t *__restrict__ colors,    // [..., N, CDIM] or [nnz, CDIM]  //
                                            // Gaussian colors or ND features.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width] // Optional
                                    // tile mask to skip rendering GS to masked
                                    // tiles.
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t
        *__restrict__ tile_offsets, // [..., tile_height, tile_width]    //
                                    // Intersection offsets outputs from
                                    // `isect_offset_encode()`, this is the
                                    // result of a prefix sum, and gives the
                                    // interval that our gaussians are gonna
                                    // use.
    const int32_t *__restrict__ flatten_ids, // [n_isects] // The global flatten
                                             // indices in [I * N] or [nnz] from
                                             // `isect_tiles()`.

    // outputs
    scalar_t
        *__restrict__ render_colors, // [..., image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [..., image_height, image_width, 3]
    scalar_t *__restrict__ render_distort, // [..., image_height, image_width, 1]
                                           // // Stores the per-pixel distortion
                                           // error proposed in Mip-NeRF 360.
    scalar_t
        *__restrict__ render_median, // [..., image_height, image_width, 1]  //
                                     // Stores the median depth contribution for
                                     // each pixel "set to the depth of the
                                     // Gaussian that brings the accumulated
                                     // opacity over 0.5."
    int32_t *__restrict__ last_ids,  // [..., image_height, image_width]     //
                                     // Stores the index of the last Gaussian
                                     // that contributed to each pixel.
    int32_t *__restrict__ median_ids // [..., image_height, image_width]    //
                                     // Stores the index of the Gaussian that
                                     // contributes to the median depth for each
                                     // pixel (bring over 0.5).
) {
    // Each thread renders one pixel while the block cooperatively stages the
    // Gaussian batch for its image tile.
    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
    uint32_t tr = block.thread_rank();
    const uint32_t warp_id = tr / 32;
    const uint32_t lane_id = tr & 31;

    tile_offsets +=
        image_id * tile_height *
        tile_width; // get the global offset of the tile w.r.t the image
    render_colors +=
        image_id * image_height * image_width *
        CDIM; // get the global offset of the pixel w.r.t the image
    render_alphas +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    last_ids +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    render_normals += image_id * image_height * image_width * 3;
    render_distort += image_id * image_height * image_width;
    render_median += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    const float tile_center_x =
        block.group_index().z * tile_size + 0.5f * static_cast<float>(tile_size);
    const float tile_center_y =
        block.group_index().y * tile_size + 0.5f * static_cast<float>(tile_size);
    const float dx = px - tile_center_x;
    const float dy = py - tile_center_y;
    int32_t pix_id = i * image_width + j;

    bool inside = (i < image_height && j < image_width);
    bool done = !inside;
    bool warp_done = (__ballot_sync(~0, done) == (~0));

    if (masks != nullptr && !masks[tile_id]) {
        if (inside) {
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
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    __shared__ int32_t id_batch[kGemmBlockSize];
    __shared__ __half pixel_gaussian_matrix[kGemmBlockSize][kGemmVecLen];
    __shared__ __half gaussian_matrix[kGemmBlockSize][kGemmVecLen];
    __shared__ __half power_matrix_1[kGemmBatchSize][kGemmBlockSize + 8]; // padding avoid bank conflict
    __shared__ __half power_matrix_2[kGemmBatchSize][kGemmBlockSize + 8];
    __shared__ vec3 xy_opacity_batch[kGemmBlockSize];



    pixel_gaussian_matrix[tr][0] = __float2half(1.0f);
    // Build the per-pixel polynomial basis in tile-local coordinates. The
    // local origin keeps the squared terms in fp16 range while still allowing
    // the per-Gaussian coefficients to reconstruct the same quadratic form.
    pixel_gaussian_matrix[tr][1] = __float2half(dx);
    pixel_gaussian_matrix[tr][2] = __float2half(dy);
    pixel_gaussian_matrix[tr][3] = __float2half(dx * dx);
    pixel_gaussian_matrix[tr][4] = __float2half(dx * dy);
    pixel_gaussian_matrix[tr][5] = __float2half(dy * dy);
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

    const __half h_zero = __float2half(0.0f);
    const __half h_one = __float2half(1.0f);
    const __half h_max_alpha = __float2half(MAX_ALPHA);
    const __half h_alpha_threshold = __float2half(ALPHA_THRESHOLD);
    const __half h_transmittance_threshold =
        __float2half(TRANSMITTANCE_THRESHOLD);

    const uint32_t gid = lane_id >> 2;
    const uint32_t tid4 = lane_id & 3;
    const uint32_t col0 = tid4 * 2;
    const uint32_t row0 = gid;
    const uint32_t row1 = gid + 8;
    // Keep transmittance and output accumulation in float even though the GEMM
    // inputs remain fp16.
    float T = 1.0f;
    uint32_t cur_idx = 0;

    float distort = 0.f;
    float accum_vis_depth = 0.f;

    float median_depth = 0.f;
    uint32_t median_idx = 0.f;

    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;

        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            const vec3 u_M = {
                ray_transforms[g * 9],
                ray_transforms[g * 9 + 1],
                ray_transforms[g * 9 + 2]
            };
            const vec3 v_M = {
                ray_transforms[g * 9 + 3],
                ray_transforms[g * 9 + 4],
                ray_transforms[g * 9 + 5]
            };
            const vec3 w_M = {
                ray_transforms[g * 9 + 6],
                ray_transforms[g * 9 + 7],
                ray_transforms[g * 9 + 8]
            };

            const vec3 C = glm::cross(u_M, v_M);
            const vec3 A = glm::cross(w_M, v_M);
            const vec3 B = glm::cross(u_M, w_M);
            // Shift the quadratic form into tile-local coordinates so the
            // squared terms stay better conditioned in fp16.
            const vec3 C_shift = {
                C.x - tile_center_x * A.x - tile_center_y * B.x,
                C.y - tile_center_x * A.y - tile_center_y * B.y,
                C.z - tile_center_x * A.z - tile_center_y * B.z
            };
            // Apply one shared scale so the ratio up / low is preserved while
            // the fp16 coefficients remain in a stable numeric range.
            const float max_abs = fmaxf(
                1e-4f,
                fmaxf(
                    fmaxf(fmaxf(fabsf(A.x), fabsf(A.y)), fabsf(A.z)),
                    fmaxf(
                        fmaxf(fmaxf(fabsf(B.x), fabsf(B.y)), fabsf(B.z)),
                        fmaxf(
                            fmaxf(fabsf(C_shift.x), fabsf(C_shift.y)),
                            fabsf(C_shift.z)
                        )
                    )
                )
            );
            const float inv_scale = 1.0f / max_abs;
            const vec3 A_s = A * inv_scale;
            const vec3 B_s = B * inv_scale;
            const vec3 C_s = C_shift * inv_scale;

            __half *dst_1 = pixel_gaussian_matrix[tr];
            dst_1[0] = __float2half(C_s.x * C_s.x + C_s.y * C_s.y);
            dst_1[1] = __float2half(-2 * (A_s.x * C_s.x + A_s.y * C_s.y));
            dst_1[2] = __float2half(-2 * (B_s.x * C_s.x + B_s.y * C_s.y));
            dst_1[3] = __float2half(A_s.x * A_s.x + A_s.y * A_s.y);
            dst_1[4] = __float2half(2 * (A_s.x * B_s.x + A_s.y * B_s.y));
            dst_1[5] = __float2half(B_s.x * B_s.x + B_s.y * B_s.y);
            dst_1[6] = __float2half(0.0f);
            dst_1[7] = __float2half(0.0f);

            __half *dst_2 = gaussian_matrix[tr];
            dst_2[0] = __float2half(C_s.z * C_s.z);
            dst_2[1] = __float2half(-2 * (A_s.z * C_s.z));
            dst_2[2] = __float2half(-2 * (B_s.z * C_s.z));
            dst_2[3] = __float2half(A_s.z * A_s.z);
            dst_2[4] = __float2half(2 * (A_s.z * B_s.z));
            dst_2[5] = __float2half(B_s.z * B_s.z);
            dst_2[6] = __float2half(0.0f);
            dst_2[7] = __float2half(0.0f);
        }

        block.sync();

        const uint32_t remaining =
            static_cast<uint32_t>(range_end - batch_start);
        const uint32_t batch_size =
            remaining < kGemmBlockSize ? remaining : kGemmBlockSize;

        for (uint32_t m = 0; m < batch_size && !warp_done; m += kGemmBatchSize) {
            const __half *A_tile_ptr = &pixel_gaussian_matrix[lane_id + m][0];
            const __half *B_tile_ptr = &gaussian_matrix[lane_id + m][0];

            uint32_t vg_reg_1[2];
            uint32_t vg_reg_2[2];

            gemm_raster::load_matrix_x2(
                vg_reg_1[0], vg_reg_1[1], __cvta_generic_to_shared(A_tile_ptr)
            );
            gemm_raster::load_matrix_x2(
                vg_reg_2[0], vg_reg_2[1], __cvta_generic_to_shared(B_tile_ptr)
            );

            // Numerator terms for the ray-intersection quadratic.
            uint32_t rc0 = 0;
            uint32_t rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_1[0], vg_reg_1[1], vp_reg[0], rc0, rc1
            );
            *(uint32_t *)&power_matrix_1[row0][warp_id * 32 + col0] = rc0;
            *(uint32_t *)&power_matrix_1[row1][warp_id * 32 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_1[0], vg_reg_1[1], vp_reg[1], rc0, rc1
            );
            *(uint32_t *)&power_matrix_1[row0][warp_id * 32 + 8 + col0] = rc0;
            *(uint32_t *)&power_matrix_1[row1][warp_id * 32 + 8 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_1[0], vg_reg_1[1], vp_reg[2], rc0, rc1
            );
            *(uint32_t *)&power_matrix_1[row0][warp_id * 32 + 16 + col0] = rc0;
            *(uint32_t *)&power_matrix_1[row1][warp_id * 32 + 16 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_1[0], vg_reg_1[1], vp_reg[3], rc0, rc1
            );
            *(uint32_t *)&power_matrix_1[row0][warp_id * 32 + 24 + col0] = rc0;
            *(uint32_t *)&power_matrix_1[row1][warp_id * 32 + 24 + col0] = rc1;

            // Denominator terms for the same quadratic.
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_2[0], vg_reg_2[1], vp_reg[0], rc0, rc1
            );
            *(uint32_t *)&power_matrix_2[row0][warp_id * 32 + col0] = rc0;
            *(uint32_t *)&power_matrix_2[row1][warp_id * 32 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_2[0], vg_reg_2[1], vp_reg[1], rc0, rc1
            );
            *(uint32_t *)&power_matrix_2[row0][warp_id * 32 + 8 + col0] = rc0;
            *(uint32_t *)&power_matrix_2[row1][warp_id * 32 + 8 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_2[0], vg_reg_2[1], vp_reg[2], rc0, rc1
            );
            *(uint32_t *)&power_matrix_2[row0][warp_id * 32 + 16 + col0] = rc0;
            *(uint32_t *)&power_matrix_2[row1][warp_id * 32 + 16 + col0] = rc1;
            rc0 = 0;
            rc1 = 0;
            gemm_raster::mma_16x8x8_fp16(
                rc0, rc1, vg_reg_2[0], vg_reg_2[1], vp_reg[3], rc0, rc1
            );
            *(uint32_t *)&power_matrix_2[row0][warp_id * 32 + 24 + col0] = rc0;
            *(uint32_t *)&power_matrix_2[row1][warp_id * 32 + 24 + col0] = rc1;
            const uint32_t chunk =
                min(kGemmBatchSize, static_cast<uint32_t>(batch_size - m));
#pragma unroll
            for (uint32_t t = 0; t < chunk && !done; ++t) {
                const __half gauss_weight_3d_h_up  = power_matrix_1[t][tr];
                const __half gauss_weight_3d_h_low = power_matrix_2[t][tr];

                const float up  = __half2float(gauss_weight_3d_h_up);
                const float low = __half2float(gauss_weight_3d_h_low);
                const bool invalid_ratio =
                    !isfinite(up) || !isfinite(low) || (low <= kGemmDenomMin) ||
                    (up < 0.0f);
                if (invalid_ratio) {
                    continue;
                }
                const int32_t g = id_batch[t + m];
                const float gauss_weight_3d = up / low;
                const vec3 xy_opac = xy_opacity_batch[m + t];
                const float opac = xy_opac.z;
                // Clamp the 3D intersection kernel against the projected 2D
                // footprint, matching the legacy rasterization rule.
                const vec2 d = {xy_opac.x - px, xy_opac.y - py};
                const float gauss_weight_2d =
                    FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);

                const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                const float sigma = 0.5f * gauss_weight;
                float alpha = min(MAX_ALPHA, opac * __expf(-sigma));

                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);
                if (next_T <= TRANSMITTANCE_THRESHOLD) {
                    done = true;
                    break;
                }

                const float vis = alpha * T;
                const float *c_ptr = colors + g * CDIM;
    #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }

                const float *n_ptr = normals + g * 3;
    #pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    normal_out[k] += n_ptr[k] * vis;
                }

                if (render_distort != nullptr) {
                    const float depth = c_ptr[CDIM - 1];
                    const float distort_bi_0 = vis * depth * (1.0f - T);
                    const float distort_bi_1 = vis * accum_vis_depth;
                    distort += 2.0f * (distort_bi_0 - distort_bi_1);
                    accum_vis_depth += vis * depth;
                }

                if (T > 0.5) {
                    median_depth = c_ptr[CDIM - 1];
                    median_idx = batch_start + t + m;
                }

                cur_idx = batch_start + t + m;

                T = next_T;
            }
            if (__ballot_sync(~0, done) == (~0)) {
                warp_done = true;
            }
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
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if (render_distort != nullptr) {
            render_distort[pix_id] = distort;
        }

        render_median[pix_id] = median_depth;
        // index in bin of gaussian that contributes to median depth
        median_ids[pix_id] = static_cast<int32_t>(median_idx);
    }
}


////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_fwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    const vec2
        *__restrict__ means2d, // Projected Gaussian means. [..., N, 2] if
                               // packed is False, [nnz, 2] if packed is True.
    const scalar_t
        *__restrict__ ray_transforms, // transformation matrices that transforms
                                      // xy-planes in pixel spaces into splat
                                      // coordinates. [..., N, 3, 3] if packed is
                                      // False, [nnz, channels] if packed is
                                      // True. This is (KWH)^{-1} in the paper
                                      // (takes screen [x,y] and map to [u,v])
    const scalar_t *__restrict__ colors,    // [..., N, CDIM] or [nnz, CDIM]  //
                                            // Gaussian colors or ND features.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width] // Optional
                                    // tile mask to skip rendering GS to masked
                                    // tiles.
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t
        *__restrict__ tile_offsets, // [..., tile_height, tile_width]    //
                                    // Intersection offsets outputs from
                                    // `isect_offset_encode()`, this is the
                                    // result of a prefix sum, and gives the
                                    // interval that our gaussians are gonna
                                    // use.
    const int32_t *__restrict__ flatten_ids, // [n_isects] // The global flatten
                                             // indices in [I * N] or [nnz] from
                                             // `isect_tiles()`.

    // outputs
    scalar_t
        *__restrict__ render_colors, // [..., image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [..., image_height, image_width, 3]
    scalar_t *__restrict__ render_distort, // [..., image_height, image_width, 1]
                                           // // Stores the per-pixel distortion
                                           // error proposed in Mip-NeRF 360.
    scalar_t
        *__restrict__ render_median, // [..., image_height, image_width, 1]  //
                                     // Stores the median depth contribution for
                                     // each pixel "set to the depth of the
                                     // Gaussian that brings the accumulated
                                     // opacity over 0.5."
    int32_t *__restrict__ last_ids,  // [..., image_height, image_width]     //
                                     // Stores the index of the last Gaussian
                                     // that contributed to each pixel.
    int32_t *__restrict__ median_ids // [..., image_height, image_width]    //
                                     // Stores the index of the Gaussian that
                                     // contributes to the median depth for each
                                     // pixel (bring over 0.5).
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    /**
     * ==============================
     * Thread and block setup:
     * This sets up the thread and block indices, determining which image,
     * tile, and pixel each thread will process. The grid structure is assigend
     * as: I * tile_height * tile_width blocks (3d grid), each block is a tile.
     * Each thread is responsible for one pixel. (blockSize = tile_size *
     * tile_size)
     * ==============================
     */
    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets +=
        image_id * tile_height *
        tile_width; // get the global offset of the tile w.r.t the image
    render_colors +=
        image_id * image_height * image_width *
        CDIM; // get the global offset of the pixel w.r.t the image
    render_alphas +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    last_ids +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    render_normals += image_id * image_height * image_width * 3;
    render_distort += image_id * image_height * image_width;
    render_median += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    // get the global offset of the background and mask
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // find the center of the pixel
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

    // print
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        // see if this is the last tile in the image
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    /**
     * ==============================
     * Register computing variables:
     * For each pixel, we need to find its uv intersection with the gaussian
     * primitives. then we retrieve the kernel's parameters and kernel weights
     * do the splatting rendering equation.
     * ==============================
     */
    // Shared memory layout:
    // This memory is laid out as follows:
    // | gaussian indices | x : y : alpha | u | v | w |
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    // stores the concatination for projected primitive source (x, y) and
    // opacity alpha
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]

    // these are row vectors of the ray transformation matrices for the current
    // batch of gaussians
    vec3 *u_Ms_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *v_Ms_batch =
        reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]); // [block_size]
    vec3 *w_Ms_batch =
        reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    // The coefficient for volumetric rendering for our responsible pixel.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    // Per-pixel distortion error proposed in Mip-NeRF 360.
    // Implemented reference:
    // https://github.com/nerfstudio-project/nerfacc/blob/master/nerfacc/losses.py#L7
    float distort = 0.f;
    float accum_vis_depth = 0.f; // accumulate vis * depth

    // keep track of median depth contribution
    float median_depth = 0.f;
    uint32_t median_idx = 0.f;

    /**
     * ==============================
     * Per-pixel rendering: (2DGS Differntiable Rasterizer Forward Pass)
     * This section is responsible for rendering a single pixel.
     * It processes batches of gaussians and accumulates the pixel color and
     * normal.
     * ==============================
     */

    // TODO (WZ): merge pix_out and normal_out to
    //  float pix_out[CDIM + 3] = {0.f}
    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
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

        // only threads within the range of the tile will fetch gaussians
        /**
         * Launch this block with each thread responsible for one gaussian.
         */
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            u_Ms_batch[tr] = {
                ray_transforms[g * 9],
                ray_transforms[g * 9 + 1],
                ray_transforms[g * 9 + 2]
            };
            v_Ms_batch[tr] = {
                ray_transforms[g * 9 + 3],
                ray_transforms[g * 9 + 4],
                ray_transforms[g * 9 + 5]
            };
            w_Ms_batch[tr] = {
                ray_transforms[g * 9 + 6],
                ray_transforms[g * 9 + 7],
                ray_transforms[g * 9 + 8]
            };
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        /**
         * ==================================================
         * Forward rasterization pass:
         * ==================================================
         *
         * GSplat computes rasterization point of intersection as:
         * 1. Generate 2 homogeneous plane parameter vectors as sets of points
         * in UV space
         * 2. Find the set of points that satisfy both conditions with the cross
         * product
         * 3. Find where this solution set intersects with UV plane using
         * projective flattening
         *
         * For each gaussian G_i and pixel q_xy:
         *
         * 1. Compute homogeneous plane parameters:
         *    h_u = p_x * M_w - M_u
         *    h_v = p_y * M_w - M_v
         *    where M_u, M_v, M_w are rows of the KWH transform
         *
         * Note: this works because:
         *    for any vector q_uv [u, v, 1], applying co-vector h_u will yield
         * the following expression: h_u * [u, v, 1]^T = P_x * (M_w * q_uv) -
         * M_u * q_uv = P_x * q_ray.z - q_ray.x * q_ray.z
         *    - where P_x is the x-coordinate of the ray origin
         *    Thus: h_u  defines a set of q_uv where q_uv's projected x
         * coordinate in ray space is P_x which aligns with the homogeneous
         * plane definition in original 2DGS paper (similar for h_v)
         *
         * 2. Compute intersection:
         *    zeta = h_u × h_v
         *    This cross product is the only solution that satisfies both
         * homogeneous plane equations (dot product == 0)
         *
         * 3. Project to UV space:
         *    s_uv = [zeta_1/zeta_3, zeta_2/zeta_3]
         *    - since UV space is essentially another ray space, and arbitrary
         * scale of q_uv will not change the result of dot product over
         * orthogonality
         *    - thus, the result is the point of intersection in UV space
         *
         * 4. Evaluate gaussian kernel:
         *    G_i = exp(-(s_u^2 + s_v^2)/2)
         *
         * 5. Accumulate color:
         *    p_xy += alpha_i * c_i * G_i * prod(1 - alpha_j * G_j)
         *
         * This method efficiently computes the point of intersection and
         * evaluates the gaussian kernel in UV space.
         * Note: in some cases, we use the minimum of ray-intersection kernels
         * and 2D projected gaussian kernels
         */
        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {

            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;

            const vec3 u_M = u_Ms_batch[t];
            const vec3 v_M = v_Ms_batch[t];
            const vec3 w_M = w_Ms_batch[t];

            // h_u and h_v are the homogeneous plane representations (they are
            // contravariant to the points on the primitive plane)
            const vec3 h_u = px * w_M - u_M;
            const vec3 h_v = py * w_M - v_M;

            const vec3 ray_cross = glm::cross(h_u, h_v);
            if (ray_cross.z == 0.0)
                continue;

            const vec2 s =
                vec2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            // IMPORTANT: This is where the gaussian kernel is evaluated!!!!!

            // point of interseciton in uv space
            const float gauss_weight_3d = s.x * s.x + s.y * s.y;

            // projected gaussian kernel
            const vec2 d = {xy_opac.x - px, xy_opac.y - py};
            // #define FILTER_INV_SQUARE_2DGS 2.0f
            const float gauss_weight_2d =
                FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);

            // merge ray-intersection kernel and 2d gaussian kernel
            const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

            const float sigma = 0.5f * gauss_weight;
            // evaluation of the gaussian exponential term
            float alpha = min(MAX_ALPHA, opac * __expf(-sigma));

            // ignore transparent gaussians
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }

            // run volumetric rendering..
            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            const float *n_ptr = normals + g * 3;
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                normal_out[k] += n_ptr[k] * vis;
            }

            if (render_distort != nullptr) {
                // the last channel of colors is depth
                const float depth = c_ptr[CDIM - 1];
                // in nerfacc, loss_bi_0 = weights * t_mids *
                // exclusive_sum(weights)
                const float distort_bi_0 = vis * depth * (1.0f - T);
                // in nerfacc, loss_bi_1 = weights * exclusive_sum(weights *
                // t_mids)
                const float distort_bi_1 = vis * accum_vis_depth;
                distort += 2.0f * (distort_bi_0 - distort_bi_1);
                accum_vis_depth += vis * depth;
            }

            // compute median depth
            if (T > 0.5) {
                median_depth = c_ptr[CDIM - 1];
                median_idx = batch_start + t;
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
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if (render_distort != nullptr) {
            render_distort[pix_id] = distort;
        }

        render_median[pix_id] = median_depth;
        // index in bin of gaussian that contributes to median depth
        median_ids[pix_id] = static_cast<int32_t>(median_idx);
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3] or [nnz, 3]
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
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_distort, // [..., image_height, image_width]
    at::Tensor render_median,  // [..., image_height, image_width]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids      // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size = tile_size * tile_size *
                         (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) +
                          sizeof(vec3) + sizeof(vec3));

    const auto python_impl =
        rasterize_to_pixels_2dgs_fwd_impl_from_id(rasterize_fwd_impl);
    bool use_gemm =
        should_use_rasterize_to_pixels_2dgs_fwd_gemm(tile_size, python_impl);
    if (use_gemm) {
        dim3 gemm_threads = {kGemmTileSize, kGemmTileSize, 1};
        // The 2DGS GEMM kernel consumes the original per-Gaussian tensors
        // directly, so dispatch here is just a backend selection plus launch.
        rasterize_to_pixels_2dgs_fwd_gemm_kernel<CDIM, float>
            <<<grid, gemm_threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                I,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                normals.data_ptr<float>(),
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
                render_normals.data_ptr<float>(),
                render_distort.data_ptr<float>(),
                render_median.data_ptr<float>(),
                last_ids.data_ptr<int32_t>(),
                median_ids.data_ptr<int32_t>()
            );
        return;
    }

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_2dgs_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_to_pixels_2dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
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
            render_normals.data_ptr<float>(),
            render_distort.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>()
        );
}

// Explicit instantiations matching the dispatch in Rasterization.cpp.
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_2dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor ray_transforms,                                       \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor normals,                                              \
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
        at::Tensor render_normals,                                             \
        at::Tensor render_distort,                                             \
        at::Tensor render_median,                                              \
        at::Tensor last_ids,                                                   \
        at::Tensor median_ids                                                  \
    );

GSPLAT_FOR_EACH(__INS__, GSPLAT_NUM_CHANNELS)
#undef __INS__

} // namespace gsplat

#endif
