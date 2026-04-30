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

#if GSPLAT_BUILD_3DGUT

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "Rasterization.h"
#include "RasterizeChunkCSR.h"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Utils.cuh"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

// TODO: rename tile_offsets to isect_offsets

// Compact CTA rasterizer for 3DGUT (world-space ray-gaussian evaluation).
// Same architectural pattern as the 3DGS compact CTA:
//   CTA_SIZE threads process a TILE_SIZE x TILE_SIZE tile,
//   each thread handling PIXELS_PER_THREAD pixels in a vertical stride.
// Shared memory per batch: CTA_SIZE * 80B = 2560B (vs 20480B with 256 threads).

// Per-architecture hardware cap on thread blocks per SM. ptxas rejects
// min_blocks_per_sm > HW cap under --warning-as-error.
//   sm_90, sm_100, sm_120: 32 blocks/SM
//   everything else:       16 blocks/SM (covers Ampere/Ada and anything older)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 32
#else
    #define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 16
#endif

// Range endpoints of the CDIM-indexed schedule.
#define GSPLAT_MIN_CDIM_FOR_HINT 1u
#define GSPLAT_MAX_CDIM_FOR_HINT 24u

// Target min_blocks at the endpoints of the schedule.
//   - CDIM=1 : push occupancy up to the HW cap, but not above 24 (beyond this
//              the kernel's natural register usage forces ptxas to spill).
//   - CDIM=24: 16. HW cap on pre-sm_90, 25% occ on sm_90+; fits within the
//              kernel's register footprint at full SH + extras.
#define GSPLAT_MIN_BLOCKS_AT_MIN_CDIM \
    (GSPLAT_ARCH_MAX_BLOCKS_PER_SM < 24 ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM : 24)
#define GSPLAT_MIN_BLOCKS_AT_MAX_CDIM 16u

// Per-CDIM occupancy hint for __launch_bounds__ min_blocks_per_sm.
// Linear schedule from MIN_BLOCKS_AT_MIN_CDIM down to MIN_BLOCKS_AT_MAX_CDIM
// over CDIM in [MIN_CDIM_FOR_HINT, MAX_CDIM_FOR_HINT]. Beyond that we emit
// min_blocks=1 (no effective hint) — high-channel kernels carry enough
// register pressure that forcing occupancy would spill.
//   blocks = high - ((CDIM - min_cdim) * (high - low)) / (max_cdim - min_cdim)
// Collapses to the constant low value on archs where high == low.
template <uint32_t CDIM>
constexpr uint32_t min_blocks_for_cdim() {
    if constexpr (CDIM > GSPLAT_MAX_CDIM_FOR_HINT) {
        return 1;
    } else {
        constexpr uint32_t high = GSPLAT_MIN_BLOCKS_AT_MIN_CDIM;
        constexpr uint32_t low = GSPLAT_MIN_BLOCKS_AT_MAX_CDIM;
        constexpr uint32_t cdim_span =
            GSPLAT_MAX_CDIM_FOR_HINT - GSPLAT_MIN_CDIM_FOR_HINT;
        constexpr uint32_t block_span = (high >= low) ? (high - low) : 0;
        constexpr uint32_t decrement =
            ((CDIM - GSPLAT_MIN_CDIM_FOR_HINT) * block_span) / cdim_span;
        return (high > decrement) ? (high - decrement) : low;
    }
}

template <uint32_t CDIM, uint32_t TILE_SIZE, uint32_t CTA_SIZE>
__global__ void __launch_bounds__(CTA_SIZE, min_blocks_for_cdim<CDIM>())
rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const float *__restrict__ colors,         // [B, C, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,      // [B, C, N] or [nnz]
    const float *__restrict__ backgrounds,    // [B, C, CDIM]
    const bool *__restrict__ masks,           // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    // camera model
    const float *__restrict__ viewmats0, // [B, C, 4, 4]
    const float *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const float *__restrict__ Ks,        // [B, C, 3, 3]
    const CameraModelType camera_model_type,
    // unscented transform
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const float *__restrict__ rays,                  // [B, C, H, W, 6]
    const float *__restrict__ radial_coeffs,         // [B, C, 6] or [B, C, 4] optional
    const float *__restrict__ tangential_coeffs,     // [B, C, 2] optional
    const float *__restrict__ thin_prism_coeffs,     // [B, C, 4] optional
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    // intersections
    const int32_t *__restrict__ tile_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    float *__restrict__ render_colors,        // [B, C, image_height, image_width, CDIM]
    float *__restrict__ render_alphas,        // [B, C, image_height, image_width, 1]
    float *__restrict__ render_normals,       // [B, C, image_height, image_width, 3] optional
    int32_t *__restrict__ last_ids,           // [B, C, image_height, image_width]
    int32_t *__restrict__ sample_counts,      // [B, C, image_height, image_width] optional
    // CSR chunk-state persistence (for bwd reuse). See RasterizeChunkCSR.h.
    // Storage layout: [total_chunks][pixels_per_tile][1 + CDIM + 3] fp32.
    //   - [0]: T (cumulative transmittance after the persist batch)
    //   - [1..1+CDIM): pix_out[CDIM] (cumulative color accumulator)
    //   - [1+CDIM..1+CDIM+3): normal_out[3] (only written when render_normals != nullptr)
    // Each tile owns chunks_per_tile[tile_linear] slots, starting at
    // chunk_offsets[tile_linear]. Persist slot c (c in [0, num_chunks))
    // corresponds to fwd state after logical batch (num_logical_batches - 1 - c*CHUNK_BATCHES),
    // where one logical batch covers pixels_per_tile gaussians (matches bwd's batch unit).
    // chunk_offsets_csr and fwd_chunk_state are passed null-or-non-null in
    // lockstep by the launcher (gated on `total_chunks > 0`); both null ⇔
    // persistence disabled (e.g. `n_isects == 0`).
    const int32_t *__restrict__ chunk_offsets_csr, // [num_tiles + 1]
    float *__restrict__ fwd_chunk_state // [total_chunks, pixels_per_tile, 1 + CDIM + 3]    
) {
    // FETCH_SIZE = gaussians fetched per cooperative-fetch round (one per
    // thread; sized to the CTA so threads fetch in parallel without idling).
    // LOGICAL_BATCH = gaussians per outer-loop iteration; matches the bwd's
    // batch unit (= pixels_per_tile) so chunk-state persist boundaries land
    // at the same gaussian positions the bwd's CSR view expects. Each logical
    // batch is split into FETCHES_PER_BATCH cooperative fetch rounds.
    constexpr uint32_t FETCH_SIZE        = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t LOGICAL_BATCH     = TILE_SIZE * TILE_SIZE;
    constexpr uint32_t FETCHES_PER_BATCH = LOGICAL_BATCH / FETCH_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert(PIXELS_PER_THREAD > 0, "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");
    static_assert(LOGICAL_BATCH % FETCH_SIZE == 0, "LOGICAL_BATCH must be a multiple of FETCH_SIZE");
    static_assert(FETCHES_PER_BATCH >= 1, "FETCHES_PER_BATCH must be >= 1");

    const int32_t iid = blockIdx.x;
    const uint32_t grid_width  = gridDim.z;
    const uint32_t grid_height = gridDim.y;

    const uint32_t tile_x = blockIdx.z;
    const uint32_t tile_y = blockIdx.y;
    const int32_t tile_id = blockIdx.y * grid_width + blockIdx.z;

    const uint32_t tid = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    const bool return_normals = render_normals != nullptr;

    // Offset pointers to current image
    tile_offsets += iid * grid_height * grid_width;
    render_colors += iid * image_height * image_width * CDIM;
    render_alphas += iid * image_height * image_width;
    if (render_normals != nullptr) {
        render_normals += iid * image_height * image_width * 3;
    }
    last_ids += iid * image_height * image_width;
    if (sample_counts != nullptr) {
        sample_counts += iid * image_height * image_width;
    }
    if (backgrounds != nullptr) {
        backgrounds += iid * CDIM;
    }
    if (masks != nullptr) {
        masks += iid * grid_height * grid_width;
    }
    if (rays != nullptr) {
        rays += iid * image_height * image_width * 6;
    }

    // Per-pixel coordinate setup.
    // For non-LIDAR: regular rectangular tile mapping (same as 3DGS compact CTA).
    // For LIDAR: arbitrary pixel mapping from tiles_to_elements_map.
    uint32_t pix_j[PIXELS_PER_THREAD];
    uint32_t pix_i[PIXELS_PER_THREAD];
    int32_t pix_id[PIXELS_PER_THREAD];
    uint32_t done_mask = 0;

    if (camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
#pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            const int elem = tid + p * CTA_SIZE;
            if (elem < element_count) {
                pix_j[p] = lidar_device_coeffs->tiles_to_elements_map[element_start + elem].x;
                pix_i[p] = lidar_device_coeffs->tiles_to_elements_map[element_start + elem].y;
                pix_id[p] = pix_i[p] * image_width + pix_j[p];
            } else {
                // Idle pad: mark out-of-bounds so the final `inside` check
                // (pix_i < H && pix_j < W) rejects the write. Leaving (0,0)
                // would race with the active thread covering pix (row=0, col=0).
                pix_j[p] = image_width;
                pix_i[p] = image_height;
                pix_id[p] = 0;
                done_mask |= (1u << p);
            }
        }
    } else {
        const uint32_t out_x = tile_x * TILE_SIZE + thread_x;
        done_mask = (out_x >= image_width) ? ALL_DONE : 0;
#pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            const uint32_t out_y = tile_y * TILE_SIZE + thread_y + p * ROW_STRIDE;
            pix_j[p] = out_x;
            pix_i[p] = out_y;
            pix_id[p] = out_y * image_width + out_x;
            if (out_y >= image_height) {
                done_mask |= (1u << p);
            }
        }
    }

    // Create rolling shutter parameter (shared across all pixels of this thread)
    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16
    );

    // Per-pixel ray computation
    vec3 ray_o[PIXELS_PER_THREAD] = {};
    vec3 ray_d[PIXELS_PER_THREAD] = {};

#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if (done_mask & (1u << p)) {
            continue;
        }

        const uint32_t j = pix_j[p];
        const uint32_t i = pix_i[p];
        WorldRay ray;

        // TODO: this should be templated on the sensor type or whether we're using rays as input.
        if (rays == nullptr) {
            // Create ray from pixel.
            // Each camera model's element_to_image_point converts (j, i) pixel
            // indices to the image-point convention it expects: pixel centers for
            // cameras, scaled-angle coordinates for lidar.
            if (camera_model_type == CameraModelType::PINHOLE) {
                if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                    if (external_distortion_device_params.has_value()) {
                        using CameraModel = PerfectPinholeCameraModel<extdist::BivariateWindshieldModel>;
                        CameraModel::KernelParameters kernel_params = {
                            {
                                {image_width, image_height},
                                rs_type,
                                *external_distortion_device_params,
                            },
                            Ks,
                        };
                        CameraModel camera_model(kernel_params, iid);
                        ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                    } else {
                        using CameraModel = PerfectPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                        CameraModel::KernelParameters kernel_params = {
                            {
                                {image_width, image_height},
                                rs_type,
                                {},
                            },
                            Ks,
                        };
                        CameraModel camera_model(kernel_params, iid);
                        ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                    }
                }
                else {
                    if (external_distortion_device_params.has_value()) {
                        using CameraModel = OpenCVPinholeCameraModel<extdist::BivariateWindshieldModel>;
                        CameraModel::KernelParameters kernel_params = {
                            {
                                {image_width, image_height},
                                rs_type,
                                *external_distortion_device_params,
                            },
                            Ks,
                            radial_coeffs,
                            tangential_coeffs,
                            thin_prism_coeffs,
                        };
                        CameraModel camera_model(kernel_params, iid);
                        ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                    } else {
                        using CameraModel = OpenCVPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                        CameraModel::KernelParameters kernel_params = {
                            {
                                {image_width, image_height},
                                rs_type,
                                {},
                            },
                            Ks,
                            radial_coeffs,
                            tangential_coeffs,
                            thin_prism_coeffs,
                        };
                        CameraModel camera_model(kernel_params, iid);
                        ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                    }
                }
            }
            else if (camera_model_type == CameraModelType::FISHEYE) {
                if (external_distortion_device_params.has_value()) {
                    using CameraModel = OpenCVFisheyeCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {
                            {image_width, image_height},
                            rs_type,
                            *external_distortion_device_params,
                        },
                        Ks,
                        radial_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                } else {
                    using CameraModel = OpenCVFisheyeCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {
                            {image_width, image_height},
                            rs_type,
                            {},
                        },
                        Ks,
                        radial_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            }
            else if (camera_model_type == CameraModelType::FTHETA) {
                if (external_distortion_device_params.has_value()) {
                    using CameraModel = FThetaCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {
                            {image_width, image_height},
                            rs_type,
                            *external_distortion_device_params,
                        },
                        Ks,
                        ftheta_device_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                } else {
                    using CameraModel = FThetaCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {
                            {image_width, image_height},
                            rs_type,
                            {},
                        },
                        Ks,
                        ftheta_device_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            }
            else if (camera_model_type == CameraModelType::LIDAR) {
                using CameraModel = RowOffsetStructuredSpinningLidarModel;
                assert(lidar_device_coeffs);
                CameraModel::KernelParameters kernel_params = {
                    *lidar_device_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
            else {
                assert(false);
                return;
            }
        }
        else {
            ray.ray_org = {rays[pix_id[p]*6+0], rays[pix_id[p]*6+1], rays[pix_id[p]*6+2]};
            ray.ray_dir = {rays[pix_id[p]*6+3], rays[pix_id[p]*6+4], rays[pix_id[p]*6+5]};
            ray.valid_flag = true;
        }

        if (!ray.valid_flag) {
            done_mask |= (1u << p);
        } else {
            ray_o[p] = ray.ray_org;
            ray_d[p] = ray.ray_dir;
        }
    }

    // When the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        if (done_mask != ALL_DONE) {
#pragma unroll
            for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                if (!(done_mask & (1u << p))) {
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        render_colors[pix_id[p] * CDIM + k] =
                            backgrounds == nullptr ? 0.0f : backgrounds[k];
                    }
                }
            }
        }
        return;
    }

    // Gaussian range for this tile
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (iid == (int32_t)gridDim.x - 1) &&
        (tile_id == (int32_t)(grid_width * grid_height) - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    // Logical-batch count matches the bwd's view (= ceil(range / pixels_per_tile)),
    // so persist boundaries align with the CSR-allocated slot positions.
    const uint32_t num_logical_batches =
        (range_end - range_start + LOGICAL_BATCH - 1) / LOGICAL_BATCH;

    // --- Chunk-state persistence setup (shared with bwd K1/K2) ---------------
    // Compute this tile's base slot in the CSR `fwd_chunk_state` buffer. Per
    // the CSR invariant (see `RasterizeChunkCSR.h`), the slot `c` for bwd
    // chunk c corresponds to fwd state after logical batch
    // `num_logical_batches - 1 - c*CHUNK_BATCHES` (for c in [0, num_chunks)),
    // so c=0 is the terminal state and c=num_chunks-1 is the earliest
    // persistable state. Logical batches advance by LOGICAL_BATCH gaussians,
    // matching the bwd's per-batch gaussian count, so this slot index maps
    // to the same gaussian position the bwd will read from. We precompute
    // per-pixel state write pointers here so the inner batch loop stays tight.
    //
    // chunk_offsets_csr and fwd_chunk_state are passed null-or-non-null in
    // lockstep by the launcher (gated on `total_chunks > 0`); pin the
    // invariant and read either pointer to detect "no persistence". The tile
    // index for CSR matches bwd:
    // tile_linear = iid * grid_height * grid_width + tile_id.
    assert((chunk_offsets_csr == nullptr) == (fwd_chunk_state == nullptr));
    const bool persist_chunks = chunk_offsets_csr != nullptr;
    const uint32_t tile_linear =
        iid * grid_height * grid_width + tile_id;
    const uint32_t pixels_per_tile = TILE_SIZE * TILE_SIZE;
    const uint32_t state_dim =
        FWD_CHUNK_STATE_PIX_OFFSET + CDIM + FWD_CHUNK_STATE_NORMAL_EXTRA;
    // chunk_base_slot is the slot index in fwd_chunk_state for this tile's
    // c=0 entry (i.e., the terminal state). Later c entries follow
    // contiguously in the CSR.
    const int64_t chunk_base_slot = persist_chunks
        ? static_cast<int64_t>(chunk_offsets_csr[tile_linear])
        : 0;
    // Number of chunks bwd will consume for this tile = ceil-div; we don't
    // reference `num_chunks` directly because the per-logical-batch persist
    // check `(num_logical_batches - 1 - lb) % CHUNK_BATCHES == 0` naturally
    // emits exactly `num_chunks` writes (one per persist boundary), and the
    // partial-last-chunk case (num_logical_batches % CHUNK_BATCHES != 0) maps
    // to c = num_chunks - 1 being the oldest boundary, written when
    // lb = lb_last = (num_logical_batches-1) - (num_chunks-1)*CHUNK_BATCHES.

    // Shared memory: FETCH_SIZE (= CTA_SIZE) entries; reused across each
    // logical batch's FETCHES_PER_BATCH cooperative-fetch rounds.
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [FETCH_SIZE]
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[FETCH_SIZE]); // [FETCH_SIZE]
    mat3 *iscl_rot_batch =
        reinterpret_cast<mat3 *>(&xyz_opacity_batch[FETCH_SIZE]); // [FETCH_SIZE]
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&iscl_rot_batch[FETCH_SIZE]); // [FETCH_SIZE]
    vec3 *normal_batch =
        reinterpret_cast<vec3 *>(&scale_batch[FETCH_SIZE]); // [FETCH_SIZE]

    // Per-pixel state
    int32_t cur_idx[PIXELS_PER_THREAD];
    float T[PIXELS_PER_THREAD];
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        cur_idx[p] = -1;
        T[p] = 1.0f;
    }
    int32_t n_accumulated[PIXELS_PER_THREAD] = {0};
    float pix_out[PIXELS_PER_THREAD][CDIM] = {0.f};
    vec3 normal_out[PIXELS_PER_THREAD] = {};

    // Lambda that persists the CURRENT per-pixel cumulative state (T, pix_out,
    // normal_out) to chunk slot c (c in [0, num_chunks)). All threads in the
    // block call this together — one slot row per thread (tr), with thread
    // rank tr indexing the pixels_per_tile axis. Threads where `inside ==
    // false` write their trivial (T=1, pix_out=0, normal_out=0) state into
    // the slot; bwd K1 writes similar no-op states in the same positions and
    // bwd variants that consume these slots also ignore out-of-bounds pixels
    // via the same `inside` check, so the write is harmless but kept to keep
    // the memory layout fully populated.
    //
    // `c=0` corresponds to the terminal state (what bwd chunk 0 starts from);
    // `c=num_chunks-1` corresponds to the earliest persistable state. The
    // boundary-formula derivation is documented in `RasterizeChunkCSR.h`.
    auto persist_state = [&](uint32_t c) {
        if (!persist_chunks) {
            return;
        }

#pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            const uint32_t tr = tid + p * CTA_SIZE;
            const int64_t slot = chunk_base_slot + static_cast<int64_t>(c);
            const int64_t base = slot * static_cast<int64_t>(pixels_per_tile) *
                                    static_cast<int64_t>(state_dim) +
                                static_cast<int64_t>(tr) *
                                    static_cast<int64_t>(state_dim);
            fwd_chunk_state[base + FWD_CHUNK_STATE_T_OFFSET] = T[p];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + k] = pix_out[p][k];
            }
            // Always zero-fill the normal slot when return_normals is false, so
            // bwd consumers can read it unconditionally without branching on a
            // template flag they may not know.
            if (return_normals) {
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 0] = normal_out[p].x;
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 1] = normal_out[p].y;
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 2] = normal_out[p].z;
            } else {
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 0] = 0.0f;
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 1] = 0.0f;
                fwd_chunk_state[base + FWD_CHUNK_STATE_PIX_OFFSET + CDIM + 2] = 0.0f;
            }
        }
    };

#pragma unroll 1
    for (uint32_t lb = 0; lb < num_logical_batches; ++lb) {
        // Each logical batch covers LOGICAL_BATCH (= pixels_per_tile) gaussians,
        // matching the bwd's per-batch unit so persist boundaries align with
        // the CSR slot positions. Internally it issues FETCHES_PER_BATCH
        // cooperative fetch rounds of FETCH_SIZE (= CTA_SIZE) gaussians each
        // — this is what keeps the CTA at one warp.
        const uint32_t logical_batch_start = range_start + LOGICAL_BATCH * lb;
#pragma unroll
        for (uint32_t r = 0; r < FETCHES_PER_BATCH; ++r) {
            const uint32_t batch_start = logical_batch_start + FETCH_SIZE * r;
            // Skip rounds that fall past the tile's gaussian range (last
            // logical batch may be partial). Each thread votes "no fetch"
            // uniformly via the per-thread `idx >= range_end` guard below;
            // here we only skip the cooperative fetch entirely when the
            // whole round is past range_end.
            if (batch_start >= (uint32_t)range_end) {
                break;
            }

            // Each thread fetches 1 gaussian from front to back
            const uint32_t idx = batch_start + tid;
            if (idx < range_end) {
                // TODO: only support 1 camera for now so it is ok to abuse the index.
                int32_t isect_id = flatten_ids[idx];
                int32_t isect_bid = isect_id / (C * N);
                int32_t isect_gid = isect_id % N;
                id_batch[tid] = isect_id;
                const vec3 xyz = means[isect_bid * N + isect_gid];
                const float opac = opacities[isect_id];
                xyz_opacity_batch[tid] = {xyz.x, xyz.y, xyz.z, opac};

                const vec4 quat = quats[isect_bid * N + isect_gid];
                vec3 scale = scales[isect_bid * N + isect_gid];

                assert(glm::dot(quat, quat) > 0.f);
                assert(scale[0] > 0.f && scale[1] > 0.f && scale[2] > 0.f);

                mat3 R = quat_to_rotmat(quat);
                mat3 S = mat3(
                    1.0f / scale[0], 0.f, 0.f,
                    0.f, 1.0f / scale[1], 0.f,
                    0.f, 0.f, 1.0f / scale[2]
                );
                iscl_rot_batch[tid] = S * glm::transpose(R);
                scale_batch[tid] = scale;

                if (return_normals) {
                    normal_batch[tid] = R[2];
                }
            }

            if constexpr (CTA_SIZE == 32) {
                __syncwarp();
            } else {
                __syncthreads();
            }

            // Process gaussians in this fetch round
            const uint32_t batch_size = min(FETCH_SIZE, ((uint32_t)range_end - batch_start));
            for (uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                const float opac = xyz_opac[3];
                const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                const mat3 iscl_rot = iscl_rot_batch[t];
                const vec3 scale = scale_batch[t];

#pragma unroll
                for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                    if (done_mask & (1u << p)) {
                        continue;
                    }

                    const vec3 gro = iscl_rot * (ray_o[p] - xyz);
                    const vec3 grd = safe_normalize(iscl_rot * ray_d[p]);
                    const vec3 gcrod = glm::cross(grd, gro);
                    const float grayDist = glm::dot(gcrod, gcrod);
                    const float power = -0.5f * grayDist;
                    float max_response = __expf(power);
                    float alpha = min(MAX_ALPHA, opac * max_response);

                    if (alpha < ALPHA_THRESHOLD) {
                        continue;
                    }

                    float hit_distance = 0.0f;
                    if (use_hit_distance) {
                        const float hit_t = glm::dot(grd, -gro);
                        const vec3 grds = scale * (grd * hit_t);
                        hit_distance = glm::length(grds);
                    }

                    const float next_T = T[p] * (1.0f - alpha);
                    if (next_T <= TRANSMITTANCE_THRESHOLD) {
                        done_mask |= (1u << p);
                        continue;
                    }

                    int32_t isect_id = id_batch[t];
                    const float vis = alpha * T[p];
                    const float *c_ptr = colors + isect_id * CDIM;

                    if (use_hit_distance) {
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            const float value = (k == CDIM - 1) ? hit_distance : c_ptr[k];
                            pix_out[p][k] += value * vis;
                        }
                    } else {
#pragma unroll
                        for (uint32_t k = 0; k < CDIM; ++k) {
                            pix_out[p][k] += c_ptr[k] * vis;
                        }
                    }

                    if (return_normals) {
                        const vec3 unnormalized_normal = normal_batch[t];
                        const bool flipped = glm::dot(unnormalized_normal, ray_d[p]) > 0.0f;
                        const vec3 unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;
                        const vec3 normal = safe_normalize(unnormalized_flipped);
                        normal_out[p] += normal * vis;
                    }

                    cur_idx[p] = batch_start + t;
                    n_accumulated[p]++;
                    T[p] = next_T;
                }
            }
        }

        // --- Chunk-boundary persist ---------------------------------------
        // After finishing logical batch `lb`, if this batch is a persist
        // boundary write the current per-pixel state into fwd_chunk_state. A
        // logical batch is a persist boundary when (num_logical_batches - 1
        // - lb) is a non-negative multiple of CHUNK_BATCHES; the corresponding
        // chunk index is c = (num_logical_batches - 1 - lb) / CHUNK_BATCHES.
        //
        // The lambda is called uniformly across all threads in the block
        // (no divergent control): every thread's T/pix_out/normal_out is
        // valid at this program point since we're outside the per-Gaussian
        // inner loop. This keeps the writes coalesced per CSR row.
        if (persist_chunks) {
            const int32_t diff = static_cast<int32_t>(num_logical_batches) - 1 -
                                 static_cast<int32_t>(lb);
            if (diff >= 0 && (diff % CHUNK_BATCHES) == 0) {
                persist_state(static_cast<uint32_t>(diff) / CHUNK_BATCHES);
            }
        }

        // Block-level early exit + inter-batch sync barrier
        if (__syncthreads_count(done_mask == ALL_DONE) >= CTA_SIZE) {
            // Block-level early exit: every pixel has hit T <=
            // TRANSMITTANCE_THRESHOLD (or was never inside). Remaining
            // persist boundaries therefore all reflect the terminal state
            // each thread holds now. Emit them before breaking so the CSR
            // slot array is completely populated — bwd K2 will be able to
            // load any slot `c` in [0, num_chunks) without needing to know
            // which batches actually executed.
            if (persist_chunks) {
                for (uint32_t lbb = lb + 1; lbb < num_logical_batches; ++lbb) {
                    const int32_t diff =
                        static_cast<int32_t>(num_logical_batches) - 1 -
                        static_cast<int32_t>(lbb);
                    if (diff >= 0 && (diff % CHUNK_BATCHES) == 0) {
                        persist_state(static_cast<uint32_t>(diff) / CHUNK_BATCHES);
                    }
                }
            }
            break;
        }
    }

    // Write outputs for each in-bounds pixel
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        // For non-LIDAR: explicit bounds check (same as 3DGS compact CTA).
        // For LIDAR: pixels assigned from the map are guaranteed in-bounds;
        // OOB pixels had done_mask set from the start and pix_id[p]=0.
        const bool inside = (pix_i[p] < image_height && pix_j[p] < image_width);
        if (inside) {
            render_alphas[pix_id[p]] = 1.0f - T[p];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pix_id[p] * CDIM + k] =
                    backgrounds == nullptr ? pix_out[p][k]
                                           : (pix_out[p][k] + T[p] * backgrounds[k]);
            }
            if (render_normals != nullptr) {
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    render_normals[pix_id[p] * 3 + k] = normal_out[p][k];
                }
            }
            last_ids[pix_id[p]] = static_cast<int32_t>(cur_idx[p]);
            if (sample_counts != nullptr) {
                sample_counts[pix_id[p]] = n_accumulated[p];
            }
        }
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, grid_h, grid_w]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [...., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    // external distortion
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, grid_h, grid_w]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    // CSR chunk structure (precomputed by caller, shared with bwd)
    const at::Tensor chunks_per_tile, // [num_tiles] int32
    const at::Tensor chunk_offsets,   // [num_tiles + 1] int32
    const int64_t total_chunks,       // scalar; equals chunk_offsets[num_tiles]
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3]
    at::Tensor fwd_chunk_state // [total_chunks, pixels_per_tile, 1 + CDIM + 3] fp32
) {
    // Note: quats need to be normalized before passing in.
    (void)chunks_per_tile;  // reserved for future parity checks

    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed mode not supported for 3DGUT forward rasterization");

    const uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    const uint32_t B = means.numel() / (N * 3);       // number of batches
    const uint32_t C = viewmats0.size(-3);            // number of cameras
    const uint32_t I = B * C;                         // number of images
    const uint32_t grid_h = tile_offsets.size(-2);
    const uint32_t grid_w = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);

    constexpr uint32_t TILE_SIZE = 8;
    constexpr uint32_t CTA_SIZE = 32;

    const dim3 threads = {CTA_SIZE, 1, 1};
    const dim3 grid = {I, grid_h, grid_w};
    // Shared memory: id_batch + xyz_opacity_batch + iscl_rot_batch + scale_batch + normal_batch
    const int64_t shmem_size =
        CTA_SIZE * (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(vec3) + sizeof(vec3));

    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");
    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(*ftheta_coeffs);
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params = cuda::std::nullopt;
    if (external_distortion_params.has_value()) {
        TORCH_CHECK(external_distortion_params.value(), "external_distortion_params intrusive_ptr is null");
        external_distortion_device_params = extdist::BivariateWindshieldModelDeviceParams(*external_distortion_params.value());
    }

    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs = cuda::std::nullopt;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model == CameraModelType::LIDAR, "If lidar sensor coefficients are given, the camera model must be lidar");
        lidar_device_coeffs = *lidar_coeffs.value();
    }
    else
    {
        TORCH_CHECK(camera_model != CameraModelType::LIDAR, "If the sensor isn't lidar, lidar coefficients must not be given");
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK(SupportedChannels::contains(channels),
        "Unsupported number of channels: ", channels,
        " (check GSPLAT_NUM_CHANNELS)");

    auto launch_kernel = [&]<typename ChannelsT>() {
        constexpr uint32_t CDIM = ChannelsT::value;

        if (cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes)."
            );
        }

        rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, TILE_SIZE, CTA_SIZE>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<const vec3 *>(means.const_data_ptr<float>()),
                reinterpret_cast<const vec4 *>(quats.const_data_ptr<float>()),
                reinterpret_cast<const vec3 *>(scales.const_data_ptr<float>()),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                backgrounds.has_value()
                    ? backgrounds.value().const_data_ptr<float>()
                    : nullptr,
                masks.has_value() ? masks.value().const_data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                // camera model
                viewmats0.const_data_ptr<float>(),
                viewmats1.has_value() ? viewmats1.value().const_data_ptr<float>()
                                      : nullptr,
                Ks.const_data_ptr<float>(),
                camera_model,
                *ut_params,
                rs_type,
                rays.has_value() ? rays.value().const_data_ptr<float>() : nullptr,
                radial_coeffs.has_value()
                    ? radial_coeffs.value().const_data_ptr<float>()
                    : nullptr,
                tangential_coeffs.has_value()
                    ? tangential_coeffs.value().const_data_ptr<float>()
                    : nullptr,
                thin_prism_coeffs.has_value()
                    ? thin_prism_coeffs.value().const_data_ptr<float>()
                    : nullptr,
                ftheta_device_coeffs,
                lidar_device_coeffs,
                external_distortion_device_params,
                // intersections
                tile_offsets.const_data_ptr<int32_t>(),
                flatten_ids.const_data_ptr<int32_t>(),
                use_hit_distance,
                renders.data_ptr<float>(),
                alphas.data_ptr<float>(),
                normals.has_value() ? normals.value().data_ptr<float>() : nullptr,
                last_ids.data_ptr<int32_t>(),
                sample_counts.has_value()
                    ? sample_counts.value().data_ptr<int32_t>()
                    : nullptr,
                // CSR chunk state persistence. A total_chunks==0 degenerate case
                // (e.g. n_isects==0) is signaled by passing nullptr — the
                // in-kernel `persist_chunks` flag then short-circuits all writes.
                (total_chunks > 0)
                    ? chunk_offsets.const_data_ptr<int32_t>()
                    : nullptr,
                (total_chunks > 0)
                    ? fwd_chunk_state.data_ptr<float>()
                    : nullptr
            );
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

} // namespace gsplat

#endif
