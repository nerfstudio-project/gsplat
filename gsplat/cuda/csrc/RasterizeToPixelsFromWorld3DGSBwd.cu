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
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "Rasterization.h"
#include "RasterizeChunkCSR.h"
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

// Pad CDIM to an odd stride when it would be even, so that the per-thread
// STORE `rgbs_batch[tr * stride + k] = colors[...]` in bwd writes to distinct
// 32-bit banks across a warp. With a stride S, the 32 lanes in a warp land
// in banks (tr * S) mod 32 for tr in [0,32); that hits all 32 banks iff
// gcd(S, 32) == 1, i.e. S is odd. The inner loop READ from a fixed `t` and
// varying `k` is a broadcast-per-lane and is not the conflict site.
// sizeof(float) == 4 → 32-bit banks; static_assert below pins that assumption.
template <uint32_t CDIM>
constexpr uint32_t cdim_smem_stride() {
    static_assert(sizeof(float) == 4, "bank layout assumes 32-bit banks");
    return (CDIM % 2 == 0) ? CDIM + 1 : CDIM;
}

// ---------------------------------------------------------------------------
// Single-kernel chunk-parallel backward pass.
//
// fwd persists per-chunk cumulative state into `fwd_chunk_state` at every
// CHUNK_BATCHES boundary; bwd derives the per-chunk starting accumulators
// from that state via a single CDIM dot + vec3 dot per thread in its
// preamble, then runs the per-gaussian gradient walk. See
// `RasterizeChunkCSR.h` for the tensor layout and boundary formula.
//
// Math (derived from the bwd recurrence in the hot loop below):
//   T_start_c          = fwd_chunk_state[slot=c, pix, T_OFFSET]
//                      = T_fwd at fwd-batch num_batches-1-c*CHUNK_BATCHES
//   render_accum_dot_c = dot(v_render_c,
//                            pix_out_final - pix_out_fwd_at_boundary_c)
//   normal_accum_dot_c = dot(v_render_n,
//                            normal_out_final - normal_out_fwd_at_boundary_c)
//   where `_final` is the c=0 slot (terminal fwd state) — the CSR-slot
//   semantics are documented in `RasterizeChunkCSR.h`.
//
// Grid: 1D {total_chunks, 1, 1}. Chunks are independent — no sequential
// dependency — exposing ample work to the GPU scheduler and eliminating the
// tail latency on tiles with many Gaussians.
// ---------------------------------------------------------------------------

// Device kernel that computes per-tile chunk counts. This was previously
// file-scoped to this TU; it is now reachable by the fwd persist path via
// the `compute_chunk_csr` host helper below. Kept `static` because only
// this TU launches it directly.
static __global__ void compute_chunks_per_tile_kernel(
    const int32_t *__restrict__ tile_offsets,
    const uint32_t num_tiles,
    const uint32_t n_isects,
    const uint32_t pixels_per_tile,
    int32_t *__restrict__ chunks_per_tile
) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tiles) {
        return;
    }
    const int32_t start = tile_offsets[t];
    const int32_t end = (t + 1 < num_tiles)
        ? tile_offsets[t + 1]
        : static_cast<int32_t>(n_isects);
    const int32_t range = end - start;
    if (range <= 0) {
        chunks_per_tile[t] = 0;
        return;
    }
    const uint32_t num_batches =
        (static_cast<uint32_t>(range) + pixels_per_tile - 1) / pixels_per_tile;
    const uint32_t num_chunks =
        (num_batches + CHUNK_BATCHES - 1) / CHUNK_BATCHES;
    chunks_per_tile[t] = static_cast<int32_t>(num_chunks);
}

// Host helper: see declaration in `RasterizeChunkCSR.h`. Launches the
// device kernel above, builds the [num_tiles+1] CSR offsets via cumsum+cat,
// then reads back `total_chunks` with a blocking `.item<int32_t>()`. Shared
// by fwd (persist buffer sizing) and bwd (gradient-kernel grid sizing).
std::tuple<at::Tensor, at::Tensor, int64_t> compute_chunk_csr(
    const at::Tensor &tile_offsets,
    int64_t n_isects,
    uint32_t num_tiles,
    uint32_t pixels_per_tile,
    at::TensorOptions dummy_options
) {
    auto int_opts = dummy_options.dtype(at::kInt);
    auto chunks_per_tile_t =
        at::empty({static_cast<int64_t>(num_tiles)}, int_opts);
    if (num_tiles > 0) {
        const uint32_t threads_per_block = 256;
        const uint32_t blocks =
            (num_tiles + threads_per_block - 1) / threads_per_block;
        compute_chunks_per_tile_kernel
            <<<blocks, threads_per_block, 0,
               at::cuda::getCurrentCUDAStream()>>>(
                tile_offsets.const_data_ptr<int32_t>(),
                num_tiles,
                static_cast<uint32_t>(n_isects),
                pixels_per_tile,
                chunks_per_tile_t.data_ptr<int32_t>());
    }
    // chunk_offsets[0..num_tiles]: exclusive prefix sum with a trailing sum.
    auto chunk_offsets_tail = at::cumsum(chunks_per_tile_t, 0, at::kInt);
    auto chunk_offsets_t = at::cat(
        {at::zeros({1}, int_opts), chunk_offsets_tail});
    const int64_t total_chunks = num_tiles == 0
        ? int64_t{0}
        : static_cast<int64_t>(
              chunk_offsets_t[static_cast<int64_t>(num_tiles)]
                  .item<int32_t>());
    return std::make_tuple(chunks_per_tile_t, chunk_offsets_t, total_chunks);
}

// --- Helpers for CSR-packed chunk state ---
//
// Each tile contributes `chunks_per_tile[t]` chunk-slots to the flat
// fwd_chunk_state buffer, with `chunk_offsets[t]` giving the starting slot
// of tile t. `chunk_offsets[num_tiles]` is the total number of chunk-slots
// and equals the CTA count for the gradient kernel. The flat layout
// avoids the per-tile padding (`num_tiles × max_chunks`) that would
// otherwise create a dense-scene OOM vector driven by the worst tile.
//
// `CHUNK_BATCHES` and `compute_chunks_per_tile_kernel` live in
// `RasterizeChunkCSR.h` so fwd and bwd can share a single source of truth.

// Binary-search helper: given ascending `chunk_offsets[num_tiles + 1]`, find
// tile t such that chunk_offsets[t] <= bid < chunk_offsets[t + 1]. Called
// once per CTA; O(log num_tiles) ≈ 14 global-mem reads on a 10K-tile scene
// and the reads hit L2 after the first CTA in the launch wave.
__device__ __forceinline__ uint32_t find_tile_for_block(
    const int32_t *__restrict__ chunk_offsets,
    uint32_t num_tiles,
    uint32_t bid
) {
    uint32_t lo = 0;
    uint32_t hi = num_tiles;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (static_cast<uint32_t>(chunk_offsets[mid + 1]) <= bid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Shared ray-generation helper. Takes the full camera-model switch
// (pinhole / fisheye / ftheta / lidar × {windshield, empty distortion})
// plus the "explicit rays" path, and returns the WorldRay for pixel
// (j, i) on image `iid`. Marked `__forceinline__` so the gradient kernel
// keeps its register count after inlining at the call site.
template <typename scalar_t>
__device__ __forceinline__ WorldRay compute_world_ray_bwd(
    const uint32_t iid,
    const uint32_t j,
    const uint32_t i,
    const int32_t pix_id,
    const bool inside,
    const scalar_t *__restrict__ rays,
    const scalar_t *__restrict__ viewmats0,
    const scalar_t *__restrict__ viewmats1,
    const scalar_t *__restrict__ Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const scalar_t *__restrict__ radial_coeffs,
    const scalar_t *__restrict__ tangential_coeffs,
    const scalar_t *__restrict__ thin_prism_coeffs,
    const FThetaCameraDistortionDeviceParams& ftheta_device_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice>& lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams>& external_distortion_device_params
) {
    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);
    WorldRay ray;
    if (inside && rays == nullptr) {
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                if (external_distortion_device_params.has_value()) {
                    using CameraModel = PerfectPinholeCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        { {image_width, image_height}, rs_type, *external_distortion_device_params },
                        Ks,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                } else {
                    using CameraModel = PerfectPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        { {image_width, image_height}, rs_type, {} },
                        Ks,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            } else {
                if (external_distortion_device_params.has_value()) {
                    using CameraModel = OpenCVPinholeCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        { {image_width, image_height}, rs_type, *external_distortion_device_params },
                        Ks, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                } else {
                    using CameraModel = OpenCVPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        { {image_width, image_height}, rs_type, {} },
                        Ks, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            }
        } else if (camera_model_type == CameraModelType::FISHEYE) {
            if (external_distortion_device_params.has_value()) {
                using CameraModel = OpenCVFisheyeCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_device_params },
                    Ks, radial_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            } else {
                using CameraModel = OpenCVFisheyeCameraModel<extdist::EmptyExternalDistortionModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, {} },
                    Ks, radial_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
        } else if (camera_model_type == CameraModelType::FTHETA) {
            if (external_distortion_device_params.has_value()) {
                using CameraModel = FThetaCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_device_params },
                    Ks, ftheta_device_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            } else {
                using CameraModel = FThetaCameraModel<extdist::EmptyExternalDistortionModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, {} },
                    Ks, ftheta_device_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
        } else if (camera_model_type == CameraModelType::LIDAR) {
            using CameraModel = RowOffsetStructuredSpinningLidarModel;
            assert(lidar_device_coeffs);
            CameraModel::KernelParameters kernel_params = { *lidar_device_coeffs };
            CameraModel camera_model(kernel_params, iid);
            ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
        } else {
            assert(false);
            ray.valid_flag = false;
        }
    } else {
        // Explicit rays path — rays may be nullptr for inactive threads when inside==false.
        ray.valid_flag = false;
        if (inside) {
            assert(rays != nullptr);
            // TODO: use at least 3x64b loads instead of 6x32b
            ray.ray_org = {rays[pix_id * 6 + 0], rays[pix_id * 6 + 1], rays[pix_id * 6 + 2]};
            ray.ray_dir = {rays[pix_id * 6 + 3], rays[pix_id * 6 + 4], rays[pix_id * 6 + 5]};
            ray.valid_flag = true;
        }
    }
    return ray;
}

// Pixel-coordinate resolution. Handles the LIDAR-vs-camera split: for
// LIDAR, looks up (row, col) from the per-tile element map and marks
// inside=false for threads past element_count; for camera paths, derives
// (row, col) from the tile origin + per-thread 2D rank and sets inside
// from a bounds check. Returns the clamped linear pix_id used by every
// downstream load/store. Marked `__forceinline__` so the call site keeps
// its register count after inlining.
struct PixelCoords
{
    uint32_t row;
    uint32_t col;
    int32_t pix_id;
    bool inside;
};

__device__ __forceinline__ PixelCoords compute_pixel_coords_bwd(
    const CameraModelType camera_model_type,
    const uint32_t tile_id,
    const uint32_t tile_row,
    const uint32_t tile_col,
    const uint32_t tile_size,
    const uint32_t tr,
    const uint32_t image_width,
    const uint32_t image_height,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice>& lidar_device_coeffs
)
{
    PixelCoords out;
    if (camera_model_type == CameraModelType::LIDAR)
    {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int tile_element_id = static_cast<int>(tr);
        if (tile_element_id < element_count)
        {
            out.col = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].x; // col_azimuth
            out.row = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].y; // row_elevation
            assert(0 <= out.row);
            assert(out.row < image_height);
            assert(0 <= out.col);
            assert(out.col < image_width);
            out.inside = true;
        }
        else
        {
            out.row = 0;
            out.col = 0;
            out.inside = false;
        }
    }
    else
    {
        out.row = tile_row * tile_size + threadIdx.y;
        out.col = tile_col * tile_size + threadIdx.x;
        out.inside = (out.row < image_height && out.col < image_width);
    }
    // Clamp to last pixel for out-of-bounds threads (both branches produce
    // valid (row, col) in range, so this is a min() rather than a branch).
    out.pix_id = min(static_cast<int32_t>(out.row * image_width + out.col),
                     static_cast<int32_t>(image_width * image_height) - 1);
    return out;
}

// Gradient pass. Per-CTA: (1) chunk_id decoded from grid x via binary
// search on chunk_offsets, (2) initial state materialised from the
// fwd-persisted `fwd_chunk_state` tensor via one CDIM dot + one vec3 dot
// per thread, (3) batch range limited to CHUNK_BATCHES, (4) v_rays
// accumulated via atomicAdd (multiple chunks per pixel).
//
// Grid: 1D {total_chunks, 1, 1}; each CTA's (tile_linear, chunk_id) is decoded
// via a binary search on chunk_offsets in the preamble.
template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_gradient_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // fwd inputs
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const scalar_t *__restrict__ colors,      // [B, C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [B, C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [B, C, CDIM]
    const bool *__restrict__ masks,           // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const scalar_t *__restrict__ viewmats0, // [B, C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const scalar_t *__restrict__ rays,              // [B, C, H, W, 6]
    const scalar_t *__restrict__ radial_coeffs,     // [B, C, 6] or [B, C, 4] optional
    const scalar_t *__restrict__ tangential_coeffs, // [B, C, 2] optional
    const scalar_t *__restrict__ thin_prism_coeffs, // [B, C, 4] optional
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs, // shared parameters for all cameras
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    // intersections
    const int32_t *__restrict__ tile_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [B, C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [B, C, image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [B, C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [B, C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_normals, // [B, C, image_height, image_width, 3] optional
    // grad inputs
    vec3 *__restrict__ v_means,        // [B, N, 3]
    vec4 *__restrict__ v_quats,        // [B, N, 4]
    vec3 *__restrict__ v_scales,       // [B, N, 3]
    scalar_t *__restrict__ v_colors,   // [B, C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [B, C, N] or [nnz]
    scalar_t *__restrict__ v_rays,     // [B, C, image_height, image_width, 6]
    // Fwd-persisted per-chunk cumulative state (CSR-packed; see
    // `RasterizeChunkCSR.h`). Layout per slot: [T, pix_out[CDIM],
    // normal_out[3]]. Slot c=0 is the terminal fwd state (used as
    // `_final` reference); slot c=chunk_id is this CTA's boundary.
    const scalar_t *__restrict__ fwd_chunk_state, // [total_chunks, pixels_per_tile, 1+CDIM+3]
    const int32_t *__restrict__ chunk_offsets     // [num_tiles + 1]
)
{
    // ---- Preamble: CSR chunk decode, pixel map, ray gen, fwd-state load ----
    // The chunk-start accumulators are derived from `fwd_chunk_state` further
    // down via one CDIM dot + one vec3 dot per thread.
    auto block = cg::this_thread_block();
    const uint32_t num_tiles_total = (B * C) * tile_height * tile_width;
    const uint32_t bid = block.group_index().x;
    const uint32_t tile_linear =
        find_tile_for_block(chunk_offsets, num_tiles_total, bid);
    const uint32_t chunk_id =
        bid - static_cast<uint32_t>(chunk_offsets[tile_linear]);
    const uint32_t iid = tile_linear / (tile_height * tile_width);
    const uint32_t tile_id = tile_linear - iid * (tile_height * tile_width);
    const uint32_t tile_row = tile_id / tile_width;
    const uint32_t tile_col = tile_id - tile_row * tile_width;
    const uint32_t tr = block.thread_rank();
    const uint32_t block_size = block.size();

    tile_offsets += iid * tile_height * tile_width;
    render_alphas += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    v_render_colors += iid * image_height * image_width * CDIM;
    v_render_alphas += iid * image_height * image_width;
    if (v_render_normals != nullptr) {
        v_render_normals += iid * image_height * image_width * 3;
    }
    if (backgrounds != nullptr) {
        backgrounds += iid * CDIM;
    }
    if (masks != nullptr) {
        masks += iid * tile_height * tile_width;
    }
    if(rays != nullptr) {
        rays += iid*image_height*image_width*6;
    }
    if (v_rays != nullptr) {
        v_rays += iid*image_height*image_width*6;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;
    const uint32_t pixels_per_tile = tile_size * tile_size;
    // Grid sized to total_chunks (CSR); every launched block has a valid slot.

    // Pixel mapping (camera vs lidar) — shared helper.
    const PixelCoords pc = compute_pixel_coords_bwd(
        camera_model_type, tile_id, tile_row, tile_col, tile_size, tr,
        image_width, image_height, lidar_device_coeffs);
    const uint32_t i = pc.row;
    const uint32_t j = pc.col;
    const bool inside = pc.inside;
    const int32_t pix_id = pc.pix_id;

    // Ray generation — uses the shared compute_world_ray_bwd helper above.
    WorldRay ray = compute_world_ray_bwd<scalar_t>(
        iid, j, i, pix_id, inside,
        rays, viewmats0, viewmats1, Ks,
        image_width, image_height,
        camera_model_type, rs_type,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_device_coeffs, lidar_device_coeffs,
        external_distortion_device_params);
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;
    const bool pixel_valid = inside && ray.valid_flag;
    const int32_t bin_final = pixel_valid ? last_ids[pix_id] : 0;

    const float T_final = pixel_valid ? 1.0f - render_alphas[pix_id] : 1.0f;

    // ---- Per-pixel gradients (same as original kernel) ----
    // These are loaded BEFORE the chunk-state materialisation below so the
    // CDIM / vec3 dots can be fused in the same arithmetic region without
    // a second pass over v_render_c / v_render_n.
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = pixel_valid ? v_render_colors[pix_id * CDIM + k] : 0.f;
    }
    const float v_render_a = pixel_valid ? v_render_alphas[pix_id] : 0.f;

    vec3 v_render_n = vec3(0.f);
    if (v_render_normals != nullptr && pixel_valid) {
        v_render_n.x = v_render_normals[pix_id * 3 + 0];
        v_render_n.y = v_render_normals[pix_id * 3 + 1];
        v_render_n.z = v_render_normals[pix_id * 3 + 2];
    }

    // ---- Materialise chunk-start state from `fwd_chunk_state` ----------------
    // The monolithic bwd recurrence satisfies, at the start of bwd chunk c:
    //   T_start_c          = T_fwd at fwd-batch num_batches-1-c*CHUNK_BATCHES
    //   render_accum_dot_c = dot(v_render_c, pix_out_final - pix_out_boundary_c)
    //   normal_accum_dot_c = dot(v_render_n, normal_out_final - normal_out_boundary_c)
    // where slot c=0 IS the terminal fwd state (see `RasterizeChunkCSR.h`).
    //
    // CSR slot layout: `[T, pix_out[CDIM], normal_out[3]]` per pixel-per-slot.
    //   - This CTA's boundary slot: `bid` (=chunk_offsets[tile]+chunk_id).
    //   - The terminal slot for this tile: `chunk_offsets[tile_linear]` (c=0).
    //
    // int64 offset arithmetic: `total_chunks * pixels_per_tile * STATE_DIM` can
    // exceed 2^31 on dense scenes at CDIM=24+ (≈18M intersections). The fwd
    // persist path already uses int64; bwd matches.
    constexpr uint32_t STATE_DIM = 1 + CDIM + 3;
    constexpr uint32_t PIX_OFFSET = 1;
    constexpr uint32_t NORMAL_OFFSET = 1 + CDIM;
    const int64_t terminal_slot =
        static_cast<int64_t>(chunk_offsets[tile_linear]);
    const int64_t ppt64 = static_cast<int64_t>(pixels_per_tile);
    const int64_t sd64 = static_cast<int64_t>(STATE_DIM);
    const int64_t tr64 = static_cast<int64_t>(tr);
    const int64_t terminal_base =
        terminal_slot * ppt64 * sd64 + tr64 * sd64;
    const int64_t boundary_base =
        static_cast<int64_t>(bid) * ppt64 * sd64 + tr64 * sd64;

    // T starts at the persisted fwd cumulative transmittance at this chunk's
    // boundary — no further scaling needed (see math in file header).
    float T = fwd_chunk_state[boundary_base + 0];

    // render_accum_dot = dot(v_render_c, pix_out_final - pix_out_boundary)
    // Streaming per-k load keeps the register footprint scalar (one `delta`
    // in flight at a time) rather than materialising pix_out_boundary[CDIM]
    // as live registers — critical for the kernel's already-tight register budget.
    float render_accum_dot = 0.f;
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        const float pix_final_k = fwd_chunk_state[terminal_base + PIX_OFFSET + k];
        const float pix_boundary_k = fwd_chunk_state[boundary_base + PIX_OFFSET + k];
        const float delta_k = pix_final_k - pix_boundary_k;
        render_accum_dot += delta_k * v_render_c[k];
    }

    // normal_accum_dot = dot(v_render_n, normal_final - normal_boundary).
    // fwd always zero-fills the normal slot when `return_normals` is false,
    // so reading unconditionally is safe. Guarded by the nullptr check to
    // skip three loads + three muls when normals are disabled.
    float normal_accum_dot = 0.f;
    if (v_render_normals != nullptr) {
        const float nx_final = fwd_chunk_state[terminal_base + NORMAL_OFFSET + 0];
        const float ny_final = fwd_chunk_state[terminal_base + NORMAL_OFFSET + 1];
        const float nz_final = fwd_chunk_state[terminal_base + NORMAL_OFFSET + 2];
        const float nx_boundary = fwd_chunk_state[boundary_base + NORMAL_OFFSET + 0];
        const float ny_boundary = fwd_chunk_state[boundary_base + NORMAL_OFFSET + 1];
        const float nz_boundary = fwd_chunk_state[boundary_base + NORMAL_OFFSET + 2];
        normal_accum_dot = (nx_final - nx_boundary) * v_render_n.x
                         + (ny_final - ny_boundary) * v_render_n.y
                         + (nz_final - nz_boundary) * v_render_n.z;
    }
    float background_render_dot = 0.f;
    if (pixel_valid && backgrounds != nullptr) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            background_render_dot += backgrounds[k] * v_render_c[k];
        }
    }
    const float v_alpha_ind_coeff = v_render_a - background_render_dot;

    vec3 v_ray_o = {0.f, 0.f, 0.f};
    vec3 v_ray_d = {0.f, 0.f, 0.f};

    extern __shared__ int s[];  // id_batch prefix + xyz_opacity + scale + quat + rgbs
    int32_t *id_batch = (int32_t *)s;
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[block_size]);
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&xyz_opacity_batch[block_size]);
    vec4 *quat_batch =
        reinterpret_cast<vec4 *>(&scale_batch[block_size]);
    float *rgbs_batch = (float *)&quat_batch[block_size];

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    // ---- Batch loop: same as original kernel, chunk-limited ----
    const uint32_t b_start = chunk_id * CHUNK_BATCHES;
    const uint32_t b_end = min(b_start + CHUNK_BATCHES, num_batches);

    for (uint32_t b = b_start; b < b_end; ++b) {
        block.sync();
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size =
            min(static_cast<int32_t>(block_size), batch_end + 1 - range_start);
        const int32_t idx = batch_end - static_cast<int32_t>(tr);

        if (idx >= range_start) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C; // intersection camera index
            int32_t isect_gid = isect_id % N;          // intersection gaussian index
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            scale_batch[tr] = scales[isect_bid * N + isect_gid];
            quat_batch[tr] = quats[isect_bid * N + isect_gid];
            assert(glm::dot(quat_batch[tr], quat_batch[tr]) > 0.f);
            assert(scale_batch[tr][0] > 0.f && scale_batch[tr][1] > 0.f && scale_batch[tr][2] > 0.f);
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * cdim_smem_stride<CDIM>() + k] = colors[isect_id * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = pixel_valid;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float vis;

            mat3 R, S;
            vec3 xyz;
            vec3 scale;
            vec4 quat;
            mat3 Mt;
            vec3 o_minus_mu, gro, grd, grd_n, gcrod;
            float grayDist, power;
            // Per-pixel hit-distance state. Three values:
            // - `hit_distance`: world-frame length of `grds` (per-pixel)
            // - `hit_t`:        whitened parametric closest-point distance
            // - `grds`:         scale-weighted whitened ray direction
            //
            // Held in registers (not in shmem `rgbs_batch`) because they
            // depend on ray_o/ray_d (per-pixel), whereas `rgbs_batch` is
            // per-Gaussian (shared across all pixels in the tile).
            //
            // Computed once in the alpha-recompute block below and reused by:
            // - the per-pixel rgb-render-dot path (hit_distance only)
            // - the hit-distance VJP block       (all three)
            float hit_distance = 0.f;
            float hit_t = 0.f;
            vec3 grds = vec3(0.f);
            if (valid) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                opac = xyz_opac[3];
                xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = scale_batch[t];
                quat = quat_batch[t];

                R = quat_to_rotmat(quat);
                S = mat3(
                    1.0f / scale[0],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[1],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[2]
                );
                // Match fwd's Mt construction expression form (S * Rᵀ vs
                // (R * S)ᵀ are mathematically equal for diagonal S, but
                // compilers may pick different FFMA fusions per source
                // form). Today nvcc emits identical SASS for both — this
                // is forward-defensive: a future compiler / surrounding-
                // code change could break the bit-equivalence we get
                // today, and source-form match keeps it stable. Softer
                // guarantee than `safe_normalize`'s explicit-intrinsic
                // pinning, but no intrinsics exist for matrix
                // construction expressions.
                Mt = S * glm::transpose(R);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;
                grd_n = safe_normalize(grd);
                // hit_t < 0 → closest approach behind camera; skip this gaussian.
                // Declared in outer scope so the hit-distance VJP block can reuse it.
                hit_t = -glm::dot(grd_n, gro);
                if (hit_t < 0.f) {
                    valid = false;
                }
                else {
                    gcrod = glm::cross(grd_n, gro);
                    grayDist = glm::dot(gcrod, gcrod);
                    power = -0.5f * grayDist;

                    vis = __expf(power);
                    alpha = min(MAX_ALPHA, opac * vis);
                    // grayDist = dot(gcrod, gcrod) is a sum of three squares, so
                    // grayDist >= 0 and therefore power = -0.5 * grayDist <= 0
                    // under any IEEE-754 evaluation order on finite inputs.
                    // Assert the invariant and use the same skip predicate as
                    // fwd. The assert also fires on NaN power, which would
                    // itself indicate an upstream numerics bug feeding NaN
                    // into gcrod.
                    assert(power <= 0.f);
                    if (alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }

                    if (use_hit_distance) {
                        grds = scale * (grd_n * hit_t);
                        hit_distance = glm::length(grds);
                    }
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_mean_local = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;

            // initialize everything to 0, only set if the lane is valid
            vec3 normal = {0.f, 0.f, 0.f};
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T *= ra;
                // Per-Gaussian color VJP: v_rgb_local[k] = fac * v_render_c[k].
                //
                // Last-channel special case when `use_hit_distance` is on:
                // - fwd substitutes per-pixel hit_distance for colors[g,CDIM-1]
                //   in pix_out, so colors[..., CDIM-1] is structurally absent
                //   from the rendered output.
                // - Therefore d(loss)/d(colors[..., CDIM-1]) = 0.
                // - Zero v_rgb_local[CDIM-1] so the warp-reduced atomic add
                //   leaves v_colors[..., CDIM-1] at 0.
                // - The hit_distance VJP below pulls the depth-channel
                //   gradient straight from v_render_c[CDIM-1], not from this
                //   slot.
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                if (use_hit_distance) {
                    v_rgb_local[CDIM - 1] = 0.f;
                }

                // Precompute normal if needed. Used by:
                // - v_alpha computation (this block)
                // - normal-gradient VJP block below
                //
                // Hoist `flipped` and `unnormalized_flipped` to outer scope so
                // the gradient block reuses the values without recomputing
                // R[2] / the dot / the flip.
                bool flipped = false;
                vec3 unnormalized_flipped = vec3(0.f);
                if (v_render_normals != nullptr) {
                    // Recompute normal from forward pass
                    // normal = R * (0, 0, 1) = R[:, 2] (third column)
                    const vec3 unnormalized_normal = R[2];

                    // Direction resolution: flip if facing away from ray
                    flipped = glm::dot(unnormalized_normal, ray_d) > 0.0f;
                    unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;

                    // Normalize
                    normal = safe_normalize(unnormalized_flipped);
                }
                
                // contribution from this pixel
                float rgb_render_dot = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    // For the last channel with use_hit_distance, use the per-pixel
                    // hit_distance instead of per-Gaussian rgbs_batch (which is
                    // shared memory and would race across pixels in the tile).
                    const float rgb_k = (use_hit_distance && k == CDIM - 1)
                        ? hit_distance
                        : rgbs_batch[t * cdim_smem_stride<CDIM>() + k];
                    rgb_render_dot += rgb_k * v_render_c[k];
                }

                // Per-Gaussian projection of the rendered-normal grad onto the
                // Gaussian's normal vector — used in the product-rule term for
                // v_alpha below.
                float normal_render_dot = 0.f;
                if (v_render_normals != nullptr) {
                    normal_render_dot = glm::dot(normal, v_render_n);
                }
                
                // Add contribution from normals to v_alpha (product rule term)
                float v_alpha = rgb_render_dot * T - render_accum_dot * ra
                              + T_final * ra * v_alpha_ind_coeff;
                // Forward: render_normals += normal * vis (where vis = alpha * T)
                // So v_alpha_normals = dot(normal * T - normal_accum * ra, v_render_n)
                if (v_render_normals != nullptr) {
                    v_alpha += normal_render_dot * T - normal_accum_dot * ra;
                }

                // Add contribution from hit distance (if enabled)
                vec3 v_grd_n_hit = vec3(0.f);
                vec3 v_gro_hit = vec3(0.f);
                if (use_hit_distance) {
                    // Depth-channel gradient for the hit_distance VJP.
                    // v_rgb_local[CDIM-1] was zeroed above (fwd replaces
                    // that color channel with hit_distance, so the color
                    // VJP must be 0). Compute v_depth directly from
                    // v_render_c instead of reading the zeroed slot.
                    const float v_depth = fac * v_render_c[CDIM - 1];

                    // hit_t / grds / hit_distance were computed in the
                    // alpha-recompute block above and hoisted to outer scope;
                    // reuse them here to avoid recomputation.

                    // Backward through length(grds)
                    vec3 v_grds = vec3(0.f);
                    if (hit_distance > 1e-8f) {
                        v_grds = (grds / hit_distance) * v_depth;
                    }

                    // Backward through grds = scale * grd_n * hit_t (element-wise multiply)
                    // d/d(hit_t): scale * grd_n
                    const float v_hit_t = glm::dot(scale * grd_n, v_grds);
                    // d/d(grd_n): scale * hit_t (element-wise)
                    v_grd_n_hit = (scale * hit_t) * v_grds;  // element-wise
                    // d/d(scale): grd_n * hit_t (element-wise)
                    v_scale_local += (grd_n * hit_t) * v_grds;  // element-wise

                    // Backward through hit_t = dot(grd_n, -gro)
                    v_grd_n_hit += -gro * v_hit_t;
                    v_gro_hit = -grd_n * v_hit_t;
                }

                if (opac * vis <= MAX_ALPHA) {
                    const float v_vis = opac * v_alpha;
                    float v_gradDist = -0.5f * vis * v_vis;
                    vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                    vec3 v_grd_n = -glm::cross(v_gcrod, gro) + v_grd_n_hit;
                    vec3 v_gro = glm::cross(v_gcrod, grd_n) + v_gro_hit;
                    vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                    mat3 v_Mt = glm::outerProduct(v_grd, ray_d) + 
                        glm::outerProduct(v_gro, o_minus_mu);
                    vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    
                    // Compute ray gradients
                    // From o_minus_mu = ray_o - xyz, we get:
                    v_ray_o += v_o_minus_mu;
                    // From grd = Mt * ray_d, we get:
                    v_ray_d += glm::transpose(Mt) * v_grd;

                    quat_scale_to_preci_half_vjp(
                        quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;
                    
                    // Compute normal gradient contribution (if computing normals)
                    // Note: normal was precomputed above for v_alpha contribution
                    if (v_render_normals != nullptr) {
                        // Compute gradient contribution
                        // Forward: render_normals += normal * fac (where fac = alpha * T)
                        const vec3 v_normal_local = v_render_n * fac;
                        
                        // Forward: normal = safe_normalize(unnormalized_flipped)
                        // unnormalized_flipped was computed in the v_alpha
                        // precompute block above and reused here.
                        const vec3 v_unnormalized_flipped = safe_normalize_bw(unnormalized_flipped, v_normal_local);

                        // Forward: unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal
                        const vec3 v_unnormalized = flipped ? -v_unnormalized_flipped : v_unnormalized_flipped;

                        // Forward: R[2][:] = unnormalized_normal
                        const mat3 v_R = mat3(vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 0.f), v_unnormalized);

                        // backward through R = quat_to_rotmat(quat)
                        quat_to_rotmat_vjp(quat, v_R, v_quat_local);
                    }
                }

                render_accum_dot += rgb_render_dot * fac;
                
                // Accumulate normal contribution (for product rule in next iterations).
                if (v_render_normals != nullptr) {
                    normal_accum_dot += normal_render_dot * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t isect_id = id_batch[t]; // flatten index in [B * C * N] or [nnz]
                int32_t isect_bid = isect_id / (C * N);   // intersection batch index
                // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
                int32_t isect_gid = isect_id % N;         // intersection gaussian index
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * isect_id;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_mean_ptr = (float *)(v_means) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                float *v_scale_ptr = (float *)(v_scales) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_scale_ptr, v_scale_local.x);
                gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y);
                gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z);

                float *v_quat_ptr = (float *)(v_quats) + 4 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                gpuAtomicAdd(v_opacities + isect_id, v_opacity_local);
            }
        }
    }

    // v_rays: atomicAdd since multiple chunks contribute per pixel.
    if (v_rays != nullptr && pixel_valid) {
        float *vr = (float *)(v_rays) + 6 * pix_id;
        gpuAtomicAdd(vr + 0, v_ray_o.x);
        gpuAtomicAdd(vr + 1, v_ray_o.y);
        gpuAtomicAdd(vr + 2, v_ray_o.z);
        gpuAtomicAdd(vr + 3, v_ray_d.x);
        gpuAtomicAdd(vr + 4, v_ray_d.y);
        gpuAtomicAdd(vr + 5, v_ray_d.z);
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets,    // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    const bool use_hit_distance,
    // forward outputs
    const at::Tensor render_alphas,   // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., C, image_height, image_width, 1]
    const at::optional<at::Tensor> v_render_normals, // [..., C, image_height, image_width, 3]
    // CSR chunk structure precomputed by the forward pass
    const at::Tensor chunks_per_tile,  // [num_tiles] int32
    const at::Tensor chunk_offsets,    // [num_tiles + 1] int32
    const int64_t total_chunks,        // scalar, equals chunk_offsets[num_tiles]
    // Per-chunk cumulative state persisted by the fwd kernel at every
    // CHUNK_BATCHES boundary. The bwd kernel reads it directly and derives
    // the per-chunk starting accumulators in its preamble — no separate
    // state-scan or prefix-scan pass needed.
    const at::Tensor fwd_chunk_state,  // [total_chunks, pixels_per_tile, 1+CDIM+3] fp32
    // outputs
    at::Tensor v_means,      // [..., N, 3]
    at::Tensor v_quats,      // [..., N, 4]
    at::Tensor v_scales,     // [..., N, 3]
    at::Tensor v_colors,     // [..., C, N, 3] or [nnz, 3]
    at::Tensor v_opacities,  // [..., C, N] or [nnz]
    at::optional<at::Tensor> v_rays // [..., C, image_height, image_width, 6]
) {
    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed opacities are not supported in the 3DGS world-space backward");
    (void)chunks_per_tile;  // only used for the shape check below

    uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    uint32_t B = means.numel() / (N * 3);       // number of batches
    uint32_t C = viewmats0.size(-3);            // number of cameras
    uint32_t I = B * C;                         // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.

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
    } else {
        TORCH_CHECK(camera_model != CameraModelType::LIDAR, "If the sensor isn't lidar, lidar coefficients must not be given");
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(SupportedChannels::contains(channels),
        "Unsupported number of color channels: ", channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h).");

    const uint32_t pixels_per_tile = tile_size * tile_size;
    const uint32_t num_tiles = I * tile_height * tile_width;

    // CSR chunk structure was computed in the forward pass (shared kernel
    // `compute_chunks_per_tile_kernel` lives in `RasterizeChunkCSR.h`) and
    // is threaded through `save_for_backward`. We consume the precomputed
    // tensors directly, avoiding the kernel launch + cumsum + blocking
    // `.item<int32_t>()` readback that used to live here on every bwd.
    TORCH_CHECK(chunks_per_tile.numel() == static_cast<int64_t>(num_tiles),
                "chunks_per_tile has wrong size");
    TORCH_CHECK(chunk_offsets.numel() == static_cast<int64_t>(num_tiles) + 1,
                "chunk_offsets has wrong size");

    auto launch_kernel = [&]<typename ChannelsT>() {
        constexpr uint32_t CDIM = ChannelsT::value;

        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * cdim_smem_stride<CDIM>());

        // Shape parity check for the fwd-persisted state. fwd allocates with
        // shape `[total_chunks, pixels_per_tile, 1 + CDIM + 3]` and fills all
        // boundaries (including early-exit tiles) with terminal state, so the
        // bwd kernel can read every slot. Check each dim — matching numel
        // alone would accept a transposed tensor with the same total element
        // count but wrong stride, silently corrupting bwd's reads.
        const int64_t state_dim =
            /*T*/ 1 + static_cast<int64_t>(CDIM) + /*normal*/ 3;
        TORCH_CHECK(fwd_chunk_state.dim() == 3,
                    "fwd_chunk_state must be 3-D (got ",
                    fwd_chunk_state.dim(), "-D)");
        TORCH_CHECK(fwd_chunk_state.size(0) == total_chunks &&
                        fwd_chunk_state.size(1) == static_cast<int64_t>(pixels_per_tile) &&
                        fwd_chunk_state.size(2) == state_dim,
                    "fwd_chunk_state has wrong shape (got [",
                    fwd_chunk_state.size(0), ", ", fwd_chunk_state.size(1),
                    ", ", fwd_chunk_state.size(2), "], expected [",
                    total_chunks, ", ", pixels_per_tile, ", ", state_dim, "])");

        // Common kernel argument block.
        const auto *means_ptr = reinterpret_cast<const vec3 *>(means.const_data_ptr<float>());
        const auto *quats_ptr = reinterpret_cast<const vec4 *>(quats.const_data_ptr<float>());
        const auto *scales_ptr = reinterpret_cast<const vec3 *>(scales.const_data_ptr<float>());
        const auto *colors_ptr = colors.const_data_ptr<float>();
        const auto *opacities_ptr = opacities.const_data_ptr<float>();
        const auto *backgrounds_ptr = backgrounds.has_value()
            ? backgrounds.value().const_data_ptr<float>() : nullptr;
        const auto *masks_ptr = masks.has_value()
            ? masks.value().const_data_ptr<bool>() : nullptr;
        const auto *viewmats0_ptr = viewmats0.const_data_ptr<float>();
        const auto *viewmats1_ptr = viewmats1.has_value()
            ? viewmats1.value().const_data_ptr<float>() : nullptr;
        const auto *Ks_ptr = Ks.const_data_ptr<float>();
        const auto *rays_ptr = rays.has_value()
            ? rays.value().const_data_ptr<float>() : nullptr;
        const auto *radial_ptr = radial_coeffs.has_value()
            ? radial_coeffs.value().const_data_ptr<float>() : nullptr;
        const auto *tangential_ptr = tangential_coeffs.has_value()
            ? tangential_coeffs.value().const_data_ptr<float>() : nullptr;
        const auto *thin_prism_ptr = thin_prism_coeffs.has_value()
            ? thin_prism_coeffs.value().const_data_ptr<float>() : nullptr;
        const auto *tile_off_gpu = tile_offsets.const_data_ptr<int32_t>();
        const auto *flatten_ptr = flatten_ids.const_data_ptr<int32_t>();
        const auto *render_alphas_ptr = render_alphas.const_data_ptr<float>();
        const auto *last_ids_ptr = last_ids.const_data_ptr<int32_t>();
        const auto *v_render_c_ptr = v_render_colors.const_data_ptr<float>();
        const auto *v_render_a_ptr = v_render_alphas.const_data_ptr<float>();
        const auto *v_render_n_ptr = v_render_normals.has_value()
            ? v_render_normals.value().const_data_ptr<float>() : nullptr;
        auto *v_means_ptr = reinterpret_cast<vec3 *>(v_means.data_ptr<float>());
        auto *v_quats_ptr = reinterpret_cast<vec4 *>(v_quats.data_ptr<float>());
        auto *v_scales_ptr = reinterpret_cast<vec3 *>(v_scales.data_ptr<float>());
        auto *v_colors_ptr = v_colors.data_ptr<float>();
        auto *v_opacities_ptr = v_opacities.data_ptr<float>();
        auto *v_rays_ptr = v_rays.has_value()
            ? v_rays.value().data_ptr<float>() : nullptr;
        const auto *fwd_chunk_state_ptr =
            fwd_chunk_state.const_data_ptr<float>();

        const auto *chunk_offsets_ptr = chunk_offsets.const_data_ptr<int32_t>();

        // ---- Gradient kernel: 1D grid over CSR chunk-slots ----
        // The preamble loads per-chunk boundary and terminal state from
        // `fwd_chunk_state` and materialises the starting accumulators via
        // one CDIM dot + one vec3 dot per thread, at the cost of one extra
        // CSR slot read per chunk (the terminal slot).
        dim3 grad_grid = {static_cast<uint32_t>(total_chunks), 1, 1};
        if (cudaFuncSetAttribute(
                rasterize_gradient_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size) != cudaSuccess) {
            AT_ERROR("Failed to set shmem for gradient kernel (",
                     shmem_size, " bytes).");
        }
        rasterize_gradient_bwd_kernel<CDIM, float>
            <<<grad_grid, threads, shmem_size,
               at::cuda::getCurrentCUDAStream()>>>(
                B, C, N, n_isects,
                means_ptr, quats_ptr, scales_ptr,
                colors_ptr, opacities_ptr, backgrounds_ptr,
                masks_ptr,
                image_width, image_height, tile_size,
                tile_width, tile_height,
                viewmats0_ptr, viewmats1_ptr, Ks_ptr,
                camera_model,
                rs_type,
                rays_ptr,
                radial_ptr, tangential_ptr, thin_prism_ptr,
                ftheta_device_coeffs, lidar_device_coeffs,
                external_distortion_device_params,
                tile_off_gpu, flatten_ptr,
                use_hit_distance,
                render_alphas_ptr, last_ids_ptr,
                v_render_c_ptr, v_render_a_ptr, v_render_n_ptr,
                v_means_ptr, v_quats_ptr, v_scales_ptr,
                v_colors_ptr, v_opacities_ptr, v_rays_ptr,
                fwd_chunk_state_ptr, chunk_offsets_ptr);
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

} // namespace gsplat

#endif
