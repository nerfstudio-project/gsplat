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
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

// Pad CDIM to an odd stride when it would be even, so that the per-thread
// STORE `rgbs_batch[tr * stride + k] = colors[...]` (K1 line ~453 / K2 line ~1030)
// writes to distinct 32-bit banks across a warp.  With a stride S, the 32 lanes
// in a warp land in banks (tr * S) mod 32 for tr in [0,32); that hits all 32
// banks iff gcd(S, 32) == 1, i.e. S is odd.  The inner loop READ from a fixed
// `t` and varying `k` is a broadcast-per-lane and is not the conflict site.
// sizeof(float) == 4 → 32-bit banks; static_assert below pins that assumption.
template <uint32_t CDIM>
constexpr uint32_t cdim_smem_stride() {
    static_assert(sizeof(float) == 4, "bank layout assumes 32-bit banks");
    return (CDIM % 2 == 0) ? CDIM + 1 : CDIM;
}

// ---------------------------------------------------------------------------
// Three-kernel batch-parallel backward pass.
//
// Instead of a single monolithic kernel that processes all Gaussians
// sequentially per pixel, the backward pass is split into three kernels:
//
//   Kernel 1 (state scan): same CTA layout as the forward pass (256 threads,
//     256 pixels). Processes all Gaussians state-only (geometry -> alpha ->
//     state update, NO gradient chain). Every CHUNK_BATCHES batches, writes
//     the per-pixel compositing state (product P, render_accum_dot S_render,
//     normal_accum_dot S_normal) to a temporary global buffer.
//
//   Kernel 1.5 (prologue scan): in-place prefix scan over the per-chunk
//     states, turning each chunk's local state into the correct prologue
//     (starting T, starting accumulators) that the gradient kernel needs.
//     Uses shared-memory buffering (SCAN_TILE_CHUNKS at a time) to avoid
//     global-memory round-trip stalls.
//
//   Kernel 2 (gradient): expanded grid -- one CTA per (tile x chunk). Reads
//     its chunk's starting prologue from the temp buffer. Processes
//     CHUNK_BATCHES batches of Gaussians with the full gradient chain.
//     Chunks are INDEPENDENT -- no sequential dependency between them --
//     exposing more work to the GPU scheduler and reducing tail latency on
//     tiles with many Gaussians.
// ---------------------------------------------------------------------------

// Number of batches per chunk. Each chunk is an independently-schedulable
// unit of work in the gradient kernel. Smaller = more parallelism but more
// temp memory and launch overhead.
constexpr uint32_t CHUNK_BATCHES = 4;

// Number of state floats per pixel per chunk: P, S_render, S_normal.
constexpr uint32_t CHUNK_STATE_DIM = 3;

// Shared ray-generation helper used by K1 and K2. Takes the full camera-model
// switch (pinhole / fisheye / ftheta / lidar × {windshield, empty distortion})
// plus the "explicit rays" path, and returns the WorldRay for pixel (j, i) on
// image `iid`. Marked `__forceinline__` so K2 keeps the register counts it had
// when the block was inlined twice verbatim; the comment at K2's preamble
// previously justified the duplication on occupancy grounds.
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

// Kernel 1: per-chunk state computation. Expanded grid (tiles × chunks).
// Each CTA processes CHUNK_BATCHES batches, state-only (no gradient chain),
// starting from zero prologue (T=1, accum=0, normal_accum=0). Writes the
// chunk's composed (P, S_render, S_normal) per pixel.
//
// chunk_PS layout: [num_tiles][max_chunks][pixels_per_tile][CHUNK_STATE_DIM]
template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_state_scan_bwd_kernel(
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
        *__restrict__ v_render_normals, // [B, C, image_height, image_width, 3] optional
    // chunk state output
    scalar_t *__restrict__ chunk_PS,     // [num_tiles, max_chunks, pixels_per_tile, 3]
    scalar_t *__restrict__ t_final_buf,  // [num_tiles * pixels_per_tile]
    const uint32_t max_chunks_per_tile
)
{
    auto block = cg::this_thread_block();
    const uint32_t iid = block.group_index().x;
    const uint32_t tile_row = block.group_index().y;
    const uint32_t combined_z = block.group_index().z;
    const uint32_t tile_col = combined_z % tile_width;
    const uint32_t chunk_id = combined_z / tile_width;
    const uint32_t tile_id = tile_row * tile_width + tile_col;
    const uint32_t tr = block.thread_rank();
    const uint32_t block_size = block.size();

    // Advance per-image pointers to the current image (iid).
    tile_offsets += iid * tile_height * tile_width;
    render_alphas += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    v_render_colors += iid * image_height * image_width * CDIM;
    if (v_render_normals != nullptr) {
        v_render_normals += iid * image_height * image_width * 3;
    }
    if (masks != nullptr) {
        masks += iid * tile_height * tile_width;
    }
    if(rays != nullptr) {
        rays += iid * image_height * image_width * 6;
    }

    // When the mask is provided, do nothing and return if
    // this tile is labeled as False.
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    // Gaussian range for this tile and chunk/batch decomposition.
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;
    const uint32_t num_chunks =
        (num_batches + CHUNK_BATCHES - 1) / CHUNK_BATCHES;
    const uint32_t tile_linear = iid * tile_height * tile_width + tile_id;
    const uint32_t pixels_per_tile = tile_size * tile_size;

    // Early-out: chunks beyond this tile's range write identity state.
    if (chunk_id >= num_chunks) {
        // Identity: P=1, S_render=0, S_normal=0.
        const uint32_t base = (tile_linear * max_chunks_per_tile + chunk_id) *
                              pixels_per_tile * CHUNK_STATE_DIM +
                              tr * CHUNK_STATE_DIM;
        chunk_PS[base + 0] = 1.0f;
        chunk_PS[base + 1] = 0.0f;
        chunk_PS[base + 2] = 0.0f;
        return;
    }

    // Pixel mapping (camera vs lidar — same as original kernel).
    uint32_t i, j;
    bool inside;
    if(camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int tile_element_id = tr;
        if(tile_element_id < element_count)
        {
            j = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].x; // col_azimuth
            i = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].y; // row_elevation
            assert(0 <= i);
            assert(i < image_height);
            assert(0 <= j);
            assert(j < image_width);
            inside = true;
        }
        else
        {
            i = 0;
            j = 0;
            inside = false;
        }
    }
    else
    {
        i = tile_row * tile_size + block.thread_index().y;
        j = tile_col * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

    // Clamp to last pixel for out-of-bounds threads.
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // Ray generation (camera-model switch + explicit-rays path) is factored
    // into the shared compute_world_ray_bwd helper; see its definition above.
    WorldRay ray = compute_world_ray_bwd<scalar_t>(
        iid, j, i, pix_id, inside,
        rays, viewmats0, viewmats1, Ks,
        image_width, image_height,
        camera_model_type, rs_type,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_device_coeffs, lidar_device_coeffs,
        external_distortion_device_params);
    const vec3 ray_o = ray.ray_org;
    const vec3 ray_d = ray.ray_dir;
    const bool pixel_valid = inside && ray.valid_flag;
    const int32_t bin_final = pixel_valid ? last_ids[pix_id] : 0;

    // Per-pixel gradients needed for the state recurrence.
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = pixel_valid ? v_render_colors[pix_id * CDIM + k] : 0.f;
    }
    vec3 v_render_n = vec3(0.f);
    if (v_render_normals != nullptr && pixel_valid) {
        v_render_n.x = v_render_normals[pix_id * 3 + 0];
        v_render_n.y = v_render_normals[pix_id * 3 + 1];
        v_render_n.z = v_render_normals[pix_id * 3 + 2];
    }

    // Shared memory: [xyz_opacity | scale | quat | rgbs] per thread.
    extern __shared__ int s[];
    vec4 *xyz_opacity_batch = reinterpret_cast<vec4 *>(s);
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&xyz_opacity_batch[block_size]);
    vec4 *quat_batch =
        reinterpret_cast<vec4 *>(&scale_batch[block_size]);
    float *rgbs_batch = (float *)&quat_batch[block_size];

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    // Process this chunk with zero prologue.
    float T = 1.0f;
    float accum_render = 0.0f;
    float accum_normal = 0.0f;
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
            const int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            const int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // const int32_t isect_cid = (isect_id / N) % C; // intersection camera index
            const int32_t isect_gid = isect_id % N;          // intersection gaussian index
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            scale_batch[tr] = scales[isect_bid * N + isect_gid];
            quat_batch[tr] = quats[isect_bid * N + isect_gid];
            // Projection kernel culls degenerate Gaussians (zero quaternion,
            // zero scale) by setting radii = 0, preventing them from entering
            // the intersection list. Mirror K2's asserts here so NaN from
            // division-by-zero doesn't silently poison the scan state.
            assert(glm::dot(quat_batch[tr], quat_batch[tr]) > 0.f);
            assert(scale_batch[tr][0] > 0.f && scale_batch[tr][1] > 0.f && scale_batch[tr][2] > 0.f);
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * cdim_smem_stride<CDIM>() + k] = colors[isect_id * CDIM + k];
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final);
             t < batch_size; ++t) {
            bool valid = pixel_valid;
            if (batch_end - t > bin_final) {
                valid = false;
            }
            float ra = 1.0f;
            float normal_render_dot = 0.f;
            float local_hit_dist = 0.f;
            if (valid) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                const float opac = xyz_opac[3];
                const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                const vec3 scale = scale_batch[t];
                const vec4 quat = quat_batch[t];
                const mat3 R = quat_to_rotmat(quat);
                const mat3 S_mat = mat3(
                    1.0f / scale[0], 0.f, 0.f,
                    0.f, 1.0f / scale[1], 0.f,
                    0.f, 0.f, 1.0f / scale[2]);
                const mat3 Mt = glm::transpose(R * S_mat);
                const vec3 o_minus_mu = ray_o - xyz;
                const vec3 gro = Mt * o_minus_mu;
                const vec3 grd = Mt * ray_d;
                const vec3 grd_n = safe_normalize(grd);
                const vec3 gcrod = glm::cross(grd_n, gro);
                const float grayDist = glm::dot(gcrod, gcrod);
                const float power = -0.5f * grayDist;
                const float vis = __expf(power);
                float alpha = min(MAX_ALPHA, opac * vis);
                if (power > 0.f || alpha < ALPHA_THRESHOLD) {
                    alpha = 0.f;
                }
                ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);

                if (use_hit_distance) {
                    const float hit_t = glm::dot(grd_n, -gro);
                    const vec3 grds = scale * (grd_n * hit_t);
                    local_hit_dist = glm::length(grds);
                }
                if (v_render_normals != nullptr) {
                    const vec3 unnormalized_normal = R[2];
                    const bool flipped = glm::dot(unnormalized_normal, ray_d) > 0.0f;
                    const vec3 unnormalized_flipped =
                        flipped ? -unnormalized_normal : unnormalized_normal;
                    const vec3 normal = safe_normalize(unnormalized_flipped);
                    normal_render_dot = glm::dot(normal, v_render_n);
                }
            }
            T *= ra;
            const float fac = (1.0f - 1.0f / ra) * T;
            float rgb_render_dot = 0.f;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                const float rgb_k = (use_hit_distance && k == CDIM - 1)
                    ? local_hit_dist
                    : rgbs_batch[t * cdim_smem_stride<CDIM>() + k];
                rgb_render_dot += rgb_k * v_render_c[k];
            }
            accum_render += rgb_render_dot * fac;
            if (v_render_normals != nullptr) {
                accum_normal += normal_render_dot * fac;
            }
        }
    }

    // Write (P, S_render, S_normal) for this chunk.
    const uint32_t base = (tile_linear * max_chunks_per_tile + chunk_id) *
                          pixels_per_tile * CHUNK_STATE_DIM +
                          tr * CHUNK_STATE_DIM;
    chunk_PS[base + 0] = T;
    chunk_PS[base + 1] = accum_render;
    chunk_PS[base + 2] = accum_normal;

    // Chunk 0 writes T_final for the scan kernel (works for both camera and lidar
    // because K1 already resolved the correct pix_id via the proper pixel mapping).
    if (chunk_id == 0) {
        t_final_buf[tile_linear * pixels_per_tile + tr] =
            pixel_valid ? 1.0f - render_alphas[pix_id] : 1.0f;
    }
}

// Kernel 1.5: shared-memory-buffered prefix scan.
//
// Loads SCAN_TILE_CHUNKS chunks at a time (coalesced) into shared memory,
// scans from shared memory (no global latency), writes back (coalesced).
// This avoids the stride-768 global memory round-trips that caused 98.8%
// no-eligible-warp stalls in the naive sequential scan.
//
// Grid: {num_tiles, 1, 1}. Block: {pixels_per_tile, 1, 1} = {256, 1, 1}.
// Shared memory: SCAN_TILE_CHUNKS * pixels_per_tile * CHUNK_STATE_DIM floats.
constexpr uint32_t SCAN_TILE_CHUNKS = 32;

template <typename scalar_t>
__global__ void rasterize_prologue_scan_kernel(
    const scalar_t *__restrict__ t_final_buf,
    const uint32_t pixels_per_tile,
    scalar_t *__restrict__ chunk_PS,
    const uint32_t max_chunks_per_tile
)
{
    const uint32_t tile_linear = blockIdx.x;
    const uint32_t pixel = threadIdx.x;
    if (pixel >= pixels_per_tile) {
        return;
    }

    const float T_final =
        t_final_buf[tile_linear * pixels_per_tile + pixel];

    // Global memory base for this tile. Layout: [tile][chunk][pixel][STATE_DIM]
    const uint32_t tile_base =
        tile_linear * max_chunks_per_tile * pixels_per_tile * CHUNK_STATE_DIM;
    const uint32_t pixel_stride = pixels_per_tile * CHUNK_STATE_DIM;

    // Shared memory: [SCAN_TILE_CHUNKS][pixels_per_tile][CHUNK_STATE_DIM]
    extern __shared__ float shmem[];

    float exc_P = 1.0f;
    float exc_Sr = 0.0f;
    float exc_Sn = 0.0f;

    for (uint32_t c_base = 0; c_base < max_chunks_per_tile;
         c_base += SCAN_TILE_CHUNKS) {
        const uint32_t c_end =
            min(c_base + SCAN_TILE_CHUNKS, max_chunks_per_tile);
        const uint32_t tile_count = c_end - c_base;

        // ---- Coalesced load: all threads load their pixel for each chunk ----
        for (uint32_t c = 0; c < tile_count; ++c) {
            const uint32_t g_idx =
                tile_base + (c_base + c) * pixel_stride + pixel * CHUNK_STATE_DIM;
            const uint32_t s_idx =
                c * pixels_per_tile * CHUNK_STATE_DIM + pixel * CHUNK_STATE_DIM;
            shmem[s_idx + 0] = chunk_PS[g_idx + 0];
            shmem[s_idx + 1] = chunk_PS[g_idx + 1];
            shmem[s_idx + 2] = chunk_PS[g_idx + 2];
        }
        __syncthreads();

        // ---- Scan from shared memory (no global latency) ----
        for (uint32_t c = 0; c < tile_count; ++c) {
            const uint32_t s_idx =
                c * pixels_per_tile * CHUNK_STATE_DIM + pixel * CHUNK_STATE_DIM;
            const float Pc = shmem[s_idx + 0];
            const float Src = shmem[s_idx + 1];
            const float Snc = shmem[s_idx + 2];
            const float new_P = Pc * exc_P;
            const float new_Sr = Src * exc_P + exc_Sr;
            const float new_Sn = Snc * exc_P + exc_Sn;
            // Write resolved prologue to shared memory.
            shmem[s_idx + 0] = exc_P * T_final;
            shmem[s_idx + 1] = exc_Sr * T_final;
            shmem[s_idx + 2] = exc_Sn * T_final;
            exc_P = new_P;
            exc_Sr = new_Sr;
            exc_Sn = new_Sn;
        }
        __syncthreads();

        // ---- Coalesced write back ----
        for (uint32_t c = 0; c < tile_count; ++c) {
            const uint32_t g_idx =
                tile_base + (c_base + c) * pixel_stride + pixel * CHUNK_STATE_DIM;
            const uint32_t s_idx =
                c * pixels_per_tile * CHUNK_STATE_DIM + pixel * CHUNK_STATE_DIM;
            chunk_PS[g_idx + 0] = shmem[s_idx + 0];
            chunk_PS[g_idx + 1] = shmem[s_idx + 1];
            chunk_PS[g_idx + 2] = shmem[s_idx + 2];
        }
        __syncthreads();
    }
}

// Kernel 2: gradient pass. Same CTA structure and inner loop as the original
// kernel but with: (1) chunk_id decoded from grid z, (2) initial state read
// from chunk_states, (3) batch range limited to CHUNK_BATCHES, (4) v_rays
// via atomicAdd (multiple chunks per pixel).
//
// Grid: {I, tile_height, tile_width * max_chunks_per_tile}
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
    // chunk state input (from K1.5 prefix scan)
    const scalar_t *__restrict__ chunk_states, // [num_tiles, max_chunks, pixels_per_tile, 3]
    const uint32_t max_chunks_per_tile
)
{
    // ---- Preamble: identical to K1 (chunk decode, pixel map, ray gen) ----
    // NOTE: This duplicates K1's preamble intentionally. Factoring into a
    // __device__ function risks register spills from the additional call frame,
    // which would reduce occupancy on the register-pressure-sensitive K2 path.
    auto block = cg::this_thread_block();
    const uint32_t iid = block.group_index().x;
    const uint32_t tile_row = block.group_index().y;
    const uint32_t combined_z = block.group_index().z;
    const uint32_t tile_col = combined_z % tile_width;
    const uint32_t chunk_id = combined_z / tile_width;
    const uint32_t tile_id = tile_row * tile_width + tile_col;
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
    const uint32_t num_chunks =
        (num_batches + CHUNK_BATCHES - 1) / CHUNK_BATCHES;
    const uint32_t tile_linear = iid * tile_height * tile_width + tile_id;
    const uint32_t pixels_per_tile = tile_size * tile_size;

    if (chunk_id >= num_chunks) {
        return;
    }

    // Pixel mapping (camera vs lidar).
    uint32_t i, j;
    bool inside;
    if(camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int tile_element_id = tr;
        if(tile_element_id < element_count) {
            j = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].x; // col_azimuth
            i = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].y; // row_elevation
            assert(0 <= i);
            assert(i < image_height);
            assert(0 <= j);
            assert(j < image_width);
            inside = true;
        } else {
            i = 0;
            j = 0;
            inside = false;
        }
    } else {
        i = tile_row * tile_size + block.thread_index().y;
        j = tile_col * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // Ray generation — shared with K1 via compute_world_ray_bwd helper above.
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

    // ---- Read chunk prologue from temp memory ----
    const uint32_t cs_base = (tile_linear * max_chunks_per_tile + chunk_id) *
                             pixels_per_tile * CHUNK_STATE_DIM +
                             tr * CHUNK_STATE_DIM;
    float T = chunk_states[cs_base + 0];
    float render_accum_dot = chunk_states[cs_base + 1];
    float normal_accum_dot = chunk_states[cs_base + 2];

    const float T_final = pixel_valid ? 1.0f - render_alphas[pix_id] : 1.0f;

    // ---- Per-pixel gradients (same as original kernel) ----
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

    extern __shared__ int s[];  // same layout as K1, plus id_batch prefix
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
            // Per-pixel hit distance — stored in a register, NOT in shared memory
            // rgbs_batch, because hit_distance depends on ray_o/ray_d (per-pixel)
            // while rgbs_batch is per-Gaussian (shared across all pixels in the tile).
            float local_hit_dist = 0.f;
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
                Mt = glm::transpose(R * S);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;
                grd_n = safe_normalize(grd);
                gcrod = glm::cross(grd_n, gro);
                grayDist = glm::dot(gcrod, gcrod);
                power = -0.5f * grayDist;

                vis = __expf(power);
                alpha = min(MAX_ALPHA, opac * vis);
                if (power > 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }

                if (use_hit_distance) {
                    const float hit_t = glm::dot(grd_n, -gro);
                    const vec3 grds = scale * (grd_n * hit_t);
                    local_hit_dist = glm::length(grds);
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
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                
                // Precompute normal if needed (for both v_alpha contribution and gradient)
                bool flipped = false;
                if (v_render_normals != nullptr) {
                    // Recompute normal from forward pass
                    // normal = R * (0, 0, 1) = R[:, 2] (third column)
                    const vec3 unnormalized_normal = R[2];
                    
                    // Direction resolution: flip if facing away from ray
                    flipped = glm::dot(unnormalized_normal, ray_d) > 0.0f;
                    const vec3 unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;
                    
                    // Normalize
                    normal = safe_normalize(unnormalized_flipped);
                }
                
                // contribution from this pixel
                float rgb_render_dot = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    // For the last channel with use_hit_distance, use the per-pixel
                    // local_hit_dist instead of per-Gaussian rgbs_batch (which is
                    // shared memory and would race across pixels in the tile).
                    const float rgb_k = (use_hit_distance && k == CDIM - 1)
                        ? local_hit_dist
                        : rgbs_batch[t * cdim_smem_stride<CDIM>() + k];
                    rgb_render_dot += rgb_k * v_render_c[k];
                }

                float normal_render_dot = 0.f;
                // contribution from background pixel
                if (v_render_normals != nullptr) {
                    normal_render_dot = glm::dot(normal, v_render_n);
                }
                
                // Add contribution from normals to v_alpha (product rule term)
                float v_alpha = rgb_render_dot * T - render_accum_dot * ra
                              + T_final * ra * v_alpha_ind_coeff;
                // Forward: render_normals += normal * vis (where vis = alpha * T)
                // So v_alpha_normals = dot(normal * T - normal_buffer * ra, v_render_n)
                if (v_render_normals != nullptr) {
                    v_alpha += normal_render_dot * T - normal_accum_dot * ra;
                }

                // Add contribution from hit distance (if enabled)
                vec3 v_grd_n_hit = vec3(0.f);
                vec3 v_gro_hit = vec3(0.f);
                if (use_hit_distance) {
                    const float v_depth = v_rgb_local[CDIM - 1];  // gradient from depth channel (last channel)

                    // From forward:
                    const float hit_t = glm::dot(grd_n, -gro);
                    const vec3 grds = scale * (grd_n * hit_t);
                    const float hit_dist_len = glm::length(grds);

                    // Backward through length(grds)
                    vec3 v_grds = vec3(0.f);
                    if (hit_dist_len > 1e-8f) {
                        v_grds = (grds / hit_dist_len) * v_depth;
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
                        const vec3 unnormalized_normal = R[2];
                        const vec3 unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;
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
                
                // Update normal buffer (for product rule in next iterations)
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
    TORCH_CHECK(SupportedChannels::contains(channels),
        "Unsupported number of channels: ", channels,
        " (check GSPLAT_NUM_CHANNELS)");

    auto launch_kernel = [&]<typename ChannelsT>() {
        constexpr uint32_t CDIM = ChannelsT::value;

        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * cdim_smem_stride<CDIM>());

        const uint32_t pixels_per_tile = tile_size * tile_size;
        const uint32_t num_tiles = I * tile_height * tile_width;

        // Compute max batches per tile to size the temp buffer.
        auto tile_off_cpu = tile_offsets.cpu();
        const int32_t *tile_off_ptr = tile_off_cpu.const_data_ptr<int32_t>();
        uint32_t max_range = 0;
        for (uint32_t t = 0; t < num_tiles; ++t) {
            const int32_t start = tile_off_ptr[t];
            const int32_t end =
                (t + 1 < num_tiles) ? tile_off_ptr[t + 1] : n_isects;
            const uint32_t range = static_cast<uint32_t>(end - start);
            if (range > max_range) {
                max_range = range;
            }
        }
        const uint32_t max_batches =
            (max_range + pixels_per_tile - 1) / pixels_per_tile;
        const uint32_t max_chunks =
            (max_batches + CHUNK_BATCHES - 1) / CHUNK_BATCHES;

        // Allocate temp buffer: [num_tiles][max_chunks][pixels_per_tile][CHUNK_STATE_DIM]
        const int64_t chunk_state_numel =
            static_cast<int64_t>(num_tiles) * max_chunks * pixels_per_tile *
            CHUNK_STATE_DIM;
        auto chunk_states = at::empty(
            {chunk_state_numel}, means.options().dtype(at::kFloat));
        chunk_states.zero_();

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
        auto *chunk_ptr = chunk_states.data_ptr<float>();

        // T_final buffer for the scan kernel (avoids lidar pixel mapping in scan).
        auto t_final = at::empty(
            {static_cast<int64_t>(num_tiles * pixels_per_tile)},
            means.options().dtype(at::kFloat));
        auto *t_final_ptr = t_final.data_ptr<float>();

        // ---- Kernel 1: state scan (expanded grid) ----
        dim3 k1_grid = {I, tile_height, tile_width * max_chunks};
        int64_t k1_shmem =
            pixels_per_tile *
            (sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * cdim_smem_stride<CDIM>());
        if (cudaFuncSetAttribute(
                rasterize_state_scan_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                k1_shmem) != cudaSuccess) {
            AT_ERROR("Failed to set shmem for state-scan kernel (",
                     k1_shmem, " bytes).");
        }
        rasterize_state_scan_bwd_kernel<CDIM, float>
            <<<k1_grid, threads, k1_shmem,
               at::cuda::getCurrentCUDAStream()>>>(
                B, C, N, n_isects,
                means_ptr, quats_ptr, scales_ptr,
                colors_ptr, opacities_ptr,
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
                v_render_c_ptr, v_render_n_ptr,
                chunk_ptr, t_final_ptr, max_chunks);

        // ---- Kernel 1.5: prologue scan (shared-memory buffered) ----
        {
            dim3 scan_grid = {num_tiles, 1, 1};
            dim3 scan_threads = {pixels_per_tile, 1, 1};
            int64_t scan_shmem =
                SCAN_TILE_CHUNKS * pixels_per_tile * CHUNK_STATE_DIM *
                sizeof(float);
            if (cudaFuncSetAttribute(
                    rasterize_prologue_scan_kernel<float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    scan_shmem) != cudaSuccess) {
                AT_ERROR("Failed to set shmem for scan kernel (",
                         scan_shmem, " bytes).");
            }
            rasterize_prologue_scan_kernel<float>
                <<<scan_grid, scan_threads, scan_shmem,
                   at::cuda::getCurrentCUDAStream()>>>(
                    t_final_ptr, pixels_per_tile,
                    chunk_ptr, max_chunks);
        }

        // ---- Kernel 2: gradient (expanded grid) ----
        dim3 k2_grid = {I, tile_height, tile_width * max_chunks};
        if (cudaFuncSetAttribute(
                rasterize_gradient_bwd_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size) != cudaSuccess) {
            AT_ERROR("Failed to set shmem for gradient kernel (",
                     shmem_size, " bytes).");
        }
        rasterize_gradient_bwd_kernel<CDIM, float>
            <<<k2_grid, threads, shmem_size,
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
                chunk_ptr, max_chunks);
    };
    const bool dispatched = dispatch::dispatch(SupportedChannels{channels}, std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}
} // namespace gsplat

#endif
