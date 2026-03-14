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

#if GSPLAT_BUILD_3DGUT

#include <ATen/Dispatch.h>
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
#include "MacroUtils.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const scalar_t *__restrict__ colors,      // [B, C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [B, C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [B, C, CDIM] or [nnz, CDIM]
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
    // uncented transform
    const UnscentedTransformParameters ut_params,    
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
    scalar_t *__restrict__ v_rays      // [B, C, image_height, image_width, 6]
) {
    auto block = cg::this_thread_block();
    uint32_t iid = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;

    uint32_t i, j;
    bool inside;
    if(camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int tile_element_id = block.thread_rank();
        if(tile_element_id < element_count)
        {
            j = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].x; // row_elevation
            i = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].y; // col_azimuth
            assert(0 <= i);
            assert(i < image_height);
            assert(0 <= j);
            assert(j < image_width);
            inside = true;
        }
        else
        {
            inside = false;
        }
    }
    else
    {
        i = block.group_index().y * tile_size + block.thread_index().y;
        j = block.group_index().z * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

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

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16
    );

    WorldRay ray;
    if(rays == nullptr)
    {
        // shift pointers to the current camera. note that glm is colume-major.
        const vec2 focal_length = {Ks[iid * 9 + 0], Ks[iid * 9 + 4]};
        const vec2 principal_point = {Ks[iid * 9 + 2], Ks[iid * 9 + 5]};
        
        // Create ray from pixel
        if (camera_model_type == CameraModelType::PINHOLE) {
            if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
                PerfectPinholeCameraModel::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = { principal_point.x, principal_point.y };
                cm_params.focal_length = { focal_length.x, focal_length.y };
                cm_params.external_distortion_params = external_distortion_device_params.has_value() ? 
                    &external_distortion_device_params.value() : nullptr;
                PerfectPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            } else {
                OpenCVPinholeCameraModel<>::Parameters cm_params = {};
                cm_params.resolution = {image_width, image_height};
                cm_params.shutter_type = rs_type;
                cm_params.principal_point = { principal_point.x, principal_point.y };
                cm_params.focal_length = { focal_length.x, focal_length.y };
                if (radial_coeffs != nullptr) {
                    cm_params.radial_coeffs = make_array<float, 6>(radial_coeffs + iid * 6);
                }
                if (tangential_coeffs != nullptr) {
                    cm_params.tangential_coeffs = make_array<float, 2>(tangential_coeffs + iid * 2);
                }
                if (thin_prism_coeffs != nullptr) {
                    cm_params.thin_prism_coeffs = make_array<float, 4>(thin_prism_coeffs + iid * 4);
                }
                cm_params.external_distortion_params = external_distortion_device_params.has_value() ? 
                    &external_distortion_device_params.value() : nullptr;
                OpenCVPinholeCameraModel camera_model(cm_params);
                ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
            }
        } else if (camera_model_type == CameraModelType::FISHEYE) {
            OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs != nullptr) {
                cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + iid * 4);
            }
            cm_params.external_distortion_params = external_distortion_device_params.has_value() ? 
                &external_distortion_device_params.value() : nullptr;
            OpenCVFisheyeCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else if (camera_model_type == CameraModelType::FTHETA) {
            FThetaCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.dist = ftheta_device_coeffs;
            cm_params.external_distortion_params = external_distortion_device_params.has_value() ?
                &external_distortion_device_params.value() : nullptr;
            FThetaCameraModel camera_model(cm_params);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else if (camera_model_type == CameraModelType::LIDAR) {
            assert(lidar_device_coeffs);
            RowOffsetStructuredSpinningLidarModel camera_model(*lidar_device_coeffs);
            ray = camera_model.image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            // should never reach here
            assert(false);
            return;
        }
    }
    else
    {
        assert(rays != nullptr);
        ray.valid_flag = false;
        if(inside)
        {
            // TODO: use at least 3x64b loads instead of 6x32b
            ray.ray_org = {rays[pix_id*6+0], rays[pix_id*6+1], rays[pix_id*6+2]};
            ray.ray_dir = {rays[pix_id*6+3], rays[pix_id*6+4], rays[pix_id*6+5]};
            ray.valid_flag = true;
        }
    }
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;

    // keep not rasterizing threads around for reading data
    bool done = inside && ray.valid_flag;

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[block_size]); // [block_size]
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&xyz_opacity_batch[block_size]); // [block_size]
    vec4 *quat_batch =
        reinterpret_cast<vec4 *>(&scale_batch[block_size]); // [block_size]
    float *rgbs_batch =
        (float *)&quat_batch[block_size]; // [block_size * CDIM]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // the contribution from gaussians behind the current one (for normals)
    vec3 normal_buffer = {0.f, 0.f, 0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = done ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];
    
    // df/d_normal for this pixel (only if computing normals)
    vec3 v_render_n = vec3(0.f, 0.f, 0.f);
    if (v_render_normals != nullptr) {
        v_render_n.x = v_render_normals[pix_id * 3 + 0];
        v_render_n.y = v_render_normals[pix_id * 3 + 1];
        v_render_n.z = v_render_normals[pix_id * 3 + 2];
    }

    vec3 v_ray_o = {0.f, 0.f, 0.f};
    vec3 v_ray_d = {0.f, 0.f, 0.f};

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
            int32_t isect_gid = isect_id % N;         // intersection gaussian index
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            scale_batch[tr] = scales[isect_bid * N + isect_gid];
            quat_batch[tr] = quats[isect_bid * N + isect_gid];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[isect_id * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = done;
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
                if (power > 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }

                // Recompute hit_distance to match forward pass when use_hit_distance=True
                if (use_hit_distance) {
                    const float hit_t = glm::dot(grd_n, -gro);
                    const vec3 grds = scale * (grd_n * hit_t);
                    const float hit_dist = glm::length(grds);
                    // Replace last channel in rgbs_batch with recomputed hit_distance
                    rgbs_batch[t * CDIM + (CDIM - 1)] = hit_dist;
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
            vec3 normal = {0.f, 0.f, 0.f};  // pre-declare for use in v_alpha and later
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
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
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }
                
                // Add contribution from normals to v_alpha (product rule term)
                // Forward: render_normals += normal * vis (where vis = alpha * T)
                // So v_alpha_normals = dot(normal * T - normal_buffer * ra, v_render_n)
                if (v_render_normals != nullptr) {
                    v_alpha += glm::dot(normal * T - normal_buffer * ra, v_render_n);
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

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
                
                // Update normal buffer (for product rule in next iterations)
                if (v_render_normals != nullptr) {
                    normal_buffer += normal * fac;
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

    if (v_rays != nullptr && inside) {
        float *v_ray_ptr = (float *)(v_rays) + 6 * pix_id;
        v_ray_ptr[0] = v_ray_o.x;
        v_ray_ptr[1] = v_ray_o.y;
        v_ray_ptr[2] = v_ray_o.z;
        v_ray_ptr[3] = v_ray_d.x;
        v_ray_ptr[4] = v_ray_d.y;
        v_ray_ptr[5] = v_ray_d.z;
    }
}

template <uint32_t CDIM>
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
    assert (packed == false); // only support non-packed for now

    uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    uint32_t B = means.numel() / (N * 3);       // number of batches
    uint32_t C = viewmats0.size(-3);            // number of cameras
    uint32_t I = B * C;                         // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * CDIM);

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

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

    rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            // camera model
            viewmats0.data_ptr<float>(),
            viewmats1.has_value() ? viewmats1.value().data_ptr<float>()
                                : nullptr,
            Ks.data_ptr<float>(),
            camera_model,
            // uncented transform
            *ut_params,
            rs_type,
            rays.has_value() ? rays.value().data_ptr<float>()
                           : nullptr,
            radial_coeffs.has_value() ? radial_coeffs.value().data_ptr<float>()
                                    : nullptr,
            tangential_coeffs.has_value()
                ? tangential_coeffs.value().data_ptr<float>()
                : nullptr,
            thin_prism_coeffs.has_value()
                ? thin_prism_coeffs.value().data_ptr<float>()
                : nullptr,
            ftheta_device_coeffs,
            lidar_device_coeffs,
            external_distortion_device_params,
            // intersections
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            use_hit_distance,
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.has_value() ? v_render_normals.value().data_ptr<float>() : nullptr,
            // outputs
            reinterpret_cast<vec3 *>(v_means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(v_quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_scales.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_rays.has_value() ? v_rays.value().data_ptr<float>() : nullptr
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel<CDIM>( \
        const at::Tensor means,                                                \
        const at::Tensor quats,                                                \
        const at::Tensor scales,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor viewmats0,                                            \
        const at::optional<at::Tensor> viewmats1,                              \
        const at::Tensor Ks,                                                   \
        const CameraModelType camera_model,                                    \
        const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,     \
        const ShutterType rs_type,                                             \
        const at::optional<at::Tensor> rays,                                   \
        const at::optional<at::Tensor> radial_coeffs,                          \
        const at::optional<at::Tensor> tangential_coeffs,                      \
        const at::optional<at::Tensor> thin_prism_coeffs,                      \
        const c10::intrusive_ptr<FThetaCameraDistortionParameters>             \
            &ftheta_coeffs,                                                    \
        const at::optional<c10::intrusive_ptr<                                 \
            RowOffsetStructuredSpinningLidarModelParametersExt>>               \
            &lidar_coeffs,                                                     \
        const at::optional<                                                    \
            c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>   \
            &external_distortion_params,                                       \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const bool use_hit_distance,                                           \
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        const at::optional<at::Tensor> v_render_normals,                       \
        at::Tensor v_means,                                                    \
        at::Tensor v_quats,                                                    \
        at::Tensor v_scales,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities,                                                \
        at::optional<at::Tensor> v_rays                                        \
    );

GSPLAT_FOR_EACH(__INS__, GSPLAT_NUM_CHANNELS)
    
#undef __INS__

} // namespace gsplat

#endif
