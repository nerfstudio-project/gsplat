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
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "Rasterization.h"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Utils.cuh"
#include "MacroUtils.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
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
    // uncented transform
    const UnscentedTransformParameters ut_params,    
    const ShutterType rs_type,
    const float *__restrict__ rays,                  // [B, C, H, W, 6]
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
    scalar_t
        *__restrict__ render_colors,      // [B, C, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [B, C, image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [B, C, image_height, image_width, 3] optional (can be nullptr)
    int32_t *__restrict__ last_ids,       // [B, C, image_height, image_width]
    int32_t *__restrict__ sample_counts   // [B, C, image_height, image_width] optional (can be nullptr)
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t iid = block.group_index().x;
    int32_t tile_id =
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

    bool return_normals = render_normals != nullptr;

    tile_offsets += iid * tile_height * tile_width;
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
        masks += iid * tile_height * tile_width;
    }
    if(rays != nullptr) {
        rays += iid * image_height * image_width * 6;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // Create rolling shutter parameter
    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16
    );

    WorldRay ray;

    // TODO: this should be templated on the sensor type or whether we're using rays as input.
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

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool done = (!inside) || (!ray.valid_flag);

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[block_size]); // [block_size]
    mat3 *iscl_rot_batch =
        reinterpret_cast<mat3 *>(&xyz_opacity_batch[block_size]); // [block_size]
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&iscl_rot_batch[block_size]); // [block_size]
        vec3 *normal_batch =
        reinterpret_cast<vec3 *>(&scale_batch[block_size]); // [block_size] (only used if return_normals)
    // Normal is the third column of rotation matrix R (canonical normal (0,0,1) transformed to world)

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    int32_t cur_idx = -1;
    // count of samples accumulated (only tracked if sample_counts != nullptr)
    int32_t n_accumulated = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    vec3 normal_out = {0.f, 0.f, 0.f};  // Accumulated normal (only used if return_normals)
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
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
            int32_t isect_gid = isect_id % N;         // intersection gaussian index
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            
            const vec4 quat = quats[isect_bid * N + isect_gid];
            vec3 scale = scales[isect_bid * N + isect_gid];
            
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
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
            mat3 iscl_rot = S * glm::transpose(R);
            iscl_rot_batch[tr] = iscl_rot;
            scale_batch[tr] = scale;
            
            // Store normal if computing normals
            // Normal = R * (0, 0, 1) = third column of R
            if (return_normals) {
                normal_batch[tr] = R[2];
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac[3];
            const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
            const mat3 iscl_rot = iscl_rot_batch[t];
            const vec3 scale = scale_batch[t];

            const vec3 gro = iscl_rot * (ray_o - xyz);
            const vec3 grd = safe_normalize(iscl_rot * ray_d);
            const vec3 gcrod = glm::cross(grd, gro);
            const float grayDist = glm::dot(gcrod, gcrod);
            const float power = -0.5f * grayDist;
            float max_response = __expf(power);
            float alpha = min(MAX_ALPHA, opac * max_response);
            if (alpha < 1.f / 255.f || max_response <= MAX_KERNEL_DENSITY_CUTOFF) {
                continue;
            }

            // Compute hit distance if needed
            float hit_distance = 0.0f;
            if (use_hit_distance) {
                const float hit_t = glm::dot(grd, -gro);
                const vec3 grds = scale * (grd * hit_t);
                hit_distance = glm::length(grds);
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t isect_id = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + isect_id * CDIM;

            if (use_hit_distance) {
                // Use hit distance for depth channel
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    const float value = (k == CDIM - 1) ? hit_distance : c_ptr[k];
                    pix_out[k] += value * vis;
                }
            } else {
                // Use stored depth from colors
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    pix_out[k] += c_ptr[k] * vis;
                }
            }
            
            // Accumulate normal if computing normals
            if (return_normals) {
                const vec3 unnormalized_normal = normal_batch[t];
                
                // Direction resolution: flip if facing away from ray
                const bool flipped = glm::dot(unnormalized_normal, ray_d) > 0.0f;
                const vec3 unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;
                
                // Normalize (should already be unit length, but ensure stability)
                const vec3 normal = safe_normalize(unnormalized_flipped);

                normal_out += normal * vis;
            }
            
            cur_idx = batch_start + t;
            n_accumulated++;  // Increment sample count

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
        // Write accumulated normals if computing normals
        if (render_normals != nullptr) {
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                render_normals[pix_id * 3 + k] = normal_out[k];
            }
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
        // number of samples accumulated (only write if requested)
        if (sample_counts != nullptr) {
            sample_counts[pix_id] = n_accumulated;
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
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
    const at::optional<at::Tensor> rays,              // [...., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    // external distortion
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals  // [..., C, image_height, image_width, 3]
) {
    // Note: quats need to be normalized before passing in.

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

    // Shared memory: id_batch + xyz_opacity_batch + iscl_rot_batch + scale_batch + normal_batch
    int64_t shmem_size =
        tile_size * tile_size * 
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(vec3) + sizeof(vec3));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
        rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>,
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

    rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM, float>
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
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            normals.has_value() ? normals.value().data_ptr<float>() : nullptr,
            last_ids.data_ptr<int32_t>(),
            sample_counts.has_value() ? sample_counts.value().data_ptr<int32_t>() : nullptr
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel<CDIM>( \
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
        const at::Tensor renders,                                              \
        const at::Tensor alphas,                                               \
        const at::Tensor last_ids,                                             \
        const at::optional<at::Tensor> sample_counts,                          \
        const at::optional<at::Tensor> normals                                 \
    );

GSPLAT_FOR_EACH(__INS__, GSPLAT_NUM_CHANNELS)
#undef __INS__

} // namespace gsplat

#endif
