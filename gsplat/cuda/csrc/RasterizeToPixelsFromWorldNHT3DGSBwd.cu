/*
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
 *
 * NHT backward rasterizer
 * 
 * Supports FP16 color fetches (vectorized half loads into FP32 shmem).
 * All gradient math runs in FP32.
 */

#include "Config.h"

#if GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "CommonNHT.h"
#include "ExternalDistortion.cuh"
#include "HalfVectorLoads.cuh"
#include "RasterizationNHT.h"
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Interpolation.cuh"

namespace gsplat {

template <typename T>
constexpr T constexpr_max(T a, T b) {
    return a > b ? a : b;
}

namespace cg = cooperative_groups;

inline __device__ float nht_shutter_relative_frame_time(
    const float image_x,
    const float image_y,
    const uint32_t image_width,
    const uint32_t image_height,
    const ShutterType rs_type
) {
    switch (rs_type) {
    case ShutterType::ROLLING_TOP_TO_BOTTOM:
        return floorf(image_y) / (image_height - 1);
    case ShutterType::ROLLING_LEFT_TO_RIGHT:
        return floorf(image_x) / (image_width - 1);
    case ShutterType::ROLLING_BOTTOM_TO_TOP:
        return (image_height - ceilf(image_y)) / (image_height - 1);
    case ShutterType::ROLLING_RIGHT_TO_LEFT:
        return (image_width - ceilf(image_x)) / (image_width - 1);
    case ShutterType::GLOBAL:
    default:
        return 0.f;
    }
}

inline __device__ WorldRay nht_ortho_world_ray(
    const int j,
    const int i,
    const uint32_t image_width,
    const uint32_t image_height,
    const vec2 focal_length,
    const vec2 principal_point,
    const ShutterType rs_type,
    const RollingShutterParameters &rs_params
) {
    const float image_x = static_cast<float>(j) + 0.5f;
    const float image_y = static_cast<float>(i) + 0.5f;
    const vec2 uv = (vec2(image_x, image_y) - principal_point) / focal_length;
    const vec3 camera_org = {uv.x, uv.y, 0.f};
    const vec3 camera_dir = {0.f, 0.f, 1.f};
    const ShutterPose pose = interpolate_shutter_pose(
        nht_shutter_relative_frame_time(image_x, image_y, image_width, image_height, rs_type),
        rs_params
    );
    const mat3 R_inv = glm::mat3_cast(glm::inverse(pose.q));
    return {R_inv * (camera_org - pose.t), R_inv * camera_dir, true};
}

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const float *__restrict__ opacities,
    const float *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const float *__restrict__ viewmats0,
    const float *__restrict__ viewmats1,
    const float *__restrict__ Ks,
    const CameraModelType camera_model_type,
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const float *__restrict__ radial_coeffs,
    const float *__restrict__ tangential_coeffs,
    const float *__restrict__ thin_prism_coeffs,
    // See the Fwd kernel comment: the device-friendly variant is what the
    // FThetaCameraModel<>::Parameters API expects when running on device.
    const FThetaCameraDistortionDeviceParams ftheta_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // Fused-aux forward replay inputs (only read when the corresponding
    // gradient pointer is non-null; otherwise harmlessly ignored).
    const float *__restrict__ depths_per_gauss,
    const bool use_hit_distance,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    const float *__restrict__ v_render_depth,    // [B*C*H*W] or nullptr
    const float *__restrict__ v_render_normals,  // [B*C*H*W*3] or nullptr
    vec3 *__restrict__ v_means,
    vec4 *__restrict__ v_quats,
    vec3 *__restrict__ v_scales,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    float *__restrict__ v_depths_per_gauss       // [B*C*N] or nullptr
) {
    constexpr uint32_t OUT_CHANNELS = (CDIM / VERTEX_PER_PRIM) * ENCF;
    constexpr uint32_t BASE_CDIM = constexpr_max(1U, CDIM / 4);
    constexpr uint32_t OUT_DIM = constexpr_max(1U, CDIM * ENCF / 4);

    const auto block = cg::this_thread_block();
    const uint32_t iid = block.group_index().x;
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = 0, j = 0;
    bool inside = false;
    if (camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int tile_element_id = block.thread_rank();
        if (tile_element_id < element_count) {
            j = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].x; // col_azimuth
            i = lidar_device_coeffs->tiles_to_elements_map[element_start + tile_element_id].y; // row_elevation
            assert(0 <= i);
            assert(i < image_height);
            assert(0 <= j);
            assert(j < image_width);
            inside = true;
        }
    } else {
        i = block.group_index().y * tile_size + block.thread_index().y;
        j = block.group_index().z * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

    tile_offsets += iid * tile_height * tile_width;
    render_alphas += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    v_render_colors += iid * image_height * image_width * OUT_CHANNELS;
    v_render_alphas += iid * image_height * image_width;
    const bool bwd_d = (v_render_depth != nullptr);
    const bool bwd_n = (v_render_normals != nullptr);
    if (bwd_d) v_render_depth += iid * image_height * image_width;
    if (bwd_n) v_render_normals += iid * image_height * image_width * 3;
    if (backgrounds != nullptr) backgrounds += iid * OUT_CHANNELS;
    if (masks != nullptr) masks += iid * tile_height * tile_width;

    if (masks != nullptr && !masks[tile_id]) return;

    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);
    const vec2 focal_length = {Ks[iid * 9 + 0], Ks[iid * 9 + 4]};
    const vec2 principal_point = {Ks[iid * 9 + 2], Ks[iid * 9 + 5]};

    WorldRay ray;
    if (inside && camera_model_type == CameraModelType::PINHOLE) {
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
    } else if (inside && camera_model_type == CameraModelType::ORTHO) {
        ray = nht_ortho_world_ray(
            j,
            i,
            image_width,
            image_height,
            focal_length,
            principal_point,
            rs_type,
            rs_params
        );
    } else if (inside && camera_model_type == CameraModelType::FISHEYE) {
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
    } else if (inside && camera_model_type == CameraModelType::FTHETA) {
        if (external_distortion_device_params.has_value()) {
            using CameraModel = FThetaCameraModel<extdist::BivariateWindshieldModel>;
            CameraModel::KernelParameters kernel_params = {
                { {image_width, image_height}, rs_type, *external_distortion_device_params },
                Ks, ftheta_coeffs,
            };
            CameraModel camera_model(kernel_params, iid);
            ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
        } else {
            using CameraModel = FThetaCameraModel<extdist::EmptyExternalDistortionModel>;
            CameraModel::KernelParameters kernel_params = {
                { {image_width, image_height}, rs_type, {} },
                Ks, ftheta_coeffs,
            };
            CameraModel camera_model(kernel_params, iid);
            ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
        }
    } else if (inside && camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        using CameraModel = RowOffsetStructuredSpinningLidarModel;
        CameraModel::KernelParameters kernel_params = *lidar_device_coeffs;
        CameraModel camera_model(kernel_params, iid);
        ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
    } else {
        ray.valid_flag = false;
        if (inside) {
            assert(false);
            return;
        }
    }
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;

    bool done = inside && ray.valid_flag;


    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
        ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    constexpr uint32_t RGBS_STRIDE = CDIM + ((CDIM % 32 == 0) ? 1 : 0);

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec4 *xyz_opacity_batch = reinterpret_cast<vec4 *>(&id_batch[block_size]);
    vec4 *quat_batch = reinterpret_cast<vec4 *>(&xyz_opacity_batch[block_size]);
    vec3 *scale_batch = reinterpret_cast<vec3 *>(&quat_batch[block_size]);
    static_assert(sizeof(mat3) >= sizeof(vec4) + sizeof(vec3), "layout check for quat+scale in mat3 slot");
    float *rgbs_batch = (float *)((char *)(&xyz_opacity_batch[block_size]) + sizeof(mat3) * block_size);

    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[OUT_DIM] = {0.f};
    // Aux buffers for fused depth / normal backward (zero-initialised, only
    // updated when the corresponding gradient stream is active). These mirror
    // the standard kernel's `buffer` / `normal_buffer` used to compute the
    // v_alpha contribution from gaussians "behind" the current one.
    float depth_buffer = 0.f;
    vec3  normal_buffer = {0.f, 0.f, 0.f};
    const int32_t bin_final = done ? last_ids[pix_id] : 0;

    float v_render_c[OUT_DIM];
    #pragma unroll
    for (uint32_t k = 0; k < OUT_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * OUT_DIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];
    const float v_render_d = bwd_d ? v_render_depth[pix_id] : 0.f;
    vec3 v_render_n = {0.f, 0.f, 0.f};
    if (bwd_n) {
        v_render_n.x = v_render_normals[pix_id * 3 + 0];
        v_render_n.y = v_render_normals[pix_id * 3 + 1];
        v_render_n.z = v_render_normals[pix_id * 3 + 2];
    }

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t isect_id = flatten_ids[idx];
            int32_t isect_bid = isect_id / (C * N);
            int32_t isect_gid = isect_id % N;
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            quat_batch[tr]  = quats[isect_bid * N + isect_gid];
            scale_batch[tr] = scales[isect_bid * N + isect_gid];

            if constexpr (sizeof(scalar_t) == 2 && CDIM % 8 == 0) {
                const __half* hp = reinterpret_cast<const __half*>(colors + isect_id * CDIM);
                for (uint32_t k = 0; k < CDIM; k += 8) {
                    float tmp[8];
                    load_8_halves_ld128(hp + k, tmp);
                    #pragma unroll
                    for (uint32_t ii = 0; ii < 8; ++ii)
                        rgbs_batch[tr * RGBS_STRIDE + k + ii] = tmp[ii];
                }
            } else if constexpr (sizeof(scalar_t) == 2 && CDIM % 4 == 0) {
                const __half* hp = reinterpret_cast<const __half*>(colors + isect_id * CDIM);
                for (uint32_t k = 0; k < CDIM; k += 4) {
                    float tmp[4];
                    load_4_halves_ld64(hp + k, tmp);
                    #pragma unroll
                    for (uint32_t ii = 0; ii < 4; ++ii)
                        rgbs_batch[tr * RGBS_STRIDE + k + ii] = tmp[ii];
                }
            } else {
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    rgbs_batch[tr * RGBS_STRIDE + k] = static_cast<float>(colors[isect_id * CDIM + k]);
                }
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size; ++t) {
            bool valid = done;
            if (batch_end - t > bin_final) valid = 0;

            float alpha, opac, vis;
            vec3 xyz, scale;
            vec4 quat;
            mat3 R, S, Mt;
            vec3 o_minus_mu, gro, grd, grd_n;
            float3 sample_pos;

            if (valid) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                opac = xyz_opac[3];
                xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = scale_batch[t];
                quat = quat_batch[t];

                R = quat_to_rotmat(quat);
                S = mat3(
                    1.0f / scale[0], 0.f, 0.f,
                    0.f, 1.0f / scale[1], 0.f,
                    0.f, 0.f, 1.0f / scale[2]
                );
                Mt = glm::transpose(R * S);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;
                grd_n = safe_normalize(grd);
                const float t_closest = -glm::dot(gro, grd_n);
                const vec3 sp = gro + t_closest * grd_n;
                sample_pos = make_float3(sp.x, sp.y, sp.z);
                const float grayDist = glm::dot(sp, sp);

                const float power = -0.5f * grayDist;
                vis = __expf(power);
                alpha = min(MAX_ALPHA, opac * vis);
                if (alpha < ALPHA_THRESHOLD) valid = false;
            }

            if (!warp.any(valid)) continue;

            float v_rgb_local[CDIM] = {0.f};
            vec3 v_mean_local = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;

            const float *f_base_ptr = &rgbs_batch[t * RGBS_STRIDE];
            const float *v0 = f_base_ptr + 0 * BASE_CDIM;
            const float *v1 = f_base_ptr + 1 * BASE_CDIM;
            const float *v2 = f_base_ptr + 2 * BASE_CDIM;
            const float *v3 = f_base_ptr + 3 * BASE_CDIM;
            float f_interp[BASE_CDIM];
            float3 v_sample_pos_local = make_float3(0.f, 0.f, 0.f);
            float *v_v0_local = v_rgb_local + 0 * BASE_CDIM;
            float *v_v1_local = v_rgb_local + 1 * BASE_CDIM;
            float *v_v2_local = v_rgb_local + 2 * BASE_CDIM;
            float *v_v3_local = v_rgb_local + 3 * BASE_CDIM;

            // Recompute aux quantities used by the depth / normal backward.
            // Declared here so that the warp-reduce step at the bottom can
            // pull `fac` / `v_depth_local` regardless of which inner `if`
            // branches were taken; they're zeroed when `valid` is false so
            // the reduction lane stays neutral.
            float fac_for_aux = 0.f;
            float depth_i = 0.f;
            bool flipped = false;
            vec3 normal = {0.f, 0.f, 0.f};
            float hit_t_for_hit = 0.f;
            vec3 grds_for_hit = vec3(0.f);
            float hit_dist_len = 0.f;

            if (valid) {
                const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;
                fac_for_aux = fac;

                // Recompute per-Gaussian normal (R[2] flipped + normalised)
                // and depth (per-Gaussian or hit-distance) for both the
                // v_alpha contribution below and the gradient terms farther
                // down. Mirrors the forward kernel's accumulation.
                if (bwd_n) {
                    const vec3 unnormalized = R[2];
                    flipped = glm::dot(unnormalized, ray_d) > 0.0f;
                    const vec3 oriented = flipped ? -unnormalized : unnormalized;
                    normal = safe_normalize(oriented);
                }
                if (bwd_d) {
                    if (use_hit_distance) {
                        hit_t_for_hit = -glm::dot(gro, grd_n);
                        grds_for_hit = scale * (grd_n * hit_t_for_hit);
                        hit_dist_len = glm::length(grds_for_hit);
                        depth_i = hit_dist_len;
                    } else {
                        depth_i = (depths_per_gauss != nullptr)
                                      ? depths_per_gauss[id_batch[t]]
                                      : 0.f;
                    }
                }

                barycentric_interpolate_cuda_fwd<BASE_CDIM>(sample_pos,
                    (float *)v0, (float *)v1, (float *)v2, (float *)v3,
                    reinterpret_cast<float *>(f_interp));

                // Fused (k, freq) loop: a single __sincosf(bv * freq_val) call
                // per iteration feeds three accumulators
                //   (a) v_f_interp_local : ∂L/∂f_interp via harmonic_encoding_bwd
                //                          (d_sin = c*freq_val, d_cos = -s*freq_val).
                //   (b) v_alpha (feature part) : over-operator chain term using
                //                                the pre-Gaussian buffer[] values.
                //   (c) buffer[] update : adds (s, c) * fac for the next Gaussian.
                // buffer is read for (b) and written for (c) at the same FREQ_IDX pair
                float v_f_interp_local[BASE_CDIM] = {};
                float v_alpha = 0.f;
                #pragma unroll
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    const float bv = f_interp[k];
                    #pragma unroll
                    for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                        const float freq_val = get_encoding_frequency(freq);
                        float s, c;
                        __sincosf(bv * freq_val, &s, &c);
                        const float d_sin =  c * freq_val;
                        const float d_cos = -s * freq_val;

                        const uint32_t idx_s = FREQ_IDX(k, 2 * freq);
                        const uint32_t idx_c = FREQ_IDX(k, 2 * freq + 1);
                        const float vc_s = v_render_c[idx_s];
                        const float vc_c = v_render_c[idx_c];

                        v_f_interp_local[k] = fmaf(fac * d_sin, vc_s, v_f_interp_local[k]);
                        v_f_interp_local[k] = fmaf(fac * d_cos, vc_c, v_f_interp_local[k]);

                        v_alpha = fmaf(fmaf(s, T, -buffer[idx_s] * ra), vc_s, v_alpha);
                        v_alpha = fmaf(fmaf(c, T, -buffer[idx_c] * ra), vc_c, v_alpha);

                        buffer[idx_s] = fmaf(s, fac, buffer[idx_s]);
                        buffer[idx_c] = fmaf(c, fac, buffer[idx_c]);
                    }
                }
                v_alpha = fmaf(T_final * ra, v_render_a, v_alpha);

                float v_v0_l[BASE_CDIM] = {};
                float v_v1_l[BASE_CDIM] = {};
                float v_v2_l[BASE_CDIM] = {};
                float v_v3_l[BASE_CDIM] = {};
                barycentric_interpolate_cuda_bwd<BASE_CDIM>(sample_pos,
                    (float *)v0, (float *)v1, (float *)v2, (float *)v3,
                    reinterpret_cast<float *>(v_f_interp_local),
                    &v_sample_pos_local,
                    (float *)v_v0_l, (float *)v_v1_l, (float *)v_v2_l, (float *)v_v3_l);
                #pragma unroll
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    v_v0_local[k] += v_v0_l[k];
                    v_v1_local[k] += v_v1_l[k];
                    v_v2_local[k] += v_v2_l[k];
                    v_v3_local[k] += v_v3_l[k];
                }

                if (backgrounds != nullptr) {
                    float accum = 0.f;
                    #pragma unroll
                    for (uint32_t ii = 0; ii < OUT_CHANNELS; ++ii) {
                        accum = fmaf(backgrounds[ii], v_render_c[ii], accum);
                    }
                    v_alpha = fmaf(-T_final * ra, accum, v_alpha);
                }

                // Fused-aux v_alpha contributions (product rule for the
                // alpha-blend of depth and normals). Standard kernel does the
                // same in RasterizeToPixelsFromWorld3DGSBwd.cu lines ~469 and
                // ~487; we add them after the feature / background terms so
                // the existing accumulators stay untouched.
                if (bwd_d) {
                    v_alpha = fmaf(depth_i * T - depth_buffer * ra,
                                   v_render_d, v_alpha);
                }
                if (bwd_n) {
                    v_alpha += glm::dot(normal * T - normal_buffer * ra,
                                        v_render_n);
                }

                if (opac * vis <= MAX_ALPHA) {
                    const float v_vis = opac * v_alpha;
                    const float v_gradDist = -0.5f * vis * v_vis;

                    // Combined gradient: grayDist path + interpolation path
                    // v_sample_pos = v_sample_pos_interp + 2*sample_pos*v_gradDist
                    vec3 v_sp = vec3(v_sample_pos_local.x, v_sample_pos_local.y, v_sample_pos_local.z);
                    v_sp += 2.0f * v_gradDist * vec3(sample_pos.x, sample_pos.y, sample_pos.z);

                    // Backprop through sample_pos = gro - dot(gro, grd_n) * grd_n
                    const float t_val = -glm::dot(gro, grd_n);
                    const float v_t = glm::dot(v_sp, grd_n);
                    vec3 v_gro = v_sp - v_t * grd_n;
                    vec3 v_grd_n = t_val * v_sp - v_t * gro;

                    // Hit-distance backward (only when the depth slot was
                    // populated from ray hit distance). Adds into v_grd_n /
                    // v_gro / v_scale_local before the geometry chain rule.
                    if (bwd_d && use_hit_distance && hit_dist_len > 1e-8f) {
                        const vec3 v_grds = (grds_for_hit / hit_dist_len) * v_render_d;
                        const float v_hit_t = glm::dot(scale * grd_n, v_grds);
                        // grds = scale * grd_n * hit_t  (element-wise)
                        const vec3 v_grd_n_hit = (scale * hit_t_for_hit) * v_grds - gro * v_hit_t;
                        const vec3 v_gro_hit   = -grd_n * v_hit_t;
                        v_scale_local += (grd_n * hit_t_for_hit) * v_grds;
                        v_grd_n += v_grd_n_hit;
                        v_gro   += v_gro_hit;
                    }

                    vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                    mat3 v_Mt = glm::outerProduct(v_grd, ray_d) +
                                glm::outerProduct(v_gro, o_minus_mu);
                    vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    quat_scale_to_preci_half_vjp(
                        quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;

                    // Normal backward: gradient on the per-Gaussian rotation
                    // (only the third column of R carries information, since
                    // the canonical normal is +Z). Standard kernel pattern.
                    if (bwd_n) {
                        const vec3 v_normal_local = v_render_n * fac;
                        const vec3 unnormalized = R[2];
                        const vec3 oriented = flipped ? -unnormalized : unnormalized;
                        const vec3 v_oriented =
                            safe_normalize_bw(oriented, v_normal_local);
                        const vec3 v_unnormalized = flipped ? -v_oriented : v_oriented;
                        const mat3 v_R_n = mat3(
                            vec3(0.f, 0.f, 0.f),
                            vec3(0.f, 0.f, 0.f),
                            v_unnormalized);
                        quat_to_rotmat_vjp(quat, v_R_n, v_quat_local);
                    }
                }

                // Update aux buffers (gaussians behind us in the next
                // back-to-front step). The harmonic `buffer[]` was already
                // updated in the fused (k, freq) loop above.
                if (bwd_d) {
                    depth_buffer = fmaf(depth_i, fac, depth_buffer);
                }
                if (bwd_n) {
                    normal_buffer.x = fmaf(normal.x, fac, normal_buffer.x);
                    normal_buffer.y = fmaf(normal.y, fac, normal_buffer.y);
                    normal_buffer.z = fmaf(normal.z, fac, normal_buffer.z);
                }
            }

            // v_depths_per_gauss accumulates the per-Gaussian depth gradient
            // contribution from THIS lane's pixel. Each lane's value is
            // `fac * v_render_d` (set to 0 on invalid lanes via fac_for_aux);
            // warp-reduce + atomic-add per Gaussian. Only meaningful when
            // depths_per_gauss was the source of the depth slot (not the
            // hit-distance branch; that path threads its gradient through
            // means/scales/quats).
            const float v_depth_local =
                (bwd_d && !use_hit_distance && v_depths_per_gauss != nullptr)
                    ? fac_for_aux * v_render_d
                    : 0.f;

            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            const float v_depth_reduced = cg::reduce(
                warp, v_depth_local, cg::plus<float>());
            if (warp.thread_rank() == 0) {
                int32_t isect_id = id_batch[t];
                int32_t isect_bid = isect_id / (C * N);
                int32_t isect_gid = isect_id % N;
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

                if (bwd_d && !use_hit_distance && v_depths_per_gauss != nullptr) {
                    gpuAtomicAdd(v_depths_per_gauss + isect_id, v_depth_reduced);
                }
            }
        }
    }
}

template <uint32_t CDIM, typename scalar_t>
void launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor colors,
    const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor viewmats0,
    const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks,
    CameraModelType camera_model,
    const UnscentedTransformParameters &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const FThetaCameraDistortionParameters &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::optional<at::Tensor> depths_per_gauss,
    bool use_hit_distance,
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor v_render_colors,
    const at::Tensor v_render_alphas,
    const at::optional<at::Tensor> v_render_depth,
    const at::optional<at::Tensor> v_render_normals,
    at::Tensor v_means,
    at::Tensor v_quats,
    at::Tensor v_scales,
    at::Tensor v_colors,
    at::Tensor v_opacities,
    const at::optional<at::Tensor> v_depths_per_gauss
) {
    bool packed = opacities.dim() == 1;
    assert(packed == false);

    uint32_t N = packed ? 0 : means.size(-2);
    uint32_t B = means.numel() / (N * 3);
    uint32_t C = viewmats0.size(-3);
    uint32_t I = B * C;
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    if (n_isects == 0) return;

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    constexpr uint32_t RGBS_STRIDE = CDIM + ((CDIM % 32 == 0) ? 1 : 0);
    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(float) * RGBS_STRIDE);

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<CDIM, scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size) != cudaSuccess) {
        AT_ERROR("Failed to set shmem size (requested ", shmem_size, " bytes).");
    }

    const float *bg_ptr = nullptr;
    const bool  *mask_ptr = nullptr;
    const float *vm1_ptr = nullptr;
    const float *rc_ptr = nullptr;
    const float *tc_ptr = nullptr;
    const float *tp_ptr = nullptr;
    const float *depths_ptr = nullptr;
    const float *v_render_depth_ptr = nullptr;
    const float *v_render_normals_ptr = nullptr;
    float       *v_depths_per_gauss_ptr = nullptr;
    if (backgrounds.has_value()) bg_ptr = backgrounds.value().data_ptr<float>();
    if (masks.has_value()) mask_ptr = masks.value().data_ptr<bool>();
    if (viewmats1.has_value()) vm1_ptr = viewmats1.value().data_ptr<float>();
    if (radial_coeffs.has_value()) rc_ptr = radial_coeffs.value().data_ptr<float>();
    if (tangential_coeffs.has_value()) tc_ptr = tangential_coeffs.value().data_ptr<float>();
    if (thin_prism_coeffs.has_value()) tp_ptr = thin_prism_coeffs.value().data_ptr<float>();
    if (depths_per_gauss.has_value()) depths_ptr = depths_per_gauss.value().data_ptr<float>();
    if (v_render_depth.has_value()) v_render_depth_ptr = v_render_depth.value().data_ptr<float>();
    if (v_render_normals.has_value()) v_render_normals_ptr = v_render_normals.value().data_ptr<float>();
    if (v_depths_per_gauss.has_value()) v_depths_per_gauss_ptr = v_depths_per_gauss.value().data_ptr<float>();

    // Convert host CustomClassHolder to its device-friendly counterpart
    // (matches main's RasterizeToPixelsFromWorld3DGSBwd.cu).
    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(ftheta_coeffs);
    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs = cuda::std::nullopt;
    if (lidar_coeffs.has_value()) {
        assert(camera_model == CameraModelType::LIDAR);
        lidar_device_coeffs = *lidar_coeffs.value();
    }
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params = cuda::std::nullopt;
    if (external_distortion_params.has_value()) {
        external_distortion_device_params =
            extdist::BivariateWindshieldModelDeviceParams(*external_distortion_params.value());
    }

    rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<CDIM, scalar_t>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B, C, N, n_isects, packed,
            reinterpret_cast<vec3 *>(means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(scales.data_ptr<float>()),
            colors.data_ptr<scalar_t>(),
            opacities.data_ptr<float>(),
            bg_ptr, mask_ptr,
            image_width, image_height, tile_size, tile_width, tile_height,
            viewmats0.data_ptr<float>(),
            vm1_ptr,
            Ks.data_ptr<float>(),
            camera_model, ut_params, rs_type,
            rc_ptr, tc_ptr, tp_ptr,
            ftheta_device_coeffs,
            lidar_device_coeffs,
            external_distortion_device_params,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            depths_ptr,
            use_hit_distance,
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_depth_ptr,
            v_render_normals_ptr,
            reinterpret_cast<vec3 *>(v_means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(v_quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_scales.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_depths_per_gauss_ptr);
}

#define __INS__(CDIM, SCALAR_T)                                                \
    template void launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<CDIM, SCALAR_T>( \
        const at::Tensor, const at::Tensor, const at::Tensor,                 \
        const at::Tensor, const at::Tensor,                                    \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        uint32_t, uint32_t, uint32_t,                                          \
        const at::Tensor, const at::optional<at::Tensor>, const at::Tensor,    \
        CameraModelType, const UnscentedTransformParameters &, ShutterType,    \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        const at::optional<at::Tensor>, const FThetaCameraDistortionParameters &, \
        const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &, \
        const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &, \
        const at::Tensor, const at::Tensor,                                    \
        const at::optional<at::Tensor>, bool,                                  \
        const at::Tensor, const at::Tensor,                                    \
        const at::Tensor, const at::Tensor,                                    \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor,            \
        const at::optional<at::Tensor>);

__INS__(4, at::Half) __INS__(8, at::Half) __INS__(12, at::Half)
__INS__(16, at::Half) __INS__(20, at::Half) __INS__(24, at::Half)
__INS__(28, at::Half) __INS__(32, at::Half) __INS__(36, at::Half)
__INS__(40, at::Half) __INS__(44, at::Half) __INS__(48, at::Half)
__INS__(49, at::Half) __INS__(64, at::Half) __INS__(65, at::Half)
__INS__(80, at::Half) __INS__(96, at::Half) __INS__(128, at::Half)
__INS__(129, at::Half) __INS__(256, at::Half) __INS__(257, at::Half)
__INS__(512, at::Half) __INS__(513, at::Half)
#undef __INS__

} // namespace gsplat

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS
