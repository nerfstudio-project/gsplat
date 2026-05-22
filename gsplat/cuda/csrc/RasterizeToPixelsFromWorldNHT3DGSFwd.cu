/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NHT forward rasterizer
 *
 * Supports FP16 color fetches interpolation/encoding in fp32, accumulation fp32
 */

#include "Config.h"

#if GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "CommonNHT.h"
#include "ExternalDistortion.cuh"
#include "HalfVectorLoads.cuh"
#include "RasterizationNHT.h"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "Utils.cuh"
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

////////////////////////////////////////////////////////////////
// Forward kernel – tetrahedral feature interpolation + harmonic encoding.
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,      // [B, C, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,
    const scalar_t *__restrict__ backgrounds, // [B, C, CDIM]
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
    // FThetaCameraDistortionParameters is a torch::CustomClassHolder (host-only);
    // its device equivalent FThetaCameraDistortionDeviceParams is the plain
    // struct that the FThetaCameraModel<>::Parameters API expects on device.
    const FThetaCameraDistortionDeviceParams ftheta_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const bool center_ray_mode,
    const float *__restrict__ center_ray_dirs, // [I, 3] or nullptr
    const float ray_dir_scale,                 // tcnn mapping: (v*scale+1)/2
    const float *__restrict__ depths_per_gauss,  // [B*C*N] per-Gaussian projection depths, or nullptr
    const bool use_hit_distance,                 // emit ray hit distance instead of projection depth
    float *__restrict__ render_colors,
    float *__restrict__ render_alphas,
    float *__restrict__ render_depth,            // [B*C*H*W] or nullptr (skip depth)
    float *__restrict__ render_normals,          // [B*C*H*W*3] or nullptr (skip normals)
    int32_t *__restrict__ last_ids
) {
    const auto block      = cg::this_thread_block();
    const int32_t iid     = block.group_index().x;
    const int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;

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
        } else {
            inside = false;
        }
    } else {
        i = block.group_index().y * tile_size + block.thread_index().y;
        j = block.group_index().z * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

    constexpr uint32_t OUT_CDIM = CDIM / VERTEX_PER_PRIM;
    constexpr uint32_t FEAT_OUT = OUT_CDIM * ENCF;
    constexpr uint32_t PIXEL_STRIDE = FEAT_OUT + 3;  // features + ray_dirs

    const bool render_d = (render_depth != nullptr);
    const bool render_n = (render_normals != nullptr);

    tile_offsets  += iid * tile_height * tile_width;
    render_colors += iid * image_height * image_width * PIXEL_STRIDE;
    render_alphas += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    if (backgrounds != nullptr) backgrounds += iid * FEAT_OUT;
    if (masks != nullptr)       masks += iid * tile_height * tile_width;
    // depths_per_gauss is indexed directly by isect_id (like opacities/colors).
    if (render_d) render_depth += iid * image_height * image_width;
    if (render_n) render_normals += iid * image_height * image_width * 3;

    int32_t pix_id = i * image_width + j;

    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);
    const vec2 focal_length    = {Ks[iid * 9 + 0], Ks[iid * 9 + 4]};
    const vec2 principal_point = {Ks[iid * 9 + 2], Ks[iid * 9 + 5]};

    WorldRay ray;
    if (inside && camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            cm_params.external_distortion_params = external_distortion_device_params.has_value()
                ? &external_distortion_device_params.value()
                : nullptr;
            ray = PerfectPinholeCameraModel(cm_params).element_to_world_ray_shutter_pose(j, i, rs_params);
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs)     cm_params.radial_coeffs     = make_array<float, 6>(radial_coeffs + iid * 6);
            if (tangential_coeffs) cm_params.tangential_coeffs  = make_array<float, 2>(tangential_coeffs + iid * 2);
            if (thin_prism_coeffs) cm_params.thin_prism_coeffs  = make_array<float, 4>(thin_prism_coeffs + iid * 4);
            cm_params.external_distortion_params = external_distortion_device_params.has_value()
                ? &external_distortion_device_params.value()
                : nullptr;
            ray = OpenCVPinholeCameraModel(cm_params).element_to_world_ray_shutter_pose(j, i, rs_params);
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
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.focal_length = { focal_length.x, focal_length.y };
        if (radial_coeffs) cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + iid * 4);
        cm_params.external_distortion_params = external_distortion_device_params.has_value()
            ? &external_distortion_device_params.value()
            : nullptr;
        ray = OpenCVFisheyeCameraModel(cm_params).element_to_world_ray_shutter_pose(j, i, rs_params);
    } else if (inside && camera_model_type == CameraModelType::FTHETA) {
        FThetaCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.dist = ftheta_coeffs;
        cm_params.external_distortion_params = external_distortion_device_params.has_value()
            ? &external_distortion_device_params.value()
            : nullptr;
        ray = FThetaCameraModel(cm_params).element_to_world_ray_shutter_pose(j, i, rs_params);
    } else if (inside && camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_device_coeffs);
        ray = RowOffsetStructuredSpinningLidarModel(*lidar_device_coeffs)
            .element_to_world_ray_shutter_pose(j, i, rs_params);
    } else {
        ray.valid_flag = false;
        if (!inside) {
            // Inactive lidar tile lanes land here; they should not rasterize.
        } else {
            assert(false);
            return;
        }
    }
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;

    bool done = (!inside) || (!ray.valid_flag);


    if (masks != nullptr && inside && !masks[tile_id]) {
        #pragma unroll
        for (uint32_t k = 0; k < FEAT_OUT; ++k) {
            render_colors[pix_id * PIXEL_STRIDE + k] =
                (backgrounds == nullptr) ? 0.0f : static_cast<float>(backgrounds[k]);
        }
        render_colors[pix_id * PIXEL_STRIDE + FEAT_OUT + 0] = 0.0f;
        render_colors[pix_id * PIXEL_STRIDE + FEAT_OUT + 1] = 0.0f;
        render_colors[pix_id * PIXEL_STRIDE + FEAT_OUT + 2] = 0.0f;
        if (render_d) render_depth[pix_id] = 0.0f;
        if (render_n) {
            render_normals[pix_id * 3 + 0] = 0.0f;
            render_normals[pix_id * 3 + 1] = 0.0f;
            render_normals[pix_id * 3 + 2] = 0.0f;
        }
        return;
    }

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (iid == B * C - 1) && (tile_id == tile_width * tile_height - 1)
        ? n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared-memory layout: id, xyz+opacity, iscl_rot, scale, normal. Scale
    // and normal slots are always reserved so the layout is deterministic
    // across builds; the kernel writes to them only when the corresponding
    // aux output is requested. 
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec4 *xyz_opacity_batch = reinterpret_cast<vec4 *>(&id_batch[block_size]);
    mat3 *iscl_rot_batch = reinterpret_cast<mat3 *>(&xyz_opacity_batch[block_size]);
    vec3 *scale_batch = reinterpret_cast<vec3 *>(&iscl_rot_batch[block_size]);
    vec3 *normal_batch = reinterpret_cast<vec3 *>(&scale_batch[block_size]);

    float T = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();

    constexpr uint32_t OUT_DIM = constexpr_max(1U, CDIM * ENCF / 4);
    float pix_out[OUT_DIM] = {0.f};
    // Aux accumulators for fused depth / normals (zero-initialised, only
    // updated when the corresponding output is requested).
    float depth_out = 0.f;
    vec3  normal_out = {0.f, 0.f, 0.f};
#define ACC_ADD(idx, val, vis) do { acc_add_float(pix_out[idx], val, vis); } while(0)

    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;

        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t isect_id = flatten_ids[idx];
            int32_t isect_bid = isect_id / (C * N);
            int32_t isect_gid = isect_id % N;
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};

            const vec4 quat = quats[isect_bid * N + isect_gid];
            const vec3 scale = scales[isect_bid * N + isect_gid];
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
                1.0f / scale[0], 0.f, 0.f,
                0.f, 1.0f / scale[1], 0.f,
                0.f, 0.f, 1.0f / scale[2]
            );
            iscl_rot_batch[tr] = S * glm::transpose(R);
            // Stash scale + canonical normal (R[2], the third column of R,
            // i.e. R * (0,0,1) = the +Z axis transformed to world frame).
            // Loads are unconditional so all lanes in the warp execute the
            // same memory ops; the kernel only USES the values when the
            // corresponding output is requested.
            scale_batch[tr] = scale;
            normal_batch[tr] = R[2];
        }
        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec4 xyz_opac = xyz_opacity_batch[t];
            const float opac = xyz_opac[3];
            const vec3 xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
            const mat3 iscl_rot = iscl_rot_batch[t];

            const vec3 gro = iscl_rot * (ray_o - xyz);
            const vec3 grd_raw = iscl_rot * ray_d;
            const vec3 grd = safe_normalize(grd_raw);
            const float t_closest = -glm::dot(gro, grd);
            const vec3 sample_pos_v = gro + t_closest * grd;
            const float3 sample_pos = make_float3(sample_pos_v.x, sample_pos_v.y, sample_pos_v.z);
            const float grayDist = glm::dot(sample_pos_v, sample_pos_v);

            const float power = -0.5f * grayDist;
            const float density = __expf(power);
            float alpha = min(MAX_ALPHA, opac * density);
            if (power > 0.f || alpha < 1.f / 255.f || density <= MAX_KERNEL_DENSITY_CUTOFF) continue;

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { done = true; break; }

            int32_t isect_id = id_batch[t];
            const float vis = alpha * T;

            if (render_d) {
                float depth_i;
                if (use_hit_distance) {
                    // Ray hit distance in world space at the closest point on
                    // the Gaussian's response, mirroring
                    // RasterizeToPixelsFromWorld3DGSFwd. `grd` is already in
                    // the iscl_rot-transformed frame; un-scale to world.
                    const vec3 scale_t = scale_batch[t];
                    const vec3 grds = scale_t * (grd * t_closest);
                    depth_i = glm::length(grds);
                } else {
                    // Use the precomputed per-Gaussian projection depth.
                    // depths_per_gauss must be non-null when render_depth is
                    // requested and use_hit_distance is false.
                    depth_i = depths_per_gauss[isect_id];
                }
                depth_out = fmaf(depth_i, vis, depth_out);
            }
            if (render_n) {
                // Per-Gaussian normal is the +Z axis rotated by the
                // Gaussian's rotation matrix (R[2]). Match the regular
                // eval3d gsplat convention: flip normals toward the camera ray.
                const vec3 unnormalized = normal_batch[t];
                const bool flipped = glm::dot(unnormalized, ray_d) > 0.0f;
                const vec3 oriented = flipped ? -unnormalized : unnormalized;
                const vec3 nrm = safe_normalize(oriented);
                normal_out.x = fmaf(nrm.x, vis, normal_out.x);
                normal_out.y = fmaf(nrm.y, vis, normal_out.y);
                normal_out.z = fmaf(nrm.z, vis, normal_out.z);
            }

            // Tetrahedral feature interpolation
            {
                float w0, w1, w2, w3;
                tetrahedron_barycentric_weights(sample_pos, w0, w1, w2, w3);

                constexpr uint32_t BASE_CDIM = constexpr_max(1U, CDIM / 4);
                const scalar_t *f_base_ptr = colors + isect_id * CDIM;

                const float weights[4] = {w0, w1, w2, w3};

                if constexpr (sizeof(scalar_t) == 2 && BASE_CDIM % 8 == 0) {
                    for (uint32_t k = 0; k < BASE_CDIM; k += 8) {
                        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v) {
                            const float w = weights[v];
                            float tmp[8];
                            load_8_halves_ld128(reinterpret_cast<const __half*>(f_base_ptr + v * BASE_CDIM + k), tmp);
                            #pragma unroll
                            for (uint32_t ii = 0; ii < 8; ++ii) acc[ii] = fmaf(w, tmp[ii], acc[ii]);
                        }
                        #pragma unroll
                        for (uint32_t ii = 0; ii < 8; ++ii) {
                            #pragma unroll
                            for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                                float s, c;
                                harmonic_encoding_fwd(acc[ii], freq, s, c);
                                ACC_ADD(FREQ_IDX(k + ii, 2 * freq),     s, vis);
                                ACC_ADD(FREQ_IDX(k + ii, 2 * freq + 1), c, vis);
                            }
                        }
                    }
                } else if constexpr (sizeof(scalar_t) == 2 && BASE_CDIM % 4 == 0) {
                    for (uint32_t k = 0; k < BASE_CDIM; k += 4) {
                        float acc[4] = {0.f, 0.f, 0.f, 0.f};
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v) {
                            const float w = weights[v];
                            float tmp[4];
                            load_4_halves_ld64(reinterpret_cast<const __half*>(f_base_ptr + v * BASE_CDIM + k), tmp);
                            #pragma unroll
                            for (uint32_t ii = 0; ii < 4; ++ii) acc[ii] = fmaf(w, tmp[ii], acc[ii]);
                        }
                        #pragma unroll
                        for (uint32_t ii = 0; ii < 4; ++ii) {
                            #pragma unroll
                            for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                                float s, c;
                                harmonic_encoding_fwd(acc[ii], freq, s, c);
                                ACC_ADD(FREQ_IDX(k + ii, 2 * freq),     s, vis);
                                ACC_ADD(FREQ_IDX(k + ii, 2 * freq + 1), c, vis);
                            }
                        }
                    }
                } else {
                    for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                        float bv = 0.f;
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v) {
                            bv = fmaf(weights[v], static_cast<float>(f_base_ptr[v * BASE_CDIM + k]), bv);
                        }
                        #pragma unroll
                        for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                            float s, c;
                            harmonic_encoding_fwd(bv, freq, s, c);
                            ACC_ADD(FREQ_IDX(k, 2 * freq),     s, vis);
                            ACC_ADD(FREQ_IDX(k, 2 * freq + 1), c, vis);
                        }
                    }
                }
            }

            cur_idx = batch_start + t;
            T = next_T;
        }
    }
#undef ACC_ADD

    if (inside) {
        render_alphas[pix_id] = 1.0f - T;

        const uint32_t pix_base = pix_id * PIXEL_STRIDE;

        #pragma unroll
        for (uint32_t k = 0; k < OUT_CDIM; ++k) {
            #pragma unroll
            for (uint32_t f = 0; f < ENCF; ++f) {
                const uint32_t ii = k * ENCF + f;
                render_colors[pix_base + ii] = (backgrounds != nullptr)
                    ? fmaf(T, static_cast<float>(backgrounds[ii]), pix_out[ii]) : pix_out[ii];
            }
        }

        {
            float dx, dy, dz;
            if (center_ray_mode && center_ray_dirs != nullptr) {
                dx = center_ray_dirs[iid * 3 + 0];
                dy = center_ray_dirs[iid * 3 + 1];
                dz = center_ray_dirs[iid * 3 + 2];
            } else {
                dx = ray_d.x; dy = ray_d.y; dz = ray_d.z;
            }
            render_colors[pix_base + FEAT_OUT + 0] = fmaf(dx, ray_dir_scale, 1.0f) * 0.5f;
            render_colors[pix_base + FEAT_OUT + 1] = fmaf(dy, ray_dir_scale, 1.0f) * 0.5f;
            render_colors[pix_base + FEAT_OUT + 2] = fmaf(dz, ray_dir_scale, 1.0f) * 0.5f;
        }

        // Write fused depth / normals outputs (skipped when their pointers
        // are null). Both are linear alpha-blends of per-Gaussian quantities,
        // so depth_out / normal_out already carry the final accumulation.
        if (render_d) {
            render_depth[pix_id] = depth_out;
        }
        if (render_n) {
            render_normals[pix_id * 3 + 0] = normal_out.x;
            render_normals[pix_id * 3 + 1] = normal_out.y;
            render_normals[pix_id * 3 + 2] = normal_out.z;
        }

        last_ids[pix_id] = static_cast<int32_t>(cur_idx);
    }
}

////////////////////////////////////////////////////////////////
// Launch wrapper
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
void launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel(
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
    bool center_ray_mode,
    const at::Tensor center_ray_dirs,
    float ray_dir_scale,
    const at::optional<at::Tensor> depths_per_gauss,
    bool use_hit_distance,
    at::Tensor renders,
    at::Tensor alphas,
    const at::optional<at::Tensor> render_depth,
    const at::optional<at::Tensor> render_normals,
    at::Tensor last_ids
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

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    // Shared-memory layout matches the in-kernel pointer layout:
    // id_batch (int32) | xyz_opacity_batch (vec4) | iscl_rot_batch (mat3)
    //                                       | scale_batch (vec3) | normal_batch (vec3)
    // scale_batch / normal_batch are always reserved so the launch param
    // matches the kernel regardless of whether the aux outputs are
    // requested. See the in-kernel comment for details.
    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3)
         + sizeof(vec3) + sizeof(vec3));

    if (cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel<CDIM, scalar_t>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size) != cudaSuccess) {
        AT_ERROR("Failed to set shmem size (requested ", shmem_size, " bytes).");
    }

    const scalar_t *bg_ptr = nullptr;
    const bool     *mask_ptr = nullptr;
    const float    *vm1_ptr = nullptr;
    const float    *rc_ptr = nullptr;
    const float    *tc_ptr = nullptr;
    const float    *tp_ptr = nullptr;
    const float    *depths_ptr = nullptr;
    float          *render_depth_ptr = nullptr;
    float          *render_normals_ptr = nullptr;
    if (backgrounds.has_value()) bg_ptr = backgrounds.value().data_ptr<scalar_t>();
    if (masks.has_value()) mask_ptr = masks.value().data_ptr<bool>();
    if (viewmats1.has_value()) vm1_ptr = viewmats1.value().data_ptr<float>();
    if (radial_coeffs.has_value()) rc_ptr = radial_coeffs.value().data_ptr<float>();
    if (tangential_coeffs.has_value()) tc_ptr = tangential_coeffs.value().data_ptr<float>();
    if (thin_prism_coeffs.has_value()) tp_ptr = thin_prism_coeffs.value().data_ptr<float>();
    if (depths_per_gauss.has_value()) depths_ptr = depths_per_gauss.value().data_ptr<float>();
    if (render_depth.has_value()) render_depth_ptr = render_depth.value().data_ptr<float>();
    if (render_normals.has_value()) render_normals_ptr = render_normals.value().data_ptr<float>();
    const float *crd_ptr = center_ray_dirs.data_ptr<float>();

    // Caller invariants: when render_depth is requested the kernel needs
    // either per-Gaussian depths or the use_hit_distance flag set.
    TORCH_CHECK(
        render_depth_ptr == nullptr || depths_ptr != nullptr || use_hit_distance,
        "render_depth output requested but neither depths_per_gauss nor "
        "use_hit_distance was supplied to launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel."
    );

    // Convert host-only FThetaCameraDistortionParameters to its device-friendly
    // POD-like equivalent before the kernel launch (mirrors main's standard
    // RasterizeToPixelsFromWorld3DGSFwd.cu pattern).
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

    rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel<CDIM, scalar_t>
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
            center_ray_mode,
            crd_ptr,
            ray_dir_scale,
            depths_ptr,
            use_hit_distance,
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_depth_ptr,
            render_normals_ptr,
            last_ids.data_ptr<int32_t>());
}

// Explicit instantiation — match the full channel list from the reference.
#define __INS__(CDIM, SCALAR_T)                                                \
    template void launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel<CDIM, SCALAR_T>( \
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
        bool, const at::Tensor, float,                                         \
        const at::optional<at::Tensor>, bool,                                  \
        at::Tensor, at::Tensor,                                                \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        at::Tensor);                                                           \

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
