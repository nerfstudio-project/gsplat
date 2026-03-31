/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NHT backward rasterizer
 * 
 * Supports FP16 color fetches (vectorized half loads into FP32 shmem).
 * All gradient math runs in FP32.
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "HalfVectorLoads.cuh"
#include "RasterizationNHT.h"
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Interpolation.cuh"

namespace gsplat {

template <typename T>
constexpr T constexpr_max(T a, T b) {
    return a > b ? a : b;
}

namespace cg = cooperative_groups;

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
    const FThetaCameraDistortionParameters ftheta_coeffs,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const float *__restrict__ render_alphas,
    const int32_t *__restrict__ last_ids,
    const float *__restrict__ v_render_colors,
    const float *__restrict__ v_render_alphas,
    vec3 *__restrict__ v_means,
    vec4 *__restrict__ v_quats,
    vec3 *__restrict__ v_scales,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities
) {
    constexpr uint32_t OUT_CHANNELS = (CDIM / VERTEX_PER_PRIM) * ENCF;
    constexpr uint32_t BASE_CDIM = constexpr_max(1U, CDIM / 4);
    constexpr uint32_t OUT_DIM = constexpr_max(1U, CDIM * ENCF / 4);

    const auto block = cg::this_thread_block();
    const uint32_t iid = block.group_index().x;
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    const uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    const uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += iid * tile_height * tile_width;
    render_alphas += iid * image_height * image_width;
    last_ids += iid * image_height * image_width;
    v_render_colors += iid * image_height * image_width * OUT_CHANNELS;
    v_render_alphas += iid * image_height * image_width;
    if (backgrounds != nullptr) backgrounds += iid * OUT_CHANNELS;
    if (masks != nullptr) masks += iid * tile_height * tile_width;

    if (masks != nullptr && !masks[tile_id]) return;

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);
    const vec2 focal_length = {Ks[iid * 9 + 0], Ks[iid * 9 + 4]};
    const vec2 principal_point = {Ks[iid * 9 + 2], Ks[iid * 9 + 5]};

    WorldRay ray;
    if (camera_model_type == CameraModelType::PINHOLE) {
        if (radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            PerfectPinholeCameraModel::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            ray = PerfectPinholeCameraModel(cm_params).image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        } else {
            OpenCVPinholeCameraModel<>::Parameters cm_params = {};
            cm_params.resolution = {image_width, image_height};
            cm_params.shutter_type = rs_type;
            cm_params.principal_point = { principal_point.x, principal_point.y };
            cm_params.focal_length = { focal_length.x, focal_length.y };
            if (radial_coeffs)     cm_params.radial_coeffs     = make_array<float, 6>(radial_coeffs + iid * 6);
            if (tangential_coeffs) cm_params.tangential_coeffs  = make_array<float, 2>(tangential_coeffs + iid * 2);
            if (thin_prism_coeffs) cm_params.thin_prism_coeffs  = make_array<float, 4>(thin_prism_coeffs + iid * 4);
            ray = OpenCVPinholeCameraModel(cm_params).image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
        }
    } else if (camera_model_type == CameraModelType::FISHEYE) {
        OpenCVFisheyeCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.focal_length = { focal_length.x, focal_length.y };
        if (radial_coeffs) cm_params.radial_coeffs = make_array<float, 4>(radial_coeffs + iid * 4);
        ray = OpenCVFisheyeCameraModel(cm_params).image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
    } else if (camera_model_type == CameraModelType::FTHETA) {
        FThetaCameraModel<>::Parameters cm_params = {};
        cm_params.resolution = {image_width, image_height};
        cm_params.shutter_type = rs_type;
        cm_params.principal_point = { principal_point.x, principal_point.y };
        cm_params.dist = ftheta_coeffs;
        ray = FThetaCameraModel(cm_params).image_point_to_world_ray_shutter_pose(vec2(px, py), rs_params);
    } else {
        assert(false);
        return;
    }
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;

    bool done = (i < image_height && j < image_width) && ray.valid_flag;

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
    const int32_t bin_final = done ? last_ids[pix_id] : 0;

    float v_render_c[OUT_DIM];
    #pragma unroll
    for (uint32_t k = 0; k < OUT_DIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * OUT_DIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

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
                alpha = min(0.999f, opac * vis);
                if (power > 0.f || alpha < 1.f / 255.f) valid = false;
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

            if (valid) {
                const float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                barycentric_interpolate_cuda_fwd<BASE_CDIM>(sample_pos,
                    (float *)v0, (float *)v1, (float *)v2, (float *)v3,
                    reinterpret_cast<float *>(f_interp));

                float v_f_interp_local[BASE_CDIM] = {};
                #pragma unroll
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    const float bv = f_interp[k];
                    #pragma unroll
                    for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                        float d_sin, d_cos;
                        harmonic_encoding_bwd(bv, freq, d_sin, d_cos);
                        v_f_interp_local[k] = fmaf(fac * d_sin, v_render_c[FREQ_IDX(k, 2 * freq)], v_f_interp_local[k]);
                        v_f_interp_local[k] = fmaf(fac * d_cos, v_render_c[FREQ_IDX(k, 2 * freq + 1)], v_f_interp_local[k]);
                    }
                }

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

                float v_alpha = 0.f;
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    const float bv = f_interp[k];
                    #pragma unroll
                    for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                        float s, c;
                        harmonic_encoding_fwd(bv, freq, s, c);
                        v_alpha = fmaf(fmaf(s, T, -buffer[FREQ_IDX(k, 2 * freq)] * ra), v_render_c[FREQ_IDX(k, 2 * freq)], v_alpha);
                        v_alpha = fmaf(fmaf(c, T, -buffer[FREQ_IDX(k, 2 * freq + 1)] * ra), v_render_c[FREQ_IDX(k, 2 * freq + 1)], v_alpha);
                    }
                }
                v_alpha = fmaf(T_final * ra, v_render_a, v_alpha);

                if (backgrounds != nullptr) {
                    float accum = 0.f;
                    #pragma unroll
                    for (uint32_t ii = 0; ii < OUT_CHANNELS; ++ii) {
                        accum = fmaf(backgrounds[ii], v_render_c[ii], accum);
                    }
                    v_alpha = fmaf(-T_final * ra, accum, v_alpha);
                }

                if (opac * vis <= 0.999f) {
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

                    vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                    mat3 v_Mt = glm::outerProduct(v_grd, ray_d) +
                                glm::outerProduct(v_gro, o_minus_mu);
                    vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    quat_scale_to_preci_half_vjp(
                        quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;
                }

                #pragma unroll
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    const float bv = f_interp[k];
                    #pragma unroll
                    for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                        float s, c;
                        harmonic_encoding_fwd(bv, freq, s, c);
                        buffer[FREQ_IDX(k, 2 * freq)] = fmaf(s, fac, buffer[FREQ_IDX(k, 2 * freq)]);
                        buffer[FREQ_IDX(k, 2 * freq + 1)] = fmaf(c, fac, buffer[FREQ_IDX(k, 2 * freq + 1)]);
                    }
                }
            }

            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
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
    UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const at::Tensor render_alphas,
    const at::Tensor last_ids,
    const at::Tensor v_render_colors,
    const at::Tensor v_render_alphas,
    at::Tensor v_means,
    at::Tensor v_quats,
    at::Tensor v_scales,
    at::Tensor v_colors,
    at::Tensor v_opacities
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
    if (backgrounds.has_value()) bg_ptr = backgrounds.value().data_ptr<float>();
    if (masks.has_value()) mask_ptr = masks.value().data_ptr<bool>();
    if (viewmats1.has_value()) vm1_ptr = viewmats1.value().data_ptr<float>();
    if (radial_coeffs.has_value()) rc_ptr = radial_coeffs.value().data_ptr<float>();
    if (tangential_coeffs.has_value()) tc_ptr = tangential_coeffs.value().data_ptr<float>();
    if (thin_prism_coeffs.has_value()) tp_ptr = thin_prism_coeffs.value().data_ptr<float>();

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
            ftheta_coeffs,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_alphas.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            reinterpret_cast<vec3 *>(v_means.data_ptr<float>()),
            reinterpret_cast<vec4 *>(v_quats.data_ptr<float>()),
            reinterpret_cast<vec3 *>(v_scales.data_ptr<float>()),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>());
}

#define __INS__(CDIM, SCALAR_T)                                                \
    template void launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<CDIM, SCALAR_T>( \
        const at::Tensor, const at::Tensor, const at::Tensor,                 \
        const at::Tensor, const at::Tensor,                                    \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        uint32_t, uint32_t, uint32_t,                                          \
        const at::Tensor, const at::optional<at::Tensor>, const at::Tensor,    \
        CameraModelType, UnscentedTransformParameters, ShutterType,            \
        const at::optional<at::Tensor>, const at::optional<at::Tensor>,        \
        const at::optional<at::Tensor>, FThetaCameraDistortionParameters,      \
        const at::Tensor, const at::Tensor,                                    \
        const at::Tensor, const at::Tensor,                                    \
        const at::Tensor, const at::Tensor,                                    \
        at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);

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

#endif // GSPLAT_BUILD_3DGS
