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
 * Fused NHT training backward.
 *
 * One kernel that backpropagates from dL/dRGB all the way to the splat
 * parameters AND the MLP weights:
 *
 *   1. Prologue (per pixel, warp-cooperative): recompute the MLP forward from
 *      the per-pixel feature buffer saved by the fused forward, then
 *      backpropagate the loss-scaled RGB gradient through
 *      sigmoid → output layer → hidden layers → input encoding.
 *        - dL/d(accumulated features) stays in registers (v_render_c)
 *        - dL/dW accumulated per block (smem fp16 slab) → fp32 global atomics
 *          in tcnn LINEAR parameter layout (still scaled by loss_scale)
 *   2. Main loop: identical to the NHT training backward — re-walk Gaussians
 *      back-to-front and accumulate gradients for means/quats/scales/
 *      features/opacities, consuming v_render_c directly from registers
 *      (the [H*W, FEAT_OUT] fp32 v_render_colors round trip never exists).
 *
 * Not supported vs the unfused backward (by design, matches the fused fwd):
 * backgrounds (composite in RGB space outside), masks, depth/normal grads,
 * packed mode, ORTHO camera.
 *
 * Requires __CUDA_ARCH__ >= 800.
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
#include "NHTFusedMLPDevice.cuh"

namespace gsplat {

template <typename T>
constexpr T constexpr_max_fb(T a, T b) { return a > b ? a : b; }

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t,
          uint32_t MLP_HIDDEN_T = 64u, uint32_t N_HIDDEN_LAYERS_T = 2u>
__global__ void rasterize_to_pixels_from_world_nht_3dgs_fused_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const float *__restrict__ opacities,
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
    const FThetaCameraDistortionDeviceParams ftheta_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_params,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const bool   center_ray_mode,
    const float *__restrict__ center_ray_dirs,
    const float  ray_dir_scale,
    // MLP (native fragment layout) + loss scaling
    const __half *__restrict__ mlp_params,
    const float   loss_scale,
    // Saved forward state
    const float   *__restrict__ render_feat,     // [I*H*W, FEAT_OUT] fp32
    const float   *__restrict__ render_alphas,   // [I*H*W]
    const int32_t *__restrict__ last_ids,        // [I*H*W]
    // Upstream gradients
    const float *__restrict__ v_render_rgb,      // [I*H*W, 3] fp32
    const float *__restrict__ v_render_alphas,   // [I*H*W]
    // Outputs
    vec3  *__restrict__ v_means,
    vec4  *__restrict__ v_quats,
    vec3  *__restrict__ v_scales,
    float *__restrict__ v_colors,
    float *__restrict__ v_opacities,
    float *__restrict__ v_mlp_params             // [n_params] fp32, linear layout
) {
    static_assert(CDIM >= 4 && CDIM % VERTEX_PER_PRIM == 0,
        "CDIM must be >= 4 and divisible by VERTEX_PER_PRIM");

    constexpr uint32_t OUT_CDIM   = CDIM / VERTEX_PER_PRIM;
    constexpr uint32_t FEAT_OUT   = OUT_CDIM * ENCF;
    constexpr uint32_t ENC_DIM    = nht_mlp::enc_dim_v<FEAT_OUT>;
    constexpr uint32_t MLP_HIDDEN = MLP_HIDDEN_T;
    constexpr uint32_t N_HIDDEN_LAYERS = N_HIDDEN_LAYERS_T;
    constexpr uint32_t BASE_CDIM  = constexpr_max_fb(1U, CDIM / 4);
    constexpr uint32_t OUT_DIM    = constexpr_max_fb(1U, CDIM * ENCF / 4);

    const auto block = cg::this_thread_block();
    const uint32_t iid     = block.group_index().x;
    const uint32_t tile_id = block.group_index().y * tile_width + block.group_index().z;

    uint32_t i = 0, j = 0;
    bool inside = false;
    if (camera_model_type == CameraModelType::LIDAR) {
        assert(lidar_coeffs);
        const int element_start = lidar_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_coeffs->tiles_pack_info[tile_id].y;
        if (block.thread_rank() < (uint32_t)element_count) {
            j = lidar_coeffs->tiles_to_elements_map[element_start + block.thread_rank()].x;
            i = lidar_coeffs->tiles_to_elements_map[element_start + block.thread_rank()].y;
            inside = true;
        }
    } else {
        i = block.group_index().y * tile_size + block.thread_index().y;
        j = block.group_index().z * tile_size + block.thread_index().x;
        inside = (i < image_height && j < image_width);
    }

    tile_offsets    += iid * tile_height * tile_width;
    render_feat     += (size_t)iid * image_height * image_width * FEAT_OUT;
    render_alphas   += iid * image_height * image_width;
    last_ids        += iid * image_height * image_width;
    v_render_rgb    += (size_t)iid * image_height * image_width * 3;
    v_render_alphas += iid * image_height * image_width;

    const int32_t pix_id = min(i * image_width + j, image_width * image_height - 1);

    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16);

    WorldRay ray;
    if (inside) {
        if (camera_model_type == CameraModelType::PINHOLE &&
            radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr) {
            if (external_distortion_params.has_value()) {
                using CameraModel = PerfectPinholeCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_params },
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
        } else if (camera_model_type == CameraModelType::PINHOLE) {
            if (external_distortion_params.has_value()) {
                using CameraModel = OpenCVPinholeCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_params },
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
        } else if (camera_model_type == CameraModelType::FISHEYE) {
            if (external_distortion_params.has_value()) {
                using CameraModel = OpenCVFisheyeCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_params },
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
            if (external_distortion_params.has_value()) {
                using CameraModel = FThetaCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    { {image_width, image_height}, rs_type, *external_distortion_params },
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
        } else if (camera_model_type == CameraModelType::LIDAR) {
            assert(lidar_coeffs);
            using CameraModel = RowOffsetStructuredSpinningLidarModel;
            CameraModel::KernelParameters kernel_params = *lidar_coeffs;
            CameraModel camera_model(kernel_params, iid);
            ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
        } else {
            // Unsupported camera model — degrade gracefully (no early return:
            // every thread must reach the warp-cooperative MLP backward).
            ray.valid_flag = false;
            inside = false;
        }
    } else {
        ray.valid_flag = false;
    }

    // For inactive/invalid pixels (partial edge tiles or rays the camera model
    // rejected) ``ray`` is default-constructed and ``ray_dir``/``ray_org`` are
    // uninitialized garbage. These threads still participate in the
    // warp-cooperative MLP backward below (with zero upstream gradient), so the
    // garbage ray must not reach the SH encoding: a non-finite encoded input
    // yields non-finite hidden activations, and ``0 * inf = NaN`` in the shared
    // dW outer-product would poison the whole tile's MLP weight gradient.
    const bool active = inside && ray.valid_flag;
    const vec3 ray_d = active ? ray.ray_dir : vec3(0.f, 0.f, 0.f);
    const vec3 ray_o = active ? ray.ray_org : vec3(0.f, 0.f, 0.f);

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end   = ((int32_t)iid == (int32_t)(B * C) - 1 &&
                                  tile_id == tile_width * tile_height - 1)
                                    ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size  = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;
    const uint32_t tr          = block.thread_rank();
    const uint32_t warp_id     = tr / nht_mlp::WARP;
    const uint32_t num_warps   = block_size / nht_mlp::WARP;

    // ── Shared memory: the MLP prologue region and the Gaussian-batch region
    //    overlap (union) — the prologue completes before the loop's first
    //    block.sync(), after which the same bytes hold the Gaussian batches.
    //    smem size = max(gauss_bytes, mlp_bytes); see launcher.
    extern __shared__ int s[];
    constexpr uint32_t RGBS_STRIDE = CDIM + ((CDIM % 32 == 0) ? 1 : 0);
    int32_t *id_batch          = (int32_t *)s;
    vec4    *xyz_opacity_batch = reinterpret_cast<vec4 *>(&id_batch[block_size]);
    vec4    *quat_batch        = reinterpret_cast<vec4 *>(&xyz_opacity_batch[block_size]);
    vec3    *scale_batch       = reinterpret_cast<vec3 *>(&quat_batch[block_size]);
    static_assert(sizeof(mat3) >= sizeof(vec4) + sizeof(vec3), "layout check");
    float   *rgbs_batch = (float *)((char *)(&xyz_opacity_batch[block_size]) + sizeof(mat3) * block_size);

    // The dW slab shares the same bytes as the Gaussian batch region (the MLP
    // prologue completes before the first batch is staged).
    __half *dw_slab = (__half *)s;

    // ── Prologue: fused MLP backward → v_render_c (dL/d accumulated features)
    float pix_feat[FEAT_OUT];
    float v_rgb[3] = {0.f, 0.f, 0.f};
    #pragma unroll
    for (uint32_t k = 0; k < FEAT_OUT; ++k)
        pix_feat[k] = active ? render_feat[(size_t)pix_id * FEAT_OUT + k] : 0.f;
    if (active) {
        v_rgb[0] = v_render_rgb[(size_t)pix_id * 3 + 0];
        v_rgb[1] = v_render_rgb[(size_t)pix_id * 3 + 1];
        v_rgb[2] = v_render_rgb[(size_t)pix_id * 3 + 2];
    }
    float rx, ry, rz;
    if (center_ray_mode && center_ray_dirs != nullptr) {
        rx = fmaf(center_ray_dirs[iid * 3 + 0], ray_dir_scale, 1.f) * 0.5f;
        ry = fmaf(center_ray_dirs[iid * 3 + 1], ray_dir_scale, 1.f) * 0.5f;
        rz = fmaf(center_ray_dirs[iid * 3 + 2], ray_dir_scale, 1.f) * 0.5f;
    } else {
        rx = fmaf(ray_d.x, ray_dir_scale, 1.f) * 0.5f;
        ry = fmaf(ray_d.y, ray_dir_scale, 1.f) * 0.5f;
        rz = fmaf(ray_d.z, ray_dir_scale, 1.f) * 0.5f;
    }

    const auto v_feat = nht_mlp::nht_fused_shade_bwd<
        FEAT_OUT, ENC_DIM, MLP_HIDDEN, N_HIDDEN_LAYERS>(
            pix_feat, rx, ry, rz, v_rgb, loss_scale,
            mlp_params, v_mlp_params,
            dw_slab, warp_id, num_warps, tr, block_size);

    float v_render_c[OUT_DIM];
    #pragma unroll
    for (uint32_t k = 0; k < OUT_DIM; ++k) v_render_c[k] = v_feat[k];

    // ── Rasterization backward (identical to the unfused NHT backward) ─────
    const float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    float buffer[OUT_DIM] = {0.f};
    const int32_t bin_final = active ? last_ids[pix_id] : 0;
    const float v_render_a = v_render_alphas[pix_id];

    bool done = active;

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>());

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        const int32_t batch_end  = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            const int32_t isect_id  = flatten_ids[idx];
            const int32_t isect_bid = isect_id / (C * N);
            const int32_t isect_gid = isect_id % N;
            id_batch[tr] = isect_id;
            const vec3 xyz  = means[isect_bid * N + isect_gid];
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
                for (uint32_t k = 0; k < CDIM; ++k)
                    rgbs_batch[tr * RGBS_STRIDE + k] = static_cast<float>(colors[isect_id * CDIM + k]);
            }
        }
        block.sync();

        for (uint32_t t = max(0, batch_end - warp_bin_final); t < (uint32_t)batch_size; ++t) {
            bool valid = done;
            if (batch_end - (int32_t)t > bin_final) valid = false;

            float alpha, opac, vis;
            vec3 xyz, scale;
            vec4 quat;
            mat3 R, S, Mt;
            vec3 o_minus_mu, gro, grd, grd_n;
            float3 sample_pos;

            if (valid) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                opac  = xyz_opac[3];
                xyz   = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = scale_batch[t];
                quat  = quat_batch[t];

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
            vec3 v_mean_local  = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local  = {0.f, 0.f, 0.f, 0.f};
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
                const float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T *= ra;
                const float fac = alpha * T;

                barycentric_interpolate_cuda_fwd<BASE_CDIM>(sample_pos,
                    (float *)v0, (float *)v1, (float *)v2, (float *)v3,
                    reinterpret_cast<float *>(f_interp));

                float v_f_interp_local[BASE_CDIM] = {};
                float v_alpha = 0.f;
                #pragma unroll
                for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                    const float bv = f_interp[k];
                    #pragma unroll
                    for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                        const float freq_val = get_encoding_frequency(freq);
                        float sv, cv;
                        __sincosf(bv * freq_val, &sv, &cv);
                        const float d_sin =  cv * freq_val;
                        const float d_cos = -sv * freq_val;

                        const uint32_t idx_s = FREQ_IDX(k, 2 * freq);
                        const uint32_t idx_c = FREQ_IDX(k, 2 * freq + 1);
                        const float vc_s = v_render_c[idx_s];
                        const float vc_c = v_render_c[idx_c];

                        v_f_interp_local[k] = fmaf(fac * d_sin, vc_s, v_f_interp_local[k]);
                        v_f_interp_local[k] = fmaf(fac * d_cos, vc_c, v_f_interp_local[k]);

                        v_alpha = fmaf(fmaf(sv, T, -buffer[idx_s] * ra), vc_s, v_alpha);
                        v_alpha = fmaf(fmaf(cv, T, -buffer[idx_c] * ra), vc_c, v_alpha);

                        buffer[idx_s] = fmaf(sv, fac, buffer[idx_s]);
                        buffer[idx_c] = fmaf(cv, fac, buffer[idx_c]);
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

                if (opac * vis <= MAX_ALPHA) {
                    const float v_vis = opac * v_alpha;
                    const float v_gradDist = -0.5f * vis * v_vis;

                    vec3 v_sp = vec3(v_sample_pos_local.x, v_sample_pos_local.y, v_sample_pos_local.z);
                    v_sp += 2.0f * v_gradDist * vec3(sample_pos.x, sample_pos.y, sample_pos.z);

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
            }

            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                const int32_t isect_id  = id_batch[t];
                const int32_t isect_bid = isect_id / (C * N);
                const int32_t isect_gid = isect_id % N;
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * isect_id;
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_mean_ptr = (float *)(v_means) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_mean_ptr,     v_mean_local.x);
                gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                float *v_scale_ptr = (float *)(v_scales) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_scale_ptr,     v_scale_local.x);
                gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y);
                gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z);

                float *v_quat_ptr = (float *)(v_quats) + 4 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_quat_ptr,     v_quat_local.x);
                gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                gpuAtomicAdd(v_opacities + isect_id, v_opacity_local);
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// Launch wrapper
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t,
          uint32_t MLP_HIDDEN_T = 64u, uint32_t N_HIDDEN_LAYERS_T = 2u>
void launch_rasterize_to_pixels_from_world_nht_3dgs_fused_bwd_kernel(
    const at::Tensor& means, const at::Tensor& quats, const at::Tensor& scales,
    const at::Tensor& colors, const at::Tensor& opacities,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor& viewmats0, const at::optional<at::Tensor>& viewmats1,
    const at::Tensor& Ks, CameraModelType camera_model,
    const UnscentedTransformParameters& ut_params, ShutterType rs_type,
    const at::optional<at::Tensor>& radial_coeffs,
    const at::optional<at::Tensor>& tangential_coeffs,
    const at::optional<at::Tensor>& thin_prism_coeffs,
    const FThetaCameraDistortionParameters& ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>>& lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>& external_distortion_params,
    const at::Tensor& tile_offsets, const at::Tensor& flatten_ids,
    bool center_ray_mode, const at::Tensor& center_ray_dirs, float ray_dir_scale,
    const at::Tensor& mlp_params, float loss_scale,
    const at::Tensor& render_feat, const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_rgb, const at::Tensor& v_render_alphas,
    at::Tensor& v_means, at::Tensor& v_quats, at::Tensor& v_scales,
    at::Tensor& v_colors, at::Tensor& v_opacities, at::Tensor& v_mlp_params
) {
    constexpr uint32_t RGBS_STRIDE = CDIM + ((CDIM % 32 == 0) ? 1 : 0);

    const uint32_t N    = (uint32_t)means.size(-2);
    const uint32_t B    = (uint32_t)(means.numel() / (N * 3));
    const uint32_t C_n  = (uint32_t)viewmats0.size(-3);
    const uint32_t I    = B * C_n;
    const uint32_t tile_height = (uint32_t)tile_offsets.size(-2);
    const uint32_t tile_width  = (uint32_t)tile_offsets.size(-1);
    const uint32_t n_isects    = (uint32_t)flatten_ids.size(0);
    const uint32_t block_size  = tile_size * tile_size;

    if (n_isects == 0) return;

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid    = {I, tile_height, tile_width};

    const int64_t shmem_gauss = (int64_t)block_size *
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(float) * RGBS_STRIDE);
    // dW slab (largest layer) only — fragment staging is shuffle-based.
    const int64_t shmem_mlp =
        (int64_t)MLP_HIDDEN_T * MLP_HIDDEN_T * sizeof(__half);
    const int64_t shmem_total = shmem_gauss > shmem_mlp ? shmem_gauss : shmem_mlp;

    const auto kernel_fn =
        rasterize_to_pixels_from_world_nht_3dgs_fused_bwd_kernel<
            CDIM, scalar_t, MLP_HIDDEN_T, N_HIDDEN_LAYERS_T>;

    TORCH_CHECK(cudaFuncSetAttribute(kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_total) == cudaSuccess,
        "NHT fused bwd: failed to request ", shmem_total, " bytes of shared memory.");

    const float *vmat1_ptr = viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr;
    const float *rad_ptr   = radial_coeffs.has_value()     ? radial_coeffs.value().data_ptr<float>()     : nullptr;
    const float *tan_ptr   = tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>() : nullptr;
    const float *prism_ptr = thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>() : nullptr;

    FThetaCameraDistortionDeviceParams ftheta_dev(ftheta_coeffs);
    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_dev =
        cuda::std::nullopt;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model == CameraModelType::LIDAR,
            "lidar_coeffs requires camera_model=LIDAR");
        lidar_dev = *lidar_coeffs.value();
    }
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> extdist_dev =
        cuda::std::nullopt;
    if (external_distortion_params.has_value()) {
        extdist_dev = extdist::BivariateWindshieldModelDeviceParams(
            *external_distortion_params.value());
    }

    kernel_fn<<<grid, threads, (size_t)shmem_total, at::cuda::getCurrentCUDAStream()>>>(
        B, C_n, N, n_isects,
        (const vec3*)means.data_ptr<float>(),
        (const vec4*)quats.data_ptr<float>(),
        (const vec3*)scales.data_ptr<float>(),
        (const scalar_t*)colors.data_ptr<at::Half>(),
        opacities.data_ptr<float>(),
        image_width, image_height, tile_size, tile_width, tile_height,
        viewmats0.data_ptr<float>(), vmat1_ptr, Ks.data_ptr<float>(),
        camera_model, ut_params, rs_type,
        rad_ptr, tan_ptr, prism_ptr, ftheta_dev,
        lidar_dev, extdist_dev,
        tile_offsets.data_ptr<int32_t>(),
        flatten_ids.data_ptr<int32_t>(),
        center_ray_mode, center_ray_dirs.data_ptr<float>(), ray_dir_scale,
        reinterpret_cast<const __half*>(mlp_params.data_ptr<at::Half>()),
        loss_scale,
        render_feat.data_ptr<float>(),
        render_alphas.data_ptr<float>(),
        last_ids.data_ptr<int32_t>(),
        v_render_rgb.data_ptr<float>(),
        v_render_alphas.data_ptr<float>(),
        reinterpret_cast<vec3*>(v_means.data_ptr<float>()),
        reinterpret_cast<vec4*>(v_quats.data_ptr<float>()),
        reinterpret_cast<vec3*>(v_scales.data_ptr<float>()),
        v_colors.data_ptr<float>(),
        v_opacities.data_ptr<float>(),
        // empty tensor => skip in-kernel MLP weight gradients
        v_mlp_params.numel() > 0 ? v_mlp_params.data_ptr<float>() : nullptr
    );
}

// ── Per-config thin wrappers (same MSVC C2760 workaround as the fused fwd) ──
#define __NHT_DEF_BWD_WRAPPER__(C, H, L) \
static void nht_fbwd_wrapper_##C##_##H##_##L( \
    const at::Tensor& means, const at::Tensor& quats, const at::Tensor& scales, \
    const at::Tensor& colors, const at::Tensor& opacities, \
    uint32_t image_width, uint32_t image_height, uint32_t tile_size, \
    const at::Tensor& viewmats0, const at::optional<at::Tensor>& viewmats1, \
    const at::Tensor& Ks, CameraModelType camera_model, \
    const UnscentedTransformParameters& ut_params, ShutterType rs_type, \
    const at::optional<at::Tensor>& radial_coeffs, \
    const at::optional<at::Tensor>& tangential_coeffs, \
    const at::optional<at::Tensor>& thin_prism_coeffs, \
    const FThetaCameraDistortionParameters& ftheta_coeffs, \
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>>& lidar_coeffs, \
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>& external_distortion_params, \
    const at::Tensor& tile_offsets, const at::Tensor& flatten_ids, \
    bool center_ray_mode, const at::Tensor& center_ray_dirs, float ray_dir_scale, \
    const at::Tensor& mlp_params, float loss_scale, \
    const at::Tensor& render_feat, const at::Tensor& render_alphas, \
    const at::Tensor& last_ids, \
    const at::Tensor& v_render_rgb, const at::Tensor& v_render_alphas, \
    at::Tensor& v_means, at::Tensor& v_quats, at::Tensor& v_scales, \
    at::Tensor& v_colors, at::Tensor& v_opacities, at::Tensor& v_mlp_params) \
{ \
    launch_rasterize_to_pixels_from_world_nht_3dgs_fused_bwd_kernel \
        <C, at::Half, H, L>( \
            means, quats, scales, colors, opacities, image_width, image_height, tile_size, \
            viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
            lidar_coeffs, external_distortion_params, tile_offsets, flatten_ids, \
            center_ray_mode, center_ray_dirs, ray_dir_scale, mlp_params, loss_scale, \
            render_feat, render_alphas, last_ids, v_render_rgb, v_render_alphas, \
            v_means, v_quats, v_scales, v_colors, v_opacities, v_mlp_params); \
}

// Channel set {4,8,12,16,24,32,48,64,96} × each (hidden, layers) config.
#define __NHT_DEF_BWD_WRAPPER_ALL_C__(H, L) \
    __NHT_DEF_BWD_WRAPPER__(4,  H, L) __NHT_DEF_BWD_WRAPPER__(8,  H, L) __NHT_DEF_BWD_WRAPPER__(12, H, L) \
    __NHT_DEF_BWD_WRAPPER__(16, H, L) __NHT_DEF_BWD_WRAPPER__(24, H, L) __NHT_DEF_BWD_WRAPPER__(32, H, L) \
    __NHT_DEF_BWD_WRAPPER__(48, H, L) __NHT_DEF_BWD_WRAPPER__(64, H, L) __NHT_DEF_BWD_WRAPPER__(96, H, L)
__NHT_DEF_BWD_WRAPPER_ALL_C__(64,  2)
__NHT_DEF_BWD_WRAPPER_ALL_C__(64,  3)
__NHT_DEF_BWD_WRAPPER_ALL_C__(128, 2)
__NHT_DEF_BWD_WRAPPER_ALL_C__(128, 3)
#undef __NHT_DEF_BWD_WRAPPER_ALL_C__
#undef __NHT_DEF_BWD_WRAPPER__

// ── Public dispatch entry point ──────────────────────────────────────────────
void dispatch_rasterize_to_pixels_from_world_nht_3dgs_fused_bwd(
    const at::Tensor& means, const at::Tensor& quats, const at::Tensor& scales,
    const at::Tensor& colors, const at::Tensor& opacities,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor& viewmats0, const at::optional<at::Tensor>& viewmats1,
    const at::Tensor& Ks, CameraModelType camera_model,
    const UnscentedTransformParameters& ut_params, ShutterType rs_type,
    const at::optional<at::Tensor>& radial_coeffs,
    const at::optional<at::Tensor>& tangential_coeffs,
    const at::optional<at::Tensor>& thin_prism_coeffs,
    const FThetaCameraDistortionParameters& ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>>& lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>& external_distortion_params,
    const at::Tensor& tile_offsets, const at::Tensor& flatten_ids,
    bool center_ray_mode, const at::Tensor& center_ray_dirs, float ray_dir_scale,
    const at::Tensor& mlp_params,
    uint32_t mlp_hidden_dim, uint32_t mlp_num_layers,
    float loss_scale,
    const at::Tensor& render_feat, const at::Tensor& render_alphas,
    const at::Tensor& last_ids,
    const at::Tensor& v_render_rgb, const at::Tensor& v_render_alphas,
    at::Tensor& v_means, at::Tensor& v_quats, at::Tensor& v_scales,
    at::Tensor& v_colors, at::Tensor& v_opacities, at::Tensor& v_mlp_params
) {
    const uint32_t channels = (uint32_t)colors.size(-1);

#define __NHT_FB__(C,H,L) nht_fbwd_wrapper_##C##_##H##_##L( \
        means, quats, scales, colors, opacities, image_width, image_height, tile_size, \
        viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
        lidar_coeffs, external_distortion_params, tile_offsets, flatten_ids, \
        center_ray_mode, center_ray_dirs, ray_dir_scale, mlp_params, loss_scale, \
        render_feat, render_alphas, last_ids, v_render_rgb, v_render_alphas, \
        v_means, v_quats, v_scales, v_colors, v_opacities, v_mlp_params)

#define __NHT_FB_SWITCH__(H_VAL, L_VAL) \
    switch (channels) { \
        case 4:  __NHT_FB__(4,  H_VAL, L_VAL); break; \
        case 8:  __NHT_FB__(8,  H_VAL, L_VAL); break; \
        case 12: __NHT_FB__(12, H_VAL, L_VAL); break; \
        case 16: __NHT_FB__(16, H_VAL, L_VAL); break; \
        case 24: __NHT_FB__(24, H_VAL, L_VAL); break; \
        case 32: __NHT_FB__(32, H_VAL, L_VAL); break; \
        case 48: __NHT_FB__(48, H_VAL, L_VAL); break; \
        case 64: __NHT_FB__(64, H_VAL, L_VAL); break; \
        case 96: __NHT_FB__(96, H_VAL, L_VAL); break; \
        default: AT_ERROR("NHT fused bwd: unsupported channels=", channels, \
                          " (supported: 4, 8, 12, 16, 24, 32, 48, 64, 96)"); \
    }

    if      (mlp_hidden_dim ==  64 && mlp_num_layers == 2) { __NHT_FB_SWITCH__( 64, 2) }
    else if (mlp_hidden_dim ==  64 && mlp_num_layers == 3) { __NHT_FB_SWITCH__( 64, 3) }
    else if (mlp_hidden_dim == 128 && mlp_num_layers == 2) { __NHT_FB_SWITCH__(128, 2) }
    else if (mlp_hidden_dim == 128 && mlp_num_layers == 3) { __NHT_FB_SWITCH__(128, 3) }
    else {
        AT_ERROR("NHT fused bwd: unsupported mlp_hidden_dim=", (int)mlp_hidden_dim,
                 " mlp_num_layers=", (int)mlp_num_layers,
                 ". Supported: hidden in {64,128}, n_layers in {2,3}.");
    }
#undef __NHT_FB__
#undef __NHT_FB_SWITCH__
}

} // namespace gsplat

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS
