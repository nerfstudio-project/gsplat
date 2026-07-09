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
 * Fused NHT forward rasterizer (inference AND training).
 *
 * This kernel replaces the feature rasterizer + separate TCNN MLP call with a
 * single fused pass that:
 *   1. Unpacks the per-pixel world ray from the camera model
 *   2. Iterates depth-sorted Gaussians, computes 3D ray–Gaussian intersection,
 *      tetrahedral barycentric weights, harmonic encoding, and alpha blending
 *      (identical to the unfused NHT kernel, no UT re-projection inside)
 *   3. Encodes the accumulated features + mapped ray direction inline
 *   4. Evaluates the N-layer FullyFusedMLP using warp-cooperative WMMA
 *      (weights in tcnn "native memory" fragment layout — convert once via
 *      gsplat.nht.convert_mlp_params_to_fused_native)
 *   5. Applies sigmoid, writes RGB fp16 + alpha fp32 directly
 *
 * Inference mode (render_feat == last_ids == nullptr) omits all backward
 * state. Training mode additionally writes the per-pixel accumulated feature
 * buffer (fp32) and last_ids, which the fused backward
 * (RasterizeToPixelsFromWorldNHTFusedBwd.cu) consumes.
 *
 * Not produced in either mode: depth / normal outputs, in-kernel backgrounds,
 * masks (composite backgrounds in RGB space outside the kernel).
 *
 * Grid: {I, tile_height, tile_width}  (identical to the unfused kernel)
 * Block: {tile_size, tile_size, 1}
 *
 * Requires __CUDA_ARCH__ >= 800 for mma.sync.aligned.m16n8k16.
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
#include "NHTFusedMLPDevice.cuh"

namespace gsplat {

template <typename T>
constexpr T constexpr_max_inf(T a, T b) { return a > b ? a : b; }

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Fused inference forward kernel
////////////////////////////////////////////////////////////////

// __launch_bounds__(256, 2): cap registers so two blocks fit per SM. Without
// it the MLP epilogue pushes allocation to 255 regs → 1 block/SM, and the
// latency-bound Gaussian loop loses to the unfused rasterizer (72 regs) on
// high-depth-complexity scenes. The forced spills land in the once-per-pixel
// epilogue, which is the right trade (same approach as tcnn's fused kernels).
template <uint32_t CDIM, typename scalar_t,
          uint32_t MLP_HIDDEN_T = 64u, uint32_t N_HIDDEN_LAYERS_T = 2u>
__global__ void __launch_bounds__(256, 2)
rasterize_to_pixels_from_world_nht_3dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool     packed,
    const vec3  *__restrict__ means,
    const vec4  *__restrict__ quats,
    const vec3  *__restrict__ scales,
    const scalar_t *__restrict__ colors,   // [B, C, N, CDIM] fp16
    const float    *__restrict__ opacities,
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
    const bool    center_ray_mode,
    const float  *__restrict__ center_ray_dirs,
    const float   ray_dir_scale,
    // MLP weights (TCNN native format, network params at offset 0)
    const __half *__restrict__ mlp_params,
    // Outputs
    __half *__restrict__ render_rgb,     // [B*C*H*W, 3]  fp16
    float  *__restrict__ render_alphas,  // [B*C*H*W]     fp32
    // Optional training-state outputs (null in pure inference)
    float   *__restrict__ render_feat,   // [B*C*H*W, FEAT_OUT] fp32 or null
    int32_t *__restrict__ last_ids       // [B*C*H*W] or null
) {
    static_assert(CDIM >= 4 && CDIM % VERTEX_PER_PRIM == 0,
        "CDIM must be >= 4 and divisible by VERTEX_PER_PRIM");

    constexpr uint32_t OUT_CDIM       = CDIM / VERTEX_PER_PRIM;
    constexpr uint32_t FEAT_OUT       = OUT_CDIM * ENCF;
    constexpr uint32_t ENC_DIM        = nht_mlp::enc_dim_v<FEAT_OUT>;
    constexpr uint32_t MLP_HIDDEN     = MLP_HIDDEN_T;
    constexpr uint32_t N_HIDDEN_LAYERS = N_HIDDEN_LAYERS_T;

    auto block = cg::this_thread_block();
    const int32_t iid     = block.group_index().x;
    const int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;

    // Per-thread pixel coordinates
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

    const int32_t pix_id = (int32_t)(i * image_width + j);

    // Camera geometry for this pixel
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
            ray.valid_flag = false;
        }
    } else {
        ray.valid_flag = false;
    }

    // See RasterizeToPixelsFromWorldNHTFusedBwd.cu for why this guard matters:
    // for inactive pixels ``ray`` is default-constructed and ray_dir/ray_org
    // are uninitialized garbage. The forward MLP eval is row-independent (no
    // cross-pixel reduction) and the output write below is gated on `inside`,
    // so a garbage ray here can't corrupt other pixels' outputs — zeroing it
    // is just hygiene (avoids relying on UB / propagating NaN through SH enc).
    const bool active = inside && ray.valid_flag;
    const vec3 ray_d = active ? ray.ray_dir : vec3(0.f, 0.f, 0.f);
    const vec3 ray_o = active ? ray.ray_org : vec3(0.f, 0.f, 0.f);
    bool done = !active;

    // Offset into per-image data
    tile_offsets  += iid * tile_height * tile_width;
    render_rgb    += (size_t)iid * image_height * image_width * 3;
    render_alphas += iid * image_height * image_width;
    if (render_feat != nullptr)
        render_feat += (size_t)iid * image_height * image_width * FEAT_OUT;
    if (last_ids != nullptr)
        last_ids += iid * image_height * image_width;

    // Tile range
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end   = (iid == (int32_t)(B * C) - 1 &&
                                  tile_id == (int32_t)(tile_width * tile_height) - 1)
                                    ? (int32_t)n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // ── Shared memory layout (Gaussian batching only; the MLP epilogue is
    //    smem-free — fragment staging happens via warp shuffles) ─────────────
    // [0]  id_batch:          block_size × int32
    // [1]  xyz_opacity_batch: block_size × vec4
    // [2]  iscl_rot_batch:    block_size × mat3
    extern __shared__ uint8_t s[];
    int32_t *id_batch          = (int32_t*)s;
    vec4    *xyz_opacity_batch  = (vec4  *)(&id_batch[block_size]);
    mat3    *iscl_rot_batch     = (mat3  *)(&xyz_opacity_batch[block_size]);

    // ── Feature accumulators ─────────────────────────────────────────────────
    constexpr uint32_t OUT_DIM = constexpr_max_inf(1U, CDIM * ENCF / 4);
    float pix_out[OUT_DIM] = {};
    float T = 1.0f;
    uint32_t cur_idx = 0;
    uint32_t tr = block.thread_rank();

    // ── Main Gaussian loop (identical to training kernel) ────────────────────
    for (uint32_t b = 0; b < num_batches; ++b) {
        if (__syncthreads_count(done) >= block_size) break;

        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx         = batch_start + tr;

        if (idx < (uint32_t)range_end) {
            const int32_t isect_id  = flatten_ids[idx];
            const int32_t isect_bid = isect_id / (int32_t)(C * N);
            const int32_t isect_gid = isect_id % (int32_t)N;
            id_batch[tr] = isect_id;

            const vec3 xyz  = means   [isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};

            const vec4 quat  = quats [isect_bid * N + isect_gid];
            const vec3 scale = scales[isect_bid * N + isect_gid];
            mat3 R = quat_to_rotmat(quat);
            mat3 S = mat3(
                1.0f / scale[0], 0.f, 0.f,
                0.f, 1.0f / scale[1], 0.f,
                0.f, 0.f, 1.0f / scale[2]
            );
            iscl_rot_batch[tr] = S * glm::transpose(R);
        }
        block.sync();

        const uint32_t batch_size = min(block_size, (uint32_t)(range_end - batch_start));
        for (uint32_t t = 0; t < batch_size && !done; ++t) {
            const vec4  xyz_opac = xyz_opacity_batch[t];
            const float opac     = xyz_opac[3];
            const vec3  xyz      = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
            const mat3  iscl_rot = iscl_rot_batch[t];

            // 3D Gaussian intersection in Gaussian-local space
            const vec3 gro         = iscl_rot * (ray_o - xyz);
            const vec3 grd_raw     = iscl_rot * ray_d;
            const vec3 grd         = safe_normalize(grd_raw);
            const float t_closest  = -glm::dot(gro, grd);
            const vec3  sample_pos_v = gro + t_closest * grd;
            const float3 sample_pos = make_float3(
                sample_pos_v.x, sample_pos_v.y, sample_pos_v.z);

            const float power   = -0.5f * glm::dot(sample_pos_v, sample_pos_v);
            const float density = __expf(power);
            const float alpha   = fminf(MAX_ALPHA, opac * density);
            if (alpha < ALPHA_THRESHOLD) continue;

            const float next_T = T * (1.0f - alpha);
            if (next_T <= TRANSMITTANCE_THRESHOLD) { done = true; break; }

            const int32_t isect_id = id_batch[t];
            const float   vis      = alpha * T;
            T = next_T;
            cur_idx = batch_start + t;

            // Tetrahedral feature interpolation + harmonic encoding
            {
                float w0, w1, w2, w3;
                tetrahedron_barycentric_weights(sample_pos, w0, w1, w2, w3);

                constexpr uint32_t BASE_CDIM = constexpr_max_inf(1U, CDIM / 4);
                const scalar_t *f_base_ptr = colors + isect_id * CDIM;
                const float weights[4] = {w0, w1, w2, w3};

                if constexpr (sizeof(scalar_t) == 2 && BASE_CDIM % 8 == 0) {
                    for (uint32_t k = 0; k < BASE_CDIM; k += 8) {
                        float acc[8] = {};
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v) {
                            float tmp[8];
                            load_8_halves_ld128(
                                reinterpret_cast<const __half*>(f_base_ptr + v * BASE_CDIM + k), tmp);
                            #pragma unroll
                            for (uint32_t ii = 0; ii < 8; ++ii)
                                acc[ii] = fmaf(weights[v], tmp[ii], acc[ii]);
                        }
                        #pragma unroll
                        for (uint32_t ii = 0; ii < 8; ++ii) {
                            #pragma unroll
                            for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                                float s, c;
                                harmonic_encoding_fwd(acc[ii], (int)freq, s, c);
                                acc_add_float(pix_out[FREQ_IDX(k + ii, 2 * freq)],     s, vis);
                                acc_add_float(pix_out[FREQ_IDX(k + ii, 2 * freq + 1)], c, vis);
                            }
                        }
                    }
                } else if constexpr (sizeof(scalar_t) == 2 && BASE_CDIM % 4 == 0) {
                    for (uint32_t k = 0; k < BASE_CDIM; k += 4) {
                        float acc[4] = {};
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v) {
                            float tmp[4];
                            load_4_halves_ld64(
                                reinterpret_cast<const __half*>(f_base_ptr + v * BASE_CDIM + k), tmp);
                            #pragma unroll
                            for (uint32_t ii = 0; ii < 4; ++ii)
                                acc[ii] = fmaf(weights[v], tmp[ii], acc[ii]);
                        }
                        #pragma unroll
                        for (uint32_t ii = 0; ii < 4; ++ii) {
                            #pragma unroll
                            for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                                float s, c;
                                harmonic_encoding_fwd(acc[ii], (int)freq, s, c);
                                acc_add_float(pix_out[FREQ_IDX(k + ii, 2 * freq)],     s, vis);
                                acc_add_float(pix_out[FREQ_IDX(k + ii, 2 * freq + 1)], c, vis);
                            }
                        }
                    }
                } else {
                    for (uint32_t k = 0; k < BASE_CDIM; ++k) {
                        float acc = 0.f;
                        #pragma unroll 4
                        for (uint32_t v = 0; v < 4; ++v)
                            acc = fmaf(weights[v], static_cast<float>(f_base_ptr[v * BASE_CDIM + k]), acc);
                        #pragma unroll
                        for (uint32_t freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
                            float s, c;
                            harmonic_encoding_fwd(acc, (int)freq, s, c);
                            acc_add_float(pix_out[FREQ_IDX(k, 2 * freq)],     s, vis);
                            acc_add_float(pix_out[FREQ_IDX(k, 2 * freq + 1)], c, vis);
                        }
                    }
                }
            }
        }
        block.sync();
    } // end Gaussian loop

    // ── Fused MLP evaluation + output write ──────────────────────────────────
    // All threads participate in the warp-cooperative MMA, including out-of-tile
    // threads (they provide dummy pix_out values, their RGB is discarded).

    // Map ray direction to tcnn [0,1] range
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

    // Warp-cooperative MLP evaluation (all 32 lanes must participate)
    auto rgb = nht_mlp::nht_fused_shade<FEAT_OUT, ENC_DIM, MLP_HIDDEN, N_HIDDEN_LAYERS>(
        pix_out, rx, ry, rz, mlp_params);

    // Write outputs (only inside pixels)
    if (inside) {
        render_alphas[pix_id] = 1.f - T;
        render_rgb[(size_t)pix_id * 3 + 0] = rgb[0];
        render_rgb[(size_t)pix_id * 3 + 1] = rgb[1];
        render_rgb[(size_t)pix_id * 3 + 2] = rgb[2];
        if (render_feat != nullptr) {
            #pragma unroll
            for (uint32_t k = 0; k < FEAT_OUT; ++k)
                render_feat[(size_t)pix_id * FEAT_OUT + k] = pix_out[k];
        }
        if (last_ids != nullptr)
            last_ids[pix_id] = (int32_t)cur_idx;
    }
}

////////////////////////////////////////////////////////////////
// Launch wrapper
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t,
          uint32_t MLP_HIDDEN_T = 64u, uint32_t N_HIDDEN_LAYERS_T = 2u>
void launch_rasterize_to_pixels_from_world_nht_3dgs_fused_fwd_kernel(
    const at::Tensor& means,
    const at::Tensor& quats,
    const at::Tensor& scales,
    const at::Tensor& colors,
    const at::Tensor& opacities,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor& viewmats0,
    const at::optional<at::Tensor>& viewmats1,
    const at::Tensor& Ks,
    CameraModelType camera_model,
    const UnscentedTransformParameters& ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor>& radial_coeffs,
    const at::optional<at::Tensor>& tangential_coeffs,
    const at::optional<at::Tensor>& thin_prism_coeffs,
    const FThetaCameraDistortionParameters& ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>>& lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>& external_distortion_params,
    const at::Tensor& tile_offsets,
    const at::Tensor& flatten_ids,
    bool center_ray_mode,
    const at::Tensor& center_ray_dirs,
    float ray_dir_scale,
    const at::Tensor& mlp_params,
    at::Tensor& renders_rgb,
    at::Tensor& alphas,
    const at::optional<at::Tensor>& render_feat,   // training-state outputs
    const at::optional<at::Tensor>& last_ids
) {
    const uint32_t N    = (uint32_t)means.size(-2);
    const uint32_t B    = (uint32_t)(means.numel() / (N * 3));
    const uint32_t C_n  = (uint32_t)viewmats0.size(-3);
    const uint32_t I    = B * C_n;
    const uint32_t tile_height  = (uint32_t)tile_offsets.size(-2);
    const uint32_t tile_width   = (uint32_t)tile_offsets.size(-1);
    const uint32_t n_isects     = (uint32_t)flatten_ids.size(0);
    const uint32_t block_size   = tile_size * tile_size;

    TORCH_CHECK(block_size <= 256,
        "NHT fused fwd: tile_size^2 must be <= 256 (kernel __launch_bounds__), got ",
        block_size);

    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid    = {I, tile_height, tile_width};

    // Gaussian batching only — the MLP epilogue uses no shared memory.
    const int64_t shmem_total = (int64_t)block_size *
        (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3));

    const auto kernel_fn =
        rasterize_to_pixels_from_world_nht_3dgs_fused_fwd_kernel<
            CDIM, scalar_t, MLP_HIDDEN_T, N_HIDDEN_LAYERS_T>;

    TORCH_CHECK(cudaFuncSetAttribute(kernel_fn,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shmem_total) == cudaSuccess,
        "NHT inference: failed to request ", shmem_total, " bytes of shared memory.");

    // Extract raw pointers
    const float  *vmat1_ptr = viewmats1.has_value() ? viewmats1.value().data_ptr<float>() : nullptr;
    const float  *rad_ptr   = radial_coeffs.has_value()     ? radial_coeffs.value().data_ptr<float>()     : nullptr;
    const float  *tan_ptr   = tangential_coeffs.has_value() ? tangential_coeffs.value().data_ptr<float>() : nullptr;
    const float  *prism_ptr = thin_prism_coeffs.has_value() ? thin_prism_coeffs.value().data_ptr<float>() : nullptr;
    const float  *crd_ptr   = center_ray_dirs.data_ptr<float>();

    // Convert host-only FTheta params to device-friendly POD equivalent
    // (mirrors the pattern used in RasterizeToPixelsFromWorldNHT3DGSFwd.cu)
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
        B, C_n, N, n_isects, /*packed=*/false,
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
        center_ray_mode, crd_ptr, ray_dir_scale,
        reinterpret_cast<const __half*>(mlp_params.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(renders_rgb.data_ptr<at::Half>()),
        alphas.data_ptr<float>(),
        render_feat.has_value() ? render_feat.value().data_ptr<float>() : nullptr,
        last_ids.has_value() ? last_ids.value().data_ptr<int32_t>() : nullptr
    );
}

// ── Per-config thin wrappers ──────────────────────────────────────────────────
// Each wrapper calls a specific (CDIM, H, L) kernel instance.  They live in the
// .cu compilation unit where all template definitions are in scope, so the
// host compiler never sees < > template-argument syntax outside a template
// body — which avoids the MSVC C2760 "unexpected '<'" issue.
//
// NB: these are NOT __device__/__global__; they run on the host and launch GPU kernels.

#define __NHT_DEF_WRAPPER__(C, H, L) \
static void nht_ffwd_wrapper_##C##_##H##_##L( \
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
    const at::Tensor& mlp_params, at::Tensor& renders_rgb, at::Tensor& alphas, \
    const at::optional<at::Tensor>& render_feat, const at::optional<at::Tensor>& last_ids) \
{ \
    launch_rasterize_to_pixels_from_world_nht_3dgs_fused_fwd_kernel \
        <C, at::Half, H, L>( \
            means, quats, scales, colors, opacities, image_width, image_height, tile_size, \
            viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
            radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
            lidar_coeffs, external_distortion_params, tile_offsets, flatten_ids, \
            center_ray_mode, center_ray_dirs, ray_dir_scale, mlp_params, renders_rgb, alphas, \
            render_feat, last_ids); \
}

// Channel set {4,8,12,16,24,32,48,64,96} × each (hidden, layers) config,
// covering both training and inference-only callers.
#define __NHT_DEF_WRAPPER_ALL_C__(H, L) \
    __NHT_DEF_WRAPPER__(4,  H, L) __NHT_DEF_WRAPPER__(8,  H, L) __NHT_DEF_WRAPPER__(12, H, L) \
    __NHT_DEF_WRAPPER__(16, H, L) __NHT_DEF_WRAPPER__(24, H, L) __NHT_DEF_WRAPPER__(32, H, L) \
    __NHT_DEF_WRAPPER__(48, H, L) __NHT_DEF_WRAPPER__(64, H, L) __NHT_DEF_WRAPPER__(96, H, L)
__NHT_DEF_WRAPPER_ALL_C__(64, 2)   // H=64,  L=2
__NHT_DEF_WRAPPER_ALL_C__(64, 3)   // H=64,  L=3
__NHT_DEF_WRAPPER_ALL_C__(128, 2)  // H=128, L=2
__NHT_DEF_WRAPPER_ALL_C__(128, 3)  // H=128, L=3 (default training config)
#undef __NHT_DEF_WRAPPER_ALL_C__
#undef __NHT_DEF_WRAPPER__

// ── Public dispatch entry point (called from RasterizationNHT.cpp) ───────────
void dispatch_rasterize_to_pixels_from_world_nht_3dgs_fused_fwd(
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
    at::Tensor& renders_rgb, at::Tensor& alphas,
    const at::optional<at::Tensor>& render_feat,
    const at::optional<at::Tensor>& last_ids
) {
    const uint32_t channels = (uint32_t)colors.size(-1);

    // Call the appropriate per-config wrapper (defined just above).
    // No template <> syntax here — each wrapper is a plain named function.
#define __NHT_W__(C,H,L) nht_ffwd_wrapper_##C##_##H##_##L( \
        means, quats, scales, colors, opacities, image_width, image_height, tile_size, \
        viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
        lidar_coeffs, external_distortion_params, tile_offsets, flatten_ids, \
        center_ray_mode, center_ray_dirs, ray_dir_scale, mlp_params, renders_rgb, alphas, \
        render_feat, last_ids)

#define __NHT_SWITCH__(H_VAL, L_VAL) \
    switch (channels) { \
        case 4:  __NHT_W__(4,  H_VAL, L_VAL); break; \
        case 8:  __NHT_W__(8,  H_VAL, L_VAL); break; \
        case 12: __NHT_W__(12, H_VAL, L_VAL); break; \
        case 16: __NHT_W__(16, H_VAL, L_VAL); break; \
        case 24: __NHT_W__(24, H_VAL, L_VAL); break; \
        case 32: __NHT_W__(32, H_VAL, L_VAL); break; \
        case 48: __NHT_W__(48, H_VAL, L_VAL); break; \
        case 64: __NHT_W__(64, H_VAL, L_VAL); break; \
        case 96: __NHT_W__(96, H_VAL, L_VAL); break; \
        default: AT_ERROR("NHT inference: unsupported channels=", channels, \
                          " for hidden=", (int)mlp_hidden_dim, " layers=", (int)mlp_num_layers); \
    }

    if      (mlp_hidden_dim ==  64 && mlp_num_layers == 2) { __NHT_SWITCH__( 64, 2) }
    else if (mlp_hidden_dim ==  64 && mlp_num_layers == 3) { __NHT_SWITCH__( 64, 3) }
    else if (mlp_hidden_dim == 128 && mlp_num_layers == 2) { __NHT_SWITCH__(128, 2) }
    else if (mlp_hidden_dim == 128 && mlp_num_layers == 3) { __NHT_SWITCH__(128, 3) }
    else {
        AT_ERROR("NHT inference: unsupported mlp_hidden_dim=", (int)mlp_hidden_dim,
                 " mlp_num_layers=", (int)mlp_num_layers,
                 ". Supported: hidden in {64,128}, n_layers in {2,3}.");
    }
#undef __NHT_W__
#undef __NHT_SWITCH__
}

} // namespace gsplat

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS
