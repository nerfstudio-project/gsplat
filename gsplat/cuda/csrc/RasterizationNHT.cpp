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
 */

#include "Config.h"

#if GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

#    include <ATen/TensorUtils.h>
#    include <ATen/core/Tensor.h>
#    include <ATen/Functions.h>
#    include <ATen/NativeFunctions.h>
#    include <ATen/core/grad_mode.h>
#    include <c10/cuda/CUDAGuard.h>
#    include <cuda_fp16.h>
#    include <torch/csrc/autograd/custom_function.h>
#    include <torch/library.h>

#    include <iterator>
#    include <tuple>
#    include <vector>

#    include "Common.h"
#    include "CommonNHT.h"
#    include "OpsNHT.h"
#    include "RasterizationNHT.h"
#    include "TorchUtils.h"

namespace gsplat {

int nht_encoding_expansion_factor() { return ENCF; }
int nht_num_encoding_frequencies() { return NUM_ENCODING_FREQUENCIES; }
int nht_feature_divisor() { return VERTEX_PER_PRIM; }

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_fwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &viewmats0, const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks, int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    bool center_ray_mode,
    double ray_dir_scale,
    const at::optional<at::Tensor> &depths_per_gauss,
    bool use_hit_distance,
    bool with_normals
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");

    // Always rasterize in half: cast if needed (FP32 during training, FP16 during eval)
    at::Tensor colors_h = (colors.scalar_type() == at::kHalf) ? colors : colors.to(at::kHalf);
    at::optional<at::Tensor> backgrounds_h = backgrounds;
    if (backgrounds.has_value() && backgrounds.value().scalar_type() != at::kHalf) {
        backgrounds_h = backgrounds.value().to(at::kHalf);
    }

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t C = viewmats0.size(-3);
    uint32_t channels = colors_h.size(-1);
    uint32_t feat_output_channels =
        (channels / VERTEX_PER_PRIM) * ENCF;
    uint32_t total_output_channels = feat_output_channels + 3;

    // Narrow torch-friendly int64_t/double inputs to the kernel's native types.
    uint32_t image_width_u = static_cast<uint32_t>(image_width);
    uint32_t image_height_u = static_cast<uint32_t>(image_height);
    uint32_t tile_size_u = static_cast<uint32_t>(tile_size);
    CameraModelType camera_model_e = static_cast<CameraModelType>(camera_model);
    ShutterType rs_type_e = static_cast<ShutterType>(rs_type);
    float ray_dir_scale_f = static_cast<float>(ray_dir_scale);
    const UnscentedTransformParameters &ut_params_v = *ut_params;
    const FThetaCameraDistortionParameters &ftheta_coeffs_v = *ftheta_coeffs;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model_e == CameraModelType::LIDAR, "If lidar sensor coefficients are given, the camera model must be lidar");
        TORCH_CHECK(lidar_coeffs.value(), "lidar_coeffs intrusive_ptr is null");
    } else {
        TORCH_CHECK(camera_model_e != CameraModelType::LIDAR, "If the sensor is lidar, lidar coefficients must be given");
    }

    at::DimVector renders_shape(batch_dims);
    renders_shape.append({C, image_height_u, image_width_u, total_output_channels});
    at::Tensor renders = at::empty(renders_shape, opt);

    at::DimVector alphas_shape(batch_dims);
    alphas_shape.append({C, image_height_u, image_width_u, 1});
    at::Tensor alphas = at::empty(alphas_shape, opt);

    at::DimVector last_ids_shape(batch_dims);
    last_ids_shape.append({C, image_height_u, image_width_u});
    at::Tensor last_ids = at::empty(last_ids_shape, opt.dtype(at::kInt));

    // Fused-aux outputs. Allocated only when requested; pass at::nullopt
    // otherwise so the kernel skips the corresponding accumulation. A null
    // tensor is returned to the caller for consistent tuple arity.
    const bool render_d_req = depths_per_gauss.has_value() || use_hit_distance;
    const bool render_n_req = with_normals;
    if (depths_per_gauss.has_value()) {
        CHECK_INPUT(depths_per_gauss.value());
    }
    at::optional<at::Tensor> render_depth_opt = at::nullopt;
    at::optional<at::Tensor> render_normals_opt = at::nullopt;
    at::Tensor render_depth_t;
    at::Tensor render_normals_t;
    if (render_d_req) {
        at::DimVector depth_shape(batch_dims);
        depth_shape.append({C, image_height_u, image_width_u, 1});
        render_depth_t = at::empty(depth_shape, opt);
        render_depth_opt = render_depth_t;
    }
    if (render_n_req) {
        at::DimVector normals_shape(batch_dims);
        normals_shape.append({C, image_height_u, image_width_u, 3});
        render_normals_t = at::empty(normals_shape, opt);
        render_normals_opt = render_normals_t;
    }

    at::Tensor center_ray_dirs = viewmats0.select(-2, 2).narrow(-1, 0, 3).contiguous();

#define __NHT_FWD_LAUNCH__(N, SCALAR_T) \
    launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel<N, SCALAR_T>( \
        means, quats, scales, colors_h, opacities, backgrounds_h, masks, \
        image_width_u, image_height_u, tile_size_u, \
        viewmats0, viewmats1, Ks, camera_model_e, ut_params_v, rs_type_e, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs_v, \
        lidar_coeffs, external_distortion_params, \
        tile_offsets, flatten_ids, \
        center_ray_mode, center_ray_dirs, ray_dir_scale_f, \
        depths_per_gauss, use_hit_distance, \
        renders, alphas, render_depth_opt, render_normals_opt, last_ids)

    switch (channels) {
        case 4:   __NHT_FWD_LAUNCH__(4,   at::Half); break;
        case 8:   __NHT_FWD_LAUNCH__(8,   at::Half); break;
        case 12:  __NHT_FWD_LAUNCH__(12,  at::Half); break;
        case 16:  __NHT_FWD_LAUNCH__(16,  at::Half); break;
        case 20:  __NHT_FWD_LAUNCH__(20,  at::Half); break;
        case 24:  __NHT_FWD_LAUNCH__(24,  at::Half); break;
        case 28:  __NHT_FWD_LAUNCH__(28,  at::Half); break;
        case 32:  __NHT_FWD_LAUNCH__(32,  at::Half); break;
        case 36:  __NHT_FWD_LAUNCH__(36,  at::Half); break;
        case 40:  __NHT_FWD_LAUNCH__(40,  at::Half); break;
        case 44:  __NHT_FWD_LAUNCH__(44,  at::Half); break;
        case 48:  __NHT_FWD_LAUNCH__(48,  at::Half); break;
        case 64:  __NHT_FWD_LAUNCH__(64,  at::Half); break;
        case 80:  __NHT_FWD_LAUNCH__(80,  at::Half); break;
        case 96:  __NHT_FWD_LAUNCH__(96,  at::Half); break;
        case 128: __NHT_FWD_LAUNCH__(128, at::Half); break;
        case 256: __NHT_FWD_LAUNCH__(256, at::Half); break;
        default: AT_ERROR("NHT fwd: unsupported channels: ", channels);
    }
#undef __NHT_FWD_LAUNCH__

    // Always return five tensors (placeholder empty tensors when the
    // corresponding aux output wasn't requested). torch ops can't return
    // Optional<Tensor>, and empty placeholders are zero-cost.
    if (!render_d_req) {
        render_depth_t = at::empty({0}, opt);
    }
    if (!render_n_req) {
        render_normals_t = at::empty({0}, opt);
    }
    return std::make_tuple(
        renders, alphas, render_depth_t, render_normals_t, last_ids);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_bwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &colors, const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &viewmats0, const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks, int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    const at::optional<at::Tensor> &depths_per_gauss,
    bool use_hit_distance,
    const at::Tensor &render_alphas, const at::Tensor &last_ids,
    const at::Tensor &v_render_colors, const at::Tensor &v_render_alphas,
    const at::optional<at::Tensor> &v_render_depth,
    const at::optional<at::Tensor> &v_render_normals
) {
    DEVICE_GUARD(means);
    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");

    // Cast colors to half for the kernel (same as forward)
    at::Tensor colors_h = (colors.scalar_type() == at::kHalf) ? colors : colors.to(at::kHalf);
    uint32_t channels = colors_h.size(-1);

    // Narrow torch-friendly int64_t inputs to the kernel's native types.
    uint32_t image_width_u = static_cast<uint32_t>(image_width);
    uint32_t image_height_u = static_cast<uint32_t>(image_height);
    uint32_t tile_size_u = static_cast<uint32_t>(tile_size);
    CameraModelType camera_model_e = static_cast<CameraModelType>(camera_model);
    ShutterType rs_type_e = static_cast<ShutterType>(rs_type);
    const UnscentedTransformParameters &ut_params_v = *ut_params;
    const FThetaCameraDistortionParameters &ftheta_coeffs_v = *ftheta_coeffs;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model_e == CameraModelType::LIDAR, "If lidar sensor coefficients are given, the camera model must be lidar");
        TORCH_CHECK(lidar_coeffs.value(), "lidar_coeffs intrusive_ptr is null");
    } else {
        TORCH_CHECK(camera_model_e != CameraModelType::LIDAR, "If the sensor is lidar, lidar coefficients must be given");
    }

    at::Tensor v_means = at::zeros_like(means);
    at::Tensor v_quats = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_opacities = at::zeros_like(opacities);

    at::Tensor v_render_colors_bwd = v_render_colors.to(at::kFloat);
    at::optional<at::Tensor> backgrounds_bwd = backgrounds;
    if (backgrounds.has_value()) {
        backgrounds_bwd = backgrounds.value().to(at::kFloat);
    }
    // v_colors accumulated in FP32, converted to input dtype at the end
    at::Tensor v_colors_bwd = at::zeros(colors_h.sizes(), means.options().dtype(at::kFloat));

    // Fused-aux gradients. v_depths_per_gauss is allocated iff the forward
    // used a per-Gaussian depth (not the hit-distance branch). For
    // hit-distance, the gradient is threaded directly into v_means / v_scales
    // / v_quats inside the kernel, so v_depths_per_gauss stays unallocated.
    at::optional<at::Tensor> v_render_depth_f32 = at::nullopt;
    at::optional<at::Tensor> v_render_normals_f32 = at::nullopt;
    if (v_render_depth.has_value()) {
        v_render_depth_f32 = v_render_depth.value().to(at::kFloat);
    }
    if (v_render_normals.has_value()) {
        v_render_normals_f32 = v_render_normals.value().to(at::kFloat);
    }
    at::optional<at::Tensor> v_depths_per_gauss_opt = at::nullopt;
    at::Tensor v_depths_per_gauss_t;
    if (v_render_depth.has_value() && !use_hit_distance && depths_per_gauss.has_value()) {
        v_depths_per_gauss_t = at::zeros_like(depths_per_gauss.value(), means.options().dtype(at::kFloat));
        v_depths_per_gauss_opt = v_depths_per_gauss_t;
    }

#define __NHT_BWD_LAUNCH__(N) \
    launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<N, at::Half>( \
        means, quats, scales, colors_h, opacities, backgrounds_bwd, masks, \
        image_width_u, image_height_u, tile_size_u, \
        viewmats0, viewmats1, Ks, camera_model_e, ut_params_v, rs_type_e, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs_v, \
        lidar_coeffs, external_distortion_params, \
        tile_offsets, flatten_ids, \
        depths_per_gauss, use_hit_distance, \
        render_alphas, last_ids, \
        v_render_colors_bwd, v_render_alphas, \
        v_render_depth_f32, v_render_normals_f32, \
        v_means, v_quats, v_scales, v_colors_bwd, v_opacities, \
        v_depths_per_gauss_opt)

    switch (channels) {
        case 4:   __NHT_BWD_LAUNCH__(4);   break;
        case 8:   __NHT_BWD_LAUNCH__(8);   break;
        case 12:  __NHT_BWD_LAUNCH__(12);  break;
        case 16:  __NHT_BWD_LAUNCH__(16);  break;
        case 20:  __NHT_BWD_LAUNCH__(20);  break;
        case 24:  __NHT_BWD_LAUNCH__(24);  break;
        case 28:  __NHT_BWD_LAUNCH__(28);  break;
        case 32:  __NHT_BWD_LAUNCH__(32);  break;
        case 36:  __NHT_BWD_LAUNCH__(36);  break;
        case 40:  __NHT_BWD_LAUNCH__(40);  break;
        case 44:  __NHT_BWD_LAUNCH__(44);  break;
        case 48:  __NHT_BWD_LAUNCH__(48);  break;
        case 64:  __NHT_BWD_LAUNCH__(64);  break;
        case 80:  __NHT_BWD_LAUNCH__(80);  break;
        case 96:  __NHT_BWD_LAUNCH__(96);  break;
        case 128: __NHT_BWD_LAUNCH__(128); break;
        case 256: __NHT_BWD_LAUNCH__(256); break;
        default: AT_ERROR("NHT bwd: unsupported channels: ", channels);
    }
#undef __NHT_BWD_LAUNCH__

    at::Tensor v_colors = v_colors_bwd.to(colors.scalar_type());

    if (!v_depths_per_gauss_opt.has_value()) {
        v_depths_per_gauss_t = at::empty({0}, means.options().dtype(at::kFloat));
    }
    return std::make_tuple(
        v_means, v_quats, v_scales, v_colors, v_opacities, v_depths_per_gauss_t);
}

// ── Fully-fused NHT inference (rasterization + encoding + MLP) ───────────────

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_fused_fwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &colors, const at::Tensor &opacities,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &viewmats0, const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks, int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    bool center_ray_mode, double ray_dir_scale,
    const at::Tensor &mlp_params,
    int64_t mlp_hidden_dim,
    int64_t mlp_num_layers,
    bool save_state
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means); CHECK_INPUT(quats); CHECK_INPUT(scales);
    CHECK_INPUT(colors); CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets); CHECK_INPUT(flatten_ids);
    CHECK_INPUT(mlp_params);
    TORCH_CHECK(ut_params,     "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");
    TORCH_CHECK(colors.scalar_type() == at::kHalf, "NHT inference: colors must be fp16");
    TORCH_CHECK(mlp_params.scalar_type() == at::kHalf, "NHT inference: mlp_params must be fp16");
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model == (int64_t)CameraModelType::LIDAR,
            "lidar_coeffs requires camera_model=LIDAR");
        TORCH_CHECK(lidar_coeffs.value(), "lidar_coeffs intrusive_ptr is null");
    }

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    const uint32_t C_cam  = (uint32_t)viewmats0.size(-3);
    const uint32_t im_h   = (uint32_t)image_height;
    const uint32_t im_w   = (uint32_t)image_width;
    const uint32_t channels = (uint32_t)colors.size(-1);

    at::DimVector rgb_shape(batch_dims);
    rgb_shape.append({C_cam, im_h, im_w, 3});
    at::Tensor renders_rgb = at::empty(rgb_shape,
        at::TensorOptions().dtype(at::kHalf).device(means.device()));

    at::DimVector alpha_shape(batch_dims);
    alpha_shape.append({C_cam, im_h, im_w});
    at::Tensor alphas = at::zeros(alpha_shape, opt);

    at::Tensor center_ray_dirs = viewmats0.select(-2, 2).narrow(-1, 0, 3).contiguous();

    // Optional training-state outputs (saved for the fused backward).
    const uint32_t feat_out = (channels / 4) * 2;  // OUT_CDIM * ENCF
    at::Tensor render_feat, last_ids;
    at::optional<at::Tensor> render_feat_opt, last_ids_opt;
    if (save_state) {
        at::DimVector feat_shape(batch_dims);
        feat_shape.append({C_cam, im_h, im_w, (int64_t)feat_out});
        render_feat = at::zeros(feat_shape, opt);
        at::DimVector lid_shape(batch_dims);
        lid_shape.append({C_cam, im_h, im_w});
        last_ids = at::zeros(lid_shape, opt.dtype(at::kInt));
        render_feat_opt = render_feat;
        last_ids_opt = last_ids;
    } else {
        render_feat = at::empty({0}, opt);
        last_ids = at::empty({0}, opt.dtype(at::kInt));
    }

    // All channel/hidden/layer dispatch is handled in the .cu file.
    dispatch_rasterize_to_pixels_from_world_nht_3dgs_fused_fwd(
        means, quats, scales, colors, opacities,
        im_w, im_h, (uint32_t)tile_size,
        viewmats0, viewmats1, Ks,
        (CameraModelType)camera_model, *ut_params, (ShutterType)rs_type,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        *ftheta_coeffs, lidar_coeffs, external_distortion_params,
        tile_offsets, flatten_ids,
        center_ray_mode, center_ray_dirs, (float)ray_dir_scale,
        mlp_params,
        (uint32_t)mlp_hidden_dim, (uint32_t)mlp_num_layers,
        renders_rgb, alphas, render_feat_opt, last_ids_opt);

    return {renders_rgb, alphas, render_feat, last_ids};
}

// ── Fused NHT training backward ──────────────────────────────────────────────

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_fused_bwd(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &colors, const at::Tensor &opacities,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::Tensor &viewmats0, const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks, int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets, const at::Tensor &flatten_ids,
    bool center_ray_mode, double ray_dir_scale,
    const at::Tensor &mlp_params,
    int64_t mlp_hidden_dim, int64_t mlp_num_layers,
    double loss_scale,
    const at::Tensor &render_feat,
    const at::Tensor &render_alphas,
    const at::Tensor &last_ids,
    const at::Tensor &v_render_rgb,
    const at::Tensor &v_render_alphas,
    bool compute_mlp_grad
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means); CHECK_INPUT(quats); CHECK_INPUT(scales);
    CHECK_INPUT(colors); CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets); CHECK_INPUT(flatten_ids);
    CHECK_INPUT(mlp_params);
    CHECK_INPUT(render_feat); CHECK_INPUT(render_alphas); CHECK_INPUT(last_ids);
    CHECK_INPUT(v_render_rgb); CHECK_INPUT(v_render_alphas);
    TORCH_CHECK(ut_params,     "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");
    TORCH_CHECK(colors.scalar_type() == at::kHalf, "NHT fused bwd: colors must be fp16");
    TORCH_CHECK(mlp_params.scalar_type() == at::kHalf, "NHT fused bwd: mlp_params must be fp16");
    TORCH_CHECK(render_feat.scalar_type() == at::kFloat, "render_feat must be fp32");
    TORCH_CHECK(v_render_rgb.scalar_type() == at::kFloat, "v_render_rgb must be fp32");
    TORCH_CHECK(v_render_alphas.scalar_type() == at::kFloat, "v_render_alphas must be fp32");
    TORCH_CHECK(last_ids.scalar_type() == at::kInt, "last_ids must be int32");

    const uint32_t im_h = (uint32_t)image_height;
    const uint32_t im_w = (uint32_t)image_width;

    at::Tensor v_means     = at::zeros_like(means);
    at::Tensor v_quats     = at::zeros_like(quats);
    at::Tensor v_scales    = at::zeros_like(scales);
    at::Tensor v_colors    = at::zeros(colors.sizes(),
        colors.options().dtype(at::kFloat));
    at::Tensor v_opacities = at::zeros_like(opacities);
    at::Tensor v_mlp_params = at::zeros(
        {compute_mlp_grad ? mlp_params.numel() : 0},
        mlp_params.options().dtype(at::kFloat));

    at::Tensor center_ray_dirs = viewmats0.select(-2, 2).narrow(-1, 0, 3).contiguous();

    dispatch_rasterize_to_pixels_from_world_nht_3dgs_fused_bwd(
        means, quats, scales, colors, opacities,
        im_w, im_h, (uint32_t)tile_size,
        viewmats0, viewmats1, Ks,
        (CameraModelType)camera_model, *ut_params, (ShutterType)rs_type,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        *ftheta_coeffs, lidar_coeffs, external_distortion_params,
        tile_offsets, flatten_ids,
        center_ray_mode, center_ray_dirs, (float)ray_dir_scale,
        mlp_params,
        (uint32_t)mlp_hidden_dim, (uint32_t)mlp_num_layers,
        (float)loss_scale,
        render_feat, render_alphas, last_ids,
        v_render_rgb, v_render_alphas,
        v_means, v_quats, v_scales, v_colors, v_opacities, v_mlp_params);

    // v_mlp_params is still multiplied by loss_scale; the Python wrapper
    // divides after the kernel (keeps the fp16 fragment path well-scaled).
    return {v_means, v_quats, v_scales, v_colors, v_opacities, v_mlp_params};
}

// ── Autograd-aware NHT rasterizer ────────────────────────────────────────────
// Mirrors `RasterizeToPixelsFromWorld3DGSAutograd` in Rasterization.cpp: a
// custom autograd Function around the fwd/bwd pair, plus a dispatcher that
// skips it entirely under no-grad. This is what `rasterization_3dgs` calls, so
// the whole NHT render path lives in the C++ orchestrator rather than in
// Python.

namespace
{
    // Channel counts the NHT kernel switch is instantiated for. Feature inputs
    // are padded up to the next entry; the padded output columns are trimmed
    // back off after the render.
    constexpr int64_t kNHTSupportedChannels[] = {4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 80, 96, 128, 256};

    int64_t next_supported_nht_channels(int64_t channels)
    {
        for(const int64_t supported: kNHTSupportedChannels)
        {
            if(supported >= channels)
            {
                return supported;
            }
        }
        return -1;
    }

    bool is_supported_nht_channels(int64_t channels)
    {
        for(const int64_t supported: kNHTSupportedChannels)
        {
            if(supported == channels)
            {
                return true;
            }
        }
        return false;
    }

    // Pad vertex features up to the next compiled CDIM. Features are laid out as
    // VERTEX_PER_PRIM contiguous groups of `fvdim` channels, so the padding has
    // to be inserted per group rather than appended to the flat channel axis.
    // Backgrounds live in *output* (encoded) space, so they grow by the encoded
    // width of the added feature columns.
    struct NHTPaddedFeatures
    {
        at::Tensor colors;
        at::optional<at::Tensor> backgrounds;
    };

    NHTPaddedFeatures pad_nht_features(
        const at::Tensor &colors, const at::optional<at::Tensor> &backgrounds, int64_t divisor, int64_t encf
    )
    {
        const int64_t input_channels = colors.size(-1);
        if(is_supported_nht_channels(input_channels))
        {
            return {colors, backgrounds};
        }

        const int64_t target_channels = next_supported_nht_channels(input_channels);
        TORCH_CHECK(
            target_channels > 0,
            "NHT: unsupported input channel count ",
            input_channels,
            "; the largest compiled NHT channel count is ",
            kNHTSupportedChannels[std::size(kNHTSupportedChannels) - 1]
        );

        const int64_t fvdim         = input_channels / divisor;
        const int64_t target_fvdim  = target_channels / divisor;
        const int64_t pad_per_group = target_fvdim - fvdim;

        at::Tensor padded = colors.unflatten(-1, {divisor, fvdim});
        padded            = at::constant_pad_nd(padded, {0, pad_per_group}, 0.0);
        padded            = padded.flatten(-2, -1);

        at::optional<at::Tensor> padded_backgrounds = backgrounds;
        if(backgrounds.has_value())
        {
            const int64_t padded_output      = (target_fvdim - fvdim) * encf;
            const at::Tensor &bg             = backgrounds.value();
            std::vector<int64_t> zeros_shape = bg.sizes().vec();
            zeros_shape.back()               = padded_output;
            padded_backgrounds               = at::cat({bg, at::zeros(zeros_shape, bg.options())}, -1);
        }

        return {padded, padded_backgrounds};
    }

    class RasterizeToPixelsFromWorldNHT3DGSAutograd
        : public torch::autograd::Function<RasterizeToPixelsFromWorldNHT3DGSAutograd>
    {
    public:
        // Forward-input positions; COUNT sizes the returned grad list and mirrors
        // apply()'s argument order one-to-one.
        struct FwdInput
        {
            enum
            {
                MEANS,
                QUATS,
                SCALES,
                COLORS,
                OPACITIES,
                BACKGROUNDS,
                MASKS,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                TILE_SIZE,
                VIEWMATS0,
                VIEWMATS1,
                KS,
                CAMERA_MODEL,
                UT_PARAMS,
                RS_TYPE,
                RADIAL_COEFFS,
                TANGENTIAL_COEFFS,
                THIN_PRISM_COEFFS,
                FTHETA_COEFFS,
                LIDAR_COEFFS,
                EXTERNAL_DISTORTION_PARAMS,
                TILE_OFFSETS,
                FLATTEN_IDS,
                CENTER_RAY_MODE,
                RAY_DIR_SCALE,
                DEPTHS_PER_GAUSS,
                USE_HIT_DISTANCE,
                WITH_NORMALS,
                COUNT,
            };
        };

        // Forward-output positions, in forward()'s return order.
        struct FwdOutput
        {
            enum
            {
                RENDERS,
                ALPHAS,
                DEPTH,
                NORMALS,
                COUNT
            };
        };

        static torch::autograd::variable_list forward(
            torch::autograd::AutogradContext *ctx,
            const at::Tensor &means,
            const at::Tensor &quats,
            const at::Tensor &scales,
            const at::Tensor &colors,
            const at::Tensor &opacities,
            const at::optional<at::Tensor> &backgrounds,
            const at::optional<at::Tensor> &masks,
            int64_t image_width,
            int64_t image_height,
            int64_t tile_size,
            const at::Tensor &viewmats0,
            const at::optional<at::Tensor> &viewmats1,
            const at::Tensor &Ks,
            int64_t camera_model,
            const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
            int64_t rs_type,
            const at::optional<at::Tensor> &radial_coeffs,
            const at::optional<at::Tensor> &tangential_coeffs,
            const at::optional<at::Tensor> &thin_prism_coeffs,
            const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
            const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
            const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>>
                &external_distortion_params,
            const at::Tensor &tile_offsets,
            const at::Tensor &flatten_ids,
            bool center_ray_mode,
            double ray_dir_scale,
            const at::optional<at::Tensor> &depths_per_gauss,
            bool use_hit_distance,
            bool with_normals
        )
        {
            static_assert(
                FwdInput::COUNT == fwd_input_count<&forward>(), "FwdInput must have one enumerator per forward input"
            );

            auto [renders, alphas, render_depth, render_normals, last_ids]
                = rasterize_to_pixels_from_world_nht_3dgs_fwd(
                    means,
                    quats,
                    scales,
                    colors,
                    opacities,
                    backgrounds,
                    masks,
                    image_width,
                    image_height,
                    tile_size,
                    viewmats0,
                    viewmats1,
                    Ks,
                    camera_model,
                    ut_params,
                    rs_type,
                    radial_coeffs,
                    tangential_coeffs,
                    thin_prism_coeffs,
                    ftheta_coeffs,
                    lidar_coeffs,
                    external_distortion_params,
                    tile_offsets,
                    flatten_ids,
                    center_ray_mode,
                    ray_dir_scale,
                    depths_per_gauss,
                    use_hit_distance,
                    with_normals
                );

            // depths_per_gauss is saved as an empty sentinel when absent so the
            // saved-tensor list keeps a fixed arity.
            ctx->save_for_backward(
                {means,
                 quats,
                 scales,
                 colors,
                 opacities,
                 as_tensor(backgrounds),
                 as_tensor(masks),
                 viewmats0,
                 as_tensor(viewmats1),
                 Ks,
                 as_tensor(radial_coeffs),
                 as_tensor(tangential_coeffs),
                 as_tensor(thin_prism_coeffs),
                 tile_offsets,
                 flatten_ids,
                 alphas,
                 last_ids,
                 depths_per_gauss.value_or(at::empty({0}, means.options()))}
            );
            ctx->saved_data["image_width"]          = image_width;
            ctx->saved_data["image_height"]         = image_height;
            ctx->saved_data["tile_size"]            = tile_size;
            ctx->saved_data["camera_model"]         = camera_model;
            ctx->saved_data["rs_type"]              = rs_type;
            ctx->saved_data["use_hit_distance"]     = use_hit_distance;
            ctx->saved_data["has_depth"]            = render_depth.numel() > 0;
            ctx->saved_data["has_normals"]          = render_normals.numel() > 0;
            ctx->saved_data["has_depths_per_gauss"] = depths_per_gauss.has_value();
            ctx->saved_data["ut_params"]            = ut_params;
            ctx->saved_data["ftheta_coeffs"]        = ftheta_coeffs;
            if(lidar_coeffs.has_value())
            {
                ctx->saved_data["lidar_coeffs"] = lidar_coeffs.value();
            }
            if(external_distortion_params.has_value())
            {
                ctx->saved_data["external_distortion_params"] = external_distortion_params.value();
            }

            if(render_depth.numel() == 0)
            {
                ctx->mark_non_differentiable({render_depth});
            }
            if(render_normals.numel() == 0)
            {
                ctx->mark_non_differentiable({render_normals});
            }

            torch::autograd::variable_list out(FwdOutput::COUNT);
            out[FwdOutput::RENDERS] = renders;
            out[FwdOutput::ALPHAS]  = alphas;
            out[FwdOutput::DEPTH]   = render_depth;
            out[FwdOutput::NORMALS] = render_normals;
            return out;
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs
        )
        {
            const torch::autograd::variable_list saved = ctx->get_saved_variables();
            const at::Tensor &means                    = saved[0];
            const at::Tensor &quats                    = saved[1];
            const at::Tensor &scales                   = saved[2];
            const at::Tensor &colors                   = saved[3];
            const at::Tensor &opacities                = saved[4];
            const at::Tensor &backgrounds              = saved[5];
            const at::Tensor &masks                    = saved[6];
            const at::Tensor &viewmats0                = saved[7];
            const at::Tensor &viewmats1                = saved[8];
            const at::Tensor &Ks                       = saved[9];
            const at::Tensor &radial_coeffs            = saved[10];
            const at::Tensor &tangential_coeffs        = saved[11];
            const at::Tensor &thin_prism_coeffs        = saved[12];
            const at::Tensor &tile_offsets             = saved[13];
            const at::Tensor &flatten_ids              = saved[14];
            const at::Tensor &render_alphas            = saved[15];
            const at::Tensor &last_ids                 = saved[16];
            const at::Tensor &saved_depths_per_gauss   = saved[17];

            const bool has_depth            = ctx->saved_data["has_depth"].toBool();
            const bool has_normals          = ctx->saved_data["has_normals"].toBool();
            const bool has_depths_per_gauss = ctx->saved_data["has_depths_per_gauss"].toBool();

            const at::Tensor &v_renders = grad_outputs[FwdOutput::RENDERS];
            TORCH_CHECK(v_renders.defined(), "NHT backward requires a gradient on the rendered features");

            // Strip the three trailing ray-direction channels: they are an
            // informational output with no gradient path back into the scene.
            // The forward renders in half precision; the backward kernel
            // accumulates in fp32.
            const at::Tensor v_render_colors
                = v_renders.narrow(-1, 0, v_renders.size(-1) - 3).contiguous().to(at::kFloat);

            const at::Tensor &v_alphas_in = grad_outputs[FwdOutput::ALPHAS];
            const at::Tensor v_render_alphas
                = v_alphas_in.defined() ? v_alphas_in.contiguous() : at::zeros_like(render_alphas);

            at::optional<at::Tensor> v_render_depth;
            if(has_depth)
            {
                const at::Tensor &grad = grad_outputs[FwdOutput::DEPTH];
                if(grad.defined() && grad.numel() > 0)
                {
                    v_render_depth = grad.contiguous();
                }
            }
            at::optional<at::Tensor> v_render_normals;
            if(has_normals)
            {
                const at::Tensor &grad = grad_outputs[FwdOutput::NORMALS];
                if(grad.defined() && grad.numel() > 0)
                {
                    v_render_normals = grad.contiguous();
                }
            }

            at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> lidar_coeffs;
            if(ctx->saved_data.count("lidar_coeffs") != 0)
            {
                lidar_coeffs = ctx->saved_data["lidar_coeffs"]
                                   .toCustomClass<RowOffsetStructuredSpinningLidarModelParametersExt>();
            }
            at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> external_distortion_params;
            if(ctx->saved_data.count("external_distortion_params") != 0)
            {
                external_distortion_params = ctx->saved_data["external_distortion_params"]
                                                 .toCustomClass<extdist::BivariateWindshieldModelParameters>();
            }

            auto [v_means, v_quats, v_scales, v_colors, v_opacities, v_depths_per_gauss]
                = rasterize_to_pixels_from_world_nht_3dgs_bwd(
                    means,
                    quats,
                    scales,
                    colors,
                    opacities,
                    as_optional_tensor(backgrounds),
                    as_optional_tensor(masks),
                    ctx->saved_data["image_width"].toInt(),
                    ctx->saved_data["image_height"].toInt(),
                    ctx->saved_data["tile_size"].toInt(),
                    viewmats0,
                    as_optional_tensor(viewmats1),
                    Ks,
                    ctx->saved_data["camera_model"].toInt(),
                    ctx->saved_data["ut_params"].toCustomClass<UnscentedTransformParameters>(),
                    ctx->saved_data["rs_type"].toInt(),
                    as_optional_tensor(radial_coeffs),
                    as_optional_tensor(tangential_coeffs),
                    as_optional_tensor(thin_prism_coeffs),
                    ctx->saved_data["ftheta_coeffs"].toCustomClass<FThetaCameraDistortionParameters>(),
                    lidar_coeffs,
                    external_distortion_params,
                    tile_offsets,
                    flatten_ids,
                    has_depths_per_gauss ? at::optional<at::Tensor>(saved_depths_per_gauss) : at::nullopt,
                    ctx->saved_data["use_hit_distance"].toBool(),
                    render_alphas,
                    last_ids,
                    v_render_colors,
                    v_render_alphas,
                    v_render_depth,
                    v_render_normals
                );

            // The backward kernel does not produce the background gradient: it is
            // the incoming color gradient weighted by the unoccluded alpha.
            at::Tensor v_backgrounds;
            if(tensor_requires_grad(backgrounds))
            {
                const at::Tensor one_minus_alpha = at::sub(at::ones_like(render_alphas), render_alphas).to(at::kFloat);
                v_backgrounds                    = at::mul(v_render_colors, one_minus_alpha).sum({-3, -2});
            }

            torch::autograd::variable_list grads(FwdInput::COUNT);
            grads[FwdInput::MEANS]       = v_means;
            grads[FwdInput::QUATS]       = v_quats;
            grads[FwdInput::SCALES]      = v_scales;
            grads[FwdInput::COLORS]      = v_colors;
            grads[FwdInput::OPACITIES]   = v_opacities;
            grads[FwdInput::BACKGROUNDS] = v_backgrounds;
            if(has_depths_per_gauss && v_depths_per_gauss.numel() > 0)
            {
                grads[FwdInput::DEPTHS_PER_GAUSS] = v_depths_per_gauss;
            }
            return grads;
        }
    };
} // namespace

RasterizeToPixelsFromWorldNHT3DGSResult rasterize_to_pixels_from_world_nht_3dgs(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &viewmats0,
    const at::optional<at::Tensor> &viewmats1,
    const at::Tensor &Ks,
    int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    bool center_ray_mode,
    double ray_dir_scale,
    bool use_hit_distance,
    bool with_normals
)
{
    constexpr int64_t divisor = VERTEX_PER_PRIM;
    constexpr int64_t encf    = ENCF;

    // The orchestrator hands us [vertex features | depth]: every render mode
    // that needs depth (and the depth-derived-normals mode) appends a single
    // projection-depth column as the last channel. NHT must not harmonically
    // encode that scalar, so it is split off here and fed to the kernel's
    // dedicated depth accumulator instead. Detect it structurally: the feature
    // block is always a multiple of VERTEX_PER_PRIM, so exactly one trailing
    // channel is the only way the total can be off by one.
    at::Tensor feature_colors                    = colors;
    at::optional<at::Tensor> feature_backgrounds = backgrounds;
    at::optional<at::Tensor> depths_per_gauss;

    const int64_t total_channels = colors.size(-1);
    const bool has_depth_column  = (total_channels % divisor != 0) && ((total_channels - 1) % divisor == 0);
    if(has_depth_column)
    {
        depths_per_gauss = colors.select(-1, total_channels - 1).contiguous();
        feature_colors   = colors.narrow(-1, 0, total_channels - 1);
        if(backgrounds.has_value())
        {
            feature_backgrounds = backgrounds.value().narrow(-1, 0, backgrounds.value().size(-1) - 1);
        }
    }
    else if(use_hit_distance)
    {
        // Hit distance is measured by the kernel itself, so any per-Gaussian depth
        // the caller supplied is not an input to it.
        depths_per_gauss = at::nullopt;
    }

    TORCH_CHECK(
        feature_colors.size(-1) % divisor == 0,
        "NHT feature channels must be divisible by ",
        divisor,
        "; got ",
        feature_colors.size(-1)
    );

    const int64_t original_output_channels = (feature_colors.size(-1) / divisor) * encf;
    NHTPaddedFeatures padded               = pad_nht_features(feature_colors, feature_backgrounds, divisor, encf);

    const at::Tensor kernel_colors = padded.colors.contiguous();
    const at::optional<at::Tensor> kernel_backgrounds
        = padded.backgrounds.has_value() ? at::optional<at::Tensor>(padded.backgrounds.value().contiguous())
                                         : at::nullopt;

    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor render_depth;
    at::Tensor render_normals;

    const bool use_custom_autograd
        = needs_custom_autograd(means, quats, scales, kernel_colors, opacities, kernel_backgrounds, depths_per_gauss);

    if(use_custom_autograd)
    {
        torch::autograd::variable_list outputs = RasterizeToPixelsFromWorldNHT3DGSAutograd::apply(
            means,
            quats,
            scales,
            kernel_colors,
            opacities,
            kernel_backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            viewmats0,
            viewmats1,
            Ks,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            lidar_coeffs,
            external_distortion_params,
            tile_offsets,
            flatten_ids,
            center_ray_mode,
            ray_dir_scale,
            depths_per_gauss,
            use_hit_distance,
            with_normals
        );
        renders        = outputs[0];
        alphas         = outputs[1];
        render_depth   = outputs[2];
        render_normals = outputs[3];
    }
    else
    {
        at::Tensor last_ids;
        std::tie(renders, alphas, render_depth, render_normals, last_ids) = rasterize_to_pixels_from_world_nht_3dgs_fwd(
            means,
            quats,
            scales,
            kernel_colors,
            opacities,
            kernel_backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            viewmats0,
            viewmats1,
            Ks,
            camera_model,
            ut_params,
            rs_type,
            radial_coeffs,
            tangential_coeffs,
            thin_prism_coeffs,
            ftheta_coeffs,
            lidar_coeffs,
            external_distortion_params,
            tile_offsets,
            flatten_ids,
            center_ray_mode,
            ray_dir_scale,
            depths_per_gauss,
            use_hit_distance,
            with_normals
        );
    }

    // Drop the columns that only exist because the features were padded up to a
    // compiled CDIM, keeping the three trailing ray-direction channels.
    const int64_t rendered_channels = renders.size(-1);
    const int64_t feature_channels  = rendered_channels - 3;
    if(feature_channels > original_output_channels)
    {
        renders
            = at::cat({renders.narrow(-1, 0, original_output_channels), renders.narrow(-1, feature_channels, 3)}, -1);
    }

    return {renders, alphas, render_depth, render_normals};
}
} // namespace gsplat

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS
