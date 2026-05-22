/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "Common.h"
#include "CommonNHT.h"
#include "OpsNHT.h"
#include "RasterizationNHT.h"

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
        case 4:   __NHT_FWD_LAUNCH__(4, at::Half); break;
        case 8:   __NHT_FWD_LAUNCH__(8, at::Half); break;
        case 12:  __NHT_FWD_LAUNCH__(12, at::Half); break;
        case 16:  __NHT_FWD_LAUNCH__(16, at::Half); break;
        case 20:  __NHT_FWD_LAUNCH__(20, at::Half); break;
        case 24:  __NHT_FWD_LAUNCH__(24, at::Half); break;
        case 28:  __NHT_FWD_LAUNCH__(28, at::Half); break;
        case 32:  __NHT_FWD_LAUNCH__(32, at::Half); break;
        case 36:  __NHT_FWD_LAUNCH__(36, at::Half); break;
        case 40:  __NHT_FWD_LAUNCH__(40, at::Half); break;
        case 44:  __NHT_FWD_LAUNCH__(44, at::Half); break;
        case 48:  __NHT_FWD_LAUNCH__(48, at::Half); break;
        case 64:  __NHT_FWD_LAUNCH__(64, at::Half); break;
        case 80:  __NHT_FWD_LAUNCH__(80, at::Half); break;
        case 96:  __NHT_FWD_LAUNCH__(96, at::Half); break;
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
        case 4:   __NHT_BWD_LAUNCH__(4); break;
        case 8:   __NHT_BWD_LAUNCH__(8); break;
        case 12:  __NHT_BWD_LAUNCH__(12); break;
        case 16:  __NHT_BWD_LAUNCH__(16); break;
        case 20:  __NHT_BWD_LAUNCH__(20); break;
        case 24:  __NHT_BWD_LAUNCH__(24); break;
        case 28:  __NHT_BWD_LAUNCH__(28); break;
        case 32:  __NHT_BWD_LAUNCH__(32); break;
        case 36:  __NHT_BWD_LAUNCH__(36); break;
        case 40:  __NHT_BWD_LAUNCH__(40); break;
        case 44:  __NHT_BWD_LAUNCH__(44); break;
        case 48:  __NHT_BWD_LAUNCH__(48); break;
        case 64:  __NHT_BWD_LAUNCH__(64); break;
        case 80:  __NHT_BWD_LAUNCH__(80); break;
        case 96:  __NHT_BWD_LAUNCH__(96); break;
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

} // namespace gsplat

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS
