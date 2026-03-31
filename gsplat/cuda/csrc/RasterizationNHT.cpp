/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#include "Config.h"
#include "Common.h"
#include "Ops.h"
#include "RasterizationNHT.h"

namespace gsplat {

#if GSPLAT_BUILD_3DGS

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_fwd(
    const at::Tensor means, const at::Tensor quats, const at::Tensor scales,
    const at::Tensor colors, const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor viewmats0, const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks, CameraModelType camera_model,
    UnscentedTransformParameters ut_params, ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor tile_offsets, const at::Tensor flatten_ids,
    bool center_ray_mode,
    float ray_dir_scale
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

    at::DimVector renders_shape(batch_dims);
    renders_shape.append({C, image_height, image_width, total_output_channels});
    at::Tensor renders = at::empty(renders_shape, opt);

    at::DimVector alphas_shape(batch_dims);
    alphas_shape.append({C, image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_shape, opt);

    at::DimVector last_ids_shape(batch_dims);
    last_ids_shape.append({C, image_height, image_width});
    at::Tensor last_ids = at::empty(last_ids_shape, opt.dtype(at::kInt));

    at::Tensor center_ray_dirs = viewmats0.select(-2, 2).narrow(-1, 0, 3).contiguous();

#define __NHT_FWD_LAUNCH__(N, SCALAR_T) \
    launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel<N, SCALAR_T>( \
        means, quats, scales, colors_h, opacities, backgrounds_h, masks, \
        image_width, image_height, tile_size, \
        viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
        tile_offsets, flatten_ids, \
        center_ray_mode, center_ray_dirs, ray_dir_scale, \
        renders, alphas, last_ids)

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
        default: AT_ERROR("NHT fwd: unsupported channels: ", channels);
    }
#undef __NHT_FWD_LAUNCH__

    return std::make_tuple(renders, alphas, last_ids);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_bwd(
    const at::Tensor means, const at::Tensor quats, const at::Tensor scales,
    const at::Tensor colors, const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor viewmats0, const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks, CameraModelType camera_model,
    UnscentedTransformParameters ut_params, ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    FThetaCameraDistortionParameters ftheta_coeffs,
    const at::Tensor tile_offsets, const at::Tensor flatten_ids,
    const at::Tensor render_alphas, const at::Tensor last_ids,
    const at::Tensor v_render_colors, const at::Tensor v_render_alphas
) {
    DEVICE_GUARD(means);

    // Cast colors to half for the kernel (same as forward)
    at::Tensor colors_h = (colors.scalar_type() == at::kHalf) ? colors : colors.to(at::kHalf);
    uint32_t channels = colors_h.size(-1);

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

#define __NHT_BWD_LAUNCH__(N) \
    launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel<N, at::Half>( \
        means, quats, scales, colors_h, opacities, backgrounds_bwd, masks, \
        image_width, image_height, tile_size, \
        viewmats0, viewmats1, Ks, camera_model, ut_params, rs_type, \
        radial_coeffs, tangential_coeffs, thin_prism_coeffs, ftheta_coeffs, \
        tile_offsets, flatten_ids, render_alphas, last_ids, \
        v_render_colors_bwd, v_render_alphas, \
        v_means, v_quats, v_scales, v_colors_bwd, v_opacities)

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
        default: AT_ERROR("NHT bwd: unsupported channels: ", channels);
    }
#undef __NHT_BWD_LAUNCH__

    at::Tensor v_colors = v_colors_bwd.to(colors.scalar_type());

    return std::make_tuple(v_means, v_quats, v_scales, v_colors, v_opacities);
}

#endif // GSPLAT_BUILD_3DGS

} // namespace gsplat
