/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include "Cameras.h"
#include "ExternalDistortion.h"
#include "Lidars.h"

namespace at {
class Tensor;
}

namespace gsplat {

struct RowOffsetStructuredSpinningLidarModelParametersExt;

// NHT "from world" 3DGS: full camera model stack + tetrahedral interpolation.
// NHT always uses world evaluation (with_eval3d + with_ut).
// Forward supports FP16 via scalar_t = at::Half (fetches half, accumulation fp32).

template <uint32_t CDIM, typename scalar_t>
void launch_rasterize_to_pixels_from_world_nht_3dgs_fwd_kernel(
    const at::Tensor means, const at::Tensor quats, const at::Tensor scales,
    const at::Tensor colors, const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor viewmats0, const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks, CameraModelType camera_model,
    const UnscentedTransformParameters &ut_params, ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const FThetaCameraDistortionParameters &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets, const at::Tensor flatten_ids,
    bool center_ray_mode, const at::Tensor center_ray_dirs,
    float ray_dir_scale,
    const at::optional<at::Tensor> depths_per_gauss,
    bool use_hit_distance,
    at::Tensor renders, at::Tensor alphas,
    const at::optional<at::Tensor> render_depth,
    const at::optional<at::Tensor> render_normals,
    at::Tensor last_ids
);

// Backward: gradient math in FP32, color fetches templated for FP16 vectorized loads.
template <uint32_t CDIM, typename scalar_t>
void launch_rasterize_to_pixels_from_world_nht_3dgs_bwd_kernel(
    const at::Tensor means, const at::Tensor quats, const at::Tensor scales,
    const at::Tensor colors, const at::Tensor opacities,
    const at::optional<at::Tensor> backgrounds,
    const at::optional<at::Tensor> masks,
    uint32_t image_width, uint32_t image_height, uint32_t tile_size,
    const at::Tensor viewmats0, const at::optional<at::Tensor> viewmats1,
    const at::Tensor Ks, CameraModelType camera_model,
    const UnscentedTransformParameters &ut_params, ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,
    const at::optional<at::Tensor> tangential_coeffs,
    const at::optional<at::Tensor> thin_prism_coeffs,
    const FThetaCameraDistortionParameters &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets, const at::Tensor flatten_ids,
    const at::optional<at::Tensor> depths_per_gauss,
    bool use_hit_distance,
    const at::Tensor render_alphas, const at::Tensor last_ids,
    const at::Tensor v_render_colors, const at::Tensor v_render_alphas,
    const at::optional<at::Tensor> v_render_depth,
    const at::optional<at::Tensor> v_render_normals,
    at::Tensor v_means, at::Tensor v_quats, at::Tensor v_scales,
    at::Tensor v_colors, at::Tensor v_opacities,
    const at::optional<at::Tensor> v_depths_per_gauss
);

} // namespace gsplat
