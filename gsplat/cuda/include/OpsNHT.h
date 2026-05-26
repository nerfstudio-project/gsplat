/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Operator declarations for the Neural Harmonic Textures (NHT) rasterization
 * path. Kept in a separate header so that core Ops.h has no NHT-specific
 * surface.
 */

#pragma once

#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>

#include "Cameras.h"
#include "ExternalDistortion.h"
#include "Lidars.h"
// Include Config.h directly so the GSPLAT_BUILD_* guards below are well-defined
#include "../csrc/Config.h"

namespace gsplat {

struct RowOffsetStructuredSpinningLidarModelParametersExt;

#if GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

// Trivial accessors for NHT compile-time constants. The values themselves
// (ENCF, NUM_ENCODING_FREQUENCIES, VERTEX_PER_PRIM) live in CommonNHT.h and
// are intentionally not exposed in this header to keep the NHT macros from
// leaking into every translation unit that pulls in Ops.h.
int nht_encoding_expansion_factor();
int nht_num_encoding_frequencies();
int nht_feature_divisor();

// NHT torch-bound rasterizer signatures. The parameter types mirror the
// standard `rasterize_to_pixels_from_world_3dgs_fwd/bwd` so torch's schema
// inference accepts them: only `int64_t` / `double` / `bool` / `Tensor` /
// `c10::intrusive_ptr<CustomClassHolder>` are allowed as argument types.
// The enum (`CameraModelType`, `ShutterType`) and float / uint32_t conversions
// happen inside the implementation (see `RasterizationNHT.cpp`).
// fwd returns:
//   (render_colors, render_alphas, render_depth, render_normals, last_ids)
//   render_depth / render_normals are empty tensors when not requested.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_fwd(
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
    const at::optional<at::Tensor> &depths_per_gauss,
    bool use_hit_distance,
    bool with_normals
);

// bwd returns:
//   (v_means, v_quats, v_scales, v_colors, v_opacities, v_depths_per_gauss)
//   v_depths_per_gauss is the (zero-init -> accumulated) gradient on the
//   forward `depths_per_gauss` input. Empty tensor when forward used the
//   hit-distance branch (gradient flows through means/scales/quats instead)
//   or when no depth was rendered.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_nht_3dgs_bwd(
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
    const at::optional<at::Tensor> &depths_per_gauss,
    bool use_hit_distance,
    const at::Tensor &render_alphas,
    const at::Tensor &last_ids,
    const at::Tensor &v_render_colors,
    const at::Tensor &v_render_alphas,
    const at::optional<at::Tensor> &v_render_depth,
    const at::optional<at::Tensor> &v_render_normals
);

#endif // GSPLAT_BUILD_NHT && GSPLAT_BUILD_3DGS

} // namespace gsplat
