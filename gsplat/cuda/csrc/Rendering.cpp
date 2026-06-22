/*
 * SPDX-FileCopyrightText: Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <tuple>
#include <vector>

#include "Cameras.h"
#include "Common.h"
#include "Config.h"
#include "DistributedCollectives.h"
#include "Intersect.h"
#include "Projection.h"
#include "Rasterization.h"
#include "SphericalHarmonics.h"
#include "TorchUtils.h"

namespace gsplat {

#if GSPLAT_BUILD_3DGS || GSPLAT_BUILD_3DGUT

struct Rasterization3DGSResult {
    at::Tensor render_colors;
    at::Tensor render_alphas;
    at::Tensor render_extra_signals;
    at::Tensor render_normals;
    at::Tensor means2d_absgrad;
    at::Tensor batch_ids;
    at::Tensor camera_ids;
    at::Tensor gaussian_ids;
    at::Tensor radii;
    at::Tensor means2d;
    at::Tensor depths;
    at::Tensor conics;
    at::Tensor opacities;
    at::Tensor tiles_per_gauss;
    at::Tensor isect_ids;
    at::Tensor flatten_ids;
    at::Tensor isect_offsets;
    int64_t tile_width;
    int64_t tile_height;
};

template <> struct TorchArgDef<Rasterization3DGSResult> {
    static auto to(const Rasterization3DGSResult &r) { return to_torch_args(
        r.render_colors, r.render_alphas, r.render_extra_signals,
        r.render_normals, r.means2d_absgrad, r.batch_ids, r.camera_ids,
        r.gaussian_ids, r.radii, r.means2d, r.depths, r.conics,
        r.opacities, r.tiles_per_gauss, r.isect_ids, r.flatten_ids,
        r.isect_offsets, r.tile_width, r.tile_height
    ); }
};

namespace {

void check_rasterization_3dgs_inputs(
    const at::Tensor &means, const at::optional<at::Tensor> &covars,
    const at::optional<at::Tensor> &quats,
    const at::optional<at::Tensor> &scales,
    const at::Tensor &opacities, const at::optional<at::Tensor> &colors,
    const at::Tensor &viewmats, const at::Tensor &Ks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    const at::optional<at::Tensor> &rays,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const at::optional<at::Tensor> &extra_signals,
    const at::optional<at::Tensor> &backgrounds,
    bool has_color, bool append_depth,
    int64_t sh_degree, int64_t extra_signals_sh_degree,
    bool packed, bool sparse_grad, bool absgrad,
    bool calc_compensations, bool classic_rasterize_mode,
    int64_t channel_chunk, CameraModelType camera_model,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    bool with_eval3d, bool with_ut, ShutterType rolling_shutter,
    bool global_z_order, bool use_hit_distance, bool return_normals,
    bool distributed
) {
    CHECK_INPUT(means);
    if (covars.has_value()) {
        CHECK_INPUT(covars.value());
    }
    if (quats.has_value()) {
        CHECK_INPUT(quats.value());
    }
    if (scales.has_value()) {
        CHECK_INPUT(scales.value());
    }
    CHECK_INPUT(opacities);
    if (colors.has_value()) {
        CHECK_INPUT(colors.value());
    }
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    if (extra_signals.has_value()) {
        CHECK_INPUT(extra_signals.value());
    }
    if (rays.has_value()) {
        CHECK_INPUT(rays.value());
    }
    if (viewmats_rs.has_value()) {
        CHECK_INPUT(viewmats_rs.value());
    }
    if (radial_coeffs.has_value()) {
        CHECK_INPUT(radial_coeffs.value());
    }
    if (tangential_coeffs.has_value()) {
        CHECK_INPUT(tangential_coeffs.value());
    }
    if (thin_prism_coeffs.has_value()) {
        CHECK_INPUT(thin_prism_coeffs.value());
    }

    // The distributed gather/scatter orchestration is only correct for the
    // unbatched classic 3DGS pinhole path; reject every other mode first, so the
    // error names the distributed limitation rather than a generic one.
    if (distributed) {
        TORCH_CHECK(means.dim() == 2, "distributed=True does not support batch dimensions");
        TORCH_CHECK(!sparse_grad, "distributed=True does not support sparse_grad=True");
        TORCH_CHECK(!absgrad, "distributed=True does not support absgrad=True");
        TORCH_CHECK(!with_ut, "distributed=True does not support with_ut=True");
        TORCH_CHECK(!with_eval3d, "distributed=True does not support with_eval3d=True");
        TORCH_CHECK(!return_normals, "distributed=True does not support return_normals=True");
        TORCH_CHECK(!rays.has_value(), "distributed=True does not support rays");
        TORCH_CHECK(
            camera_model == CameraModelType::PINHOLE,
            "distributed=True only supports camera_model='pinhole'"
        );
        TORCH_CHECK(global_z_order, "distributed=True does not support global_z_order=False");
        TORCH_CHECK(
            rolling_shutter == ShutterType::GLOBAL && !viewmats_rs.has_value(),
            "distributed=True does not support rolling shutter"
        );
        // The OpenCV-style coefficient tensors are optional and only present when
        // distortion is actually requested; ftheta is gated on camera_model
        // (rejected above) and the Python wrapper always passes a default object.
        TORCH_CHECK(
            !radial_coeffs.has_value() && !tangential_coeffs.has_value() &&
                !thin_prism_coeffs.has_value() &&
                !external_distortion_params.has_value(),
            "distributed=True does not support camera distortion"
        );
        TORCH_CHECK(
            !lidar_coeffs.has_value(),
            "distributed=True does not support lidar coefficients"
        );
        // Gaussians scatter per-Gaussian, so per-view (C, N, D) features have no
        // distributed route; only the per-Gaussian (N, D) layout is allowed. SH
        // coefficients ([N, K, D]) are shared across cameras and are fine.
        if (has_color && sh_degree < 0) {
            TORCH_CHECK(
                colors.has_value() && colors.value().dim() == 2,
                "distributed=True only supports per-Gaussian colors"
            );
        }
        if (extra_signals.has_value() && extra_signals_sh_degree < 0) {
            TORCH_CHECK(
                extra_signals.value().dim() == 2,
                "distributed=True only supports per-Gaussian extra signals"
            );
        }
    }

    const bool is_lidar_camera =
        camera_model == CameraModelType::LIDAR;
    TORCH_CHECK(
        has_color || append_depth,
        "Unsupported render_mode. Expected one of RGB, d, Ed, D, ED, "
        "RGB-d, RGB-Ed, RGB+D, or RGB+ED"
    );
    TORCH_CHECK(
        classic_rasterize_mode || (!with_eval3d && !with_ut),
        "3DGUT rendering only supports rasterize_mode='classic'"
    );
    TORCH_CHECK(
        !calc_compensations || (!with_eval3d && !with_ut),
        "Antialiased rasterization is only supported for classic 3DGS"
    );
    TORCH_CHECK(
        global_z_order || with_ut,
        "global_z_order can be false only if with_ut=True"
    );
    TORCH_CHECK(
        with_ut || camera_model != CameraModelType::FTHETA,
        "ftheta camera is only supported via UT, please set with_ut=True in the rasterization()"
    );
    TORCH_CHECK(
        is_lidar_camera == lidar_coeffs.has_value(),
        "Lidar coefficients must be given if and only if camera model is lidar"
    );
    TORCH_CHECK(
        !is_lidar_camera || with_ut,
        "Lidar camera model requires with_ut=True"
    );
    TORCH_CHECK(channel_chunk > 0, "channel_chunk must be > 0");
    if (with_eval3d) {
        TORCH_CHECK(
            tile_size == 8 || tile_size == 16,
            "Eval3D rasterization requires tile_size in {8, 16}, got ",
            tile_size
        );
    } else {
        TORCH_CHECK(
            tile_size == 4 || tile_size == 16,
            "Only tile_size in {4, 16} is supported for 3DGS rasterization, got ",
            tile_size
        );
    }
    TORCH_CHECK(
        !use_hit_distance || with_eval3d,
        "hit-distance render modes require with_eval3d=True"
    );
    TORCH_CHECK(
        !return_normals || with_eval3d,
        "return_normals=True requires with_eval3d=True"
    );
    TORCH_CHECK(
        !sparse_grad || packed,
        "sparse_grad is only supported when packed is True"
    );
    TORCH_CHECK(
        !sparse_grad || means.dim() == 2,
        "sparse_grad does not support batch dimensions"
    );

    TORCH_CHECK(
        means.dim() >= 2 && means.size(-1) == 3,
        "means must have shape [..., N, 3], got ",
        means.sizes()
    );
    const int64_t batch_ndim = means.dim() - 2;
    const int64_t N = means.size(-2);
    at::DimVector batch_shape(means.sizes().slice(0, batch_ndim));

    TORCH_CHECK(
        opacities.dim() == batch_ndim + 1,
        "opacities must have shape [..., N], got ",
        opacities.sizes()
    );
    TORCH_CHECK(
        viewmats.dim() == batch_ndim + 3,
        "viewmats must have shape [..., C, 4, 4], got ",
        viewmats.sizes()
    );
    TORCH_CHECK(
        Ks.dim() == batch_ndim + 3,
        "Ks must have shape [..., C, 3, 3], got ",
        Ks.sizes()
    );

    const int64_t C = viewmats.size(batch_ndim);
    at::DimVector opacities_shape(batch_shape);
    opacities_shape.append({N});
    TORCH_CHECK(
        opacities.sizes() == opacities_shape,
        "opacities must have shape [..., N], got ",
        opacities.sizes()
    );

    at::DimVector viewmats_shape(batch_shape);
    viewmats_shape.append({C, 4, 4});
    TORCH_CHECK(
        viewmats.sizes() == viewmats_shape,
        "viewmats must have shape [..., C, 4, 4], got ",
        viewmats.sizes()
    );

    at::DimVector Ks_shape(batch_shape);
    Ks_shape.append({C, 3, 3});
    TORCH_CHECK(
        Ks.sizes() == Ks_shape,
        "Ks must have shape [..., C, 3, 3], got ",
        Ks.sizes()
    );

    if (covars.has_value()) {
        TORCH_CHECK(
            !with_eval3d && !with_ut,
            "UT and Eval3D rasterization require quats and scales, not covars"
        );
        at::DimVector covars_shape(batch_shape);
        covars_shape.append({N, 6});
        at::DimVector covars_matrix_shape(batch_shape);
        covars_matrix_shape.append({N, 3, 3});
        TORCH_CHECK(
            covars.value().sizes() == covars_shape ||
                covars.value().sizes() == covars_matrix_shape,
            "covars must have shape [..., N, 3, 3] or [..., N, 6], got ",
            covars.value().sizes()
        );
    } else {
        TORCH_CHECK(quats.has_value(), "covars or quats is required");
        TORCH_CHECK(scales.has_value(), "covars or scales is required");

        at::DimVector quats_shape(batch_shape);
        quats_shape.append({N, 4});
        TORCH_CHECK(
            quats.value().sizes() == quats_shape,
            "quats must have shape [..., N, 4], got ",
            quats.value().sizes()
        );

        at::DimVector scales_shape(batch_shape);
        scales_shape.append({N, 3});
        TORCH_CHECK(
            scales.value().sizes() == scales_shape,
            "scales must have shape [..., N, 3], got ",
            scales.value().sizes()
        );
    }
    if (with_eval3d) {
        TORCH_CHECK(quats.has_value(), "Eval3D rasterization requires quats");
        TORCH_CHECK(scales.has_value(), "Eval3D rasterization requires scales");
        TORCH_CHECK(!packed, "Packed mode is not supported with Eval3D");
        TORCH_CHECK(!sparse_grad, "Sparse grad is not supported with Eval3D");
    }
    if (with_ut) {
        TORCH_CHECK(quats.has_value(), "UT rasterization requires quats");
        TORCH_CHECK(scales.has_value(), "UT rasterization requires scales");
        TORCH_CHECK(!packed, "Packed mode is not supported with UT");
        TORCH_CHECK(!sparse_grad, "Sparse grad is not supported with UT");
    }

    if (rolling_shutter == ShutterType::GLOBAL) {
        TORCH_CHECK(
            !viewmats_rs.has_value(),
            "viewmats_rs should be None for global rolling shutter"
        );
    } else {
        TORCH_CHECK(with_ut, "Rolling shutter requires with_ut=True");
        TORCH_CHECK(
            viewmats_rs.has_value(),
            "Rolling shutter requires viewmats_rs"
        );
    }
    if (viewmats_rs.has_value()) {
        TORCH_CHECK(
            viewmats_rs.value().dim() == batch_ndim + 3 &&
                viewmats_rs.value().size(batch_ndim) == C &&
                viewmats_rs.value().size(batch_ndim + 1) == 4 &&
                viewmats_rs.value().size(batch_ndim + 2) == 4,
            "viewmats_rs must have shape [..., C, 4, 4], got ",
            viewmats_rs.value().sizes()
        );
    }

    if (rays.has_value()) {
        const at::Tensor &rays_tensor = rays.value();
        TORCH_CHECK(with_eval3d, "Rays input is only supported with Eval3D");
        TORCH_CHECK(
            rays_tensor.dim() <= batch_ndim + 4,
            "rays must be broadcastable to [..., C, image_height, image_width, 6], got ",
            rays_tensor.sizes()
        );
        std::vector<int64_t> expected_rays_shape(batch_shape.begin(), batch_shape.end());
        expected_rays_shape.insert(
            expected_rays_shape.end(),
            {C, image_height, image_width, 6}
        );
        const int64_t rays_offset =
            static_cast<int64_t>(expected_rays_shape.size()) - rays_tensor.dim();
        for (int64_t dim = 0; dim < rays_tensor.dim(); ++dim) {
            const int64_t actual = rays_tensor.size(dim);
            const int64_t expected = expected_rays_shape[rays_offset + dim];
            TORCH_CHECK(
                actual == expected || actual == 1,
                "rays must be broadcastable to [..., C, image_height, image_width, 6], got ",
                rays_tensor.sizes()
            );
        }
        TORCH_CHECK(
            rays_tensor.scalar_type() == at::kFloat,
            "rays must be torch.float32"
        );
    }

    if (radial_coeffs.has_value()) {
        const at::Tensor &radial = radial_coeffs.value();
        TORCH_CHECK(with_ut, "Radial distortion requires with_ut=True");
        TORCH_CHECK(
            radial.dim() == batch_ndim + 2 && radial.size(batch_ndim) == C &&
                (radial.size(batch_ndim + 1) == 6 ||
                 radial.size(batch_ndim + 1) == 4),
            "radial_coeffs must have shape [..., C, 6] or [..., C, 4], got ",
            radial.sizes()
        );
    }
    if (tangential_coeffs.has_value()) {
        const at::Tensor &tangential = tangential_coeffs.value();
        TORCH_CHECK(with_ut, "Tangential distortion requires with_ut=True");
        TORCH_CHECK(
            tangential.dim() == batch_ndim + 2 &&
                tangential.size(batch_ndim) == C &&
                tangential.size(batch_ndim + 1) == 2,
            "tangential_coeffs must have shape [..., C, 2], got ",
            tangential.sizes()
        );
    }
    if (thin_prism_coeffs.has_value()) {
        const at::Tensor &thin_prism = thin_prism_coeffs.value();
        TORCH_CHECK(with_ut, "Thin-prism distortion requires with_ut=True");
        TORCH_CHECK(
            thin_prism.dim() == batch_ndim + 2 &&
                thin_prism.size(batch_ndim) == C &&
                thin_prism.size(batch_ndim + 1) == 4,
            "thin_prism_coeffs must have shape [..., C, 4], got ",
            thin_prism.sizes()
        );
    }
    TORCH_CHECK(
        !external_distortion_params.has_value() || with_ut,
        "External distortion requires with_ut=True"
    );

    if (has_color) {
        TORCH_CHECK(colors.has_value(), "colors must be provided for color render modes");
        const at::Tensor &colors_tensor = colors.value();
        if (sh_degree >= 0) {
            TORCH_CHECK(
                colors_tensor.dim() == 3 && colors_tensor.size(0) == N,
                "SH colors must have shape [N, K, D], got ",
                colors_tensor.sizes()
            );
            TORCH_CHECK(
                (sh_degree + 1) * (sh_degree + 1) <= colors_tensor.size(-2),
                "sh_degree requires more color SH coefficients than provided"
            );
        } else {
            const bool per_gaussian =
                colors_tensor.dim() == batch_ndim + 2 &&
                colors_tensor.size(batch_ndim) == N;
            const bool per_view =
                colors_tensor.dim() == batch_ndim + 3 &&
                colors_tensor.size(batch_ndim) == C &&
                colors_tensor.size(batch_ndim + 1) == N;
            TORCH_CHECK(
                per_gaussian || per_view,
                "colors must have shape [..., N, D] or [..., C, N, D], got ",
                colors_tensor.sizes()
            );
        }
    }
    TORCH_CHECK(
        has_color || sh_degree < 0,
        "sh_degree must be None when colors is None"
    );

    if (extra_signals.has_value()) {
        const at::Tensor &extra_tensor = extra_signals.value();
        if (extra_signals_sh_degree >= 0) {
            TORCH_CHECK(
                extra_tensor.dim() == 3 && extra_tensor.size(0) == N,
                "SH extra_signals must have shape [N, K, D], got ",
                extra_tensor.sizes()
            );
            TORCH_CHECK(
                (extra_signals_sh_degree + 1) *
                        (extra_signals_sh_degree + 1) <=
                    extra_tensor.size(-2),
                "extra_signals_sh_degree requires more SH coefficients than provided"
            );
        } else {
            const bool per_gaussian =
                extra_tensor.dim() == batch_ndim + 2 &&
                extra_tensor.size(batch_ndim) == N;
            const bool per_view =
                extra_tensor.dim() == batch_ndim + 3 &&
                extra_tensor.size(batch_ndim) == C &&
                extra_tensor.size(batch_ndim + 1) == N;
            TORCH_CHECK(
                per_gaussian || per_view,
                "extra_signals must have shape [..., N, E] or [..., C, N, E], got ",
                extra_tensor.sizes()
            );
        }
    }

    if (backgrounds.has_value()) {
        TORCH_CHECK(
            backgrounds.value().dim() == batch_ndim + 2 &&
                backgrounds.value().size(batch_ndim) == C,
            "backgrounds must have shape [..., C, D], got ",
            backgrounds.value().sizes()
        );
    }
}

at::optional<at::Tensor> normalize_covars_for_3dgs(
    const at::optional<at::Tensor> &covars
) {
    if (!covars.has_value()) {
        return c10::nullopt;
    }

    const at::Tensor &covars_tensor = covars.value();
    if (covars_tensor.size(-1) == 6) {
        return covars_tensor;
    }

    return at::stack(
        {
            covars_tensor.select(-2, 0).select(-1, 0),
            covars_tensor.select(-2, 0).select(-1, 1),
            covars_tensor.select(-2, 0).select(-1, 2),
            covars_tensor.select(-2, 1).select(-1, 1),
            covars_tensor.select(-2, 1).select(-1, 2),
            covars_tensor.select(-2, 2).select(-1, 2),
        },
        -1
    );
}

at::optional<at::Tensor> expand_rays_for_3dgs(
    const at::optional<at::Tensor> &rays,
    const at::Tensor &means,
    const at::Tensor &viewmats,
    int64_t image_height,
    int64_t image_width
) {
    if (!rays.has_value()) {
        return c10::nullopt;
    }

    const int64_t batch_ndim = means.dim() - 2;
    std::vector<int64_t> expected_shape;
    expected_shape.reserve(batch_ndim + 4);
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        expected_shape.push_back(means.size(dim));
    }
    expected_shape.insert(
        expected_shape.end(),
        {viewmats.size(batch_ndim), image_height, image_width, 6}
    );
    return rays.value().expand(expected_shape).contiguous();
}

std::vector<int64_t> batch_shape_with(
    const at::Tensor &means, std::initializer_list<int64_t> suffix
) {
    std::vector<int64_t> shape;
    const int64_t batch_ndim = means.dim() - 2;
    shape.reserve(batch_ndim + static_cast<int64_t>(suffix.size()));
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        shape.push_back(means.size(dim));
    }
    shape.insert(shape.end(), suffix.begin(), suffix.end());
    return shape;
}

at::Tensor viewmat_to_camera_position(const at::Tensor &viewmats) {
    at::Tensor R = viewmats.narrow(-2, 0, 3).narrow(-1, 0, 3);
    at::Tensor t = viewmats.narrow(-2, 0, 3).select(-1, 3);
    return -at::matmul(R.transpose(-1, -2), t.unsqueeze(-1)).squeeze(-1);
}

at::Tensor compute_classic_viewdirs(
    const at::Tensor &means, const at::Tensor &viewmats,
    const at::optional<at::Tensor> &viewmats_rs,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    const at::optional<at::Tensor> &indptr,
    int64_t B, int64_t C, int64_t N
) {
    at::Tensor campos = viewmat_to_camera_position(viewmats);
    if (viewmats_rs.has_value()) {
        campos =
            0.5 * (campos + viewmat_to_camera_position(viewmats_rs.value()));
    }
    at::Tensor dirs;
    if (gaussian_ids.has_value()) {
        const at::Tensor &gaussian_ids_tensor = gaussian_ids.value();
        at::Tensor means_flat = means.view({B, N, 3});
        at::Tensor campos_flat = campos.view({B, C, 3});
        if (B * C == 1) {
            dirs = means_flat.select(0, 0).index_select(0, gaussian_ids_tensor) -
                   campos_flat.select(0, 0).select(0, 0);
        } else if (
            indptr.has_value() &&
            static_cast<double>(gaussian_ids_tensor.numel()) /
                    static_cast<double>(B * C) >
                10000.0 &&
            campos_flat.is_cuda() &&
            campos_flat.requires_grad()) {
            dirs = at::empty({gaussian_ids_tensor.numel(), 3}, means.options());
            at::Tensor indptr_cpu =
                indptr.value().to(at::kCPU, at::kLong).contiguous();
            const int64_t *offsets = indptr_cpu.const_data_ptr<int64_t>();
            for (int64_t batch_idx = 0; batch_idx < B; ++batch_idx) {
                for (int64_t camera_idx = 0; camera_idx < C; ++camera_idx) {
                    const int64_t image_idx = batch_idx * C + camera_idx;
                    const int64_t start = offsets[image_idx];
                    const int64_t end = offsets[image_idx + 1];
                    if (start == end) {
                        continue;
                    }
                    at::Tensor gids = gaussian_ids_tensor.slice(0, start, end);
                    at::Tensor dirs_chunk =
                        means_flat.select(0, batch_idx).index_select(0, gids) -
                        campos_flat.select(0, batch_idx).select(0, camera_idx);
                    dirs.slice(0, start, end).copy_(dirs_chunk);
                }
            }
        } else {
            dirs = means_flat.index({batch_ids.value(), gaussian_ids_tensor}) -
                   campos_flat.index({batch_ids.value(), camera_ids.value()});
        }
    } else {
        dirs = means.unsqueeze(-3) - campos.unsqueeze(-2);
    }
    // The SH CUDA kernel normalizes directions internally and its backward
    // computes gradients through that normalization, so pass raw
    // (un-normalized) camera-to-Gaussian directions here.
    return dirs;
}

at::Tensor normalize_features_layout_3dgs(
    const at::Tensor &features, const at::Tensor &means,
    int64_t B, int64_t C, int64_t N,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids
) {
    const int64_t batch_ndim = means.dim() - 2;
    const int64_t channels = features.size(-1);

    const bool per_view =
        features.dim() == batch_ndim + 3 && features.size(batch_ndim) == C &&
        features.size(batch_ndim + 1) == N;
    if (per_view) {
        if (gaussian_ids.has_value()) {
            return features.view({B, C, N, channels})
                .index({batch_ids.value(), camera_ids.value(), gaussian_ids.value()});
        }
        return features;
    }

    TORCH_CHECK(
        features.dim() == batch_ndim + 2 && features.size(batch_ndim) == N,
        "features must have shape [..., N, D] or [..., C, N, D], got ",
        features.sizes()
    );
    if (gaussian_ids.has_value()) {
        return features.view({B, N, channels})
            .index({batch_ids.value(), gaussian_ids.value()});
    }

    std::vector<int64_t> expanded_shape = batch_shape_with(means, {C, N, channels});
    return features.unsqueeze(batch_ndim).expand(expanded_shape);
}

at::Tensor maybe_evaluate_feature_sh(
    int64_t degree,
    const at::Tensor &coeffs, const at::Tensor &dirs,
    const at::Tensor &valid_gaussians,
    const at::optional<at::Tensor> &gaussian_ids,
    bool clamp_after_bias
) {
    at::Tensor coeffs_for_visible = gaussian_ids.has_value()
        ? coeffs.index({gaussian_ids.value()})
        : coeffs;
    at::Tensor values =
        spherical_harmonics(degree, dirs, coeffs_for_visible, valid_gaussians);
    values = values + 0.5;
    return clamp_after_bias ? at::clamp_min(values, 0.0) : values;
}

at::Tensor append_depth_channel(
    const at::Tensor &features, const at::Tensor &depths,
    bool use_hit_distance
) {
    at::Tensor depth_channel =
        use_hit_distance ? at::zeros_like(depths) : depths;
    if (features.defined()) {
        return at::cat({features, depth_channel.unsqueeze(-1)}, -1);
    }
    return depth_channel.unsqueeze(-1);
}

at::optional<at::Tensor> append_background_depth_channel(
    const at::optional<at::Tensor> &backgrounds,
    const at::Tensor &means,
    int64_t C, bool had_features
) {
    if (!backgrounds.has_value()) {
        return c10::nullopt;
    }

    const at::Tensor &backgrounds_tensor = backgrounds.value();
    if (had_features) {
        at::Tensor zeros =
            at::zeros_like(backgrounds_tensor.narrow(-1, 0, 1));
        return at::cat({backgrounds_tensor, zeros}, -1);
    }

    std::vector<int64_t> shape = batch_shape_with(means, {C, 1});
    return at::zeros(shape, backgrounds_tensor.options());
}

at::Tensor channel_chunk_or_contiguous(
    const at::Tensor &features, int64_t start, int64_t end
) {
    const int64_t channels = features.size(-1);
    if (start == 0 && end == channels) {
        return features.is_contiguous() ? features : features.contiguous();
    }
    return features.slice(-1, start, end).contiguous();
}

at::Tensor empty_like_indices(const at::Tensor &means) {
    return at::empty({0}, means.options().dtype(at::kLong));
}

std::tuple<at::Tensor, at::Tensor> postprocess_render_colors(
    const at::Tensor &render_colors, const at::Tensor &render_alphas,
    bool has_extra_signals, bool append_depth, bool expected_depth,
    int64_t primary_channels, int64_t extra_signal_channels
) {
    TORCH_CHECK(primary_channels >= 0, "primary_channels must be >= 0");
    TORCH_CHECK(extra_signal_channels >= 0, "extra_signal_channels must be >= 0");
    TORCH_CHECK(
        !has_extra_signals || extra_signal_channels > 0,
        "extra_signal_channels must be > 0 when extra_signals are provided"
    );
    TORCH_CHECK(
        render_colors.size(-1) >=
            primary_channels + extra_signal_channels + (append_depth ? 1 : 0),
        "rendered channel layout is smaller than requested post-processing layout"
    );

    const int64_t depth_start = render_colors.size(-1) - 1;
    at::Tensor render_extra_signals = at::empty({0}, render_colors.options());
    at::Tensor final_render_colors = render_colors;

    if (has_extra_signals) {
        render_extra_signals = render_colors.slice(
            -1,
            primary_channels,
            primary_channels + extra_signal_channels
        );

        if (append_depth) {
            at::Tensor render_depth =
                render_colors.slice(-1, depth_start, depth_start + 1);
            if (expected_depth) {
                render_depth = render_depth / render_alphas.clamp_min(1e-10);
            }
            final_render_colors = at::cat(
                {render_colors.slice(-1, 0, primary_channels), render_depth},
                -1
            );
        } else {
            final_render_colors = render_colors.slice(-1, 0, primary_channels);
        }
    } else if (expected_depth) {
        at::Tensor render_depth =
            render_colors.slice(-1, depth_start, depth_start + 1) /
            render_alphas.clamp_min(1e-10);
        final_render_colors =
            at::cat({render_colors.slice(-1, 0, depth_start), render_depth}, -1);
    }

    return std::make_tuple(final_render_colors, render_extra_signals);
}

} // namespace

Rasterization3DGSResult rasterization_3dgs(
    const at::Tensor &means, const at::optional<at::Tensor> &covars,
    const at::optional<at::Tensor> &quats,
    const at::optional<at::Tensor> &scales,
    const at::Tensor &opacities, const at::optional<at::Tensor> &colors,
    const at::Tensor &viewmats, const at::Tensor &Ks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    float eps2d, float near_plane, float far_plane, float radius_clip,
    const at::optional<at::Tensor> &backgrounds,
    bool packed, bool sparse_grad, bool absgrad, bool calc_compensations,
    bool rasterize_mode_is_classic,
    CameraModelType camera_model, bool segmented, int64_t channel_chunk,
    bool has_color, int64_t sh_degree,
    const at::optional<at::Tensor> &extra_signals,
    int64_t extra_signals_sh_degree,
    bool append_depth, bool expected_depth,
    bool with_eval3d, bool with_ut,
    const at::optional<at::Tensor> &rays,
    const at::optional<at::Tensor> &viewmats_rs,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rolling_shutter,
    const at::optional<at::Tensor> &radial_coeffs,
    const at::optional<at::Tensor> &tangential_coeffs,
    const at::optional<at::Tensor> &thin_prism_coeffs,
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    bool global_z_order, bool use_hit_distance, bool return_normals,
    int64_t renderer_config,
    const at::optional<std::string> &process_group_name, int64_t world_size
) {
    DEVICE_GUARD(means);

    // A non-empty process-group name selects the multi-GPU distributed path.
    // Input validation, including distributed-mode rejection, lives in
    // check_rasterization_3dgs_inputs; the seams below gather/scatter when set.
    const bool distributed = process_group_name.has_value();

    check_rasterization_3dgs_inputs(
        means, covars, quats, scales,
        opacities, colors,
        viewmats, Ks,
        image_width, image_height, tile_size,
        rays, viewmats_rs,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        extra_signals, backgrounds,
        has_color, append_depth,
        sh_degree, extra_signals_sh_degree,
        packed, sparse_grad, absgrad,
        calc_compensations, rasterize_mode_is_classic,
        channel_chunk, camera_model,
        lidar_coeffs, external_distortion_params,
        with_eval3d, with_ut, rolling_shutter,
        global_z_order, use_hit_distance, return_normals,
        distributed
    );

    const int64_t batch_ndim = means.dim() - 2;
    const int64_t N = means.size(-2);
    int64_t C = viewmats.size(batch_ndim);
    const int64_t B = means.numel() / (N * 3);
    const int64_t I = B * C;
    at::optional<at::Tensor> projection_covars = normalize_covars_for_3dgs(covars);
    at::optional<at::Tensor> raster_rays =
        expand_rays_for_3dgs(rays, means, viewmats, image_height, image_width);

    // Seam A (distributed): all-gather cameras so this rank projects its local
    // Gaussians against every rank's cameras. `C` becomes the global camera count
    // for projection and feature assembly; Seam B resets it to the local count.
    // `I = B * C` above stays at the local image count consumed by tiling.
    const int64_t local_C = C;
    const int64_t local_N = N;
    at::Tensor proj_viewmats = viewmats;
    at::Tensor proj_Ks = Ks;
    std::vector<int64_t> N_world;
    std::vector<int64_t> C_world;
    if (distributed) {
        DistributedCameraGather gathered = gather_cameras_for_distributed(
            viewmats, Ks, local_N, local_C, world_size, process_group_name.value()
        );
        // The gather restores per-tensor shapes by slicing a packed buffer, so
        // the cameras come back as non-contiguous views; the projection requires
        // contiguous viewmats and Ks.
        proj_viewmats = gathered.viewmats.contiguous();
        proj_Ks = gathered.Ks.contiguous();
        N_world = std::move(gathered.N_world);
        C_world = std::move(gathered.C_world);
        C = gathered.global_C;
    }

    at::Tensor batch_ids;
    at::Tensor camera_ids;
    at::Tensor gaussian_ids;
    at::Tensor indptr;
    at::Tensor radii;
    at::Tensor means2d;
    at::Tensor depths;
    at::Tensor conics;
    at::Tensor compensations;

    // --- Project Gaussians to 2D -----------------------------------------
    // Project Gaussians to 2D. The packed branch returns sparse visible-Gaussian
    // indices; the dense branch returns per-camera tensors and uses empty index
    // sentinels where the packed batch/camera/gaussian indices would go.
    if (with_ut) {
        ProjectionUT3DGSFusedResult projection = projection_ut_3dgs_fused(
            means, quats.value(), scales.value(), opacities,
            viewmats, viewmats_rs, Ks,
            image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            calc_compensations, camera_model, global_z_order,
            ut_params, rolling_shutter,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_coeffs, lidar_coeffs, external_distortion_params
        );
        radii = projection.radii;
        means2d = projection.means2d;
        depths = projection.depths;
        conics = projection.conics;
        if (projection.compensations.has_value()) {
            compensations = projection.compensations.value();
        }
        batch_ids = empty_like_indices(means);
        camera_ids = empty_like_indices(means);
        gaussian_ids = empty_like_indices(means);
    }
#if GSPLAT_BUILD_3DGS
    else if (packed) {
        ProjectionEWA3DGSPackedResult projection = projection_ewa_3dgs_packed(
            means, projection_covars, quats, scales, opacities,
            proj_viewmats, proj_Ks,
            image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            sparse_grad, calc_compensations, camera_model
        );
        batch_ids = projection.batch_ids;
        camera_ids = projection.camera_ids;
        gaussian_ids = projection.gaussian_ids;
        indptr = projection.indptr;
        radii = projection.radii;
        means2d = projection.means2d;
        depths = projection.depths;
        conics = projection.conics;
        at::optional<at::Tensor> compensations_opt = projection.compensations;
        TORCH_CHECK(
            !calc_compensations || compensations_opt.has_value(),
            "packed projection did not return requested compensations"
        );
        if (compensations_opt.has_value()) {
            compensations = compensations_opt.value();
        }
    } else {
        ProjectionEWA3DGSFusedResult projection = projection_ewa_3dgs_fused(
            means, projection_covars, quats, scales, opacities,
            proj_viewmats, proj_Ks,
            image_width, image_height,
            eps2d, near_plane, far_plane, radius_clip,
            calc_compensations, camera_model
        );
        radii = projection.radii;
        means2d = projection.means2d;
        depths = projection.depths;
        conics = projection.conics;
        at::optional<at::Tensor> compensations_opt = projection.compensations;
        TORCH_CHECK(
            !calc_compensations || compensations_opt.has_value(),
            "fused projection did not return requested compensations"
        );
        if (compensations_opt.has_value()) {
            compensations = compensations_opt.value();
        }
        batch_ids = empty_like_indices(means);
        camera_ids = empty_like_indices(means);
        gaussian_ids = empty_like_indices(means);
    }
#else
    else {
        TORCH_CHECK(
            false,
            "rasterization_3dgs: the classic (non-UT) projection path requires "
            "building with GSPLAT_BUILD_3DGS; rebuild with 3DGS enabled or call "
            "rasterization with with_ut=True."
        );
    }
#endif

    // --- Prepare projected opacities --------------------------------------
    at::Tensor projected_opacities;
    at::optional<at::Tensor> image_ids;
    at::optional<at::Tensor> batch_ids_opt;
    at::optional<at::Tensor> camera_ids_opt;
    at::optional<at::Tensor> gaussian_ids_opt;
    at::optional<at::Tensor> indptr_opt;
    if (packed) {
        projected_opacities =
            opacities.view({B, N}).index({batch_ids, gaussian_ids});
        image_ids = batch_ids * C + camera_ids;
        batch_ids_opt = batch_ids;
        camera_ids_opt = camera_ids;
        gaussian_ids_opt = gaussian_ids;
        indptr_opt = indptr;
    } else {
        std::vector<int64_t> opacity_shape = batch_shape_with(means, {C, N});
        projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape);
    }
    if (calc_compensations) {
        projected_opacities = projected_opacities * compensations;
    }

    // Only Gaussians with a positive radius on every axis are valid SH inputs.
    at::Tensor valid_gaussians = radii.gt(0).all(-1);

    // --- Prepare feature channels ----------------------------------------
    // Turn colors/extra signals into [..., C, N, D] or [nnz, D] so the
    // rasterization kernels can process a uniform feature tensor.
    at::Tensor projected_features;

    // --- Fused fast-path: assemble [SH colors | direct extra | depth] in one
    // coalesced CUDA kernel (folds SH eval + the +0.5/relu color bias + extra
    // read + depth write + the cat()s). Eligible for the unpacked, non-
    // distributed, SH-color path with at most direct (non-SH) float32 extra
    // signals; numerically identical to the per-step path below. When taken, it
    // also writes the depth column, so the append-depth concat is skipped.
    bool fused_assembled = false;
    const bool fused_eligible =
        !packed && !distributed && has_color && sh_degree >= 0 &&
        colors.has_value() && colors.value().dim() == 3 &&
        means.is_cuda() && means.scalar_type() == at::kFloat &&
        means.dim() == batch_ndim + 2 && means.size(-1) == 3 &&
        (!extra_signals.has_value() ||
         (extra_signals_sh_degree < 0 &&
          extra_signals.value().scalar_type() == at::kFloat));
    if (fused_eligible) {
        at::Tensor campos = viewmat_to_camera_position(proj_viewmats);
        if (viewmats_rs.has_value()) {
            campos =
                0.5 * (campos + viewmat_to_camera_position(viewmats_rs.value()));
        }
        const int64_t Dc = colors.value().size(-1);
        const int64_t E =
            extra_signals.has_value() ? extra_signals.value().size(-1) : 0;

        bool extra_ok = true;
        bool extra_has_c = false;
        at::optional<at::Tensor> extra_in;
        if (extra_signals.has_value()) {
            const at::Tensor &es = extra_signals.value();
            if (es.dim() == batch_ndim + 2 && es.size(batch_ndim) == N &&
                es.size(-1) == E) {
                extra_has_c = false; // [*batch, N, E] broadcast over cameras
            } else if (es.dim() == batch_ndim + 3 &&
                       es.size(batch_ndim) == C &&
                       es.size(batch_ndim + 1) == N && es.size(-1) == E) {
                extra_has_c = true; // [*batch, C, N, E] per-view
            } else {
                extra_ok = false;
            }
            extra_in = es;
        }

        const bool has_depth = append_depth;
        const bool depth_is_zero = use_hit_distance;
        const bool depth_ok =
            !has_depth || depth_is_zero || (depths.scalar_type() == at::kFloat);
        at::optional<at::Tensor> depths_in =
            (has_depth && !depth_is_zero) ? at::optional<at::Tensor>(depths)
                                          : c10::nullopt;

        if (extra_ok && depth_ok && campos.is_cuda() &&
            campos.scalar_type() == at::kFloat) {
            projected_features = assemble_proj_features(
                sh_degree, B, C, N, Dc, E,
                /*color_post=*/2, // shift_relu (matches clamp_after_bias colors)
                /*extra_post=*/0, // none (direct extra is layout-only, no bias)
                has_depth, depth_is_zero, extra_has_c,
                means, campos, colors.value(), extra_in, depths_in,
                valid_gaussians
            );
            fused_assembled = true;
        }
    }

    if (!fused_assembled) {
        const bool needs_dirs =
            (has_color && sh_degree >= 0) ||
            (extra_signals.has_value() && extra_signals_sh_degree >= 0);
        at::Tensor dirs;
        if (needs_dirs) {
            dirs = compute_classic_viewdirs(
                means, proj_viewmats, viewmats_rs,
                batch_ids_opt, camera_ids_opt, gaussian_ids_opt, indptr_opt,
                B, C, N
            );
        }

        std::vector<at::Tensor> feature_list;
        if (has_color) {
            TORCH_CHECK(colors.has_value(), "colors must be provided for color render modes");
            // Colors are post-activation values unless an SH degree is provided.
            // SH color output is clamped after the +0.5 color bias.
            at::Tensor projected_colors = sh_degree >= 0
                ? maybe_evaluate_feature_sh(
                      sh_degree,
                      colors.value(), dirs, valid_gaussians,
                      gaussian_ids_opt,
                      true // clamp_after_bias
                  )
                : normalize_features_layout_3dgs(
                      colors.value(), means,
                      B, C, N,
                      batch_ids_opt, camera_ids_opt, gaussian_ids_opt);
            feature_list.push_back(projected_colors);
        }

        if (extra_signals.has_value()) {
            // Extra signals follow the same feature layout rules as colors, but
            // unlike RGB SH output they are not clamped after evaluation.
            at::Tensor projected_extra = extra_signals_sh_degree >= 0
                ? maybe_evaluate_feature_sh(
                      extra_signals_sh_degree,
                      extra_signals.value(), dirs, valid_gaussians,
                      gaussian_ids_opt,
                      false // clamp_after_bias
                  )
                : normalize_features_layout_3dgs(
                      extra_signals.value(), means,
                      B, C, N,
                      batch_ids_opt, camera_ids_opt, gaussian_ids_opt);
            feature_list.push_back(projected_extra);
        }

        if (feature_list.size() == 1) {
            projected_features = feature_list[0];
        } else if (feature_list.size() > 1) {
            projected_features = at::cat(feature_list, -1);
        }
    }

    // Record the returned metadata now: the pre-scatter, rank-local projection
    // (Gaussian axis = local N; the densification strategy indexes it by
    // gaussian_id). The scatter below mutates the local tensors that
    // rasterization consumes, so capturing into `result` here keeps the metadata
    // pre-scatter. For the non-distributed path nothing below mutates these.
    Rasterization3DGSResult result;
    result.batch_ids = batch_ids;
    result.camera_ids = camera_ids;
    result.gaussian_ids = gaussian_ids;
    result.radii = radii;
    result.means2d = means2d;
    result.depths = depths;
    result.conics = conics;
    result.opacities = projected_opacities;

    // Seam B (distributed): all-to-all scatter the projected Gaussians to the
    // camera-owning ranks, then continue tiling/rasterization over this rank's
    // cameras. `C` returns to the local camera count.
    if (distributed) {
        DistributedProjection in;
        in.radii = radii;
        in.means2d = means2d;
        in.depths = depths;
        in.conics = conics;
        in.opacities = projected_opacities;
        in.features = projected_features; // may be undefined (depth-only)
        in.batch_ids = batch_ids;
        in.camera_ids = camera_ids;
        in.gaussian_ids = gaussian_ids;
        DistributedProjection scattered = scatter_projection_for_distributed(
            packed, in, C_world, N_world, local_C, local_N,
            /*global_C=*/C, world_size, process_group_name.value()
        );
        radii = scattered.radii;
        means2d = scattered.means2d;
        depths = scattered.depths;
        conics = scattered.conics;
        projected_opacities = scattered.opacities;
        if (projected_features.defined()) {
            projected_features = scattered.features;
        }
        if (packed) {
            batch_ids = scattered.batch_ids;
            camera_ids = scattered.camera_ids;
            gaussian_ids = scattered.gaussian_ids;
            // camera_ids are now local; batch is always 0, so image id == camera id.
            image_ids = camera_ids;
            gaussian_ids_opt = gaussian_ids;
        }
        C = local_C;
    }

    // --- Append requested depth channel ----------------------------------
    // The fused fast-path already wrote the depth column into projected_features,
    // so only the background depth channel needs assembling in that case.
    at::optional<at::Tensor> render_backgrounds = backgrounds;
    if (append_depth) {
        if (fused_assembled) {
            render_backgrounds = append_background_depth_channel(
                backgrounds, means, C, /*had_features=*/true
            );
        } else {
            const bool had_features = projected_features.defined();
            projected_features =
                append_depth_channel(projected_features, depths, use_hit_distance);
            render_backgrounds =
                append_background_depth_channel(backgrounds, means, C, had_features);
        }
    }
    TORCH_CHECK(
        projected_features.defined(),
        "rasterization_3dgs requires at least one color, extra signal, or depth channel"
    );

    int64_t tile_width;
    int64_t tile_height;
    if (lidar_coeffs.has_value()) {
        tile_width = lidar_coeffs.value()->n_bins_azimuth;
        tile_height = lidar_coeffs.value()->n_bins_elevation;
    } else {
        tile_width = static_cast<int64_t>(
            std::ceil(image_width / static_cast<double>(tile_size))
        );
        tile_height = static_cast<int64_t>(
            std::ceil(image_height / static_cast<double>(tile_size))
        );
    }

    // --- Identify intersecting tiles -------------------------------------
    // Contiguity is kept close to the kernels that require compact inputs.
    at::Tensor kernel_means2d = means2d.contiguous();
    at::Tensor kernel_radii = radii.contiguous();
    at::Tensor kernel_depths = depths.contiguous();
    at::Tensor kernel_conics = conics.contiguous();
    at::Tensor kernel_opacities = projected_opacities.contiguous();
    at::optional<at::Tensor> kernel_image_ids =
        contiguous_optional(image_ids);
    at::optional<at::Tensor> kernel_gaussian_ids =
        contiguous_optional(gaussian_ids_opt);
    at::optional<at::Tensor> intersect_conics =
        as_optional_tensor(with_ut ? at::Tensor{} : kernel_conics);
    at::optional<at::Tensor> intersect_opacities =
        as_optional_tensor(with_ut ? at::Tensor{} : kernel_opacities);
    TileIntersectResult isects =
        lidar_coeffs.has_value()
        ? intersect_tile_lidar(
              lidar_coeffs.value(),
              kernel_means2d, kernel_radii, kernel_depths,
              kernel_image_ids, kernel_gaussian_ids,
              I,
              true, // sort
              segmented)
        : intersect_tile(
              kernel_means2d, kernel_radii, kernel_depths,
              intersect_conics, intersect_opacities,
              kernel_image_ids, kernel_gaussian_ids,
              I, tile_size, tile_width, tile_height,
              true, // sort
              segmented);

    at::Tensor isect_offsets =
        intersect_offset(isects.isect_ids, I, tile_width, tile_height);
    std::vector<int64_t> isect_offsets_shape =
        batch_shape_with(means, {C, tile_height, tile_width});
    isect_offsets = isect_offsets.reshape(isect_offsets_shape);

    // --- Rasterize feature chunks ----------------------------------------
    // Render at most channel_chunk feature channels per raster call, then
    // concatenate the chunks.
    std::vector<at::Tensor> render_color_chunks;
    at::Tensor render_alphas;
    at::Tensor render_normals = at::empty({0}, means.options());
    at::Tensor absgrad_holder;
    at::Tensor raster_isect_offsets = isect_offsets.contiguous();
    at::Tensor raster_flatten_ids = isects.flatten_ids.contiguous();
    const int64_t channels = projected_features.size(-1);
    for (int64_t start = 0; start < channels; start += channel_chunk) {
        const int64_t end = std::min(start + channel_chunk, channels);
        at::Tensor features_chunk =
            channel_chunk_or_contiguous(projected_features, start, end);
        at::optional<at::Tensor> backgrounds_chunk;
        if (render_backgrounds.has_value()) {
            backgrounds_chunk =
                channel_chunk_or_contiguous(render_backgrounds.value(), start, end);
        }

        if (with_eval3d) {
            RasterizeToPixelsFromWorld3DGSResult raster =
                rasterize_to_pixels_from_world_3dgs(
                means, quats.value(), scales.value(),
                features_chunk, kernel_opacities, backgrounds_chunk,
                c10::nullopt,
                image_width, image_height, tile_size,
                viewmats, viewmats_rs, Ks,
                static_cast<int64_t>(camera_model), ut_params,
                static_cast<int64_t>(rolling_shutter),
                raster_rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                ftheta_coeffs, lidar_coeffs, external_distortion_params,
                raster_isect_offsets, raster_flatten_ids,
                false, // return_sample_counts
                use_hit_distance,
                return_normals && start == 0,
                renderer_config,
                false, // return_last_ids
                false // unsafe_masked_tile_outputs (safe default: masked tiles write defined outputs)
            );
            render_color_chunks.push_back(raster.renders);
            if (!render_alphas.defined()) {
                render_alphas = raster.alphas;
            }
            if (return_normals && start == 0 && raster.normals.has_value()) {
                render_normals = raster.normals.value();
            }
        }
#if GSPLAT_BUILD_3DGS
        else {
            RasterizeToPixels3DGSResult raster = rasterize_to_pixels_3dgs(
                kernel_means2d, kernel_conics,
                features_chunk, kernel_opacities, backgrounds_chunk,
                c10::nullopt,
                image_width, image_height, tile_size,
                raster_isect_offsets, raster_flatten_ids,
                packed, absgrad
            );
            render_color_chunks.push_back(raster.renders);
            if (!render_alphas.defined()) {
                render_alphas = raster.alphas;
            }
            // The observable absgrad holder is the one from the final chunk's
            // rasterize call.
            absgrad_holder = raster.means2d_absgrad;
        }
#else
        else {
            TORCH_CHECK(
                false,
                "rasterization_3dgs: the classic (non-eval3d) rasterization path "
                "requires building with GSPLAT_BUILD_3DGS; rebuild with 3DGS enabled "
                "or call rasterization with with_eval3d=True."
            );
        }
#endif
    }

    // --- Reassemble output channels --------------------------------------
    at::Tensor render_colors = render_color_chunks.size() == 1
        ? render_color_chunks[0]
        : at::cat(render_color_chunks, -1);
    const int64_t primary_channels =
        (has_color && colors.has_value() && colors.value().dim() > 0)
            ? colors.value().size(-1)
            : 0;
    const int64_t extra_signal_channels =
        (extra_signals.has_value() && extra_signals.value().dim() > 0)
            ? extra_signals.value().size(-1)
            : 0;
    at::Tensor render_extra_signals;
    std::tie(render_colors, render_extra_signals) = postprocess_render_colors(
        render_colors,
        render_alphas,
        extra_signals.has_value(),
        append_depth,
        expected_depth,
        primary_channels,
        extra_signal_channels
    );

    // --- Fill in the render and tiling outputs (metadata was recorded above,
    // pre-scatter) and return ---------------------------------------------
    result.render_colors = render_colors;
    result.render_alphas = render_alphas;
    result.render_extra_signals = render_extra_signals;
    result.render_normals = render_normals;
    result.means2d_absgrad = absgrad_holder;
    result.tiles_per_gauss = isects.tiles_per_gauss;
    result.isect_ids = isects.isect_ids;
    result.flatten_ids = isects.flatten_ids;
    result.isect_offsets = isect_offsets;
    result.tile_width = tile_width;
    result.tile_height = tile_height;
    return result;
}

#endif // GSPLAT_BUILD_3DGS || GSPLAT_BUILD_3DGUT

#if GSPLAT_BUILD_2DGS

struct Rasterization2DGSResult {
    at::Tensor render_colors;
    at::Tensor render_alphas;
    at::Tensor render_normals;
    at::optional<at::Tensor> render_normals_from_depth;
    at::Tensor render_distort;
    at::Tensor render_median;
    at::Tensor means2d_absgrad;
    at::optional<at::Tensor> camera_ids;
    at::optional<at::Tensor> gaussian_ids;
    at::Tensor radii;
    at::Tensor means2d;
    at::Tensor depths;
    at::Tensor ray_transforms;
    at::Tensor opacities;
    at::Tensor normals;
    at::Tensor tiles_per_gauss;
    at::Tensor isect_ids;
    at::Tensor flatten_ids;
    at::Tensor isect_offsets;
    at::Tensor densify;
    int64_t tile_width;
    int64_t tile_height;
    int64_t n_cameras;
};

template <> struct TorchArgDef<Rasterization2DGSResult> {
    static auto to(const Rasterization2DGSResult &r) { return to_torch_args(
        r.render_colors, r.render_alphas, r.render_normals,
        r.render_normals_from_depth, r.render_distort, r.render_median,
        r.means2d_absgrad, r.camera_ids, r.gaussian_ids, r.radii,
        r.means2d, r.depths, r.ray_transforms, r.opacities, r.normals,
        r.tiles_per_gauss, r.isect_ids, r.flatten_ids, r.isect_offsets,
        r.densify, r.tile_width, r.tile_height, r.n_cameras
    ); }
};

namespace {

// Render-mode flag derivation from the mode string.
bool render_mode_has_color(const std::string &mode) {
    return mode == "RGB" || mode == "RGB-d" || mode == "RGB-Ed" ||
           mode == "RGB+D" || mode == "RGB+ED";
}
bool render_mode_has_depth(const std::string &mode) {
    return mode == "D" || mode == "ED" || mode == "RGB+D" || mode == "RGB+ED";
}
bool render_mode_has_hit_distance(const std::string &mode) {
    return mode == "d" || mode == "Ed" || mode == "RGB-d" || mode == "RGB-Ed";
}
bool render_mode_has_depth_channel(const std::string &mode) {
    return render_mode_has_depth(mode) || render_mode_has_hit_distance(mode);
}
bool render_mode_has_expected_depth(const std::string &mode) {
    return mode == "Ed" || mode == "ED" || mode == "RGB-Ed" || mode == "RGB+ED";
}

// Shape = batch dims of `means` (all but last two) followed by `suffix`.
std::vector<int64_t> batch_shape_with_2dgs(
    const at::Tensor &means, std::initializer_list<int64_t> suffix
) {
    std::vector<int64_t> shape;
    const int64_t batch_ndim = means.dim() - 2;
    shape.reserve(batch_ndim + static_cast<int64_t>(suffix.size()));
    for (int64_t dim = 0; dim < batch_ndim; ++dim) {
        shape.push_back(means.size(dim));
    }
    shape.insert(shape.end(), suffix.begin(), suffix.end());
    return shape;
}

void check_rasterization_2dgs_inputs(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &opacities, const at::Tensor &colors,
    const at::Tensor &viewmats, const at::Tensor &Ks,
    int64_t sh_degree, bool distloss, bool append_depth
) {
    TORCH_CHECK(means.dim() >= 2, "means must have shape [..., N, 3], got ", means.sizes());
    const int64_t N = means.size(-2);
    TORCH_CHECK(means.size(-1) == 3, "means must have shape [..., N, 3], got ", means.sizes());
    TORCH_CHECK(
        quats.size(-2) == N && quats.size(-1) == 4,
        "quats must have shape [..., N, 4], got ", quats.sizes()
    );
    TORCH_CHECK(
        scales.size(-2) == N && scales.size(-1) == 3,
        "scales must have shape [..., N, 3], got ", scales.sizes()
    );
    TORCH_CHECK(
        opacities.size(-1) == N, "opacities must have shape [..., N], got ", opacities.sizes()
    );
    TORCH_CHECK(
        viewmats.size(-2) == 4 && viewmats.size(-1) == 4,
        "viewmats must have shape [..., C, 4, 4], got ", viewmats.sizes()
    );
    TORCH_CHECK(
        Ks.size(-2) == 3 && Ks.size(-1) == 3,
        "Ks must have shape [..., C, 3, 3], got ", Ks.sizes()
    );
    if (sh_degree >= 0) {
        TORCH_CHECK(
            colors.dim() == 3 && colors.size(0) == N,
            "SH coefficients must have shape [N, K, D], got ", colors.sizes()
        );
        TORCH_CHECK(
            (sh_degree + 1) * (sh_degree + 1) <= colors.size(-2),
            "SH degree ", sh_degree, " too high for ", colors.size(-2), " coefficient bands"
        );
    }
    TORCH_CHECK(!distloss || append_depth, "distloss requires a depth render mode");
}

// Camera-to-Gaussian directions for SH evaluation, with the camera position
// taken as inverse(viewmats)[..., :3, 3]. Directions are left un-normalized:
// the SH kernel normalizes them internally and differentiates through it.
at::Tensor compute_viewdirs_2dgs(
    const at::Tensor &means, const at::Tensor &viewmats,
    const at::optional<at::Tensor> &batch_ids,
    const at::optional<at::Tensor> &camera_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    int64_t B, int64_t C, int64_t N
) {
    at::Tensor camtoworlds = at::linalg_inv(viewmats);
    at::Tensor campos = camtoworlds.narrow(-2, 0, 3).select(-1, 3); // [..., C, 3]
    if (gaussian_ids.has_value()) {
        at::Tensor means_flat = means.reshape({B, N, 3});
        at::Tensor campos_flat = campos.reshape({B, C, 3});
        return means_flat.index({batch_ids.value(), gaussian_ids.value()}) -
               campos_flat.index({batch_ids.value(), camera_ids.value()});
    }
    return means.unsqueeze(-3) - campos.unsqueeze(-2); // [..., C, N, 3]
}

// Evaluate SH colors, add the +0.5 bias, and clamp to non-negative values.
at::Tensor evaluate_sh_colors_2dgs(
    int64_t degree, const at::Tensor &coeffs, const at::Tensor &dirs,
    const at::Tensor &valid_gaussians, const at::optional<at::Tensor> &gaussian_ids
) {
    at::Tensor coeffs_for_visible = gaussian_ids.has_value()
        ? coeffs.index({gaussian_ids.value()})
        : coeffs;
    at::Tensor values =
        spherical_harmonics(degree, dirs, coeffs_for_visible, valid_gaussians);
    return at::clamp_min(values + 0.5, 0.0);
}

// Unproject a z-depth map to world-space points. Uses the z-depth convention,
// so per-pixel ray directions are not normalized. depths [..., H, W, 1],
// camtoworlds [..., 4, 4], Ks [..., 3, 3]; returns [..., H, W, 3].
at::Tensor depth_to_points_2dgs(
    const at::Tensor &depths, const at::Tensor &camtoworlds, const at::Tensor &Ks
) {
    const int64_t height = depths.size(-3);
    const int64_t width = depths.size(-2);
    auto opts = depths.options();
    std::vector<at::Tensor> grid =
        at::meshgrid({at::arange(width, opts), at::arange(height, opts)}, "xy");
    const at::Tensor &x = grid[0]; // [H, W]
    const at::Tensor &y = grid[1]; // [H, W]

    at::Tensor fx = Ks.select(-2, 0).select(-1, 0).unsqueeze(-1).unsqueeze(-1); // [..., 1, 1]
    at::Tensor fy = Ks.select(-2, 1).select(-1, 1).unsqueeze(-1).unsqueeze(-1);
    at::Tensor cx = Ks.select(-2, 0).select(-1, 2).unsqueeze(-1).unsqueeze(-1);
    at::Tensor cy = Ks.select(-2, 1).select(-1, 2).unsqueeze(-1).unsqueeze(-1);

    at::Tensor dx = (x - cx + 0.5) / fx; // [..., H, W]
    at::Tensor dy = (y - cy + 0.5) / fy;
    at::Tensor camera_dirs = at::stack({dx, dy}, -1);             // [..., H, W, 2]
    camera_dirs = at::constant_pad_nd(camera_dirs, {0, 1}, 1.0);  // [..., H, W, 3]

    at::Tensor R = camtoworlds.narrow(-2, 0, 3).narrow(-1, 0, 3); // [..., 3, 3]
    at::Tensor directions = at::einsum("...ij,...hwj->...hwi", {R, camera_dirs});
    at::Tensor origins = camtoworlds.narrow(-2, 0, 3).select(-1, 3); // [..., 3]
    return origins.unsqueeze(-2).unsqueeze(-2) + depths * directions; // [..., H, W, 3]
}

// Surface normals from a z-depth map: unproject to points, then normalize the
// cross product of neighbouring-pixel point differences (padded back to H×W).
at::Tensor depth_to_normal_2dgs(
    const at::Tensor &depths, const at::Tensor &camtoworlds, const at::Tensor &Ks
) {
    at::Tensor points = depth_to_points_2dgs(depths, camtoworlds, Ks); // [..., H, W, 3]
    const int64_t h_dim = points.dim() - 3;
    const int64_t w_dim = points.dim() - 2;
    const int64_t H = points.size(h_dim);
    const int64_t W = points.size(w_dim);
    at::Tensor dx = points.narrow(h_dim, 2, H - 2).narrow(w_dim, 1, W - 2) -
                    points.narrow(h_dim, 0, H - 2).narrow(w_dim, 1, W - 2);
    at::Tensor dy = points.narrow(h_dim, 1, H - 2).narrow(w_dim, 2, W - 2) -
                    points.narrow(h_dim, 1, H - 2).narrow(w_dim, 0, W - 2);
    at::Tensor normals = at::linalg_cross(dx, dy, -1);
    at::Tensor norm = at::sqrt((normals * normals).sum(-1, true)).clamp_min(1e-12);
    normals = normals / norm;
    // F.pad(normals, (0, 0, 1, 1, 1, 1)): pad W and H dims by 1 on each side.
    return at::constant_pad_nd(normals, {0, 0, 1, 1, 1, 1}, 0.0); // [..., H, W, 3]
}

} // namespace

Rasterization2DGSResult rasterization_2dgs(
    const at::Tensor &means, const at::Tensor &quats, const at::Tensor &scales,
    const at::Tensor &opacities, const at::Tensor &colors,
    const at::Tensor &viewmats, const at::Tensor &Ks,
    int64_t image_width, int64_t image_height, int64_t tile_size,
    float eps2d, float near_plane, float far_plane, float radius_clip,
    const at::optional<at::Tensor> &backgrounds,
    bool packed, bool sparse_grad, bool absgrad, bool distloss,
    at::optional<int64_t> sh_degree, const std::string &render_mode,
    const std::string &depth_mode
) {
    DEVICE_GUARD(means);

    const bool has_color = render_mode_has_color(render_mode);
    const bool append_depth = render_mode_has_depth_channel(render_mode);
    const bool expected_depth = render_mode_has_expected_depth(render_mode);
    const bool compute_normals_from_depth =
        render_mode_has_depth(render_mode) && has_color;
    const bool depth_mode_is_median = depth_mode == "median";
    const int64_t sh_degree_value = sh_degree.value_or(-1);

    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(opacities);
    CHECK_INPUT(colors);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    if (backgrounds.has_value()) {
        CHECK_INPUT(backgrounds.value());
    }
    check_rasterization_2dgs_inputs(
        means, quats, scales, opacities, colors, viewmats, Ks,
        sh_degree_value, distloss, append_depth
    );

    const int64_t batch_ndim = means.dim() - 2;
    const int64_t N = means.size(-2);
    const int64_t C = viewmats.size(batch_ndim);
    const int64_t B = means.numel() / (N * 3);
    const int64_t I = B * C;

    // --- Project 2D Gaussians (ray-splat transforms) ----------------------
    at::Tensor camera_ids;
    at::Tensor gaussian_ids;
    at::Tensor radii;
    at::Tensor means2d;
    at::Tensor depths;
    at::Tensor ray_transforms;
    at::Tensor normals;
    at::Tensor projected_opacities;
    at::optional<at::Tensor> batch_ids_opt;
    at::optional<at::Tensor> camera_ids_opt;
    at::optional<at::Tensor> gaussian_ids_opt;
    at::optional<at::Tensor> image_ids;
    if (packed) {
        Projection2DGSPackedResult projection = projection_2dgs_packed(
            means, quats, scales, viewmats, Ks,
            image_width, image_height, near_plane, far_plane, radius_clip,
            sparse_grad
        );
        at::Tensor batch_ids = projection.batch_ids;
        camera_ids = projection.camera_ids;
        gaussian_ids = projection.gaussian_ids;
        radii = projection.radii;
        means2d = projection.means2d;
        depths = projection.depths;
        ray_transforms = projection.ray_transforms;
        normals = projection.normals;
        projected_opacities =
            opacities.reshape({B, N}).index({batch_ids, gaussian_ids});
        image_ids = batch_ids * C + camera_ids;
        batch_ids_opt = batch_ids;
        camera_ids_opt = camera_ids;
        gaussian_ids_opt = gaussian_ids;
    } else {
        Projection2DGSFusedResult projection = projection_2dgs_fused(
            means, quats, scales, viewmats, Ks,
            image_width, image_height, eps2d, near_plane, far_plane, radius_clip
        );
        radii = projection.radii;
        means2d = projection.means2d;
        depths = projection.depths;
        ray_transforms = projection.ray_transforms;
        normals = projection.normals;
        std::vector<int64_t> opacity_shape = batch_shape_with_2dgs(means, {C, N});
        projected_opacities = opacities.unsqueeze(batch_ndim).expand(opacity_shape);
    }

    // Gradient accumulator for densification (surfaced via meta["gradient_2dgs"]).
    at::Tensor densify = at::zeros_like(means2d).set_requires_grad(true);

    // --- Identify intersecting tiles --------------------------------------
    const int64_t tile_width = static_cast<int64_t>(
        std::ceil(image_width / static_cast<double>(tile_size))
    );
    const int64_t tile_height = static_cast<int64_t>(
        std::ceil(image_height / static_cast<double>(tile_size))
    );
    TileIntersectResult isects = intersect_tile(
        means2d.contiguous(), radii.contiguous(), depths.contiguous(),
        c10::nullopt, c10::nullopt,
        contiguous_optional(image_ids), contiguous_optional(gaussian_ids_opt),
        I, tile_size, tile_width, tile_height,
        true, // sort
        false // segmented
    );
    at::Tensor isect_offsets =
        intersect_offset(isects.isect_ids, I, tile_width, tile_height);
    isect_offsets = isect_offsets.reshape(
        batch_shape_with_2dgs(means, {C, tile_height, tile_width})
    );

    // --- Assemble feature channels (SH colors / post-activation + depth) ---
    at::Tensor feature;
    if (has_color) {
        if (sh_degree_value >= 0) {
            at::Tensor valid_gaussians = radii.gt(0).all(-1);
            at::Tensor dirs = compute_viewdirs_2dgs(
                means, viewmats, batch_ids_opt, camera_ids_opt, gaussian_ids_opt,
                B, C, N
            );
            feature = evaluate_sh_colors_2dgs(
                sh_degree_value, colors, dirs, valid_gaussians, gaussian_ids_opt
            );
        } else {
            feature = colors;
        }
    }
    at::optional<at::Tensor> raster_backgrounds = backgrounds;
    if (append_depth) {
        if (has_color) {
            feature = at::cat({feature, depths.unsqueeze(-1)}, -1);
            if (backgrounds.has_value()) {
                raster_backgrounds = at::cat(
                    {backgrounds.value(),
                     at::zeros_like(backgrounds.value().narrow(-1, 0, 1))},
                    -1
                );
            }
        } else {
            feature = depths.unsqueeze(-1);
        }
    }
    TORCH_CHECK(
        feature.defined(),
        "rasterization_2dgs requires at least one color or depth channel"
    );

    // --- Rasterize --------------------------------------------------------
    RasterizeToPixels2DGSResult raster = rasterize_to_pixels_2dgs(
        means2d, ray_transforms, feature, projected_opacities, normals, densify,
        raster_backgrounds, c10::nullopt,
        image_width, image_height, tile_size,
        isect_offsets.contiguous(), isects.flatten_ids.contiguous(),
        packed, absgrad, distloss
    );
    at::Tensor render_colors = raster.renders;
    at::Tensor render_alphas = raster.alphas;
    at::Tensor render_normals = raster.render_normals; // camera space
    at::Tensor render_distort = raster.render_distort;
    at::Tensor render_median = raster.render_median;
    at::Tensor means2d_absgrad = raster.means2d_absgrad;

    // --- Post-process render-mode outputs ---------------------------------
    // Normalize the accumulated depth channel by alpha for expected-depth modes.
    if (expected_depth) {
        const int64_t channels = render_colors.size(-1);
        at::Tensor expected = render_colors.narrow(-1, channels - 1, 1) /
                              render_alphas.clamp_min(1e-10);
        render_colors = channels > 1
            ? at::cat({render_colors.narrow(-1, 0, channels - 1), expected}, -1)
            : expected;
    }

    // Surface normals derived from the rendered depth (world space).
    at::optional<at::Tensor> render_normals_from_depth;
    if (compute_normals_from_depth) {
        at::Tensor depth_for_normal = depth_mode_is_median
            ? render_median
            : render_colors.narrow(-1, render_colors.size(-1) - 1, 1);
        at::Tensor camtoworlds = at::linalg_inv(viewmats);
        render_normals_from_depth =
            depth_to_normal_2dgs(depth_for_normal, camtoworlds, Ks).squeeze(0);
    }

    // Rotate the rendered (camera-space) normals into world space.
    at::Tensor camtoworlds_rotation =
        at::linalg_inv(viewmats).narrow(-2, 0, 3).narrow(-1, 0, 3); // [..., C, 3, 3]
    render_normals =
        at::einsum("...ij,...hwj->...hwi", {camtoworlds_rotation, render_normals});

    return {
        .render_colors = render_colors,
        .render_alphas = render_alphas,
        .render_normals = render_normals,
        .render_normals_from_depth = render_normals_from_depth,
        .render_distort = render_distort,
        .render_median = render_median,
        .means2d_absgrad = means2d_absgrad,
        .camera_ids = camera_ids_opt,
        .gaussian_ids = gaussian_ids_opt,
        .radii = radii,
        .means2d = means2d,
        .depths = depths,
        .ray_transforms = ray_transforms,
        .opacities = projected_opacities,
        .normals = normals,
        .tiles_per_gauss = isects.tiles_per_gauss,
        .isect_ids = isects.isect_ids,
        .flatten_ids = isects.flatten_ids,
        .isect_offsets = isect_offsets,
        .densify = densify,
        .tile_width = tile_width,
        .tile_height = tile_height,
        .n_cameras = C,
    };
}

#endif // GSPLAT_BUILD_2DGS

void register_rendering_cuda_impl(torch::Library &m) {
#if GSPLAT_BUILD_3DGS || GSPLAT_BUILD_3DGUT
    m.impl("rasterization_3dgs", to_torch_op<&rasterization_3dgs>);
#endif
#if GSPLAT_BUILD_2DGS
    m.impl("rasterization_2dgs", to_torch_op<&rasterization_2dgs>);
#endif
}

void register_rendering_autograd_cuda_impl(torch::Library &m) {
#if GSPLAT_BUILD_3DGS || GSPLAT_BUILD_3DGUT
    m.impl("rasterization_3dgs", to_torch_op<&rasterization_3dgs>);
#endif
#if GSPLAT_BUILD_2DGS
    m.impl("rasterization_2dgs", to_torch_op<&rasterization_2dgs>);
#endif
}

} // namespace gsplat
