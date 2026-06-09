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
#include "Intersect.h"
#include "Projection.h"
#include "Rasterization.h"
#include "SphericalHarmonics.h"
#include "TorchUtils.h"

namespace gsplat {


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
#if GSPLAT_BUILD_2DGS
    m.impl("rasterization_2dgs", to_torch_op<&rasterization_2dgs>);
#endif
}

void register_rendering_autograd_cuda_impl(torch::Library &m) {
#if GSPLAT_BUILD_2DGS
    m.impl("rasterization_2dgs", to_torch_op<&rasterization_2dgs>);
#endif
}

} // namespace gsplat
