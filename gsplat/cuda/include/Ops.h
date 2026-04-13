/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

// A collection of operators for gsplat
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/ivalue.h>
#include <string>

#include "Lidars.h"
#include "Cameras.h"
#include "Common.h"
#include "ExternalDistortion.h"

#include <optional>

namespace gsplat {

struct RowOffsetStructuredSpinningLidarModelParametersExt : public torch::CustomClassHolder
{
    RowOffsetStructuredSpinningLidarModelParametersExt() = default;

    RowOffsetStructuredSpinningLidarModelParametersExt(
        at::Tensor row_elevations_rad,
        at::Tensor column_azimuths_rad,
        at::Tensor row_azimuth_offsets_rad,
        SpinningDirection spinning_direction,
        float spinning_frequency_hz,
        c10::intrusive_ptr<FOV> fov_vert_rad,
        c10::intrusive_ptr<FOV> fov_horiz_rad,
        float fov_eps_rad,
        at::Tensor angles_to_columns_map,
        int n_bins_azimuth,
        int n_bins_elevation,
        at::Tensor cdf_elevation,
        at::Tensor cdf_dense_ray_mask,
        at::Tensor tiles_pack_info,
        at::Tensor tiles_to_elements_map
    )
        : row_elevations_rad(std::move(row_elevations_rad)),
          column_azimuths_rad(std::move(column_azimuths_rad)),
          row_azimuth_offsets_rad(std::move(row_azimuth_offsets_rad)),
          spinning_direction(spinning_direction),
          spinning_frequency_hz(spinning_frequency_hz),
          fov_vert_rad(std::move(fov_vert_rad)),
          fov_horiz_rad(std::move(fov_horiz_rad)),
          fov_eps_rad(fov_eps_rad),
          angles_to_columns_map(std::move(angles_to_columns_map)),
          n_bins_azimuth(n_bins_azimuth),
          n_bins_elevation(n_bins_elevation),
          cdf_elevation(cdf_elevation),
          cdf_dense_ray_mask(cdf_dense_ray_mask),
          tiles_pack_info(std::move(tiles_pack_info)),
          tiles_to_elements_map(std::move(tiles_to_elements_map))
    {}

    int n_rows() const { return this->row_elevations_rad.size(0); }
    int n_columns() const { return this->column_azimuths_rad.size(0); }

    // Actual parameters directly related to the lidar model
    at::Tensor row_elevations_rad;
    at::Tensor column_azimuths_rad;
    at::Tensor row_azimuth_offsets_rad;

    SpinningDirection spinning_direction;
    float spinning_frequency_hz;

    // Now some values computed elsewhere from the above parameters.
    c10::intrusive_ptr<FOV> fov_vert_rad;
    c10::intrusive_ptr<FOV> fov_horiz_rad;
    float fov_eps_rad;

    at::Tensor angles_to_columns_map;

    // Lidar Tiling info
    int n_bins_azimuth;
    int n_bins_elevation;
    at::Tensor cdf_elevation;
    at::Tensor cdf_dense_ray_mask;
    at::Tensor tiles_pack_info;
    at::Tensor tiles_to_elements_map;
    int cdf_resolution_elevation() const { return this->cdf_dense_ray_mask.size(-2)-1; }
    int cdf_resolution_azimuth() const { return this->cdf_dense_ray_mask.size(-1)-1; }
};

// null operator for tutorial. Does nothing.
at::Tensor null(const at::Tensor input);

// Project 3D gaussians (in camera space) to 2D image planes with EWA splatting.
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd(
    const at::Tensor &means,  // [..., C, N, 3]
    const at::Tensor &covars, // [..., C, N, 3, 3]
    const at::Tensor &Ks,     // [..., C, 3, 3]
    int64_t width,
    int64_t height,
    int64_t camera_model
);
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd(
    const at::Tensor &means,  // [..., C, N, 3]
    const at::Tensor &covars, // [..., C, N, 3, 3]
    const at::Tensor &Ks,     // [..., C, 3, 3]
    int64_t width,
    int64_t height,
    int64_t camera_model,
    at::Tensor &v_means2d, // [..., C, N, 2]
    at::Tensor &v_covars2d // [..., C, N, 2, 2]
);

// Fuse the following operations:
// 1. compute covar from {quats, scales}
// 2. transform 3D gaussians from world space to camera space
//    - w/ near far plane check
// 3. projection camera space 3D gaussians to 2D image planes with EWA
// splatting.
//    - w/ minimum radius check
// 4. add a bit blurring to the 2D gaussians for anti-aliasing.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_fused_fwd(
    const at::Tensor &means,                   // [..., N, 3]
    const at::optional<at::Tensor> &covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> &scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats,                // [..., C, 4, 4]
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor &means,                // [..., N, 3]
    const at::optional<at::Tensor> &covars, // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> &scales, // [..., N, 3] optional
    const at::Tensor &viewmats,             // [..., C, 4, 4]
    const at::Tensor &Ks,                   // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    int64_t camera_model,
    // fwd outputs
    const at::Tensor &radii,                       // [..., C, N, 2]
    const at::Tensor &conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> &compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor &v_means2d,                     // [..., C, N, 2]
    const at::Tensor &v_depths,                      // [..., C, N]
    const at::Tensor &v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> &v_compensations, // [..., C, N] optional
    bool viewmats_requires_grad
);

// On top of fusing the operations like `projection_ewa_3dgs_fused_{fwd, bwd}`,
// The packed version compresses the [C, N, D] tensors (both intermidiate and
// output) into a jagged format [nnz, D], leveraging the sparsity of these
// tensors.
//
// This could lead to less memory usage than `_fused_{fwd, bwd}` if the level of
// sparsity is high, i.e., most of the gaussians are not in the camera frustum.
// But at the cost of slightly slower speed.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ewa_3dgs_packed_fwd(
    const at::Tensor &means,                   // [..., N, 3]
    const at::optional<at::Tensor> &covars,    // [..., N, 6] optional
    const at::optional<at::Tensor> &quats,     // [..., N, 4] optional
    const at::optional<at::Tensor> &scales,    // [..., N, 3] optional
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats,                // [..., C, 4, 4]
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor &means,                // [..., N, 3]
    const at::optional<at::Tensor> &covars, // [..., N, 6]
    const at::optional<at::Tensor> &quats,  // [..., N, 4]
    const at::optional<at::Tensor> &scales, // [..., N, 3]
    const at::Tensor &viewmats,             // [..., C, 4, 4]
    const at::Tensor &Ks,                   // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    int64_t camera_model,
    // fwd outputs
    const at::Tensor &batch_ids,                   // [nnz]
    const at::Tensor &camera_ids,                  // [nnz]
    const at::Tensor &gaussian_ids,                // [nnz]
    const at::Tensor &conics,                      // [nnz, 3]
    const at::optional<at::Tensor> &compensations, // [nnz] optional
    // grad outputs
    const at::Tensor &v_means2d,                     // [nnz, 2]
    const at::Tensor &v_depths,                      // [nnz]
    const at::Tensor &v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> &v_compensations, // [nnz] optional
    bool viewmats_requires_grad,
    bool sparse_grad
);

// Sphereical harmonics
at::Tensor spherical_harmonics_fwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., 3]
    const at::Tensor &coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> &masks // [...]
);
std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., 3]
    const at::Tensor &coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> &masks, // [...]
    const at::Tensor &v_colors,            // [..., 3]
    bool compute_v_dirs
);

// Fused Adam that supports a valid mask to skip updating certain parameters.
// Note skipping is not equivalent with zeroing out the gradients, which will
// still update parameters with momentum.
void adam(
    at::Tensor &param,                    // [..., D]
    const at::Tensor &param_grad,         // [..., D]
    at::Tensor &exp_avg,                  // [..., D]
    at::Tensor &exp_avg_sq,               // [..., D]
    const at::optional<at::Tensor> &valid, // [...]
    double lr,
    double b1,
    double b2,
    double eps
);

// GS Tile Intersection
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor &means2d,                           // [..., N, 2] or [nnz, 2]
    const at::Tensor &radii,                             // [..., N, 2] or [nnz, 2]
    const at::Tensor &depths,                            // [..., N] or [nnz]
    const at::optional<at::Tensor> &conics,              // [..., N, 3] or [nnz, 3] 
    const at::optional<at::Tensor> &opacities,           // [..., N] or [nnz]        
    const at::optional<at::Tensor> &image_ids,           // [nnz]
    const at::optional<at::Tensor> &gaussian_ids,        // [nnz]
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
);
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., C, N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const int64_t I,
    const bool sort,
    const bool segmented
);
at::Tensor intersect_offset(
    const at::Tensor &isect_ids, // [n_isects]
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
);

// Compute Covariance and Precision Matrices from Quaternion and Scale
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor &quats,  // [..., 4]
    const at::Tensor &scales, // [..., 3]
    bool compute_covar,
    bool compute_preci,
    bool triu
);
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor &quats,  // [..., 4]
    const at::Tensor &scales, // [..., 3]
    bool triu,
    const at::optional<at::Tensor> &v_covars, // [..., 3, 3] or [..., 6]
    const at::optional<at::Tensor> &v_precis  // [..., 3, 3] or [..., 6]
);

// Rasterize 3D Gaussian to pixels
std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor &colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor &means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor &conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor &colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor &opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., 3]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor &v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor &v_render_alphas, // [..., image_height, image_width, 1]
    // options
    bool absgrad
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels.
std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs(
    int64_t range_start,
    int64_t range_end,        // iteration steps
    const at::Tensor &transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor &means2d,   // [..., N, 2]
    const at::Tensor &conics,    // [..., N, 3]
    const at::Tensor &opacities, // [..., N]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);

// Relocate some Gaussians in the Densification Process.
// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
std::tuple<at::Tensor, at::Tensor> relocation(
    const at::Tensor &opacities, // [N]
    const at::Tensor &scales,    // [N, 3]
    const at::Tensor &ratios,    // [N]
    const at::Tensor &binoms,    // [n_max, n_max]
    int64_t n_max
);

// Projection for 2DGS
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_fused_fwd(
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    // fwd outputs
    const at::Tensor &radii,          // [..., C, N, 2]
    const at::Tensor &ray_transforms, // [..., C, N, 3, 3]
    // grad outputs
    const at::Tensor &v_means2d,        // [..., C, N, 2]
    const at::Tensor &v_depths,         // [..., C, N]
    const at::Tensor &v_normals,        // [..., C, N, 3]
    const at::Tensor &v_ray_transforms, // [..., C, N, 3, 3]
    bool viewmats_requires_grad
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_packed_fwd(
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double near_plane,
    double far_plane,
    double radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd(
    // fwd inputs
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    // fwd outputs
    const at::Tensor &batch_ids,    // [nnz]
    const at::Tensor &camera_ids,   // [nnz]
    const at::Tensor &gaussian_ids, // [nnz]
    const at::Tensor &ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const at::Tensor &v_means2d,        // [nnz, 2]
    const at::Tensor &v_depths,         // [nnz]
    const at::Tensor &v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor &v_normals,        // [nnz, 3]
    bool viewmats_requires_grad,
    bool sparse_grad
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_2dgs_fwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor &opacities,      // [..., N]  or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> &backgrounds, // [..., channels]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
rasterize_to_pixels_2dgs_bwd(
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor &colors,         // [..., N, 3] or [nnz, 3]
    const at::Tensor &opacities,      // [..., N] or [nnz]
    const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
    const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds, // [..., 3]
    const at::optional<at::Tensor> &masks,       // [..., tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // ray_crossions
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor &render_colors, // [..., image_height, image_width, COLOR_DIM]
    const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., image_height, image_width]
    const at::Tensor &median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor &v_render_colors,  // [..., image_height, image_width, 3]
    const at::Tensor &v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor &v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor &v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor &v_render_median,  // [..., image_height, image_width, 1]
    // options
    bool absgrad
);

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_2dgs(
    int64_t range_start,
    int64_t range_end,        // iteration steps
    const at::Tensor &transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3]
    const at::Tensor &opacities,      // [..., N]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
);

// Use uncented transform to project 3D gaussians to 2D. (none differentiable)
// https://arxiv.org/abs/2412.12507
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_ut_3dgs_fused(
    const at::Tensor &means,                   // [..., N, 3]
    const at::Tensor &quats,                   // [..., N, 4]
    const at::Tensor &scales,                  // [..., N, 3]
    const at::optional<at::Tensor> &opacities, // [..., N] optional
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    double eps2d,
    double near_plane,
    double far_plane,
    double radius_clip,
    bool calc_compensations,
    int64_t camera_model,
    bool global_z_order,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs,  // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params
);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_from_world_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor &means,     // [..., N, 3]
    const at::Tensor &quats,     // [..., N, 4]
    const at::Tensor &scales,    // [..., N, 3]
    const at::Tensor &colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor &opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> &masks,       // [..., C, tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // camera
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters>& ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &rays, // [..., C, H, W, 6]
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs,  // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters>& ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor &tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor &flatten_ids,   // [n_isects]
    bool use_hit_distance,
    const at::optional<at::Tensor> &sample_counts, // [..., C, image_height, image_width] optional
    const at::optional<at::Tensor> &normals // [..., C, image_height, image_width, 3] optional output tensor
);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::optional<at::Tensor>>
rasterize_to_pixels_from_world_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor &means,  // [..., N, 3]
    const at::Tensor &quats,  // [..., N, 4]
    const at::Tensor &scales, // [..., N, 3]
    const at::Tensor &colors,                    // [..., C, N, 3] or [nnz, 3]
    const at::Tensor &opacities,                 // [..., C, N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., C, 3]
    const at::optional<at::Tensor> &masks,       // [..., C, tile_height, tile_width]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // camera
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters>& ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &rays,    // [..., C, H, W, 6]
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs,  // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters>& ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor &tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    bool use_hit_distance,
    // forward outputs
    const at::Tensor &render_alphas, // [..., C, image_height, image_width, 1]
    const at::Tensor &last_ids,      // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor &v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor &v_render_alphas, // [..., C, image_height, image_width, 1]
    const at::optional<at::Tensor> &v_render_normals // [..., C, image_height, image_width, 3]
);
} // namespace gsplat
