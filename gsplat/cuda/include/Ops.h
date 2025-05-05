// A collection of operators for gsplat
#pragma once

#include <ATen/core/Tensor.h>

#include "Common.h"

namespace gsplat {

// null operator for tutorial. Does nothing.
at::Tensor null(const at::Tensor input);

// Project 3D gaussians (in camera space) to 2D image planes with EWA splatting.
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_fwd(
    const at::Tensor means,  // [B, C, N, 3]
    const at::Tensor covars, // [B, C, N, 3, 3]
    const at::Tensor Ks,     // [B, C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor> projection_ewa_simple_bwd(
    const at::Tensor means,  // [B, C, N, 3]
    const at::Tensor covars, // [B, C, N, 3, 3]
    const at::Tensor Ks,     // [B, C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [B, C, N, 2]
    const at::Tensor v_covars2d // [B, C, N, 2, 2]
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
    const at::Tensor means,                   // [B, N, 3]
    const at::optional<at::Tensor> covars,    // [B, N, 6] optional
    const at::optional<at::Tensor> quats,     // [B, N, 4] optional
    const at::optional<at::Tensor> scales,    // [B, N, 3] optional
    const at::optional<at::Tensor> opacities, // [B, N] optional
    const at::Tensor viewmats,                // [B, C, 4, 4]
    const at::Tensor Ks,                      // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,                // [B, N, 3]
    const at::optional<at::Tensor> covars, // [B, N, 6] optional
    const at::optional<at::Tensor> quats,  // [B, N, 4] optional
    const at::optional<at::Tensor> scales, // [B, N, 3] optional
    const at::Tensor viewmats,             // [B, C, 4, 4]
    const at::Tensor Ks,                   // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [B, C, N, 2]
    const at::Tensor conics,                      // [B, C, N, 3]
    const at::optional<at::Tensor> compensations, // [B, C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [B, C, N, 2]
    const at::Tensor v_depths,                      // [B, C, N]
    const at::Tensor v_conics,                      // [B, C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [B, C, N] optional
    const bool viewmats_requires_grad
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
    at::Tensor>
projection_ewa_3dgs_packed_fwd(
    const at::Tensor means,                   // [N, 3]
    const at::optional<at::Tensor> covars,    // [N, 6] optional
    const at::optional<at::Tensor> quats,     // [N, 4] optional
    const at::optional<at::Tensor> scales,    // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,                // [C, 4, 4]
    const at::Tensor Ks,                      // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const bool calc_compensations,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_ewa_3dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6]
    const at::optional<at::Tensor> quats,  // [N, 4]
    const at::optional<at::Tensor> scales, // [N, 3]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor camera_ids,                  // [nnz]
    const at::Tensor gaussian_ids,                // [nnz]
    const at::Tensor conics,                      // [nnz, 3]
    const at::optional<at::Tensor> compensations, // [nnz] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [nnz, 2]
    const at::Tensor v_depths,                      // [nnz]
    const at::Tensor v_conics,                      // [nnz, 3]
    const at::optional<at::Tensor> v_compensations, // [nnz] optional
    const bool viewmats_requires_grad,
    const bool sparse_grad
);

// Sphereical harmonics
at::Tensor spherical_harmonics_fwd(
    const uint32_t degrees_to_use,
    const at::Tensor dirs,               // [..., 3]
    const at::Tensor coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> masks // [...]
);
std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    const uint32_t K,
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
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
    const at::optional<at::Tensor> valid, // [...]
    const float lr,
    const float b1,
    const float b2,
    const float eps
);

// GS Tile Intersection
std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [B, C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [B, C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [B, C, N] or [nnz]
    const at::optional<at::Tensor> batch_ids,    // [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t B,
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort,
    const bool segmented
);
at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t B,
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
);

// Compute Covariance and Precision Matrices from Quaternion and Scale
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor quats,  // [..., 4]
    const at::Tensor scales, // [..., 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
);
std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor quats,  // [..., 4]
    const at::Tensor scales, // [..., 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [..., 3, 3] or [..., 6]
    const at::optional<at::Tensor> v_precis  // [..., 3, 3] or [..., 6]
);

// Rasterize 3D Gaussian to pixels
std::tuple<at::Tensor, at::Tensor, at::Tensor> rasterize_to_pixels_3dgs_fwd(
    // Gaussian parameters
    const at::Tensor means2d,   // [B, C, N, 2] or [nnz, 2]
    const at::Tensor conics,    // [B, C, N, 3] or [nnz, 3]
    const at::Tensor colors,    // [B, C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [B, C, N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [B, C, channels]
    const at::optional<at::Tensor> masks,       // [B, C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
rasterize_to_pixels_3dgs_bwd(
    // Gaussian parameters
    const at::Tensor means2d,                   // [B, C, N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [B, C, N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [B, C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [B, C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [B, C, 3]
    const at::optional<at::Tensor> masks,       // [B, C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [B, C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [B, C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [B, C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [B, C, image_height, image_width, 1]
    // options
    bool absgrad
);

// Rasterize 3D Gaussian, but only return the indices of gaussians and pixels.
std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_3dgs(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [B, C, image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,   // [B, C, N, 2]
    const at::Tensor conics,    // [B, C, N, 3]
    const at::Tensor opacities, // [B, C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);

// Relocate some Gaussians in the Densification Process.
// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
std::tuple<at::Tensor, at::Tensor> relocation(
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // [n_max, n_max]
    const int n_max
);

// Projection for 2DGS
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_fused_fwd(
    const at::Tensor means,    // [B, N, 3]
    const at::Tensor quats,    // [B, N, 4]
    const at::Tensor scales,   // [B, N, 3]
    const at::Tensor viewmats, // [B, C, 4, 4]
    const at::Tensor Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor means,    // [B, N, 3]
    const at::Tensor quats,    // [B, N, 4]
    const at::Tensor scales,   // [B, N, 3]
    const at::Tensor viewmats, // [B, C, 4, 4]
    const at::Tensor Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [B, C, N, 2]
    const at::Tensor ray_transforms, // [B, C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [B, C, N, 2]
    const at::Tensor v_depths,         // [B, C, N]
    const at::Tensor v_normals,        // [B, C, N, 3]
    const at::Tensor v_ray_transforms, // [B, C, N, 3, 3]
    const bool viewmats_requires_grad
);

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_2dgs_packed_fwd(
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip
);
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
projection_2dgs_packed_bwd(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor camera_ids,     // [nnz]
    const at::Tensor gaussian_ids,   // [nnz]
    const at::Tensor ray_transforms, // [nnz, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [nnz, 2]
    const at::Tensor v_depths,         // [nnz]
    const at::Tensor v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor v_normals,        // [nnz, 3]
    const bool viewmats_requires_grad,
    const bool sparse_grad
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
    const at::Tensor means2d,        // [B, C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [B, C, N, 3] or [nnz, 3]
    const at::Tensor colors,         // [B, C, N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [B, C, N]  or [nnz]
    const at::Tensor normals,        // [B, C, N, 3] or [nnz, 3]
    const at::optional<at::Tensor> backgrounds, // [B, C, channels]
    const at::optional<at::Tensor> masks,       // [B, C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
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
    const at::Tensor means2d,        // [B, C, N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [B, C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [B, C, N, 3] or [nnz, 3]
    const at::Tensor opacities,      // [B, C, N] or [nnz]
    const at::Tensor normals,        // [B, C, N, 3] or [nnz, 3]
    const at::Tensor densify,
    const at::optional<at::Tensor> backgrounds, // [B, C, 3]
    const at::optional<at::Tensor> masks,       // [B, C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [B, C, image_height, image_width, COLOR_DIM]
    const at::Tensor render_alphas, // [B, C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [B, C, image_height, image_width]
    const at::Tensor median_ids,    // [B, C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [B, C, image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [B, C, image_height, image_width, 1]
    const at::Tensor v_render_normals, // [B, C, image_height, image_width, 3]
    const at::Tensor v_render_distort, // [B, C, image_height, image_width, 1]
    const at::Tensor v_render_median,  // [B, C, image_height, image_width, 1]
    // options
    bool absgrad
);

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_2dgs(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [B, C, image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,        // [B, C, N, 2]
    const at::Tensor ray_transforms, // [B, C, N, 3, 3]
    const at::Tensor opacities,      // [B, C, N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [B, C, tile_height, tile_width]
    const at::Tensor flatten_ids   // [n_isects]
);

} // namespace gsplat
