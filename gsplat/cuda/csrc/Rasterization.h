#pragma once

#include <cstdint>
#include "Cameras.h"

namespace at {
class Tensor;
}

namespace gsplat {

#define FILTER_INV_SQUARE_2DGS 2.0f

/////////////////////////////////////////////////
// rasterize_to_pixels_3dgs
/////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
);

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets,    // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    // forward outputs
    const at::Tensor render_alphas,   // [..., image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
);

/////////////////////////////////////////////////
// rasterize_to_indices_3dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_3dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2]
    const at::Tensor conics,    // [..., N, 3]
    const at::Tensor opacities, // [..., N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor>
        chunk_starts, // [..., image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [..., image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_2dgs
/////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width, 1]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_distort, // [..., image_height, image_width, 1]
    at::Tensor render_median,  // [..., image_height, image_width, 1]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids      // [..., image_height, image_width]
);
template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [..., image_height, image_width, CDIM]
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    at::Tensor v_normals,                   // [..., N, 3] or [nnz, 3]
    at::Tensor v_densify                    // [..., N, 2] or [nnz, 2]
);

/////////////////////////////////////////////////
// rasterize_to_indices_2dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_2dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3]
    const at::Tensor opacities,      // [..., N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor>
        chunk_starts, // [..., image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [..., image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

///////////////////////////////////////////////////
// rasterize_to_pixels_from_world_3dgs
///////////////////////////////////////////////////

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const FThetaCameraDistortionParameters ftheta_coeffs, // shared parameters for all cameras
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids // [..., C, image_height, image_width]
);

template <uint32_t CDIM>
void launch_rasterize_to_pixels_from_world_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means,  // [..., N, 3]
    const at::Tensor quats,  // [..., N, 4]
    const at::Tensor scales, // [..., N, 3]
    const at::Tensor colors,                    // [..., C, N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, 3]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const UnscentedTransformParameters ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const FThetaCameraDistortionParameters ftheta_coeffs, // shared parameters for all cameras
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., C, image_height, image_width, 1]
    // outputs
    at::Tensor v_means,      // [..., N, 3]
    at::Tensor v_quats,      // [..., N, 4]
    at::Tensor v_scales,     // [..., N, 3]
    at::Tensor v_colors,     // [..., C, N, 3] or [nnz, 3]
    at::Tensor v_opacities   // [..., C, N] or [nnz]
) ;

} // namespace gsplat