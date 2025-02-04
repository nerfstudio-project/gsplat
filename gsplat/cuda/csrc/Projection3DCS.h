#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_projection_ewa_3dcs_fused_fwd_kernel(
    // inputs
    const at::Tensor convex_points,                // [N, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor delta,                       // [N]
    const at::Tensor sigma,                       // [N]
    at::Tensor scaling,                     // [N]
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                       // [C, N, 2]
    at::Tensor normals,                     // [C, total_nb_points, 2]
    at::Tensor offsets,                     // [C, total_nb_points]
    at::Tensor p_image,                     // [C, total_nb_points, 2]
    at::Tensor hull,                        // [C, 2*total_nb_points]
    at::Tensor num_points_per_convex_view,  // [C, N]
    at::Tensor indices,                     // [C, total_nb_points]
    at::Tensor means2d,                     // [C, N, 2]
    at::Tensor depths,                      // [C, N]
    at::Tensor conics,                      // [C, N, 3]
    at::optional<at::Tensor> compensations  // [C, N] optional
);
void launch_projection_ewa_3dcs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor convex_points,                // [N, K, 3]
    const at::Tensor cumsum_of_points_per_convex, // [N]
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor hull,                        // [C, 2*total_nb_points]
    const at::Tensor num_points_per_convex_view,  // [C, N]
    //const at::Tensor normals,                     // [C, total_nb_points, 2]
    const at::Tensor offsets,                     // [C, total_nb_points]
    const at::Tensor p_image,                     // [C, total_nb_points, 2]
    const at::Tensor indices,                     // [C, total_nb_points]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad inputs
    const at::Tensor v_normals,                     // [C, total_nb_points, 2]
    const at::Tensor v_offsets,                     // [C, total_nb_points]
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_convex_points,   // [N, K, 3]
    at::Tensor v_viewmats         // [C, 4, 4]
);

// void launch_projection_ewa_3dcs_packed_fwd_kernel(
//     // inputs
//     const at::Tensor means,                // [N, 3]
//     const at::optional<at::Tensor> covars, // [N, 6] optional
//     const at::optional<at::Tensor> quats,  // [N, 4] optional
//     const at::optional<at::Tensor> scales, // [N, 3] optional
//     const at::optional<at::Tensor> opacities, // [N] optional
//     const at::Tensor viewmats,             // [C, 4, 4]
//     const at::Tensor Ks,                   // [C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const float eps2d,
//     const float near_plane,
//     const float far_plane,
//     const float radius_clip,
//     const at::optional<at::Tensor>
//         block_accum, // [C * blocks_per_row] packing helper
//     const CameraModelType camera_model,
//     // outputs
//     at::optional<at::Tensor> block_cnts, // [C * blocks_per_row] packing helper
//     at::optional<at::Tensor> indptr,     // [C + 1]
//     at::optional<at::Tensor> camera_ids, // [nnz]
//     at::optional<at::Tensor> gaussian_ids, // [nnz]
//     at::optional<at::Tensor> radii,        // [nnz, 2]
//     at::optional<at::Tensor> means2d,      // [nnz, 2]
//     at::optional<at::Tensor> depths,       // [nnz]
//     at::optional<at::Tensor> conics,       // [nnz, 3]
//     at::optional<at::Tensor> compensations // [nnz] optional
// );
// void launch_projection_ewa_3dcs_packed_bwd_kernel(
//     // fwd inputs
//     const at::Tensor means,                // [N, 3]
//     const at::optional<at::Tensor> covars, // [N, 6]
//     const at::optional<at::Tensor> quats,  // [N, 4]
//     const at::optional<at::Tensor> scales, // [N, 3]
//     const at::Tensor viewmats,             // [C, 4, 4]
//     const at::Tensor Ks,                   // [C, 3, 3]
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const float eps2d,
//     const CameraModelType camera_model,
//     // fwd outputs
//     const at::Tensor camera_ids,                  // [nnz]
//     const at::Tensor gaussian_ids,                // [nnz]
//     const at::Tensor conics,                      // [nnz, 3]
//     const at::optional<at::Tensor> compensations, // [nnz] optional
//     // grad outputs
//     const at::Tensor v_means2d,                     // [nnz, 2]
//     const at::Tensor v_depths,                      // [nnz]
//     const at::Tensor v_conics,                      // [nnz, 3]
//     const at::optional<at::Tensor> v_compensations, // [nnz] optional
//     const bool sparse_grad,
//     // grad inputs
//     at::Tensor v_means,                 // [N, 3] or [nnz, 3]
//     at::optional<at::Tensor> v_covars,  // [N, 6] or [nnz, 6] Optional
//     at::optional<at::Tensor> v_quats,   // [N, 4] or [nnz, 4] Optional
//     at::optional<at::Tensor> v_scales,  // [N, 3] or [nnz, 3] Optional
//     at::optional<at::Tensor> v_viewmats // [C, 4, 4] Optional
// );

} // namespace gsplat