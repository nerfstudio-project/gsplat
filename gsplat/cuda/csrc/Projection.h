#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_projection_ewa_simple_fwd_kernel(
    // inputs
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    // outputs
    at::Tensor means2d, // [C, N, 2]
    at::Tensor covars2d // [C, N, 2, 2]
);
void launch_projection_ewa_simple_bwd_kernel(
    // inputs
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d,  // [C, N, 2]
    const at::Tensor v_covars2d, // [C, N, 2, 2]
    // outputs
    at::Tensor v_means, // [C, N, 3]
    at::Tensor v_covars // [C, N, 3, 3]
);

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
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
    at::Tensor radii,                      // [C, N, 2]
    at::Tensor means2d,                    // [C, N, 2]
    at::Tensor depths,                     // [C, N]
    at::Tensor conics,                     // [C, N, 3]
    at::optional<at::Tensor> compensations // [C, N] optional
);
void launch_projection_ewa_3dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N, 2]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> compensations, // [C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [C, N, 3]
    at::Tensor v_covars,  // [C, N, 3, 3]
    at::Tensor v_quats,   // [C, N, 4]
    at::Tensor v_scales,  // [C, N, 3]
    at::Tensor v_viewmats // [C, 4, 4]
);

void launch_projection_ewa_3dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::optional<at::Tensor> quats,  // [N, 4] optional
    const at::optional<at::Tensor> scales, // [N, 3] optional
    const at::optional<at::Tensor> opacities, // [N] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor>
        block_accum, // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    at::optional<at::Tensor> block_cnts, // [C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,     // [C + 1]
    at::optional<at::Tensor> camera_ids, // [nnz]
    at::optional<at::Tensor> gaussian_ids, // [nnz]
    at::optional<at::Tensor> radii,        // [nnz, 2]
    at::optional<at::Tensor> means2d,      // [nnz, 2]
    at::optional<at::Tensor> depths,       // [nnz]
    at::optional<at::Tensor> conics,       // [nnz, 3]
    at::optional<at::Tensor> compensations // [nnz] optional
);
void launch_projection_ewa_3dgs_packed_bwd_kernel(
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
    const bool sparse_grad,
    // grad inputs
    at::Tensor v_means,                 // [N, 3] or [nnz, 3]
    at::optional<at::Tensor> v_covars,  // [N, 6] or [nnz, 6] Optional
    at::optional<at::Tensor> v_quats,   // [N, 4] or [nnz, 4] Optional
    at::optional<at::Tensor> v_scales,  // [N, 3] or [nnz, 3] Optional
    at::optional<at::Tensor> v_viewmats // [C, 4, 4] Optional
);

void launch_projection_2dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    // outputs
    at::Tensor radii,          // [C, N, 2]
    at::Tensor means2d,        // [C, N, 2]
    at::Tensor depths,         // [C, N]
    at::Tensor ray_transforms, // [C, N, 3, 3]
    at::Tensor normals         // [C, N, 3]
);
void launch_projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [C, N, 2]
    const at::Tensor ray_transforms, // [C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [C, N, 2]
    const at::Tensor v_depths,         // [C, N]
    const at::Tensor v_normals,        // [C, N, 3]
    const at::Tensor v_ray_transforms, // [C, N, 3, 3]
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [C, N, 3]
    at::Tensor v_quats,   // [C, N, 4]
    at::Tensor v_scales,  // [C, N, 3]
    at::Tensor v_viewmats // [C, 4, 4]
);

void launch_projection_2dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,    // [N, 3]
    const at::Tensor quats,    // [N, 4]
    const at::Tensor scales,   // [N, 3]
    const at::Tensor viewmats, // [C, 4, 4]
    const at::Tensor Ks,       // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor>
        block_accum, // [C * blocks_per_row] packing helper
    // outputs
    at::optional<at::Tensor> block_cnts, // [C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,     // [C + 1]
    at::optional<at::Tensor> camera_ids, // [nnz]
    at::optional<at::Tensor> gaussian_ids,   // [nnz]
    at::optional<at::Tensor> radii,          // [nnz, 2]
    at::optional<at::Tensor> means2d,        // [nnz, 2]
    at::optional<at::Tensor> depths,         // [nnz]
    at::optional<at::Tensor> ray_transforms, // [nnz, 3, 3]
    at::optional<at::Tensor> normals         // [nnz]
);
void launch_projection_2dgs_packed_bwd_kernel(
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
    const bool sparse_grad,
    // grad inputs
    at::Tensor v_means,                 // [N, 3] or [nnz, 3]
    at::Tensor v_quats,                 // [N, 4] or [nnz, 4]
    at::Tensor v_scales,                // [N, 3] or [nnz, 3]
    at::optional<at::Tensor> v_viewmats // [C, 4, 4] Optional
);

} // namespace gsplat