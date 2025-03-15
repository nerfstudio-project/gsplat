#pragma once

#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat{


void launch_projection_3dgs_fwd_kernel(
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
void launch_projection_3dgs_bwd_kernel(
    // inputs
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_covars2d, // [C, N, 2, 2]
    // outputs
    at::Tensor v_means, // [C, N, 3]
    at::Tensor v_covars // [C, N, 3, 3]
);


void launch_projection_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> &covars, // [N, 6] optional
    const at::optional<at::Tensor> &quats,  // [N, 4] optional
    const at::optional<at::Tensor> &scales, // [N, 3] optional
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
    at::Tensor radii,          // [C, N]
    at::Tensor means2d,       // [C, N, 2]
    at::Tensor depths,        // [C, N]
    at::Tensor conics,        // [C, N, 3]
    at::optional<at::Tensor> compensations  // [C, N] optional
);
void launch_projection_3dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> &covars, // [N, 6] optional
    const at::optional<at::Tensor> &quats,  // [N, 4] optional
    const at::optional<at::Tensor> &scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [C, N]
    const at::Tensor conics,                      // [C, N, 3]
    const at::optional<at::Tensor> &compensations, // [C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [C, N, 2]
    const at::Tensor v_depths,                      // [C, N]
    const at::Tensor v_conics,                      // [C, N, 3]
    const at::optional<at::Tensor> &v_compensations, // [C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means, // [C, N, 3]
    at::Tensor v_covars, // [C, N, 3, 3]
    at::Tensor v_quats, // [C, N, 4]
    at::Tensor v_scales, // [C, N, 3]
    at::Tensor v_viewmats // [C, 4, 4]
);


void launch_projection_3dgs_packed_fwd_kernel(
    // inputs
    const at::Tensor means,                // [N, 3]
    const at::optional<at::Tensor> &covars, // [N, 6] optional
    const at::optional<at::Tensor> &quats,  // [N, 4] optional
    const at::optional<at::Tensor> &scales, // [N, 3] optional
    const at::Tensor viewmats,             // [C, 4, 4]
    const at::Tensor Ks,                   // [C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor> block_accum,    // [C * blocks_per_row] packing helper
    const CameraModelType camera_model,
    // outputs
    at::optional<at::Tensor> block_cnts,      // [C * blocks_per_row] packing helper
    at::optional<at::Tensor> indptr,          // [C + 1]
    at::optional<at::Tensor> camera_ids,      // [nnz]
    at::optional<at::Tensor> gaussian_ids,    // [nnz]
    at::optional<at::Tensor> radii,          // [nnz]
    at::optional<at::Tensor> means2d,       // [nnz, 2]
    at::optional<at::Tensor> depths,        // [nnz]
    at::optional<at::Tensor> conics,        // [nnz, 3]
    at::optional<at::Tensor> compensations  // [nnz] optional
);

}