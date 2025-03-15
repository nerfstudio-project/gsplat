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
    const bool calc_compensations,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,          // [C, N]
    at::Tensor means2d,       // [C, N, 2]
    at::Tensor depths,        // [C, N]
    at::Tensor conics,        // [C, N, 3]
    at::Tensor compensations  // [C, N] optional
);

}