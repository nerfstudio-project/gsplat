#pragma once

#include <cstdint>
#include "Cameras.h"

namespace at {
class Tensor;
}

namespace gsplat {

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                      // [..., C, N, 2]
    at::Tensor means2d,                    // [..., C, N, 2]
    at::Tensor depths,                     // [..., C, N]
    at::Tensor conics,                     // [..., C, N, 3]
    at::optional<at::Tensor> compensations // [..., C, N] optional
);
void launch_projection_ewa_3dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [..., N, 3]
    at::Tensor v_covars,  // [..., N, 3, 3]
    at::Tensor v_quats,   // [..., N, 4]
    at::Tensor v_scales,  // [..., N, 3]
    at::Tensor v_viewmats // [..., C, 4, 4]
);

} // namespace gsplat