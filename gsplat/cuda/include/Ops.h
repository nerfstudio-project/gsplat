// A collection of operators for gsplat
#pragma once

#include <ATen/core/Tensor.h>

#include "Common.h"

namespace gsplat{

// null operator for tutorial. Does nothing.
at::Tensor null(const at::Tensor input);

// Project 3D gaussians (in camera space) to 2D image planes with EWA splatting.
std::tuple<at::Tensor, at::Tensor> projection_3dgs_fwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
);
std::tuple<at::Tensor, at::Tensor> projection_3dgs_bwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_covars2d // [C, N, 2, 2]
);

// Fuse the following operations:
// 1. compute covar from {quats, scales}
// 2. transform 3D gaussians from world space to camera space
//    - w/ near far plane check
// 3. projection camera space 3D gaussians to 2D image planes with EWA splatting.
//    - w/ minimum radius check
// 4. add a bit blurring to the 2D gaussians for anti-aliasing.
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
projection_3dgs_fused_fwd(
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
    const CameraModelType camera_model
);

}
