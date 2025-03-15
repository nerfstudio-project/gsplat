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

}