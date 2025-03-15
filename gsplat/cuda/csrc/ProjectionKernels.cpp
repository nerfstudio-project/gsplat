#include <ATen/core/Tensor.h>
#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h" // where all the macros are defined
#include "ProjectionKernels.h" // where the launch function is declared
#include "Ops.h" // a collection of all gsplat operators

namespace gsplat{

std::tuple<at::Tensor, at::Tensor> projection_3dgs_fwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    at::Tensor means2d = at::empty({C, N, 2}, means.options());
    at::Tensor covars2d = at::empty({C, N, 2, 2}, covars.options());

    launch_projection_3dgs_fwd_kernel(
        // inputs
        means, 
        covars, 
        Ks, 
        width, 
        height, 
        camera_model,
        // outputs
        means2d, 
        covars2d
    );
    return std::make_tuple(means2d, covars2d);
}


std::tuple<at::Tensor, at::Tensor> projection_3dgs_bwd(
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d, // [C, N, 2]
    const at::Tensor v_covars2d // [C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    at::Tensor v_means = at::empty({C, N, 3}, means.options());
    at::Tensor v_covars = at::empty({C, N, 3, 3}, means.options());

    launch_projection_3dgs_bwd_kernel(
        // inputs
        means, 
        covars, 
        Ks, 
        width, 
        height, 
        camera_model,
        v_means2d,
        v_covars2d,
        // outputs
        v_means,
        v_covars
    );
    return std::make_tuple(v_means, v_covars);
}

} // namespace gsplat
