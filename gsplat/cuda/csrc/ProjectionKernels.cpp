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
        means, 
        covars, 
        Ks, 
        width, 
        height, 
        camera_model,
        means2d, 
        covars2d
    );
    return std::make_tuple(means2d, covars2d);
}
    
} // namespace gsplat
