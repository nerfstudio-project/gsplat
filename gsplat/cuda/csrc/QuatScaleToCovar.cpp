#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD

// TODO: replacing the following with per-operation kernels might make compile
// faster.
// https://github.com/pytorch/pytorch/blob/740ce0fa5f8c7e9e51422b614f8187ab93a60b8b/aten/src/ATen/native/cuda/ScanKernels.cpp#L8-L17
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"           // where all the macros are defined
#include "Ops.h"              // a collection of all gsplat operators
#include "QuatScaleToCovar.h" // where the launch function is declared

namespace gsplat {

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_fwd(
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);

    uint32_t N = quats.size(0);

    at::Tensor covars, precis;
    if (compute_covar) {
        if (triu) {
            covars = at::empty({N, 6}, quats.options());
        } else {
            covars = at::empty({N, 3, 3}, quats.options());
        }
    }
    if (compute_preci) {
        if (triu) {
            precis = at::empty({N, 6}, quats.options());
        } else {
            precis = at::empty({N, 3, 3}, quats.options());
        }
    }

    launch_quat_scale_to_covar_preci_fwd_kernel(
        quats,
        scales,
        triu,
        compute_covar ? at::optional<at::Tensor>(covars) : at::nullopt,
        compute_preci ? at::optional<at::Tensor>(precis) : at::nullopt
    );

    return std::make_tuple(covars, precis);
}

std::tuple<at::Tensor, at::Tensor> quat_scale_to_covar_preci_bwd(
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<at::Tensor> v_precis  // [N, 3, 3] or [N, 6]
) {
    DEVICE_GUARD(quats);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    if (v_covars.has_value()) {
        CHECK_INPUT(v_covars.value());
    }
    if (v_precis.has_value()) {
        CHECK_INPUT(v_precis.value());
    }

    uint32_t N = quats.size(0);

    // kernel with directly write values into these tensors so we could empty
    // init them.
    at::Tensor v_scales = at::empty_like(scales);
    at::Tensor v_quats = at::empty_like(quats);

    if (v_covars.has_value() || v_precis.has_value()) {
        launch_quat_scale_to_covar_preci_bwd_kernel(
            quats, scales, triu, v_covars, v_precis, v_quats, v_scales
        );
    } else {
        // if no gradients are provided, just zero out the tensors.
        v_scales.zero_();
        v_quats.zero_();
    }

    return std::make_tuple(v_quats, v_scales);
}

} // namespace gsplat
