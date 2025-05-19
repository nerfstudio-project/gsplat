#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Adam.h"   // where the launch function is declared
#include "Common.h" // where all the macros are defined
#include "Ops.h"    // a collection of all gsplat operators

namespace gsplat {

void adam(
    at::Tensor &param,                    // [N, ...]
    const at::Tensor &param_grad,         // [N, ...]
    at::Tensor &exp_avg,                  // [N, ...]
    at::Tensor &exp_avg_sq,               // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    DEVICE_GUARD(param);
    CHECK_INPUT(param);
    CHECK_INPUT(param_grad);
    CHECK_INPUT(exp_avg);
    CHECK_INPUT(exp_avg_sq);
    if (valid.has_value()) {
        CHECK_INPUT(valid.value());
        TORCH_CHECK(valid.value().dim() == 1, "valid should be 1D tensor");
        TORCH_CHECK(
            valid.value().size(0) == param.size(0),
            "valid first dimension should match param first dimension"
        );
    }

    launch_adam_kernel(
        param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps
    );
}

} // namespace gsplat
