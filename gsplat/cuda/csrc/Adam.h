#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_adam_kernel(
    at::Tensor &param,                    // [N, ...]
    const at::Tensor &param_grad,         // [N, ...]
    at::Tensor &exp_avg,                  // [N, ...]
    at::Tensor &exp_avg_sq,               // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps
);

}