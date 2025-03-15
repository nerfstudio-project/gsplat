#pragma once

#include <cstdint>

namespace at {
    class Tensor;
}

namespace gsplat{

void launch_adam_kernel(
    at::Tensor &param,               // [..., D]
    const at::Tensor &param_grad,    // [..., D]
    at::Tensor &exp_avg,             // [..., D]
    at::Tensor &exp_avg_sq,          // [..., D]
    const at::optional<at::Tensor> &valid, // [...]
    const float lr,
    const float b1,
    const float b2,
    const float eps
);

}