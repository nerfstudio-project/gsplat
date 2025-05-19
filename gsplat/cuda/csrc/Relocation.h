#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_relocation_kernel(
    // inputs
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // [n_max, n_max]
    const int n_max,
    // outputs
    at::Tensor new_opacities, // [N]
    at::Tensor new_scales     // [N, 3]
);

}