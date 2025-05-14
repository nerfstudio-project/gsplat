#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_quat_scale_to_covar_preci_fwd_kernel(
    // inputs
    const at::Tensor quats,  // [..., 4]
    const at::Tensor scales, // [..., 3]
    const bool triu,
    // outputs
    at::optional<at::Tensor> covars, // [..., 3, 3] or [..., 6]
    at::optional<at::Tensor> precis  // [..., 3, 3] or [..., 6]
);

void launch_quat_scale_to_covar_preci_bwd_kernel(
    // inputs
    const at::Tensor quats,  // [..., 4]
    const at::Tensor scales, // [..., 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [..., 3, 3] or [..., 6]
    const at::optional<at::Tensor> v_precis, // [..., 3, 3] or [..., 6]
    // outputs
    at::Tensor v_quats, // [..., 4]
    at::Tensor v_scales // [..., 3]
);

} // namespace gsplat