#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_quat_scale_to_covar_preci_fwd_kernel(
    // inputs
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    // outputs
    at::optional<at::Tensor> covars, // [N, 3, 3] or [N, 6]
    at::optional<at::Tensor> precis  // [N, 3, 3] or [N, 6]
);

void launch_quat_scale_to_covar_preci_bwd_kernel(
    // inputs
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<at::Tensor> v_precis, // [N, 3, 3] or [N, 6]
    // outputs
    at::Tensor v_quats, // [N, 4]
    at::Tensor v_scales // [N, 3]
);

} // namespace gsplat