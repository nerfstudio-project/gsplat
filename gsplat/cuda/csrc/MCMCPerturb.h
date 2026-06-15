/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace at {
class Tensor;
}

namespace gsplat {

void launch_mcmc_perturb_positions_kernel(
    at::Tensor positions,   // [N, 3] in-place, float32
    const at::Tensor &quats,     // [N, 4] wxyz pre-activation
    const at::Tensor &scales,    // [N, 3] log-scale
    const at::Tensor &opacities, // [N] logit
    const at::Tensor &noise,     // [N, 3] standard normal
    float scaler
);

} // namespace gsplat
