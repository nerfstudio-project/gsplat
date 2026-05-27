/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Config.h"

#if GSPLAT_BUILD_3DGS

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "MCMCPerturb.h" // where the launch function is declared
#include "Common.h"      // where all the macros are defined
#include "Ops.h"         // a collection of all gsplat operators

namespace gsplat {

void mcmc_perturb_positions(
    at::Tensor positions,        // [N, 3] in-place, float32
    const at::Tensor &quats,     // [N, 4] wxyz pre-activation
    const at::Tensor &scales,    // [N, 3] log-scale
    const at::Tensor &opacities, // [N] logit
    const at::Tensor &noise,     // [N, 3] standard normal
    double scaler
) {
    DEVICE_GUARD(positions);
    CHECK_INPUT(positions);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(opacities);
    CHECK_INPUT(noise);

    TORCH_CHECK(
        positions.scalar_type() == at::kFloat,
        "mcmc_perturb_positions: positions must be float32"
    );
    TORCH_CHECK(quats.scalar_type() == at::kFloat, "mcmc_perturb_positions: quats must be float32");
    TORCH_CHECK(scales.scalar_type() == at::kFloat, "mcmc_perturb_positions: scales must be float32");
    TORCH_CHECK(opacities.scalar_type() == at::kFloat, "mcmc_perturb_positions: opacities must be float32");
    TORCH_CHECK(noise.scalar_type() == at::kFloat, "mcmc_perturb_positions: noise must be float32");

    const int64_t N = positions.size(0);
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "mcmc_perturb_positions: positions must be [N, 3]");
    TORCH_CHECK(quats.sizes() == at::IntArrayRef({N, 4}), "mcmc_perturb_positions: quats must be [N, 4]");
    TORCH_CHECK(scales.sizes() == at::IntArrayRef({N, 3}), "mcmc_perturb_positions: scales must be [N, 3]");
    TORCH_CHECK(opacities.dim() == 1 && opacities.size(0) == N, "mcmc_perturb_positions: opacities must be [N]");
    TORCH_CHECK(noise.sizes() == at::IntArrayRef({N, 3}), "mcmc_perturb_positions: noise must be [N, 3]");

    launch_mcmc_perturb_positions_kernel(
        positions,
        quats,
        scales,
        opacities,
        noise,
        static_cast<float>(scaler)
    );
}

} // namespace gsplat

#endif
