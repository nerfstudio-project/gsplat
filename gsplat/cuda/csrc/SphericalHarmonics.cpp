/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "Common.h"             // where all the macros are defined
#include "SphericalHarmonics.h" // where the launch function is declared

namespace gsplat {

at::Tensor spherical_harmonics_fwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., N, 3]
    const at::Tensor &coeffs,             // [N, K, D]
    const at::optional<at::Tensor> &masks // [..., N]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    TORCH_CHECK(dirs.dim() >= 2, "dirs must have shape [..., N, 3], got ", dirs.sizes());
    TORCH_CHECK(
        coeffs.dim() == 3,
        "coeffs must have shape [N, K, D], got ",
        coeffs.sizes()
    );
    TORCH_CHECK(
        coeffs.size(-1) >= 1,
        "coeffs last dim D must be >= 1, got ",
        coeffs.size(-1)
    );
    TORCH_CHECK(
        dirs.size(-2) == coeffs.size(-3),
        "dirs N (",
        dirs.size(-2),
        ") must match coeffs N (",
        coeffs.size(-3),
        ")"
    );

    // colors dtype follows dirs; the kernel converts fp16 coeffs to fp32 on read.
    auto out_shape = dirs.sizes().vec();
    out_shape.back() = coeffs.size(-1);
    at::Tensor colors = at::empty(out_shape, dirs.options()); // [..., N, D]

    launch_spherical_harmonics_fwd_kernel(
        degrees_to_use, dirs, coeffs, masks, colors
    );
    return colors; // [..., N, D]
}

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., N, 3]
    const at::Tensor &coeffs,              // [N, K, D]
    const at::optional<at::Tensor> &masks, // [..., N]
    const at::Tensor &v_colors,            // [..., N, D]
    bool compute_v_dirs
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    TORCH_CHECK(dirs.dim() >= 2, "dirs must have shape [..., N, 3], got ", dirs.sizes());
    TORCH_CHECK(
        coeffs.dim() == 3,
        "coeffs must have shape [N, K, D], got ",
        coeffs.sizes()
    );
    TORCH_CHECK(
        coeffs.size(-1) >= 1,
        "coeffs last dim D must be >= 1, got ",
        coeffs.size(-1)
    );
    TORCH_CHECK(
        v_colors.size(-1) == coeffs.size(-1),
        "v_colors last dim (",
        v_colors.size(-1),
        ") must match coeffs last dim (",
        coeffs.size(-1),
        ")"
    );
    TORCH_CHECK(
        dirs.size(-2) == coeffs.size(-3),
        "dirs N (",
        dirs.size(-2),
        ") must match coeffs N (",
        coeffs.size(-3),
        ")"
    );

    // Always accumulate v_coeffs in fp32 to avoid precision loss when multiple
    // (batch, gaussian) elements atomic-add into the same slot. For fp32
    // coeffs the accumulator IS the output; for fp16 we cast at the end.
    at::Tensor v_coeffs_accum =
        at::zeros(coeffs.sizes(), coeffs.options().dtype(at::kFloat));
    at::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = at::zeros_like(dirs);
    }

    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use,
        dirs,
        coeffs,
        masks,
        v_colors,
        v_coeffs_accum,
        v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
    );

    at::Tensor v_coeffs = (coeffs.scalar_type() == at::kFloat)
        ? v_coeffs_accum
        : v_coeffs_accum.to(coeffs.scalar_type());
    return std::make_tuple(v_coeffs, v_dirs); // [N, K, D], [..., N, 3]
}

void register_spherical_harmonics_cuda_impl(torch::Library &m) {
    m.impl("spherical_harmonics_fwd", &spherical_harmonics_fwd);
    m.impl("spherical_harmonics_bwd", &spherical_harmonics_bwd);
}

} // namespace gsplat
