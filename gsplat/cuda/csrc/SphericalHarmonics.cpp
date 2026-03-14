/*
 * SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include "Common.h"             // where all the macros are defined
#include "Ops.h"                // a collection of all gsplat operators
#include "SphericalHarmonics.h" // where the launch function is declared

namespace gsplat {

at::Tensor spherical_harmonics_fwd(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., 3]
    const at::Tensor &coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> &masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");

    at::Tensor colors = at::empty_like(dirs); // [..., 3]

    launch_spherical_harmonics_fwd_kernel(
        degrees_to_use, dirs, coeffs, masks, colors
    );
    return colors; // [..., 3]
}

std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
    int64_t K,
    int64_t degrees_to_use,
    const at::Tensor &dirs,                // [..., 3]
    const at::Tensor &coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> &masks, // [...]
    const at::Tensor &v_colors,            // [..., 3]
    bool compute_v_dirs
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t N = dirs.numel() / 3;

    at::Tensor v_coeffs = at::zeros_like(coeffs);
    at::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = at::zeros_like(dirs);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    uint32_t n_elements = N;
    uint32_t shmem_size = 0;
    launch_spherical_harmonics_bwd_kernel(
        degrees_to_use,
        dirs,
        coeffs,
        masks,
        v_colors,
        v_coeffs,
        v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
    );
    return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
}

} // namespace gsplat
