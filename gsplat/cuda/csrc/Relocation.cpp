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

#include "Config.h"

#if GSPLAT_BUILD_RELOC

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"     // where all the macros are defined
#include "Ops.h"        // a collection of all gsplat operators
#include "Relocation.h" // where the launch function is declared

namespace gsplat {

std::tuple<at::Tensor, at::Tensor> relocation(
    const at::Tensor &opacities, // [N]
    const at::Tensor &scales,    // [N, 3]
    const at::Tensor &ratios,    // [N]
    const at::Tensor &binoms,    // [n_max, n_max]
    int64_t n_max
) {
    DEVICE_GUARD(opacities);
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(ratios);
    CHECK_INPUT(binoms);
    at::Tensor new_opacities = at::empty_like(opacities);
    at::Tensor new_scales = at::empty_like(scales);

    launch_relocation_kernel(
        opacities, scales, ratios, binoms, n_max, new_opacities, new_scales
    );
    return std::make_tuple(new_opacities, new_scales);
}

} // namespace gsplat

#endif
