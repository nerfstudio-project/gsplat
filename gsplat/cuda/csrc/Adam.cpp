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

#if GSPLAT_BUILD_ADAM

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Adam.h"   // where the launch function is declared
#include "Common.h" // where all the macros are defined
#include "Ops.h"    // a collection of all gsplat operators

namespace gsplat {

void adam(
    at::Tensor &param,                     // [N, ...]
    const at::Tensor &param_grad,          // [N, ...]
    at::Tensor &exp_avg,                   // [N, ...]
    at::Tensor &exp_avg_sq,                // [N, ...]
    const at::optional<at::Tensor> &valid, // [N]
    double lr,
    double b1,
    double b2,
    double eps
) {
    DEVICE_GUARD(param);
    CHECK_INPUT(param);
    CHECK_INPUT(param_grad);
    CHECK_INPUT(exp_avg);
    CHECK_INPUT(exp_avg_sq);
    if (valid.has_value()) {
        CHECK_INPUT(valid.value());
        TORCH_CHECK(valid.value().dim() == 1, "valid should be 1D tensor");
        TORCH_CHECK(
            valid.value().size(0) == param.size(0),
            "valid first dimension should match param first dimension"
        );
    }

    launch_adam_kernel(
        param, param_grad, exp_avg, exp_avg_sq, valid, lr, b1, b2, eps
    );
}

} // namespace gsplat

#endif
