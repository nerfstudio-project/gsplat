/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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