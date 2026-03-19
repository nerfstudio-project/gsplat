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

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    // outputs
    at::Tensor colors // [..., 2]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., 3]
    const at::Tensor coeffs,              // [..., K, 3]
    const at::optional<at::Tensor> masks, // [...]
    const at::Tensor v_colors,            // [..., 3]
    // outputs
    at::Tensor v_coeffs,            // [..., K, 3]
    at::optional<at::Tensor> v_dirs // [..., 3]
);

} // namespace gsplat