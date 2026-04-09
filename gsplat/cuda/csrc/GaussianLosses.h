/*
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

#pragma once

namespace at {
class Tensor;
}

namespace gsplat {

void launch_gaussian_losses_fwd_kernel(
    const at::Tensor &scales,           // [N, 3]
    const at::Tensor &densities,        // [N]
    const at::Tensor &z_scales,         // [N]
    const at::Tensor &positions,        // [N, 3]
    const at::Tensor &cuboid_dims,      // [N, 3]
    const at::Tensor *visibility,       // [N] float or nullptr
    float z_scale_threshold,
    at::Tensor &loss_scale,             // [N, 3]
    at::Tensor &loss_density,           // [N]
    at::Tensor &loss_z_scale,           // [N]
    at::Tensor &loss_oob               // [N, 3]
);

void launch_gaussian_losses_bwd_kernel(
    const at::Tensor &scales,           // [N, 3]
    const at::Tensor &densities,        // [N]
    const at::Tensor &z_scales,         // [N]
    const at::Tensor &positions,        // [N, 3]
    const at::Tensor &cuboid_dims,      // [N, 3]
    const at::Tensor *visibility,       // [N] float or nullptr
    float z_scale_threshold,
    const at::Tensor &v_loss_scale,     // [N, 3]
    const at::Tensor &v_loss_density,   // [N]
    const at::Tensor &v_loss_z_scale,   // [N]
    const at::Tensor &v_loss_oob,       // [N, 3]
    at::Tensor &v_scales,               // [N, 3]
    at::Tensor &v_densities,            // [N]
    at::Tensor &v_z_scales,             // [N]
    at::Tensor &v_positions             // [N, 3]
);

} // namespace gsplat
