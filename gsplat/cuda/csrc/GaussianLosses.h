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

namespace at
{
class Tensor;
}

namespace gsplat
{
// Each member owns its row count (heterogeneous regularization domains):
// scales [N_scales, 3], densities [N_densities], z_scales [N_z_scales],
// positions/cuboid_dims [N_oob, 3]; any count may be zero. `visibility`
// spans the scale and density members: [max(N_scales, N_densities)] or
// nullptr. `preactivation` selects log-space member math (exp() fused for
// scales/z_scales, |.| for densities); false reproduces the post-activation
// math bit for bit.
void launch_gaussian_losses_fwd_kernel(
    const at::Tensor &scales,      // [N_scales, 3]
    const at::Tensor &densities,   // [N_densities]
    const at::Tensor &z_scales,    // [N_z_scales]
    const at::Tensor &positions,   // [N_oob, 3]
    const at::Tensor &cuboid_dims, // [N_oob, 3]
    const at::Tensor *visibility,  // [max(N_scales, N_densities)] float or nullptr
    float z_scale_threshold,
    bool preactivation,
    at::Tensor &loss_scale,   // [N_scales, 3]
    at::Tensor &loss_density, // [N_densities]
    at::Tensor &loss_z_scale, // [N_z_scales]
    at::Tensor &loss_oob      // [N_oob, 3]
);

void launch_gaussian_losses_bwd_kernel(
    const at::Tensor &scales,      // [N_scales, 3]
    const at::Tensor &densities,   // [N_densities]
    const at::Tensor &z_scales,    // [N_z_scales]
    const at::Tensor &positions,   // [N_oob, 3]
    const at::Tensor &cuboid_dims, // [N_oob, 3]
    const at::Tensor *visibility,  // [max(N_scales, N_densities)] float or nullptr
    float z_scale_threshold,
    bool preactivation,
    const at::Tensor &v_loss_scale,   // [N_scales, 3]
    const at::Tensor &v_loss_density, // [N_densities]
    const at::Tensor &v_loss_z_scale, // [N_z_scales]
    const at::Tensor &v_loss_oob,     // [N_oob, 3]
    at::Tensor &v_scales,             // [N_scales, 3]
    at::Tensor &v_densities,          // [N_densities]
    at::Tensor &v_z_scales,           // [N_z_scales]
    at::Tensor &v_positions           // [N_oob, 3]
);
} // namespace gsplat
