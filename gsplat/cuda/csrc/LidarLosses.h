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
void launch_lidar_losses_fwd_kernel(
    const at::Tensor &flags,          // [N] int32
    const at::Tensor &distance_pred,  // [N]
    const at::Tensor &distance_gt,    // [N]
    const at::Tensor &intensity_pred, // [N]
    const at::Tensor &intensity_gt,   // [N]
    const at::Tensor &raydrop_pred,   // [N]
    const at::Tensor &raydrop_gt,     // [N]
    const at::Tensor &bg_pred,        // [N]
    float distance_factor,
    float intensity_factor,
    float raydrop_factor,
    float bg_factor,
    at::Tensor &distance_loss,  // [N]
    at::Tensor &intensity_loss, // [N]
    at::Tensor &raydrop_loss,   // [N]
    at::Tensor &bg_loss         // [N]
);

void launch_lidar_losses_bwd_kernel(
    const at::Tensor &flags,          // [N] int32
    const at::Tensor &distance_pred,  // [N]
    const at::Tensor &distance_gt,    // [N]
    const at::Tensor &intensity_pred, // [N]
    const at::Tensor &intensity_gt,   // [N]
    const at::Tensor &raydrop_pred,   // [N]
    const at::Tensor &raydrop_gt,     // [N]
    const at::Tensor &bg_pred,        // [N]
    float distance_factor,
    float intensity_factor,
    float raydrop_factor,
    float bg_factor,
    const at::Tensor &v_distance_loss,  // [N]
    const at::Tensor &v_intensity_loss, // [N]
    const at::Tensor &v_raydrop_loss,   // [N]
    const at::Tensor &v_bg_loss,        // [N]
    at::Tensor &v_distance_pred,        // [N]
    at::Tensor &v_intensity_pred,       // [N]
    at::Tensor &v_raydrop_pred,         // [N]
    at::Tensor &v_bg_pred               // [N]
);
} // namespace gsplat
