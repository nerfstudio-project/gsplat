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
void launch_ssim_losses_fwd_kernel(
    const at::Tensor &flags,  // [B, H, W] int32
    const at::Tensor &pred,   // [B, H, W, C] float32
    const at::Tensor &target, // [B, H, W, C] float32
    float factor,
    bool mask_mode_target,
    float constant_mask_value,
    at::Tensor &loss,          // [B, H, W, 1]
    at::Tensor &dm_dmu1,       // [B, C, H, W]
    at::Tensor &dm_dsigma1_sq, // [B, C, H, W]
    at::Tensor &dm_dsigma12    // [B, C, H, W]
);

void launch_ssim_losses_bwd_kernel(
    const at::Tensor &flags,  // [B, H, W] int32
    const at::Tensor &pred,   // [B, H, W, C] float32
    const at::Tensor &target, // [B, H, W, C] float32
    float factor,
    bool mask_mode_target,
    float constant_mask_value,
    const at::Tensor &v_loss,        // [B, H, W, 1]
    const at::Tensor &dm_dmu1,       // [B, C, H, W]
    const at::Tensor &dm_dsigma1_sq, // [B, C, H, W]
    const at::Tensor &dm_dsigma12,   // [B, C, H, W]
    at::Tensor &v_pred               // [B, H, W, C]
);
} // namespace gsplat
