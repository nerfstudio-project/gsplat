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
// Ground-gaussian distortion loss (HUGSIM). Constrains the height (y) and the
// roll/pitch rotation variance of gaussians that fall inside randomly placed
// depth bins along the camera z-axis.
//
// Forward: a per-point accumulation kernel fills the [B, 7] statistics buffer
// (sum/sum-of-squares of y, roll, pitch plus a count per bin), then a reductor
// kernel turns those statistics into the scalar loss.
void launch_ground_gaussians_fwd_kernel(
    const at::Tensor &positions,     // [N, 3]
    const at::Tensor &rotations,     // [N, 4] quaternion (x, y, z, w)
    const at::Tensor &cam_tquat,     // [7] camera-from-world (tx,ty,tz,qx,qy,qz,qw)
    const at::Tensor &random_values, // [B] bin offsets in [0, 1)
    float min_bias,
    float range_bias,
    float grid_len,
    float rotation_lambda,
    at::Tensor &stats, // [B, 7] accumulators (zero-initialized)
    at::Tensor &loss   // [] scalar (zero-initialized)
);

// Backward: re-derives the per-point camera-space quantities and the per-bin
// statistics (passed back from forward) to scatter gradients to positions and
// rotations.
void launch_ground_gaussians_bwd_kernel(
    const at::Tensor &positions,     // [N, 3]
    const at::Tensor &rotations,     // [N, 4] quaternion (x, y, z, w)
    const at::Tensor &cam_tquat,     // [7]
    const at::Tensor &random_values, // [B]
    const at::Tensor &stats,         // [B, 7]
    const at::Tensor &v_loss,        // [] scalar upstream gradient
    float min_bias,
    float range_bias,
    float grid_len,
    float rotation_lambda,
    at::Tensor &v_positions, // [N, 3]
    at::Tensor &v_rotations  // [N, 4]
);
} // namespace gsplat
