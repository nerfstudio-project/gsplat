/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

// Spherical Beta color model from "Deformable Beta Splatting" (SIGGRAPH 2025),
// by Rong Liu, Dylan Sun, Meida Chen, Yue Wang and Andrew Feng.
//   Paper:              https://arxiv.org/abs/2501.18630
//   Project page:       https://rongliu-leo.github.io/beta-splatting/
//   Reference impl:     https://github.com/RongLiu-Leo/beta-splatting

#pragma once

#include <cstdint>

namespace at
{
class Tensor;
}

namespace gsplat
{
// Highest number of Spherical Beta lobes supported by the evaluation and
// backward kernels. The cost of a lobe is linear, so this bound is a guard
// against pathological inputs rather than a kernel limitation.
inline constexpr int SB_MAX_LOBES = 32;

// Spherical Beta models colour only: a lobe's angular parameters are shared by
// every channel, so unlike spherical harmonics the channel count is fixed.
inline constexpr int SB_NUM_CHANNELS = 3;

// Values stored per lobe, laid out as [r, g, b, theta, phi, beta].
inline constexpr int SB_LOBE_WIDTH = SB_NUM_CHANNELS + 3;

void launch_spherical_beta_fwd_kernel(
    // inputs
    const uint32_t lobes_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor base_colors,         // [..., N, 3]
    const at::Tensor coeffs,              // [N, M, 6]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    // outputs
    at::Tensor colors // [..., N, 3]
);

void launch_spherical_beta_bwd_kernel(
    // inputs
    const uint32_t lobes_to_use,
    const at::Tensor means,
    const at::Tensor viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::Tensor coeffs,              // [N, M, 6]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    const at::optional<at::Tensor> gaussian_ids,
    const at::Tensor v_colors, // [..., N, 3]
    // outputs
    at::Tensor v_coeffs,
    at::optional<at::Tensor> v_means,
    at::optional<at::Tensor> v_viewmats,
    at::optional<at::Tensor> v_viewmats_rs
);
} // namespace gsplat
