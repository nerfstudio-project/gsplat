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

#include <cstdint>
#include <tuple>

namespace at {
class Tensor;
}

namespace gsplat {
namespace scene {

/**
 * Pack activated Gaussian splat tensors into the inference viewer-internal layout.
 *
 * Inputs (all float32 CUDA, N gaussians):
 *   means      [N, 3]           world-space positions (after activation)
 *   quats      [N, 4]           unit quaternions in wxyz order
 *   scales     [N, 3]           positive scales (after exp())
 *   opacities  [N]              opacities in [0, 1] (after sigmoid())
 *   colors     [N, 3]           RGB, or [N, K, 3] for SH coefficients
 *   sh_degree  int64            -1 for RGB mode, 0–3 for SH
 *   sh_compression_mode int64   0=none, 1=32b, 2=16b
 *
 * Outputs:
 *   means_planar [3, N]  float32  transposed means for cache-friendly viewer access
 *   qso_packed        [N, 8]  float16  packed quaternion cols 0-3,
 *                              scale cols 4-6, opacity col 7
 *   colors_packed        [N, 4] fp16 for RGB; [N, K, 3] fp32 for SH0-2;
 *                        [N, 16, 3] fp16 for SH3 mode 0;
 *                        [N, 48] fp32 for SH3 mode 1;
 *                        [N, 48] fp16 for SH3 mode 2
 */
std::tuple<at::Tensor, at::Tensor, at::Tensor> pack_gaussian_inference_scene_cuda(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    int64_t sh_degree,
    int64_t sh_compression_mode
);

} // namespace scene
} // namespace gsplat
