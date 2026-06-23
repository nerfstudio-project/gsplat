/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

namespace at
{
class Tensor;
}

namespace gsplat
{
// Autograd-aware SH color evaluation; declared here for cross-TU orchestration
// callers (Rendering.cpp). Returns colors only.
at::Tensor spherical_harmonics(
    int64_t degrees_to_use,
    const at::Tensor &dirs,               // [..., 3]
    const at::Tensor &coeffs,             // [..., K, 3]
    const at::optional<at::Tensor> &masks // [...]
);

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    // outputs
    at::Tensor colors // [..., N, D]
);

void launch_spherical_harmonics_bwd_kernel(
    // inputs
    const uint32_t degrees_to_use,
    const at::Tensor dirs,                // [..., N, 3]
    const at::Tensor coeffs,              // [N, K, D]
    const at::optional<at::Tensor> masks, // [..., N]
    const at::Tensor v_colors,            // [..., N, D]
    // outputs
    at::Tensor v_coeffs,            // [N, K, D]
    at::optional<at::Tensor> v_dirs // [..., N, 3]
);

// Fused forward assembly of proj_features = [SH colors | extra | (depth)] for
// the unpacked rasterization path. Writes each complete output row in one
// coalesced pass, replacing the SH-eval + cat(color, extra) + cat(.., depth)
// chain.
void launch_assemble_proj_features_unpacked_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t degrees_to_use,
    const uint32_t Dc,
    const uint32_t E,
    const uint32_t color_post,
    const uint32_t extra_post,
    const bool has_depth,
    const bool depth_is_zero,
    const bool extra_has_c,
    const at::Tensor means,                  // [B, N, 3]
    const at::Tensor campos,                 // [B, C, 3]
    const at::Tensor coeffs,                 // [N, K, Dc]
    const at::optional<at::Tensor> extra,    // [B, C, N, E] or [B, N, E]
    const at::optional<at::Tensor> depths,   // [B, C, N]
    const at::optional<at::Tensor> masks,    // [B, C, N]
    at::Tensor out,                          // [B, C, N, Dc + E + has_depth]
    const at::optional<at::Tensor> relu_mask // [B, C, N, Dc]
);

// Autograd-aware fused assembly of proj_features = [SH colors | extra | (depth)]
// for the unpacked rasterization path. Forward writes the whole tensor in one
// coalesced kernel (folding SH eval + the +0.5/relu post-op + extra read + depth
// write); backward routes the color-slice gradient through the SH backward and
// passes extra/depth gradients straight through. Declared for cross-TU callers
// (Rendering.cpp orchestration). Returns proj_features [*batch, C, N, width].
at::Tensor assemble_proj_features(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,
    const at::Tensor &campos,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &extra,
    const at::optional<at::Tensor> &depths,
    const at::optional<at::Tensor> &masks
);

// Forward-only dispatcher op: assembles proj_features in place into `out`.
void assemble_proj_features_unpacked_fwd(
    int64_t degrees_to_use,
    int64_t B,
    int64_t C,
    int64_t N,
    int64_t Dc,
    int64_t E,
    int64_t color_post,
    int64_t extra_post,
    bool has_depth,
    bool depth_is_zero,
    bool extra_has_c,
    const at::Tensor &means,
    const at::Tensor &campos,
    const at::Tensor &coeffs,
    const at::optional<at::Tensor> &extra,
    const at::optional<at::Tensor> &depths,
    const at::optional<at::Tensor> &masks,
    at::Tensor &out,
    const at::optional<at::Tensor> &relu_mask
);
} // namespace gsplat
