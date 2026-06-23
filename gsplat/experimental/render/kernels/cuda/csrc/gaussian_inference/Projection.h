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

#include "InferenceTypes.h"

#include <ATen/core/Tensor.h>
#include <cstdint>

// Need CameraModelType from gsplat
#include "Common.h"

namespace higs {

struct SHDecodeParams; // defined in SHCompression.h (CUDA-only)

// Projection-only launch (FUSE_SH=false). Supports arbitrary B, C.
void launch_projection_fwd_kernel(
    // inputs
    const at::Tensor means,                // [3, N]
    const at::optional<at::Tensor> covars, // [N, 6] optional
    const at::Tensor inference,                  // [N, 8] half — packed {quat(4), scale(3), opacity(1)}
    const at::Tensor viewmats,             // [1, 1, 4, 4]
    const at::Tensor Ks,                   // [1, 1, 3, 3]
    const uint32_t image_width, const uint32_t image_height, const float eps2d, const float near_plane,
    const float far_plane, const float radius_clip, const gsplat::CameraModelType camera_model,
    // outputs
    at::Tensor visible,                    // [(N+31)/32] int32 packed bitfield
    at::Tensor means2d,                    // [1, 1, N, 2]
    at::Tensor depths,                     // [1, 1, N]
    at::Tensor conics,                     // [1, 1, N, 4] half {l0,l1,l2,opacity}
    at::optional<at::Tensor> compensations // [1, 1, N] optional
);

// Fused projection + SH evaluation launch (FUSE_SH=true).
// Camera world position is derived from viewmats inside the kernel.
void launch_projection_sh_fused_kernel(
    // inputs
    const at::Tensor means,                // [B, 3, N]
    const at::optional<at::Tensor> covars, // [B, N, 6] optional
    const at::Tensor inference,                  // [B, N, 8] half
    const at::Tensor viewmats,             // [B, C, 4, 4]
    const at::Tensor Ks,                   // [B, C, 3, 3]
    const uint32_t image_width, const uint32_t image_height, const float eps2d, const float near_plane,
    const float far_plane, const float radius_clip, const gsplat::CameraModelType camera_model,
    // SH inputs
    const int32_t degrees_to_use,
    const at::Tensor sh_input, // [N, K, 3] half (uncompressed) OR [M, 4] int32 (compressed)
    const float bias, const float min_value, const SHCompressionMode mode, const SHDecodeParams *decode_params,
    // outputs
    at::Tensor visible,                    // [(N+31)/32] int32 packed bitfield
    at::Tensor means2d,                    // [1, 1, N, 2]
    at::Tensor depths,                     // [1, 1, N]
    at::Tensor conics,                     // [1, 1, N, 4] half
    at::Tensor colors,                     // [N, 4] half {R,G,B,0}
    at::optional<at::Tensor> compensations // [1, 1, N] optional
);

} // namespace higs
