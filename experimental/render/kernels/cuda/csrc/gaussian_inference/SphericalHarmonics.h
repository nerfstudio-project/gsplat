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

namespace higs {

struct SHDecodeParams; // forward declaration (defined in SHCompression.h)

void launch_spherical_harmonics_fwd_kernel(
    // inputs
    int32_t degrees_to_use,
    const at::Tensor means, // [3, N] float — gaussian centers (planar)
    float cam_x, float cam_y, float cam_z,
    const at::Tensor coeffs,              // [..., K, 3] OR packed int32 when compressed
    const at::optional<at::Tensor> masks, // [...]
    float bias, float min_value,
    // outputs
    at::Tensor colors, // [..., 4] half {R,G,B,0}
    // compressed-SH path: mode != NONE activates on-the-fly decode (requires degree 3, K=16).
    SHCompressionMode mode = SHCompressionMode::NONE, const SHDecodeParams *decode_params = nullptr);

} // namespace higs
