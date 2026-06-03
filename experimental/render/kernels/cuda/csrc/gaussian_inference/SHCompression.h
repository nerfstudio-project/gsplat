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
#include <cuda_fp16.h>
#include <cstdint>

namespace higs {

// Per-basis inverse scales for the encoder (fp32, indexed per basis).
// Index 0 = DC, set to 1.0 (unused by encoder's DC path but keeps struct well-defined).
// Shared by both 32B and 16B modes. For 16B mode, co[]/cg[] HO slots are set to 0.
struct SHEncodeParams
{
    float inv_scales[3][16];
};

// Per-basis scales for the decoder (half2, indexed per pair for SIMD).
// Pair layout: {k=0, k=1}, {k=2, k=3}, ..., {k=14, k=15}.
// Pair 0 low lane (DC) = 1.0 so the decoder can apply a uniform multiply.
// Shared by both 32B and 16B modes. For 16B mode, co[]/cg[] HO slots are 0.
struct SHDecodeParams
{
    __half2 scales[3][8];
};

// Full compression result returned by launch_sh_compress.
// packed tensor shape depends on mode: [2*N, 4] for 32B, [N, 4] for 16B.
struct SHCompressed
{
    at::Tensor packed;            // packed SH bitstream (int32)
    SHDecodeParams decode_params; // per-basis decoder scales
};

at::Tensor launch_sh_encode(const at::Tensor &coeffs, const SHEncodeParams &encode_params, SHCompressionMode mode);
// opacities: [N] float — per-gaussian opacity (sigmoid-applied). Gaussians with opacity
// below the internal threshold are excluded from percentile scale computation so their
// SH outliers don't inflate the quantization range for visible gaussians.
SHCompressed launch_sh_compress(const at::Tensor &coeffs, const at::Tensor &opacities, SHCompressionMode mode);

} // namespace higs
