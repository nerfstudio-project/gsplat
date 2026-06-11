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

#include <ATen/core/Tensor.h>
#include <cuda_runtime.h>
#include <cstdint>

// SH compression mode selector (shared by renderer, CLI, and compression kernels).
// COMPRESS_32B/16B: full YCoCg quantization codec (distinct from scene packer's PACKED layout modes)
enum class SHCompressionMode
{
    NONE,         // no compression — use fp16 coefficients
    COMPRESS_32B, // 32 bytes/gaussian: DC YCoCg fp16/fp15 + HO Y@6b + HO CoCg@4b
    COMPRESS_16B  // 16 bytes/gaussian: DC YCoCg fp16/fp11 + HO Y@6b, no HO CoCg
};

// Abstract intersection stage interface.
// Implementations produce tile-sorted intersection lists consumed by the rasterizer.
class IIntersectionStage
{
public:
    virtual ~IIntersectionStage() = default;

    virtual void execute(const at::Tensor &means2d, const at::Tensor &depths, const at::Tensor &conics,
                         const at::Tensor &visible, int32_t tile_size, int32_t tile_width, int32_t tile_height,
                         cudaStream_t stream) = 0;
};
