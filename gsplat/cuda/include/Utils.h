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

#include <c10/cuda/CUDAFunctions.h>

#include <algorithm>
#include <cstddef>
#include <tuple>

namespace gsplat
{
inline std::tuple<size_t, size_t> aligned_chunk(size_t alignment, size_t size, c10::DeviceIndex device)
{
    size_t chunk_size
        = alignment * ((size + alignment * c10::cuda::device_count() - 1) / (c10::cuda::device_count() * alignment));
    auto chunk_offset = chunk_size * device;
    if(chunk_offset + chunk_size > size)
    {
        chunk_offset = std::min(chunk_offset, size);
        chunk_size   = std::min(chunk_size, size - chunk_offset);
    }
    return std::make_tuple(chunk_offset, chunk_size);
}

inline std::tuple<size_t, size_t> chunk(size_t size, c10::DeviceIndex device)
{
    return aligned_chunk(1u, size, device);
}

void merge_streams();
} // namespace gsplat
