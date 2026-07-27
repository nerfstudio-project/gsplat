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

#include "Utils.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/irange.h>
#include <cuda_runtime_api.h>

#include <vector>

namespace gsplat
{
void merge_streams()
{
    constexpr int merge_device_id = 0;
    cudaEvent_t merge_event       = 0;
    std::vector<cudaEvent_t> events(c10::cuda::device_count());

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaEventCreate(&events[device_id], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[device_id], stream));
    }

    C10_CUDA_CHECK(cudaSetDevice(merge_device_id));
    C10_CUDA_CHECK(cudaEventCreate(&merge_event, cudaEventDisableTiming));
    auto merge_stream = c10::cuda::getCurrentCUDAStream(merge_device_id);
    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaStreamWaitEvent(merge_stream, events[device_id]));
    }
    C10_CUDA_CHECK(cudaEventRecord(merge_event, merge_stream));

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaSetDevice(device_id));
        auto stream = c10::cuda::getCurrentCUDAStream(device_id);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, merge_event));
    }

    for(const auto device_id: c10::irange(c10::cuda::device_count()))
    {
        C10_CUDA_CHECK(cudaEventDestroy(events[device_id]));
    }
    C10_CUDA_CHECK(cudaEventDestroy(merge_event));
}
} // namespace gsplat
