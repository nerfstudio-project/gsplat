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

#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <c10/util/irange.h>
#include <hip/hip_runtime_api.h>

#include <vector>

namespace gsplat
{
void merge_streams()
{
    constexpr int merge_device_id = 0;
    hipEvent_t merge_event       = 0;
    std::vector<hipEvent_t> events(c10::hip::device_count());

    for(const auto device_id: c10::irange(c10::hip::device_count()))
    {
        C10_HIP_CHECK(hipSetDevice(device_id));
        auto stream = c10::hip::getCurrentHIPStream(device_id);
        C10_HIP_CHECK(hipEventCreateWithFlags(&events[device_id], hipEventDisableTiming));
        C10_HIP_CHECK(hipEventRecord(events[device_id], stream));
    }

    C10_HIP_CHECK(hipSetDevice(merge_device_id));
    C10_HIP_CHECK(hipEventCreateWithFlags(&merge_event, hipEventDisableTiming));
    auto merge_stream = c10::hip::getCurrentHIPStream(merge_device_id);
    for(const auto device_id: c10::irange(c10::hip::device_count()))
    {
        C10_HIP_CHECK(hipStreamWaitEvent(merge_stream, events[device_id]));
    }
    C10_HIP_CHECK(hipEventRecord(merge_event, merge_stream));

    for(const auto device_id: c10::irange(c10::hip::device_count()))
    {
        C10_HIP_CHECK(hipSetDevice(device_id));
        auto stream = c10::hip::getCurrentHIPStream(device_id);
        C10_HIP_CHECK(hipStreamWaitEvent(stream, merge_event));
    }

    for(const auto device_id: c10::irange(c10::hip::device_count()))
    {
        C10_HIP_CHECK(hipEventDestroy(events[device_id]));
    }
    C10_HIP_CHECK(hipEventDestroy(merge_event));
}
} // namespace gsplat
