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

#include <torch/custom_class.h>

namespace gsplat {

// Lidar Camera Model Support

// Spinning direction enum
enum class SpinningDirection {
    CLOCKWISE = 0,
    COUNTER_CLOCKWISE = 1
};

struct FOV : public torch::CustomClassHolder
{
    FOV(float start = 0.f, float span = 0.f) : start(start), span(span) {}

    float start;
    float span;
};

// Plain FOV for device-side structs (no CustomClassHolder overhead)
struct FOVDevice
{
    FOVDevice() = default;

    FOVDevice(const c10::intrusive_ptr<FOV> &fov)
        : start{fov->start}
        , span{fov->span}
    {
    }

    float start;
    float span;
};

} // namespace gsplat

