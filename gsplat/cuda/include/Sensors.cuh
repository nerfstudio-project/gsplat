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

// Unified sensor model type system for kernel template dispatch.
//
// Combines camera models (from Cameras.cuh) and lidar models (from Lidars.cuh)
// into a single type list and variant.  The host-side kernel launch code
// stores the runtime-selected sensor's KernelParameters in a
// SensorModelKernelParamsVariant, then uses cuda::std::visit to dispatch
// to a CUDA kernel specialized on the concrete sensor model type.
//
// The reverse mapping (SensorModelFromKernelParams) recovers the full sensor
// model type from a KernelParameters alternative, enabling the visit lambda
// to instantiate the correct kernel template.

#include "Cameras.cuh"
#include "Lidars.cuh"

// Union of all camera and lidar model types.
using SensorModelTypes = TypeListCat<CameraModelTypes, LidarModelTypes>;

// A variant that can hold the KernelParameters for any sensor model.
using SensorModelKernelParamsVariant = gsplat::TypeListToKernelParamsVariant<SensorModelTypes>;

// Map a KernelParameters type back to its sensor model type.
template <typename KP>
using SensorModelFromKernelParams = gsplat::FindByKernelParams<KP, SensorModelTypes>;

// Lift a CameraModelKernelParamsVariant into the wider SensorModelKernelParamsVariant.
inline auto to_sensor_model_kernel_params(
    const CameraModelKernelParamsVariant& camera_params
) -> SensorModelKernelParamsVariant {
    return cuda::std::visit([](const auto& params) -> SensorModelKernelParamsVariant {
        return params;
    }, camera_params);
}
