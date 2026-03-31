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
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/Tensor.h>

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

/**
 * Host-side lidar parameters (Python RowOffsetStructuredSpinningLidarModelParametersExt).
 * Full definition required for CUDA host code and for MSVC when Lidar sources include Lidars.cuh.
 */
struct RowOffsetStructuredSpinningLidarModelParametersExt : torch::CustomClassHolder {
    at::Tensor row_elevations_rad;
    at::Tensor column_azimuths_rad;
    at::Tensor row_azimuth_offsets_rad;
    c10::intrusive_ptr<FOV> fov_vert_rad;
    c10::intrusive_ptr<FOV> fov_horiz_rad;
    float fov_eps_rad = 0.f;
    SpinningDirection spinning_direction = SpinningDirection::CLOCKWISE;
    float spinning_frequency_hz = 0.f;

    at::Tensor angles_to_columns_map;
    int n_bins_azimuth = 0;
    int n_bins_elevation = 0;
    at::Tensor cdf_elevation;
    at::Tensor cdf_dense_ray_mask;
    at::Tensor tiles_pack_info;
    at::Tensor tiles_to_elements_map;

    int n_rows() const {
        return static_cast<int>(row_elevations_rad.size(0));
    }
    int n_columns() const {
        return static_cast<int>(column_azimuths_rad.size(0));
    }

    int cdf_resolution_elevation() const {
        return static_cast<int>(cdf_dense_ray_mask.size(-1)) - 1;
    }
    int cdf_resolution_azimuth() const {
        return static_cast<int>(cdf_dense_ray_mask.size(-2)) - 1;
    }
};

} // namespace gsplat

