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

struct RowOffsetStructuredSpinningLidarModelParametersExt : public torch::CustomClassHolder
{
    RowOffsetStructuredSpinningLidarModelParametersExt() = default;

    RowOffsetStructuredSpinningLidarModelParametersExt(
        at::Tensor row_elevations_rad,
        at::Tensor column_azimuths_rad,
        at::Tensor row_azimuth_offsets_rad,
        SpinningDirection spinning_direction,
        float spinning_frequency_hz,
        c10::intrusive_ptr<FOV> fov_vert_rad,
        c10::intrusive_ptr<FOV> fov_horiz_rad,
        float fov_eps_rad,
        at::Tensor angles_to_columns_map,
        int n_bins_azimuth,
        int n_bins_elevation,
        at::Tensor cdf_elevation,
        at::Tensor cdf_dense_ray_mask,
        at::Tensor tiles_pack_info,
        at::Tensor tiles_to_elements_map
    )
        : row_elevations_rad(std::move(row_elevations_rad)),
          column_azimuths_rad(std::move(column_azimuths_rad)),
          row_azimuth_offsets_rad(std::move(row_azimuth_offsets_rad)),
          spinning_direction(spinning_direction),
          spinning_frequency_hz(spinning_frequency_hz),
          fov_vert_rad(std::move(fov_vert_rad)),
          fov_horiz_rad(std::move(fov_horiz_rad)),
          fov_eps_rad(fov_eps_rad),
          angles_to_columns_map(std::move(angles_to_columns_map)),
          n_bins_azimuth(n_bins_azimuth),
          n_bins_elevation(n_bins_elevation),
          cdf_elevation(std::move(cdf_elevation)),
          cdf_dense_ray_mask(std::move(cdf_dense_ray_mask)),
          tiles_pack_info(std::move(tiles_pack_info)),
          tiles_to_elements_map(std::move(tiles_to_elements_map))
    {}

    int n_rows() const { return this->row_elevations_rad.size(0); }
    int n_columns() const { return this->column_azimuths_rad.size(0); }

    // Actual parameters directly related to the lidar model
    at::Tensor row_elevations_rad;
    at::Tensor column_azimuths_rad;
    at::Tensor row_azimuth_offsets_rad;

    SpinningDirection spinning_direction;
    float spinning_frequency_hz;

    // Now some values computed elsewhere from the above parameters.
    c10::intrusive_ptr<FOV> fov_vert_rad;
    c10::intrusive_ptr<FOV> fov_horiz_rad;
    float fov_eps_rad;

    at::Tensor angles_to_columns_map;

    // Lidar Tiling info
    int n_bins_azimuth;
    int n_bins_elevation;
    at::Tensor cdf_elevation;
    at::Tensor cdf_dense_ray_mask;
    at::Tensor tiles_pack_info;
    at::Tensor tiles_to_elements_map;
    int cdf_resolution_elevation() const { return this->cdf_dense_ray_mask.size(-2)-1; }
    int cdf_resolution_azimuth() const { return this->cdf_dense_ray_mask.size(-1)-1; }
};

} // namespace gsplat

