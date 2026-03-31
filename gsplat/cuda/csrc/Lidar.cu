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

#include <ATen/core/Tensor.h>
#include <torch/types.h>
#include "Lidars.cuh"
#include "Ops.h"

RowOffsetStructuredSpinningLidarModelParametersExtDevice::RowOffsetStructuredSpinningLidarModelParametersExtDevice(
    const gsplat::RowOffsetStructuredSpinningLidarModelParametersExt &params)
    : n_rows{static_cast<int>(params.row_elevations_rad.size(0))}
    , n_columns{static_cast<int>(params.column_azimuths_rad.size(0))}
    , fov_vert_rad{params.fov_vert_rad}
    , fov_horiz_rad{params.fov_horiz_rad}
    , fov_eps_rad{params.fov_eps_rad}
    , spinning_direction{params.spinning_direction}
    , spinning_frequency_hz{params.spinning_frequency_hz}
    , angles_to_columns_map{params.angles_to_columns_map.data_ptr<int32_t>()}
    , map_dim{static_cast<int>(params.angles_to_columns_map.size(1)), // .x=width
              static_cast<int>(params.angles_to_columns_map.size(0))} // .y=height
    , n_bins_azimuth{params.n_bins_azimuth}
    , n_bins_elevation{params.n_bins_elevation}
    , cdf_resolution_elevation{params.cdf_resolution_elevation()}
    , cdf_resolution_azimuth{params.cdf_resolution_azimuth()}
    , cdf_elevation{params.cdf_elevation.data_ptr<int32_t>()}
    , cdf_dense_ray_mask{params.cdf_dense_ray_mask.data_ptr<int32_t>()}
    , tiles_pack_info{reinterpret_cast<const glm::ivec2 *>(params.tiles_pack_info.data_ptr<int32_t>())}
    , tiles_to_elements_map{reinterpret_cast<const glm::ivec2 *>(params.tiles_to_elements_map.data_ptr<int32_t>())}
{
    CHECK_INPUT(params.angles_to_columns_map);
    CHECK_INPUT(params.cdf_elevation);
    CHECK_INPUT(params.cdf_dense_ray_mask);
    CHECK_INPUT(params.tiles_pack_info);
    CHECK_INPUT(params.tiles_to_elements_map);

    TORCH_CHECK(params.angles_to_columns_map.size(0) > 1 && params.angles_to_columns_map.size(1) > 1,
                "angles_to_columns_map dimensions must be > 1");

    this->map_resolution_rad = {
        params.fov_horiz_rad->span/(params.angles_to_columns_map.size(1)-1),
        params.fov_vert_rad->span/(params.angles_to_columns_map.size(0)-1)
    };

    if(params.angles_to_columns_map.dtype() != torch::kInt32)
    {
        throw std::invalid_argument("angles_to_columns_map dtype must be torch.int32");
    }

    if(params.tiles_pack_info.dtype() != torch::kInt32)
    {
        throw std::invalid_argument("tiles_pack_info dtype must be torch.int32");
    }

    if(params.tiles_to_elements_map.dtype() != torch::kInt32)
    {
        throw std::invalid_argument("tiles_to_elements_map dtype must be torch.int32");
    }
}
