/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <cstdint>
#include <tuple>

#include "Common.h"
#include "Cameras.h"
#include "Lidars.h"
#include "ExternalDistortion.h"

namespace at {
class Tensor;
}

namespace gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::optional<at::Tensor> &conics,
    const at::optional<at::Tensor> &opacities,
    const at::optional<at::Tensor> &image_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
);

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,
    const at::Tensor radii,
    const at::Tensor depths,
    const at::optional<at::Tensor> image_ids,
    const at::optional<at::Tensor> gaussian_ids,
    int64_t I,
    bool sort,
    bool segmented
);

at::Tensor intersect_offset(
    const at::Tensor &isect_ids,
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
);

std::tuple<at::Tensor, at::Tensor> intersect_tile_sparse(
    const at::Tensor &means2d,                 // [I, N, 2] or [nnz, 2]
    const at::Tensor &radii,                   // [I, N, 2] or [nnz, 2]
    const at::Tensor &depths,                  // [I, N] or [nnz]
    const at::optional<at::Tensor> &image_ids, // [nnz] (packed mode)
    const at::Tensor &tile_mask,               // [I, tile_height, tile_width] bool
    const at::Tensor &active_tiles,            // [num_active_tiles] int32
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
);

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                           // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                             // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                            // [..., N] or [nnz]
    const at::optional<at::Tensor> conics,              // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> opacities,           // [..., N] or [nnz]      
    const at::optional<at::Tensor> image_ids,           // [nnz]
    const at::optional<at::Tensor> gaussian_ids,        // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids,     // [n_isects]
    // sparse-only: restrict enumeration to active tiles ([I, tile_height,
    // tile_width] bool). nullopt keeps the original dense behavior.
    const at::optional<at::Tensor> tile_mask = c10::nullopt
);
void launch_intersect_tile_lidar_kernel(
    // inputs
    const c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
);

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [..., tile_height, tile_width]
);

// Sparse offset encode: for each active tile (dense linear id
// image_id * n_tiles + tile_id, ascending), find the start of its intersection
// range in the sorted isect_ids via binary search on the (image, tile) key
// prefix. Produces a compacted [num_active_tiles + 1] offset table whose
// trailing sentinel equals n_isects.
void launch_intersect_offset_sparse_kernel(
    // inputs
    const at::Tensor isect_ids,    // [n_isects], sorted
    const at::Tensor active_tiles, // [num_active_tiles] int32, ascending
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [num_active_tiles + 1] int32
);

void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);

void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);

} // namespace gsplat
