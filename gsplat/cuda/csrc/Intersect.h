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
#include <optional>
#include <tuple>

#include <ATen/core/Tensor.h>

#include "Common.h"
#include "Cameras.h"
#include "Lidars.h"
#include "ExternalDistortion.h"
#include "TorchUtils.h"

namespace at {
class Tensor;
}

namespace gsplat {

struct TileIntersectResult {
    at::Tensor tiles_per_gauss;
    at::Tensor isect_ids;
    at::Tensor flatten_ids;
};

template <> struct TorchArgDef<TileIntersectResult> {
    static auto to(const TileIntersectResult &r) {
        return to_torch_args(r.tiles_per_gauss, r.isect_ids, r.flatten_ids);
    }

    template <class TT>
    static TileIntersectResult from(TT &&t) {
        return {
            .tiles_per_gauss = std::get<0>(std::forward<TT>(t)),
            .isect_ids = std::get<1>(std::forward<TT>(t)),
            .flatten_ids = std::get<2>(std::forward<TT>(t)),
        };
    }
};

TileIntersectResult intersect_tile(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::optional<at::Tensor> &conics,
    const at::optional<at::Tensor> &opacities,
    const at::optional<at::Tensor> &image_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    std::optional<int64_t> n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
);

TileIntersectResult intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,
    const at::Tensor radii,
    const at::Tensor depths,
    const at::optional<at::Tensor> image_ids,
    const at::optional<at::Tensor> gaussian_ids,
    std::optional<int64_t> n_images,
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

// Build the per-active-tile layout consumed by sparse rasterization. Pixels are
// packed: a flat [P, 2] (row, col) tensor plus a [P] image_ids tensor (the
// per-element batch index). Returns:
//   active_tiles      [AT] int32, ascending dense tile ids with >=1 active pixel
//   active_tile_mask  [n_images, tile_height, tile_width] bool
//   tile_pixel_mask   [AT, words_per_tile] int64 raster-order bitmask of active
//                     pixels per active tile (words_per_tile = ceil(T*T/64))
//   tile_pixel_cumsum [AT] int64, inclusive per-active-tile active-pixel count
//   pixel_map         [P] int64, argsort to (tile_id, in-tile) sorted order
// PRECONDITION: no duplicate (image, row, col) pixels (caller deduplicates).
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
build_sparse_tile_layout(
    const at::Tensor &pixels,    // [P, 2] int, (row, col)
    const at::Tensor &image_ids, // [P] int
    int64_t n_images,
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

// One thread per pixel: tightly packs the dense tile id and raster-order in-tile
// position into a single int64 sort key (tile_id << pos_bits | pos).
void launch_compute_tile_keys_kernel(
    // inputs
    const at::Tensor pixels,    // [P, 2] int32, (row, col)
    const at::Tensor image_ids, // [P] int32
    const int64_t n_tiles,
    const int64_t tile_size,
    const int64_t tile_width,
    const int64_t pos_bits,
    // output
    at::Tensor keys // [P] int64
);

// Bit-bounded radix sort of (key, pixel index) pairs over [0, end_bit).
// keys/pixel_index are the inputs (and double-buffer halves); the sorted results
// are exposed through keys_sorted / pixel_map.
void launch_sort_tile_keys(
    const int64_t P,
    const int end_bit,
    at::Tensor keys,        // [P] int64
    at::Tensor pixel_index, // [P] int64 (arange)
    at::Tensor keys_sorted, // [P] int64
    at::Tensor pixel_map    // [P] int64
);

// Per-active-tile raster-order bitmask. One thread per active tile ORs its
// pixels' in-tile positions (low pos_bits of each sorted key) into the tile's
// words. out_bitmask must be zeroed.
void launch_build_tile_bitmask_kernel(
    // inputs
    const at::Tensor sorted_keys,       // [P] int64, sorted (tile_id<<pos | pos)
    const at::Tensor tile_pixel_cumsum, // [num_active_tiles] int64, inclusive
    const uint32_t words_per_tile,
    const int64_t pos_bits,
    // output
    at::Tensor out_bitmask // [num_active_tiles, words_per_tile] int64 (zeroed)
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
