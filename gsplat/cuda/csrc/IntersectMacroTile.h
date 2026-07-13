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

#include <cstdint>

#include <ATen/core/Tensor.h>

namespace at
{
class Tensor;
}

namespace gsplat
{
// Gaussian–tile overlap test for macro-tile binning and render-tile refinement.
// One IntersectionType selects the test at both stages.
enum class IntersectionType : int32_t
{
    // Tight conic ellipse (conics + opacity). Fewer false positives than AABB.
    // 3DGS only: requires 2D conic footprint. Not conservative for 3DGUT.
    ELLIPSE = 0,
    // AABB from radii; looser but conservative. Required for 3DGUT; optional for 3DGS.
    AABB    = 1,
};

// Count Gaussian intersections per macro tile.
void launch_mt_binning_count(
    const IntersectionType intersection_type,
    const at::Tensor &means2d,   // [I, N, 2] float32
    const at::Tensor &radii,     // [I, N, 2] int32
    const at::Tensor &conics,    // [I, N, 3] float32; conic upper triangle (a, b, c).
    const at::Tensor &opacities, // [I, N] float32
    const int32_t tile_size,
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    at::Tensor &mt_gauss_counts // [I, n_macro_tiles] int32 (out)
);

// Write (depth_key, gauss_id) pairs into per-macro-tile segments.
void launch_mt_binning_fill(
    const IntersectionType intersection_type,
    const at::Tensor &means2d,   // [I, N, 2] float32
    const at::Tensor &radii,     // [I, N, 2] int32
    const at::Tensor &depths,    // [I, N] float32
    const at::Tensor &conics,    // [I, N, 3] float32
    const at::Tensor &opacities, // [I, N] float32
    const int32_t tile_size,
    const int32_t macro_tile_cols,
    const int32_t macro_tile_rows,
    const at::Tensor &mt_gauss_offsets, // [I * n_macro_tiles + 1] int32
    at::Tensor &mt_gauss_write_cursor,  // [I, n_macro_tiles] int32 (in/out)
    at::Tensor &mt_depth_keys,          // [n_macro_isects] int32 (out)
    at::Tensor &mt_gauss_ids            // [n_macro_isects] int32 (out)
);

// Sort (depth_key, gauss_id) pairs within each image/macro-tile segment.
at::Tensor launch_mt_segmented_sort(
    const int64_t n_macro_isects,
    const int32_t n_macro_tiles,
    const at::Tensor &mt_gauss_offsets, // [I * n_macro_tiles + 1] int32
    at::Tensor &mt_depth_keys,          // [n_macro_isects] int32 (in/scratch)
    at::Tensor &mt_gauss_ids,           // [n_macro_isects] int32 (in/scratch)
    at::Tensor &mt_tmp_depth_keys,      // [n_macro_isects] int32 (scratch)
    at::Tensor &mt_tmp_gauss_ids        // [n_macro_isects] int32 (scratch)
);

// Convert macro-tile Gaussian ranges to per-CTA batch start indices.
// mt_gauss_batch_offsets[I * n_macro_tiles] == total_gauss_batches.
void launch_mt_gauss_batch_offsets(
    const at::Tensor &mt_gauss_offsets, // [I * n_macro_tiles + 1] int32
    at::Tensor &mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32 (out)
    const int32_t total_macro_tiles
);

// Test batched macro-tile Gaussians against render tiles and write bitmasks.
void launch_rt_gauss_batch_intersection_masks(
    const IntersectionType intersection_type,
    const int32_t total_gauss_batches,
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const at::Tensor &means2d,                // [I, N, 2] float32
    const at::Tensor &conics,                 // [I, N, 3] float32
    const at::Tensor &opacities,              // [I, N] float32
    const at::Tensor &mt_gauss_offsets,       // [I * n_macro_tiles + 1] int32
    const at::Tensor &mt_gauss_ids_sorted,    // [n_macro_isects] int32
    const at::Tensor &mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
    at::Tensor &rt_bitmasks                   // [n_rt_bitmask_words] int32 (out)
);

// Derive per-render-tile counts and write offsets from bitmasks.
void launch_rt_count_gauss_and_prefix_offsets(
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const int32_t rt_tile_cols,
    const int32_t rt_tile_rows,
    const at::Tensor &mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
    const at::Tensor &rt_bitmasks,            // [n_rt_bitmask_words] int32
    at::Tensor &rt_batch_gauss_offsets,       // [n_batch_rt_slots] int32 (out)
    at::Tensor &rt_gauss_counts               // [I * n_rt] int32 (out)
);

// Scatter sorted Gaussian IDs into final per-render-tile lists.
void launch_rt_binning_fill(
    const int32_t total_gauss_batches,
    const int32_t total_macro_tiles,
    const int32_t n_macro_tiles_per_image,
    const int32_t macro_tile_cols,
    const int32_t rt_tile_size_px,
    const int32_t rt_tile_cols,
    const int32_t rt_tile_rows,
    const at::Tensor &mt_gauss_offsets,       // [I * n_macro_tiles + 1] int32
    const at::Tensor &mt_gauss_ids_sorted,    // [n_macro_isects] int32
    const at::Tensor &mt_gauss_batch_offsets, // [I * n_macro_tiles + 1] int32
    const at::Tensor &rt_gauss_offsets,       // [I * n_rt + 1] int32
    const at::Tensor &rt_batch_gauss_offsets, // [n_batch_rt_slots] int32
    const at::Tensor &rt_bitmasks,            // [n_rt_bitmask_words] int32
    at::Tensor &rt_gauss_ids                  // [n_rt_isects] int32 (out)
);
} // namespace gsplat
