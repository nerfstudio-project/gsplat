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
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cstdint>

#include "Intersect.h"

namespace gsplat {

namespace cg = cooperative_groups;

// One thread per active tile. The pixels are pre-sorted by (tile_id, in-tile
// position), so active tile ordinal t owns the contiguous sorted range
// [cumsum[t-1], cumsum[t]) (cumsum is the *inclusive* per-tile count scan). Each
// thread ORs its pixels' raster-order positions into that tile's bitmask words.
// Because each thread owns a disjoint set of output words, no atomics are
// needed. out_bitmask must be zero-initialized by the caller.
__global__ void build_tile_bitmask_kernel(
    const uint32_t num_active_tiles,
    const uint32_t words_per_tile,
    const int64_t *__restrict__ pix_in_tile_sorted, // [P]
    const int64_t *__restrict__ tile_pixel_cumsum,   // [num_active_tiles]
    uint64_t *__restrict__ out_bitmask // [num_active_tiles * words_per_tile]
) {
    uint32_t t = cg::this_grid().thread_rank();
    if (t >= num_active_tiles)
        return;

    const int64_t start = (t == 0) ? 0 : tile_pixel_cumsum[t - 1];
    const int64_t end = tile_pixel_cumsum[t];
    uint64_t *words = out_bitmask + static_cast<int64_t>(t) * words_per_tile;

    for (int64_t k = start; k < end; ++k) {
        const int64_t bit = pix_in_tile_sorted[k];
        words[bit >> 6] |= (static_cast<uint64_t>(1) << (bit & 63));
    }
}

void launch_build_tile_bitmask_kernel(
    // inputs
    const at::Tensor pix_in_tile_sorted, // [P] int64, sorted by (tile, in-tile)
    const at::Tensor tile_pixel_cumsum,  // [num_active_tiles] int64, inclusive
    const uint32_t words_per_tile,
    // output
    at::Tensor out_bitmask // [num_active_tiles, words_per_tile] int64 (zeroed)
) {
    const uint32_t num_active_tiles = tile_pixel_cumsum.size(0);
    if (num_active_tiles == 0)
        return;

    dim3 threads(256);
    dim3 grid((num_active_tiles + threads.x - 1) / threads.x);
    build_tile_bitmask_kernel<<<
        grid,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        num_active_tiles,
        words_per_tile,
        pix_in_tile_sorted.const_data_ptr<int64_t>(),
        tile_pixel_cumsum.const_data_ptr<int64_t>(),
        reinterpret_cast<uint64_t *>(out_bitmask.data_ptr<int64_t>())
    );
}

} // namespace gsplat
