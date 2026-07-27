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
#include "MathUtils.h"

namespace gsplat
{
namespace cg = cooperative_groups;

// One thread per active tile (+1 for the sentinel). For active tile i with
// dense linear id g = image_id * n_tiles + tile_id, reconstruct the gsplat
// (image, tile) sort-key prefix p = (image_id << tile_n_bits) | tile_id and
// binary-search the sorted isect_ids for the first intersection whose prefix
// is >= p. Because isect_ids is sorted by (image, tile, depth) and every
// intersection belongs to some active tile, this lower bound is exactly the
// exclusive prefix-sum start of the tile's flatten-id range.
__global__ void intersect_offset_sparse_kernel(
    const int64_t n_isects,
    const int64_t *__restrict__ isect_ids,    // [n_isects], sorted
    const int32_t *__restrict__ active_tiles, // [num_active_tiles], ascending
    const uint32_t num_active_tiles,
    const uint32_t n_tiles,
    const uint32_t tile_n_bits,
    int32_t *__restrict__ offsets // [num_active_tiles + 1]
)
{
    uint32_t i = cg::this_grid().thread_rank();
    if(i > num_active_tiles)
    {
        return;
    }

    // Trailing sentinel.
    if(i == num_active_tiles)
    {
        offsets[i] = static_cast<int32_t>(n_isects);
        return;
    }

    const int64_t g        = active_tiles[i];
    const int64_t image_id = g / n_tiles;
    const int64_t tile_id  = g % n_tiles;
    const int64_t prefix   = (image_id << tile_n_bits) | tile_id;

    // lower_bound: first index whose (image, tile) prefix >= prefix.
    int64_t lo = 0;
    int64_t hi = n_isects;
    while(lo < hi)
    {
        const int64_t mid        = (lo + hi) >> 1;
        const int64_t mid_prefix = isect_ids[mid] >> 32;
        if(mid_prefix < prefix)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    offsets[i] = static_cast<int32_t>(lo);
}

void launch_intersect_offset_sparse_kernel(
    // inputs
    const at::Tensor isect_ids,    // [n_isects], sorted
    const at::Tensor active_tiles, // [num_active_tiles] int32, ascending
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [num_active_tiles + 1] int32
)
{
    const int64_t n_isects          = isect_ids.size(0);
    const uint32_t num_active_tiles = active_tiles.size(0);
    const uint32_t n_tiles          = tile_width * tile_height;
    // Must match the (image, tile) field width used to pack isect_ids.
    const uint32_t tile_n_bits      = bits_for_count(n_tiles);

    // num_active_tiles + 1 threads (one per active tile plus the sentinel).
    dim3 threads(256);
    dim3 grid((num_active_tiles + 1 + threads.x - 1) / threads.x);
    intersect_offset_sparse_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        n_isects,
        isect_ids.const_data_ptr<int64_t>(),
        active_tiles.const_data_ptr<int32_t>(),
        num_active_tiles,
        n_tiles,
        tile_n_bits,
        offsets.data_ptr<int32_t>()
    );
}
} // namespace gsplat
