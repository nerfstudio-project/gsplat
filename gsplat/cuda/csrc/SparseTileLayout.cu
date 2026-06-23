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

// for CUB_WRAPPER (Common.h, via Intersect.h)
#include <c10/cuda/CUDACachingAllocator.h>
#include <cub/cub.cuh>

#include "Intersect.h"

namespace gsplat {

namespace cg = cooperative_groups;

// One thread per pixel. Computes the dense tile id and raster-order in-tile
// position, tightly packed into a single int64 sort key: the tile id occupies
// the bits above `pos_bits` and the in-tile position the low `pos_bits`. The
// tight packing (no wasted middle bits) lets the radix sort bound its bit range
// to ~pos_bits + bits(tile id) instead of the full 64. Fusing this into one
// kernel also avoids the chain of host-side ATen elementwise/cast ops, which is
// launch-overhead bound.
__global__ void compute_tile_keys_kernel(
    const uint32_t num_pixels,
    const int32_t *__restrict__ pixels,    // [num_pixels, 2] (row, col)
    const int32_t *__restrict__ image_ids, // [num_pixels]
    const int64_t n_tiles,
    const int32_t tile_size,
    const int32_t tile_width,
    const int32_t pos_bits,
    int64_t *__restrict__ keys // [num_pixels]
) {
    const uint32_t p = cg::this_grid().thread_rank();
    if (p >= num_pixels)
        return;

    const int32_t row = pixels[2 * p];
    const int32_t col = pixels[2 * p + 1];
    const int64_t tile_id = static_cast<int64_t>(image_ids[p]) * n_tiles +
                            static_cast<int64_t>(row / tile_size) * tile_width +
                            static_cast<int64_t>(col / tile_size);
    const int64_t pix_in_tile =
        static_cast<int64_t>(row % tile_size) * tile_size + (col % tile_size);
    keys[p] = (tile_id << pos_bits) | pix_in_tile;
}

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
) {
    const uint32_t P = keys.size(0);
    if (P == 0)
        return;

    dim3 threads(256);
    dim3 grid((P + threads.x - 1) / threads.x);
    compute_tile_keys_kernel<<<
        grid,
        threads,
        0,
        at::cuda::getCurrentCUDAStream()>>>(
        P,
        pixels.const_data_ptr<int32_t>(),
        image_ids.const_data_ptr<int32_t>(),
        n_tiles,
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tile_width),
        static_cast<int32_t>(pos_bits),
        keys.data_ptr<int64_t>()
    );
}

// Bit-bounded radix sort of the (key, pixel-index) pairs, replacing at::sort:
// CUB sorts only [0, end_bit) of the tightly-packed key (fewer passes than a
// full 64-bit sort) and the DoubleBuffer avoids extra temp allocations. The
// sorted results land in `keys` or `keys_sorted` per CUB's selector; this points
// the *_sorted tensors at whichever buffer holds them (mirrors
// radix_sort_double_buffer).
void launch_sort_tile_keys(
    const int64_t P,
    const int end_bit,
    at::Tensor keys,        // [P] int64 (input; also a sort buffer)
    at::Tensor pixel_index, // [P] int64 (input arange; also a sort buffer)
    at::Tensor keys_sorted, // [P] int64 (scratch / result)
    at::Tensor pixel_map    // [P] int64 (scratch / result)
) {
    if (P <= 0)
        return;

    cub::DoubleBuffer<int64_t> d_keys(
        keys.data_ptr<int64_t>(), keys_sorted.data_ptr<int64_t>()
    );
    cub::DoubleBuffer<int64_t> d_values(
        pixel_index.data_ptr<int64_t>(), pixel_map.data_ptr<int64_t>()
    );
    CUB_WRAPPER(
        cub::DeviceRadixSort::SortPairs,
        d_keys,
        d_values,
        P,
        0,
        end_bit,
        at::cuda::getCurrentCUDAStream()
    );
    if (d_keys.selector == 0) {
        keys_sorted.set_(keys);
    }
    if (d_values.selector == 0) {
        pixel_map.set_(pixel_index);
    }
}

// One thread per active tile. The pixels are pre-sorted by key, so active tile
// ordinal t owns the contiguous sorted range [cumsum[t-1], cumsum[t]) (cumsum is
// the *inclusive* per-tile count scan). Each thread ORs its pixels' raster-order
// in-tile positions (the low 32 bits of each sorted key) into that tile's
// bitmask words. Because each thread owns a disjoint set of output words, no
// atomics are needed. out_bitmask must be zero-initialized by the caller.
__global__ void build_tile_bitmask_kernel(
    const uint32_t num_active_tiles,
    const uint32_t words_per_tile,
    const int32_t pos_bits,
    const int64_t *__restrict__ sorted_keys,       // [P]
    const int64_t *__restrict__ tile_pixel_cumsum, // [num_active_tiles]
    uint64_t *__restrict__ out_bitmask // [num_active_tiles * words_per_tile]
) {
    uint32_t t = cg::this_grid().thread_rank();
    if (t >= num_active_tiles)
        return;

    const int64_t pos_mask = (static_cast<int64_t>(1) << pos_bits) - 1;
    const int64_t start = (t == 0) ? 0 : tile_pixel_cumsum[t - 1];
    const int64_t end = tile_pixel_cumsum[t];
    uint64_t *words = out_bitmask + static_cast<int64_t>(t) * words_per_tile;

    for (int64_t k = start; k < end; ++k) {
        // The low `pos_bits` of the key hold the raster-order in-tile position.
        const int64_t bit = sorted_keys[k] & pos_mask;
        words[bit >> 6] |= (static_cast<uint64_t>(1) << (bit & 63));
    }
}

void launch_build_tile_bitmask_kernel(
    // inputs
    const at::Tensor sorted_keys,       // [P] int64, sorted (tile_id<<pos | pos)
    const at::Tensor tile_pixel_cumsum, // [num_active_tiles] int64, inclusive
    const uint32_t words_per_tile,
    const int64_t pos_bits,
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
        static_cast<int32_t>(pos_bits),
        sorted_keys.const_data_ptr<int64_t>(),
        tile_pixel_cumsum.const_data_ptr<int64_t>(),
        reinterpret_cast<uint64_t *>(out_bitmask.data_ptr<int64_t>())
    );
}

} // namespace gsplat
