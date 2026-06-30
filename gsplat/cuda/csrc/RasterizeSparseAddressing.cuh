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

// Addressing shared by the sparse 3DGS ops (rasterizer fwd/bwd and the
// contributing-gaussian harness). Each block owns one active tile; outputs are
// packed in original-pixel order ([P, ...]). These map a block's active tile and
// a thread's in-tile pixel to the dense (image, tile) coordinates and the packed
// output slot, via the per-tile bitmask (popcount rank) and pixel_map produced
// by build_sparse_tile_layout.

namespace gsplat
{
// Decode a dense global tile id into (image, tile_x, tile_y).
__device__ __forceinline__ void sparse_decode_tile(
    const int32_t global_tile,
    const uint32_t n_tiles,
    const uint32_t tile_width,
    uint32_t &image_id,
    uint32_t &tile_x,
    uint32_t &tile_y
)
{
    const uint32_t gt            = static_cast<uint32_t>(global_tile);
    image_id                     = gt / n_tiles;
    const uint32_t tile_in_image = gt % n_tiles;
    tile_y                       = tile_in_image / tile_width;
    tile_x                       = tile_in_image % tile_width;
}

// Packed [P] output slot for the active pixel at raster-order in-tile position
// `in_tile` of an active tile. `tile_mask_words` points at that tile's bitmask
// words (tile_pixel_mask + ord * words) and `pix_start` is the exclusive start
// of the tile's active pixels in pixel_map (tile_pixel_cumsum[ord - 1], or 0).
// Returns -1 if the position holds no requested pixel. The rank is the number of
// active pixels before this in-tile position, so pixel_map maps it to the
// original (output) pixel index.
__device__ __forceinline__ int64_t sparse_pixel_slot(
    const uint64_t *__restrict__ tile_mask_words,
    const uint32_t in_tile,
    const int64_t pix_start,
    const int64_t *__restrict__ pixel_map
)
{
    const uint32_t word = in_tile >> 6;
    const uint32_t bit  = in_tile & 63u;
    if(!((tile_mask_words[word] >> bit) & 1ull))
    {
        return -1;
    }
    uint32_t rank = 0;
    for(uint32_t w = 0; w < word; ++w)
    {
        rank += __popcll(tile_mask_words[w]);
    }
    rank += __popcll(tile_mask_words[word] & (((uint64_t)1 << bit) - 1));
    return pixel_map[pix_start + rank];
}
} // namespace gsplat
