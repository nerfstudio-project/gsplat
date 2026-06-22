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
#include <cuda_runtime.h>
#include <cstdint>
#include <utility>
#include <vector>

namespace higs {

// Macro-tile rasterize: 1 warp (32 threads) per (macro-tile, batch) pair.
// Each thread loads one gaussian to shared memory per mini-batch.
// For each gaussian: thread t tests tile t overlap (__ballot_sync gives
// 32-bit tile mask), then a compile-time unrolled loop over 32 tiles
// rasterizes into register-resident per-tile accumulators.
// Writes per-pixel partial RGBT to a sparse intermediate buffer.
void launch_macro_tile_rasterize(int32_t total_ctas,                    // total batch items across all macro-tiles
                                 int32_t n_macro_tiles,                 // number of macro-tiles
                                 int32_t macro_tile_cols,               // macro-tile grid width
                                 int32_t tile_size,                     // 8 or 16
                                 int32_t tile_width,                    // render-tile grid width
                                 int32_t tile_height,                   // render-tile grid height
                                 const at::Tensor &means2d,             // [N, 2] float
                                 const at::Tensor &conics,              // [N, 4] half {l0,l1,l2,opacity}
                                 const at::Tensor &colors,              // [N, 4] half {R,G,B,0}
                                 const at::Tensor &mt_gauss_offsets,    // [n_mt+1] macro-tile segment offsets
                                 const at::Tensor &mt_gauss_ids_sorted, // [n_mt_isects] sorted gaussian IDs
                                 const at::Tensor &mt_batch_offsets,    // [n_mt+1] gaussian batch start indices
                                 at::Tensor &tile_buffer,     // fixed-layout partial RGBT
                                 at::Tensor &active_mask);    // [total_ctas] uint32

// Post-blend: 1 warp (32 threads) per render tile.  Walks batches of parent
// macro-tile in order, reads partial RGBT from sparse tile_buffer, composites
// front-to-back with transmittance early-out.
void launch_macro_tile_post_blend(int32_t tile_width,                 // render-tile grid width
                                  int32_t tile_height,                // render-tile grid height
                                  int32_t tile_size,                  // 8 or 16
                                  int32_t n_macro_tiles,              // number of macro-tiles
                                  int32_t macro_tile_cols,            // macro-tile grid width
                                  const at::Tensor &mt_batch_offsets, // [n_mt+1] gaussian batch start indices
                                  const at::Tensor &tile_buffer,      // fixed-layout partial RGBT
                                  const at::Tensor &active_mask,      // [total_ctas] uint32
                                  const at::Tensor &backgrounds,      // [4] half {R,G,B,pad}
                                  uint32_t image_width, uint32_t image_height,
                                  at::Tensor &render_colors); // [H, W, 4] half {R,G,B,T} output

} // namespace higs
