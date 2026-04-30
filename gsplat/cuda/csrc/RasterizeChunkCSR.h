/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared helpers for the CSR-packed per-(tile, chunk) state layout used by
 * the 3DGS world-space rasterization backward pass. The backward splits a
 * tile's Gaussian range into fixed-size chunks (CHUNK_BATCHES batches each)
 * and the fwd-pass persists cumulative per-pixel state (T, pix_out[CDIM],
 * normal_out[3]) at those same chunk boundaries so bwd can resume from any
 * chunk without re-walking the full Gaussian sequence.
 *
 * The tiny chunk-count device kernel `compute_chunks_per_tile_kernel` and
 * the host helper `compute_chunk_csr` that wraps it live in
 * `RasterizeToPixelsFromWorld3DGSBwd.cu` (a `.cu` translation unit, so that
 * nvcc can emit the kernel launch). This header only exposes the constants
 * + declarations so both `.cu` files and `Rasterization.cpp` (a pure C++
 * TU) can agree on the layout.
 */

#pragma once

#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>

namespace gsplat {

// Number of batches per chunk. Must match the value used by the backward
// gradient kernel and the forward chunk-boundary persist. Smaller values
// expose more chunk-level parallelism to bwd at the cost of more temp
// memory writes in fwd.
constexpr uint32_t CHUNK_BATCHES = 4;

// Base chunk-state dimension written by fwd, excluding CDIM. The full state
// tensor is `[total_chunks][pixels_per_tile][1 + CDIM + 3]` fp32, storing
// (T, pix_out[CDIM], normal_out[3]) at each chunk boundary.
constexpr uint32_t FWD_CHUNK_STATE_T_OFFSET = 0;
constexpr uint32_t FWD_CHUNK_STATE_PIX_OFFSET = 1;
// NORMAL offset depends on CDIM; callers compute it inline.
constexpr uint32_t FWD_CHUNK_STATE_NORMAL_EXTRA = 3;

// Host helper: compute the CSR chunk structure (chunks_per_tile,
// chunk_offsets, total_chunks) from `tile_offsets`. Shared by both fwd
// (to size the persist buffer) and bwd (to size the gradient-kernel grid).
// The blocking `.item<int32_t>()` readback is kept here (not compatible
// with CUDA graph capture, but cheap compared to the former host-side scan
// over tile_offsets).
//
// `dummy_options` supplies the target device; typically `means.options()`.
// Returns {chunks_per_tile [num_tiles] int32, chunk_offsets [num_tiles+1]
// int32, total_chunks int64}. Definition lives in
// `RasterizeToPixelsFromWorld3DGSBwd.cu` so the CUDA kernel launch +
// cumsum can be emitted by nvcc.
std::tuple<at::Tensor, at::Tensor, int64_t>
compute_chunk_csr(
    const at::Tensor &tile_offsets,
    int64_t n_isects,
    uint32_t num_tiles,
    uint32_t pixels_per_tile,
    at::TensorOptions dummy_options
);

} // namespace gsplat
