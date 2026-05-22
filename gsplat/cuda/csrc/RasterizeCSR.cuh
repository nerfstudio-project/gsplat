/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared helpers for the CSR-packed per-(tile, batch) state layout used by
 * the 3DGS world-space rasterization fwd persist + backward pass. The fwd
 * persists cumulative per-pixel state (T, pix_out[CDIM], normal_out[3]) at
 * each batch boundary so bwd can resume from any batch without re-walking
 * the full Gaussian sequence.
 *
 * The tiny batch-count device kernel `compute_batches_per_tile_kernel` and
 * the host helper `compute_batch_csr` that wraps it live in
 * `RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu` (a `.cu` translation unit, so that
 * nvcc can emit the kernel launch). This header only exposes the constants
 * + declarations so both `.cu` files and `Rasterization.cpp` (a pure C++
 * TU) can agree on the layout.
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <tuple>

#include <ATen/core/Tensor.h>

#include "Common.h"

namespace gsplat {

// Base batch-state element ordering written by fwd, excluding CDIM. The full
// state tensor is shaped
// `[total_batches][1 + CDIM + (ReturnNormals ? 3 : 0)][pixels_per_tile]`
// fp32. This SOA orientation makes pixels the fastest-varying axis, so
// warp-aligned loads/stores of one state element coalesce instead of striding
// by state_dim. Each slot stores (T, pix_out[CDIM]) plus normal_out[3] only
// when requested.
//
// Partials writes per-batch local state: T is the batch-local walk product and
// pix_out/normal_out are accumulated from a fresh per-batch seed. Batch-scan
// overwrites the same slot with absolute cumulative state before backward can
// read it.
constexpr uint32_t FWD_BATCH_STATE_T_OFFSET = 0;
constexpr uint32_t FWD_BATCH_STATE_PIX_OFFSET = 1;
// NORMAL offset depends on CDIM; callers compute it inline.
constexpr uint32_t FWD_BATCH_STATE_NORMAL_EXTRA = 3;

// Per-pixel compose handoff written in tile-major `[num_tiles, ppt]` order.
// Values below COMPOSE_C_STOP_INVALID_RAY name the batch where batch-scan
// handed the pixel to batch-replay. The top two uint16 values are sentinels:
// - NONE: valid pixel finished entirely in batch-scan.
// - INVALID_RAY: inside pixel whose camera/lidar ray is invalid; no batch
//   slot or backward walk is meaningful for that lane.
constexpr uint16_t COMPOSE_C_STOP_INVALID_RAY = 0xFFFEu;
constexpr uint16_t COMPOSE_C_STOP_NONE = 0xFFFFu;

GSPLAT_HOST_DEVICE inline uint16_t encode_compose_c_stop(
    int32_t c_stop
) {
    if (c_stop == -1) {
        return COMPOSE_C_STOP_NONE;
    }
    if (c_stop == -2) {
        return COMPOSE_C_STOP_INVALID_RAY;
    }
    assert(c_stop >= 0);
    assert(c_stop < static_cast<int32_t>(COMPOSE_C_STOP_INVALID_RAY));
    return static_cast<uint16_t>(c_stop);
}

GSPLAT_HOST_DEVICE inline int32_t decode_compose_c_stop(
    uint16_t packed
) {
    if (packed == COMPOSE_C_STOP_NONE) {
        return -1;
    }
    if (packed == COMPOSE_C_STOP_INVALID_RAY) {
        return -2;
    }
    return static_cast<int32_t>(packed);
}

// Host helper: compute the CSR batch structure (batches_per_tile,
// batch_offsets, total_batches) from `tile_offsets`. Shared by both fwd
// (to size the persist buffer) and bwd (to size the gradient-kernel grid).
// The blocking `.item<int32_t>()` readback is kept here (not compatible
// with CUDA graph capture, but cheap compared to the former host-side scan
// over tile_offsets).
//
// `dummy_options` supplies the target device; typically `means.options()`.
// Returns {batches_per_tile [num_tiles] int32, batch_offsets [num_tiles+1]
// int32, total_batches int64}. Definition lives in
// `RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu` so the CUDA kernel launch +
// cumsum can be emitted by nvcc.
std::tuple<at::Tensor, at::Tensor, int64_t>
compute_batch_csr(
    const at::Tensor &tile_offsets,
    int64_t n_isects,
    uint32_t num_tiles,
    int32_t pixels_per_tile,
    at::TensorOptions dummy_options
);

#ifdef __CUDACC__
template <uint32_t CDIM, bool ReturnNormals, typename scalar_t>
class FwdBatchSlotView {
public:
    static constexpr uint32_t StateDim =
        FWD_BATCH_STATE_PIX_OFFSET + CDIM +
        (ReturnNormals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0u);

    __device__ FwdBatchSlotView(
        scalar_t *__restrict__ fwd_batch_state,
        int32_t slot,
        int32_t pixels_per_tile,
        int32_t pix
    )
        : slot_base_(
              fwd_batch_state
              + slot_base_offset(slot, pixels_per_tile)),
          pixels_per_tile_(pixels_per_tile),
          pix_(pix) {
        assert(pix >= 0 && pix < pixels_per_tile);
    }

    __device__ float T() const {
        const float t = static_cast<float>(raw(FWD_BATCH_STATE_T_OFFSET));
        assert(t >= 0.0f && t <= 1.0f + 1e-5f);
        return t;
    }

    __device__ void setT(float value) {
        assert(value >= 0.0f && value <= 1.0f + 1e-5f);
        raw(FWD_BATCH_STATE_T_OFFSET) = static_cast<scalar_t>(value);
    }

    __device__ float feature(uint32_t k) const {
        assert(k < CDIM);
        return static_cast<float>(raw(FWD_BATCH_STATE_PIX_OFFSET + k));
    }

    __device__ void setFeature(uint32_t k, float value) {
        assert(k < CDIM);
        raw(FWD_BATCH_STATE_PIX_OFFSET + k) = static_cast<scalar_t>(value);
    }

    __device__ vec3 normal() const {
        static_assert(ReturnNormals,
                      "normal slots are only read when ReturnNormals=true");
        return vec3(
            static_cast<float>(raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 0)),
            static_cast<float>(raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 1)),
            static_cast<float>(raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 2)));
    }

    __device__ void setNormal(const vec3 &value) {
        static_assert(ReturnNormals,
                      "normal slots are only written when ReturnNormals=true");
        raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 0) =
            static_cast<scalar_t>(value.x);
        raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 1) =
            static_cast<scalar_t>(value.y);
        raw(FWD_BATCH_STATE_PIX_OFFSET + CDIM + 2) =
            static_cast<scalar_t>(value.z);
    }

protected:
    __device__ scalar_t &raw(uint32_t field) {
        assert(field < StateDim);
        const int32_t field_idx = static_cast<int32_t>(field);
        assert(field_idx <= (INT32_MAX - pix_) / pixels_per_tile_);
        const int32_t offset = field_idx * pixels_per_tile_ + pix_;
        return slot_base_[offset];
    }

    __device__ const scalar_t &raw(uint32_t field) const {
        assert(field < StateDim);
        const int32_t field_idx = static_cast<int32_t>(field);
        assert(field_idx <= (INT32_MAX - pix_) / pixels_per_tile_);
        const int32_t offset = field_idx * pixels_per_tile_ + pix_;
        return slot_base_[offset];
    }

private:
    __device__ static int32_t slot_base_offset(
        int32_t slot,
        int32_t pixels_per_tile
    ) {
        assert(slot >= 0);
        assert(pixels_per_tile > 0);
        const int32_t state_dim = static_cast<int32_t>(StateDim);
        assert(state_dim <= INT32_MAX / pixels_per_tile);
        const int32_t slot_stride = state_dim * pixels_per_tile;
        assert(slot_stride > 0);
        assert(slot <= INT32_MAX / slot_stride);
        return slot * slot_stride;
    }

    scalar_t *slot_base_;
    int32_t pixels_per_tile_;
    int32_t pix_;
};

template <typename PairT>
class FwdPartialsMetaView {
public:
    __device__ FwdPartialsMetaView(
        PairT *__restrict__ partials_meta,
        int32_t slot,
        int32_t pixels_per_tile,
        int32_t pix
    )
        : value_(
              partials_meta +
              offset(slot, pixels_per_tile, pix)) {}

    __device__ void set(
        int32_t last_idx_global,
        int32_t n_accumulated,
        int32_t logical_batch_start
    ) {
        *value_ = make_ushort2(
            encodeLast(last_idx_global, logical_batch_start),
            encodeCount(n_accumulated));
    }

    __device__ void reset() {
        *value_ = make_ushort2(0u, 0u);
    }

    __device__ bool needsBatchReplay() const {
        const ushort2 pair = *value_;
        return (pair.y & BATCH_REPLAY_FLAG) != 0u;
    }

    __device__ int32_t last(int32_t logical_batch_start) const {
        const ushort2 pair = *value_;
        if (pair.x == 0u) {
            return -1;
        }
        return logical_batch_start + static_cast<int32_t>(pair.x) - 1;
    }

    __device__ int32_t count() const {
        const ushort2 pair = *value_;
        return static_cast<int32_t>(pair.y & COUNT_MASK);
    }

private:
    static constexpr uint16_t BATCH_REPLAY_FLAG = 0x8000u;
    static constexpr uint16_t COUNT_MASK = 0x7fffu;

    __device__ static int32_t offset(
        int32_t slot,
        int32_t pixels_per_tile,
        int32_t pix
    ) {
        assert(slot >= 0);
        assert(pixels_per_tile > 0);
        assert(pix >= 0 && pix < pixels_per_tile);
        assert(slot <= (INT32_MAX - pix) / pixels_per_tile);
        return slot * pixels_per_tile + pix;
    }

    // Per-batch partial metadata is local to one logical
    // TILE_SIZE*TILE_SIZE Gaussian batch:
    // - x: last local gaussian index + 1, or 0 when no hit
    // - y: low 15 bits hold the number of blended gaussians in that logical
    //      batch; the high bit on pixel 0 is a CTA-level batch-replay flag.
    //
    // The full global last index is reconstructed from the batch's global
    // start. This keeps the transient partials tensor compact without changing
    // the final `last_ids` / `sample_counts` semantics.
    __device__ static uint16_t encodeLast(
        int32_t last_idx_global,
        int32_t logical_batch_start
    ) {
        if (last_idx_global >= 0) {
            assert(last_idx_global >= logical_batch_start);
            const int32_t last_local =
                last_idx_global - logical_batch_start;
            assert(last_local >= 0);
            assert(last_local < 0xFFFF);
            return static_cast<uint16_t>(last_local + 1);
        }
        return 0u;
    }

    __device__ static uint16_t encodeCount(int32_t n_accumulated) {
        assert(n_accumulated >= 0);
        assert(n_accumulated <= COUNT_MASK);
        return static_cast<uint16_t>(n_accumulated);
    }

    PairT *value_;
};

template <typename PairT>
class FwdBatchReplayPreambleView {
public:
    __device__ FwdBatchReplayPreambleView(
        PairT *__restrict__ batch_replay_preamble,
        int32_t tile_linear,
        int32_t pixels_per_tile,
        int32_t pix
    )
        : value_(
              batch_replay_preamble +
              offset(tile_linear, pixels_per_tile, pix)) {}

    __device__ void set(int32_t last_idx_global, int32_t n_accumulated) {
        *value_ = make_int2(last_idx_global, n_accumulated);
    }

    __device__ int32_t last() const {
        const int2 pair = *value_;
        return pair.x;
    }

    __device__ int32_t count() const {
        const int2 pair = *value_;
        return pair.y;
    }

private:
    __device__ static int32_t offset(
        int32_t tile_linear,
        int32_t pixels_per_tile,
        int32_t pix
    ) {
        assert(tile_linear >= 0);
        assert(pixels_per_tile > 0);
        assert(pix >= 0 && pix < pixels_per_tile);
        assert(tile_linear <= (INT32_MAX - pix) / pixels_per_tile);
        return tile_linear * pixels_per_tile + pix;
    }

    PairT *value_;
};

// Binary-search helper: given ascending `batch_offsets[num_tiles + 1]`, find
// tile t such that batch_offsets[t] <= bid < batch_offsets[t + 1]. Called
// once per CTA in per-batch-dispatched fwd/bwd kernels; O(log num_tiles)
// global-memory reads per CTA, which hit L2 after the first wave.
__device__ __forceinline__ uint32_t find_tile_for_block(
    const int32_t *__restrict__ batch_offsets,
    uint32_t num_tiles,
    uint32_t bid
) {
    uint32_t lo = 0;
    uint32_t hi = num_tiles;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (static_cast<uint32_t>(batch_offsets[mid + 1]) <= bid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}
#endif // __CUDACC__

} // namespace gsplat
