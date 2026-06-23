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

#include "Config.h"

#if GSPLAT_BUILD_3DGUT

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <cstdint>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "PrimingChainEncoding.cuh"
#include "Rasterization.h"
#include "RasterizeCSR.cuh"
#include "RasterizeToPixelsFromWorld3DGS.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "TorchUtils.h"
#include "Utils.cuh"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;
using DeviceLidarParamsOpt =
    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice>;
using DeviceExternalDistortionParamsOpt =
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams>;

namespace cg = cooperative_groups;

using PixelCoordsCompact = PixelCoords;

__device__ __forceinline__ void mark_partials_meta_needs_batch_replay(
    ushort2 *__restrict__ partials_meta,
    int32_t slot,
    int32_t pixels_per_tile
) {
    // FwdPartialsMetaView stores the batch-replay flag in the high bit of
    // pixel 0's count half. Treat that ushort2 as one 32-bit word so multiple
    // stopping lanes can publish the same flag without a CTA-wide sync.
    constexpr unsigned int BATCH_REPLAY_FLAG_WORD = 0x80000000u;
    assert(slot >= 0);
    assert(pixels_per_tile > 0);
    assert(slot <= INT32_MAX / pixels_per_tile);
    atomicOr(
        reinterpret_cast<unsigned int *>(
            partials_meta + slot * pixels_per_tile),
        BATCH_REPLAY_FLAG_WORD);
}

template <uint32_t TILE_SIZE, uint32_t CTA_SIZE>
__device__ __forceinline__ PixelCoordsCompact compute_pixel_coords_compact(
    const CameraModelType camera_model_type,
    const uint32_t tile_id,
    const uint32_t tile_row,
    const uint32_t tile_col,
    const uint32_t thread_x,
    const uint32_t thread_y,
    const uint32_t p,
    const uint32_t image_width,
    const uint32_t image_height,
    const DeviceLidarParamsOpt &lidar_device_coeffs
) {
    constexpr uint32_t ROW_STRIDE = CTA_SIZE / TILE_SIZE;
    const uint32_t tile_element_id =
        thread_y * TILE_SIZE + thread_x + p * CTA_SIZE;
    return compute_pixel_coords(
        camera_model_type,
        tile_id,
        tile_row,
        tile_col,
        TILE_SIZE,
        thread_y + p * ROW_STRIDE,
        thread_x,
        tile_element_id,
        image_width,
        image_height,
        lidar_device_coeffs);
}

////////////////////////////////////////////////////////////////
// Z-parallel forward — three kernels, all compact-CTA:
//   1. partials kernel: one CTA per (tile, batch_id). Walks its batch's
//      gaussians with `T_local` reset to 1.0 and the standard alpha /
//      transmittance threshold checks. Writes per-pixel partial state
//      `(T_p_c, pix_p_c[CDIM], normal_p_c[3])` into `fwd_batch_state` and
//      packed `(last_p_c, n_p_c)` into `partials_meta`.
//
//   2. batch-scan kernel: one CTA per tile. Walks batches front-to-back,
//      folding partial summaries into per-pixel cumulative state
//      `(T_cum, pix_cum, normal_cum)` and rewriting `fwd_batch_state` so
//      bwd reads cumulative state at every slot. The fold uses the
//      partial summary as a fast path WHEN `T_cum * T_p_c` would stay
//      above `TRANSMITTANCE_THRESHOLD` (mathematically identical to the
//      original sequential semantic in that regime). When ANY thread in
//      the CTA would see `T_cum * T_p_c <= threshold`, the whole CTA
//      drops into a per-gaussian inner loop for that batch, applying the
//      original kernel's `next_T = T_global * (1-alpha) <= threshold`
//      check exactly. That keeps `T_cum > threshold` everywhere
//      `fwd_batch_state` is written and reproduces the original
//      truncation point bit-for-bit (modulo float-add reordering across
//      the batches already folded). Pixels that need exact replay write a
//      compact preamble for the next kernel.
//
//      The exact bit-for-bit truncation described above and the kernel-3
//      batch-replay below apply to ForBackward (exact/metadata) mode. In
//      fwd-only mode there is no replay: a threshold-crossing (`stop_here`)
//      batch folds its full partial summary and finalizes, over-counting the
//      post-saturation tail by a bounded, sub-threshold amount (see the
//      `if constexpr (!ForBackward)` branch in this kernel's batch-scan).
//
//   3. batch-replay kernel: one CTA per batch. CTAs selected by batch-scan
//      replay their threshold-crossing batch, then overwrite the corresponding
//      fwd_batch_state slot with the post-saturation state that backward needs.
//
// Compact-CTA layout: 1D thread block of CTA_SIZE threads processing a
// TILE_SIZE x TILE_SIZE pixel block, each thread carrying state for
// PIXELS_PER_THREAD = TILE_SIZE*TILE_SIZE/CTA_SIZE pixels stripe-
// distributed by ROW_STRIDE = CTA_SIZE/TILE_SIZE. Cooperative gaussian
// fetches are CTA_SIZE-wide (vs TILE_SIZE^2-wide pre-compaction), so
// shared memory shrinks proportionally and many CTAs co-reside per SM.
////////////////////////////////////////////////////////////////

// Helper: shared-memory pointer pack for the per-batch gaussian inner loop.
// Compact-CTA shmem holds CTA_SIZE entries per buffer (one per thread).
struct FwdShmemBatch {
    int32_t *id_batch;
    vec4 *xyz_opacity_batch;
    mat3 *iscl_rot_batch;
    vec3 *scale_batch;
    vec3 *normal_batch;
};

__device__ __forceinline__ FwdShmemBatch
fwd_unpack_shmem(int32_t *raw, uint32_t cta_size) {
    FwdShmemBatch b;
    b.id_batch = raw;
    b.xyz_opacity_batch = reinterpret_cast<vec4 *>(&b.id_batch[cta_size]);
    b.iscl_rot_batch =
        reinterpret_cast<mat3 *>(&b.xyz_opacity_batch[cta_size]);
    b.scale_batch =
        reinterpret_cast<vec3 *>(&b.iscl_rot_batch[cta_size]);
    b.normal_batch =
        reinterpret_cast<vec3 *>(&b.scale_batch[cta_size]);
    return b;
}

// Process one logical batch using the shared serial-batch gaussian load/blend
// primitive. The parallel forward needs this wrapper because partials and
// batch-replay both seed per-pixel state differently, but both walk one
// TILE_SIZE*TILE_SIZE gaussian batch at a time.
template <
    uint32_t CDIM,
    uint32_t TILE_SIZE,
    uint32_t CTA_SIZE,
    bool ReturnNormals,
    bool UseHitDistance,
    SaturationTPolicy SaturationPolicy,
    typename scalar_t
>
__device__ __forceinline__ void process_batch_gaussians_fwd(
    const uint32_t batch_id,
    const int32_t range_start,
    const int32_t range_end,
    const uint32_t tid,
    const uint32_t C,
    const uint32_t N,
    const int32_t *__restrict__ flatten_ids,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    const vec3 (&ray_o)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    const vec3 (&ray_d)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    int32_t *__restrict__ id_batch,
    vec4 *__restrict__ xyz_opacity_batch,
    mat3 *__restrict__ iscl_rot_batch,
    vec3 *__restrict__ scale_batch,
    vec3 *__restrict__ normal_batch,
    const float (&transmittance_threshold)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    float (&T)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    float (&saturating_T)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    float (&pix_out)[TILE_SIZE * TILE_SIZE / CTA_SIZE][CDIM],
    vec3 (&normal_out)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    int32_t (&cur_idx)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    int32_t (&n_accumulated)[TILE_SIZE * TILE_SIZE / CTA_SIZE],
    uint32_t& done_mask
) {
    constexpr uint32_t LOGICAL_BATCH = TILE_SIZE * TILE_SIZE;
    constexpr uint32_t FETCH_SIZE = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t ALL_DONE = (1u << PIXELS_PER_THREAD) - 1u;

    // Early pre-batch exit: all pixels owned by every CTA thread may already be
    // inactive before the first cooperative load, e.g. invalid rays or a batch
    // replay CTA where this thread has no engaged pixels. The vote reuses the
    // same CTA-wide sync primitive used at logical-batch boundaries.
    if (cta_sync_count<CTA_SIZE>(done_mask == ALL_DONE) >=
        static_cast<int32_t>(CTA_SIZE)) {
        return;
    }

    const uint32_t logical_batch_start = range_start + batch_id * LOGICAL_BATCH;
    if (logical_batch_start >= static_cast<uint32_t>(range_end)) {
        return;
    }

    (void)process_logical_batch_gaussians<CDIM, LOGICAL_BATCH, FETCH_SIZE, CTA_SIZE, PIXELS_PER_THREAD, true, SaturationPolicy, UseHitDistance, ReturnNormals, scalar_t>(
            // CTA scratch and tile range.
            tid, id_batch, xyz_opacity_batch, iscl_rot_batch, scale_batch, normal_batch,
            logical_batch_start, range_end, flatten_ids,
            // Gaussian inputs.
            means, quats, scales, opacities, colors, C, N,
            // Per-pixel rays and accumulation state.
            ray_o, ray_d, ALL_DONE, transmittance_threshold,
            T, saturating_T, pix_out, normal_out,
            cur_idx, n_accumulated, done_mask);
}

// LIDAR's tile element count varies per tile; cameras have a fixed
// pixels_per_tile = TILE_SIZE * TILE_SIZE. compute_pixel_coords_compact
// returns inside=false past element_count for LIDAR; the kernel uses
// `done_mask` to skip those pixels in the inner walk.

////////////////////////////////////////////////////////////////
// Partials kernel — one CTA per (tile, batch_id), compact-CTA layout.
////////////////////////////////////////////////////////////////
template <
    uint32_t CDIM,
    uint32_t TILE_SIZE,
    uint32_t CTA_SIZE,
    bool ReturnNormals,
    bool UseHitDistance,
    bool ForBackward,
    typename scalar_t
>
__global__ void __launch_bounds__(
    CTA_SIZE,
    min_blocks_for_cdim<CDIM, CTA_SIZE>())
rasterize_to_pixels_from_world_3dgs_fwd_partials_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const scalar_t *__restrict__ viewmats0,
    const scalar_t *__restrict__ viewmats1,
    const scalar_t *__restrict__ Ks,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ rays,
    const scalar_t *__restrict__ radial_coeffs,
    const scalar_t *__restrict__ tangential_coeffs,
    const scalar_t *__restrict__ thin_prism_coeffs,
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs,
    const DeviceLidarParamsOpt lidar_device_coeffs,
    const DeviceExternalDistortionParamsOpt external_distortion_device_params,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // CSR + outputs. `bid_to_slot[blockIdx.x]` maps the round-major launch
    // id to the tile-major slot used by fwd_batch_state and partials_meta.
    const int32_t *__restrict__ batch_offsets_csr, // [num_tiles + 1]
    const int32_t *__restrict__ bid_to_slot, // [total_batches]
    // Per-pixel priming-chain state, packed as
    // `(!sat, fp16 T, uint16 -next_batch)`. atomicMin propagates saturated
    // chain state first, then keeps the smallest T upper bound and uses the
    // negated batch key as a latest-wave tie-break.
    int32_t *__restrict__ priming_state,    // [B, C, image_height, image_width]
    uint16_t *__restrict__ compose_c_stop,  // [num_tiles, pixels_per_tile], or
                                            // null in fwd-only mode
    scalar_t *__restrict__ fwd_batch_state, // [total_batches, state_dim, pixels_per_tile]
    ushort2 *__restrict__ partials_meta     // [total_batches, pixels_per_tile]
) {
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    static_assert(PIXELS_PER_THREAD > 0,
                  "PIXELS_PER_THREAD == 0 — CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    cg::thread_block block = cg::this_thread_block();

    // -- Locate the launch slot and owning tile -----------------------------
    const uint32_t num_tiles_total = B * C * tile_height * tile_width;
    const int32_t bid = bid_to_slot[block.group_index().x];
    assert(bid >= 0);
    const int32_t tile_linear = static_cast<int32_t>(
        find_tile_for_block(
            batch_offsets_csr, num_tiles_total, static_cast<uint32_t>(bid)));
    const int32_t batch_id = bid - batch_offsets_csr[tile_linear];
    const int32_t tile_width_count = static_cast<int32_t>(tile_width);
    const int32_t tile_height_count = static_cast<int32_t>(tile_height);
    assert(tile_width_count > 0 && tile_height_count > 0);
    assert(tile_height_count <= INT32_MAX / tile_width_count);
    const int32_t tiles_per_image = tile_height_count * tile_width_count;
    constexpr int32_t pixels_per_tile = TILE_SIZE * TILE_SIZE;
    const int32_t image_index = tile_linear / tiles_per_image;
    const int32_t tile_id = tile_linear - image_index * tiles_per_image;
    const int32_t tile_row = tile_id / tile_width_count;
    const int32_t tile_col = tile_id - tile_row * tile_width_count;
    const uint32_t tid = block.thread_rank();
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // -- Move image-local pointers to this tile's image ----------------------
    // Mirrors the backward kernel setup so persisted state and gradients agree
    // on tile/image addressing.
    tile_offsets += image_index * tiles_per_image;
    if (masks != nullptr) {
        masks += image_index * tiles_per_image;
    }
    if (masks != nullptr && !masks[tile_id]) {
        if (tid == 0u) {
            // partials_meta comes from at::empty. Pixel 0 carries this slot's
            // batch-replay flag, so clear it even though masked tiles leave all
            // per-pixel partials unwritten and batch-scan returns early.
            FwdPartialsMetaView<ushort2> summary_view(
                partials_meta, bid, pixels_per_tile, 0);
            summary_view.reset();
        }
        return;
    }

    const int32_t image_width_px = static_cast<int32_t>(image_width);
    const int32_t image_height_px = static_cast<int32_t>(image_height);
    assert(image_width_px > 0 && image_height_px > 0);
    assert(image_height_px <= INT32_MAX / image_width_px);
    const int32_t image_area = image_height_px * image_width_px;
    assert(image_index <= INT32_MAX / image_area);
    priming_state += image_index * image_area;
    if (rays != nullptr) {
        rays += image_index * image_area * 6;
    }
    const RollingShutterParameters rs_params(
        viewmats0 + image_index * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + image_index * 16);
    const uint32_t image_index_u32 = static_cast<uint32_t>(image_index);
    const uint32_t tile_id_u32 = static_cast<uint32_t>(tile_id);
    const uint32_t tile_row_u32 = static_cast<uint32_t>(tile_row);
    const uint32_t tile_col_u32 = static_cast<uint32_t>(tile_col);

    // -- Build per-thread pixel coordinates and rays -------------------------
    // Each CTA thread owns PIXELS_PER_THREAD stripe-distributed pixels.
    PixelCoordsCompact pcs[PIXELS_PER_THREAD];
    vec3 ray_o[PIXELS_PER_THREAD] = {};
    vec3 ray_d[PIXELS_PER_THREAD] = {};
    bool valid_pixel[PIXELS_PER_THREAD] = {};
    uint32_t done_mask = 0u;
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        pcs[p] = compute_pixel_coords_compact<TILE_SIZE, CTA_SIZE>(
            camera_model_type, tile_id_u32, tile_row_u32, tile_col_u32,
            thread_x, thread_y, p,
            image_width, image_height, lidar_device_coeffs);
        const WorldRay ray = compute_world_ray<scalar_t>(
            image_index_u32, pcs[p].col, pcs[p].row, pcs[p].pix_id,
            pcs[p].inside,
            rs_params, rays, Ks,
            image_width, image_height,
            camera_model_type, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_device_coeffs, lidar_device_coeffs,
            external_distortion_device_params);
        if (ray.valid_flag) {
            ray_o[p] = ray.ray_org;
            ray_d[p] = ray.ray_dir;
            valid_pixel[p] = true;
        } else {
            done_mask |= (1u << p);
        }
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (image_index == static_cast<int32_t>(B * C) - 1) &&
            (tile_id == tiles_per_image - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];
    // batch_id is the forward depth-walk index from the front. The deepest
    // batch is the only partial batch when the tile's Gaussian count is not a
    // multiple of TILE_SIZE * TILE_SIZE.
    assert(batch_id >= 0);
    const uint32_t b = static_cast<uint32_t>(batch_id);

    // -- Prepare shared memory and per-pixel accumulator state ---------------
    extern __shared__ int s[];
    FwdShmemBatch smem =
        fwd_unpack_shmem(reinterpret_cast<int32_t *>(s), CTA_SIZE);

    float T[PIXELS_PER_THREAD];
    float saturating_T[PIXELS_PER_THREAD];
    float T_init_used[PIXELS_PER_THREAD];
    float transmittance_threshold[PIXELS_PER_THREAD];
    bool chain_saturated[PIXELS_PER_THREAD];
    float pix_out[PIXELS_PER_THREAD][CDIM] = {};
    vec3 normal_out[PIXELS_PER_THREAD] = {};
    int32_t cur_idx[PIXELS_PER_THREAD];
    int32_t n_accumulated[PIXELS_PER_THREAD] = {};
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        T[p] = 1.0f;
        if constexpr (ForBackward) {
            // exact/backward mode never reads saturating_T (fwd_terminal_batch
            // is always false there), so skip its init and leave the array
            // dead for elimination.
        } else {
            saturating_T[p] = 1.0f;
        }
        T_init_used[p] = 1.0f;
        transmittance_threshold[p] = TRANSMITTANCE_THRESHOLD;
        chain_saturated[p] = false;
        cur_idx[p] = -1;
        if (pcs[p].inside && b == 0u) {
            if (!valid_pixel[p]) {
                // Make priming_state the canonical invalid-ray marker. Exact
                // mode still mirrors this to compose_c_stop below for its
                // existing batch-replay/backward handoff.
                priming_state[pcs[p].pix_id] =
                    static_cast<int32_t>(gsplat::priming::INVALID_RAY_PACKED);
            }
            if constexpr (ForBackward) {
                const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
                const int32_t pix_rank_in_tile =
                    pix_y_in_tile * TILE_SIZE + thread_x;
                assert(tile_linear <=
                       (INT32_MAX - pix_rank_in_tile) / pixels_per_tile);
                const int32_t compose_idx =
                    tile_linear * pixels_per_tile + pix_rank_in_tile;
                compose_c_stop[compose_idx] =
                    valid_pixel[p]
                        ? COMPOSE_C_STOP_NONE
                        : COMPOSE_C_STOP_INVALID_RAY;
            }
        }
    }

    // -- Read the priming chain ---------------------------------------------
    // A relaxed 32-bit atomic load keeps `T_cum`, the
    // saturation flag, and `next_batch` from tearing. A stored writer with
    // K <= b gives an upper bound on this batch's true initial T; K > b is a
    // higher-wave race and is ignored to keep the seed conservative.
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        const uint32_t bit = 1u << p;
        if (!pcs[p].inside || b == 0u || ((done_mask & bit) != 0u)) {
            continue;
        }

        const unsigned int packed =
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device>{
                *reinterpret_cast<unsigned int *>(
                    &priming_state[pcs[p].pix_id])}
                .load(cuda::memory_order_relaxed);
        const gsplat::priming::DecodeForBatchResult priming_decision =
            gsplat::priming::decode_for_batch(
                packed, static_cast<int32_t>(b));

        if (!priming_decision.use_stored_state) {
            // A downstream wave currently owns the global minimum. That T can
            // be lower than the true start for this batch, so fall back to the
            // unprimed seed.
        } else if (priming_decision.chain_saturated) {
            // Some upstream batch already crossed the threshold inside its
            // per-particle walk. Skip this batch's local walk and propagate the
            // tightest known start-T upper bound with the saturation flag.
            chain_saturated[p] = true;
            T_init_used[p] = priming_decision.T_init_used;
            done_mask |= bit;
        } else {
            // Use the tightest known upper bound on the true starting T. The
            // per-batch walk still tracks an unscaled product, but its
            // threshold tightens by the priming value.
            assert(priming_decision.T_init_used > TRANSMITTANCE_THRESHOLD);
            T_init_used[p] = priming_decision.T_init_used;
            transmittance_threshold[p] =
                TRANSMITTANCE_THRESHOLD / priming_decision.T_init_used;
        }
        assert(T_init_used[p] >= 0.0f && T_init_used[p] <= 1.0f);
    }

    // -- Walk this logical Gaussian batch -----------------------------------
    // Partials persists the unscaled per-batch walk product. Priming only
    // tightens the per-pixel threshold above; it is not baked into the stored
    // batch summary.
    process_batch_gaussians_fwd<
        CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance,
        ForBackward
            ? SaturationTPolicy::StorePostSaturationT
            : SaturationTPolicy::KeepPreAndCapturePostSaturationT,
        scalar_t>(
        b, range_start, range_end, tid, C, N,
        flatten_ids, means, quats, scales, colors, opacities,
        ray_o, ray_d,
        smem.id_batch, smem.xyz_opacity_batch, smem.iscl_rot_batch,
        smem.scale_batch, smem.normal_batch,
        transmittance_threshold,
        T, saturating_T, pix_out, normal_out, cur_idx, n_accumulated,
        done_mask);

    // -- Persist the per-batch summary --------------------------------------
    // Persist this batch's partial state only for real rays. Invalid rays
    // are represented by COMPOSE_C_STOP_INVALID_RAY, so downstream kernels
    // must not depend on slot or metadata contents for those lanes.
    //
    // SOA layout: `fwd_batch_state[slot, k, pix]` and
    // `partials_meta[slot, pix]`. With pix as the fastest-varying axis,
    // each warp writes contiguous addresses for a fixed state element.
    const int32_t slot = batch_offsets_csr[tile_linear] + batch_id;
    const int32_t logical_batch_start =
        range_start + batch_id * pixels_per_tile;
    if (tid == 0u) {
        // `partials_meta` is allocated with at::empty. Pixel 0 carries the
        // per-slot batch-replay summary bit, so initialize that word even when
        // pixel 0 itself is invalid or saturated and never writes metadata.
        FwdPartialsMetaView<ushort2> summary_view(
            partials_meta, slot, pixels_per_tile, 0);
        summary_view.reset();
    }
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
        const int32_t pix_rank_in_tile =
            pix_y_in_tile * TILE_SIZE + thread_x;
        FwdBatchSlotView<CDIM, ReturnNormals, scalar_t> slot_view(
            fwd_batch_state, slot, pixels_per_tile, pix_rank_in_tile);

        // Slot.T is a plain non-negative per-batch ratio. Exact-metadata mode
        // stores post-saturation T so batch-scan can hand the boundary to
        // batch-replay. Fwd-only mode stores pre-saturation T/pix_out and
        // marks that slot terminal, allowing batch-scan to finalize the
        // exact rendered color/alpha without replaying the killing Gaussian.
        const float walk_prod = T[p];
        const bool this_batch_saturated =
            valid_pixel[p] &&
            (chain_saturated[p] || ((done_mask & (1u << p)) != 0u));
        const bool fwd_terminal_batch =
            !ForBackward && this_batch_saturated && !chain_saturated[p];
        if (valid_pixel[p]) {
            assert(walk_prod >= 0.0f && walk_prod <= 1.0f + 1e-5f);
            slot_view.setT(walk_prod);
            if (!this_batch_saturated || !ForBackward) {
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    slot_view.setFeature(k, pix_out[p][k]);
                }
                if constexpr (ReturnNormals) {
                    slot_view.setNormal(normal_out[p]);
                }
            }
        }

        // batch-scan reads partials_meta only for batches it folds. In exact
        // mode, saturating batches are handoff boundaries and use a separate
        // batch-replay preamble tensor. In fwd-only mode, batch-scan folds the
        // pre-saturation terminal slot and then finalizes the pixel.
        if (valid_pixel[p] && (!this_batch_saturated || !ForBackward)) {
            FwdPartialsMetaView<ushort2> meta_view(
                partials_meta, slot, pixels_per_tile, pix_rank_in_tile);
            meta_view.set(
                cur_idx[p], n_accumulated[p], logical_batch_start,
                fwd_terminal_batch);
        }

        if (valid_pixel[p]) {
            // Writer-batch b publishes (T_after, b + 1, saturated). Invalid
            // rays do not publish normal chain state: exact mode records them
            // in compose_c_stop, while fwd-only mode wrote
            // INVALID_RAY_PACKED during the b==0 setup above.
            // fwd_terminal_batch can only be true in fwd-only mode; exact mode
            // always uses the pre-saturation walk product, so the saturating_T
            // read is gated out of exact mode entirely.
            float chain_walk_prod;
            if constexpr (ForBackward) {
                chain_walk_prod = walk_prod;
            } else {
                chain_walk_prod =
                    fwd_terminal_batch ? saturating_T[p] : walk_prod;
            }
            const float T_after_for_chain =
                T_init_used[p] * chain_walk_prod;
            assert(T_after_for_chain >= 0.0f &&
                   T_after_for_chain <= 1.0f + 1e-5f);
            const float T_clamped = fminf(T_after_for_chain, 1.0f);
            assert(static_cast<int32_t>(b + 1u) <
                   gsplat::priming::detail::K_LIMIT);
            const unsigned int packed_out = gsplat::priming::encode(
                T_clamped, static_cast<int32_t>(b + 1u),
                this_batch_saturated);
            atomicMin(
                reinterpret_cast<unsigned int *>(
                    &priming_state[pcs[p].pix_id]),
                packed_out);
        }
    }
}

////////////////////////////////////////////////////////////////
// Batch-scan kernel — one CTA per tile, compact-CTA.
//
// This kernel folds per-batch partial summaries independently per pixel. When
// a pixel would cross the transmittance threshold by folding the next batch, it
// records that batch in `compose_c_stop` and leaves the per-gaussian replay to
// batch-replay. This keeps the hot path at one CTA-wide replay vote per tile,
// independent of the number of batches in that tile.
////////////////////////////////////////////////////////////////
template <
    uint32_t CDIM,
    uint32_t TILE_SIZE,
    uint32_t CTA_SIZE,
    bool ReturnNormals,
    bool UseHitDistance,
    bool ForBackward,
    // When false (unsafe_masked_tile_outputs=True), the masked-tile safe-store
    // below compiles out — masked outputs are left undefined. Mirrors the
    // serial path's SAFE_MASKED_OUTPUTS opt-out.
    bool SAFE_MASKED_OUTPUTS,
    typename scalar_t
>
__global__ void __launch_bounds__(
    CTA_SIZE,
    min_blocks_for_cdim<CDIM, CTA_SIZE>())
rasterize_to_pixels_from_world_3dgs_fwd_batch_scan_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    const scalar_t *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const CameraModelType camera_model_type,
    const DeviceLidarParamsOpt lidar_device_coeffs,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    // CSR (in/out): partials on entry, cumulative on exit.
    const int32_t *__restrict__ batches_per_tile,
    const int32_t *__restrict__ batch_offsets,
    const int32_t *__restrict__ priming_state,
    scalar_t *__restrict__ fwd_batch_state,
    ushort2 *__restrict__ partials_meta,
    int2 *__restrict__ batch_replay_preamble, // [num_tiles, pixels_per_tile]
    uint16_t *__restrict__ compose_c_stop,    // [num_tiles, pixels_per_tile]
    // Final outputs for pixels that do not need batch-replay.
    scalar_t *__restrict__ render_colors,
    scalar_t *__restrict__ render_alphas,
    scalar_t *__restrict__ render_normals,
    int32_t *__restrict__ last_ids,
    int32_t *__restrict__ sample_counts
) {
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    static_assert(PIXELS_PER_THREAD > 0,
                  "PIXELS_PER_THREAD == 0 — CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");

    cg::thread_block block = cg::this_thread_block();

    // -- Locate tile and move image-local pointers --------------------------
    const int32_t image_index = static_cast<int32_t>(block.group_index().x);
    const int32_t tile_row = static_cast<int32_t>(block.group_index().y);
    const int32_t tile_col = static_cast<int32_t>(block.group_index().z);
    const int32_t tile_width_count = static_cast<int32_t>(tile_width);
    const int32_t tile_height_count = static_cast<int32_t>(tile_height);
    assert(tile_width_count > 0 && tile_height_count > 0);
    assert(tile_height_count <= INT32_MAX / tile_width_count);
    const int32_t tiles_per_image = tile_height_count * tile_width_count;
    const int32_t tile_id = tile_row * tile_width_count + tile_col;
    const uint32_t tid = block.thread_rank();
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // Output, background, mask, and fwd-only priming pointers are indexed
    // image-locally below.
    const int32_t image_width_px = static_cast<int32_t>(image_width);
    const int32_t image_height_px = static_cast<int32_t>(image_height);
    assert(image_width_px > 0 && image_height_px > 0);
    assert(image_height_px <= INT32_MAX / image_width_px);
    const int32_t image_area = image_height_px * image_width_px;
    tile_offsets += image_index * tiles_per_image;
    render_colors += image_index * image_area * CDIM;
    render_alphas += image_index * image_area;
    if constexpr (ReturnNormals) {
        render_normals += image_index * image_area * 3;
    }
    if constexpr (ForBackward) {
        last_ids += image_index * image_area;
        if (sample_counts != nullptr) {
            sample_counts += image_index * image_area;
        }
    }
    if (backgrounds != nullptr) {
        backgrounds += image_index * CDIM;
    }
    if (masks != nullptr) {
        masks += image_index * tiles_per_image;
    }
    if constexpr (!ForBackward) {
        priming_state += image_index * image_area;
    }

    // -- Build the per-thread pixel list ------------------------------------
    // Batch-scan does not need rays; it folds summaries emitted by partials.
    PixelCoordsCompact pcs[PIXELS_PER_THREAD];
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        pcs[p] = compute_pixel_coords_compact<TILE_SIZE, CTA_SIZE>(
            camera_model_type,
            static_cast<uint32_t>(tile_id),
            static_cast<uint32_t>(tile_row),
            static_cast<uint32_t>(tile_col),
            thread_x, thread_y, p,
            image_width, image_height, lidar_device_coeffs);
    }

    // Masked tiles: under the default (SAFE_MASKED_OUTPUTS) write background /
    // empty defaults so the masked region reads as a defined empty image; under
    // unsafe_masked_tile_outputs the stores compile out and outputs stay
    // undefined. Either way there are no batch slots to touch, so we bail.
    if (masks != nullptr && !masks[tile_id]) {
        if constexpr (SAFE_MASKED_OUTPUTS) {
            #pragma unroll
            for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                if (!pcs[p].inside) {
                    continue;
                }
                const int32_t pix_id = pcs[p].pix_id;
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    render_colors[pix_id * CDIM + k] =
                        backgrounds == nullptr ? 0.0f : static_cast<float>(backgrounds[k]);
                }
                render_alphas[pix_id] = 0.0f;
                if constexpr (ReturnNormals) {
                    render_normals[pix_id * 3 + 0] = 0.0f;
                    render_normals[pix_id * 3 + 1] = 0.0f;
                    render_normals[pix_id * 3 + 2] = 0.0f;
                }
                if constexpr (ForBackward) {
                    last_ids[pix_id] = -1;
                    if (sample_counts != nullptr) {
                        sample_counts[pix_id] = 0;
                    }
                }
            }
        }
        return;
    }

    // -- Decode initial per-pixel stop state --------------------------------
    const int32_t tile_linear = image_index * tiles_per_image + tile_id;
    const uint32_t num_batches_this_tile = (batches_per_tile != nullptr)
        ? static_cast<uint32_t>(batches_per_tile[tile_linear])
        : 0u;
    const int32_t batch_base = (batch_offsets != nullptr)
        ? batch_offsets[tile_linear]
        : 0;
    constexpr int32_t pixels_per_tile = TILE_SIZE * TILE_SIZE;
    const int32_t range_start = tile_offsets[tile_id];
    // Exact-metadata mode reuses the ray-validity sentinel that partials wrote
    // into compose_c_stop. Fwd-only mode does not allocate compose_c_stop,
    // so it reads the equivalent sentinel from priming_state and uses the same
    // decoded c_stop values only in registers.
    uint32_t done_mask = 0u;
    int32_t c_stop[PIXELS_PER_THREAD];
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        c_stop[p] = decode_compose_c_stop(COMPOSE_C_STOP_NONE);
        if (num_batches_this_tile == 0 || !pcs[p].inside) {
            done_mask |= (1u << p);
            continue;
        }

        int32_t initial_c_stop = decode_compose_c_stop(COMPOSE_C_STOP_NONE);
        if constexpr (!ForBackward) {
            const uint32_t packed =
                static_cast<uint32_t>(priming_state[pcs[p].pix_id]);
            initial_c_stop = gsplat::priming::is_invalid_ray(packed)
                ? decode_compose_c_stop(COMPOSE_C_STOP_INVALID_RAY)
                : decode_compose_c_stop(COMPOSE_C_STOP_NONE);
        } else {
            const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
            const int32_t pix_rank_in_tile =
                pix_y_in_tile * TILE_SIZE + thread_x;
            assert(tile_linear <=
                   (INT32_MAX - pix_rank_in_tile) / pixels_per_tile);
            const int32_t compose_idx =
                tile_linear * pixels_per_tile + pix_rank_in_tile;
            initial_c_stop = decode_compose_c_stop(
                compose_c_stop[compose_idx]);
            assert(
                initial_c_stop == decode_compose_c_stop(COMPOSE_C_STOP_NONE) ||
                initial_c_stop == decode_compose_c_stop(COMPOSE_C_STOP_INVALID_RAY));
        }
        c_stop[p] = initial_c_stop;
        if (initial_c_stop == decode_compose_c_stop(COMPOSE_C_STOP_INVALID_RAY)) {
            done_mask |= (1u << p);
        }
    }

    // -- Scan batch summaries front-to-back ---------------------------------
    float T_cum[PIXELS_PER_THREAD];
    float pix_cum[PIXELS_PER_THREAD][CDIM] = {};
    vec3 normal_cum[PIXELS_PER_THREAD] = {};
    int32_t last_idx_global[PIXELS_PER_THREAD];
    int32_t n_acc_global[PIXELS_PER_THREAD] = {};
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        T_cum[p] = 1.0f;
        last_idx_global[p] = -1;
    }
    uint32_t batch_replay_mask = 0u;

    // Iterate batches front-to-back: c = 0 (front-most) up to
    // c = num_batches-1 (terminal slot). fwd_batch_state and partials_meta
    // both use pixel rank as the fastest-varying axis.
    for (int32_t c = 0; c < static_cast<int32_t>(num_batches_this_tile); ++c) {
        const int32_t slot = batch_base + c;
        const int32_t logical_batch_start =
            range_start + c * pixels_per_tile;
        const uint32_t batch_replay_mask_before_batch = batch_replay_mask;

        // Per-pixel: either fold this batch's partial, or stop just before it
        // and save enough state for batch-replay to replay the batch
        // gaussian-by-gaussian. Pixels that are outside, invalid, or already
        // stopped simply carry their current accumulator through the slot write
        // below so bwd never observes uninitialized batch state.
        #pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
            const int32_t pix_rank_in_tile =
                pix_y_in_tile * TILE_SIZE + thread_x;
            const uint32_t bit = 1u << p;
            const bool inactive = ((done_mask | batch_replay_mask) & bit) != 0u;
            const FwdBatchSlotView<CDIM, ReturnNormals, scalar_t> partial_slot(
                fwd_batch_state, slot, pixels_per_tile, pix_rank_in_tile);
            const FwdPartialsMetaView<const ushort2> meta_view(
                partials_meta, slot, pixels_per_tile, pix_rank_in_tile);
            const float walk_prod = inactive ? 1.0f : partial_slot.T();
            const bool terminal_batch =
                !ForBackward && !inactive && meta_view.isTerminalBatch();
            const bool fold_this =
                !inactive &&
                (((T_cum[p] * walk_prod) > TRANSMITTANCE_THRESHOLD) ||
                 terminal_batch);
            const bool stop_here = !inactive && !fold_this;

            if (fold_this) {
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    const float pix_p_c = partial_slot.feature(k);
                    pix_cum[p][k] =
                        pix_cum[p][k] + T_cum[p] * pix_p_c;
                }
                if constexpr (ReturnNormals) {
                    const vec3 normal_p = partial_slot.normal();
                    normal_cum[p].x =
                        normal_cum[p].x + T_cum[p] * normal_p.x;
                    normal_cum[p].y =
                        normal_cum[p].y + T_cum[p] * normal_p.y;
                    normal_cum[p].z =
                        normal_cum[p].z + T_cum[p] * normal_p.z;
                }
                T_cum[p] = T_cum[p] * walk_prod;
                if constexpr (ForBackward) {
                    const int32_t last_p_c =
                        meta_view.last(logical_batch_start);
                    const int32_t n_p_c = meta_view.count();
                    if (last_p_c >= 0) {
                        last_idx_global[p] = last_p_c;
                    }
                    n_acc_global[p] += n_p_c;
                }
                if (terminal_batch) {
                    done_mask |= bit;
                }
            } else if (stop_here) {
                if constexpr (!ForBackward) {
                    // Fwd-only has no batch-replay fallback. Partials should
                    // have flagged this saturating batch terminal, but under
                    // unordered partials-CTA execution a stale upper-bound
                    // priming T_init can leave isTerminalBatch() false for the
                    // true terminal batch (the local threshold
                    // THRESH/T_init is then too tight). Rather than drop the
                    // batch and lose its whole contribution, fold its full
                    // partial summary and finalize. This over-counts the
                    // gaussians past the exact saturation point, but they are
                    // weighted by the already-sub-threshold T_cum so the error
                    // is bounded — fwd-only render is therefore not bit-exact
                    // vs the exact/backward path in this case.
                    #pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        pix_cum[p][k] =
                            pix_cum[p][k] + T_cum[p] * partial_slot.feature(k);
                    }
                    if constexpr (ReturnNormals) {
                        const vec3 normal_p = partial_slot.normal();
                        normal_cum[p].x = normal_cum[p].x + T_cum[p] * normal_p.x;
                        normal_cum[p].y = normal_cum[p].y + T_cum[p] * normal_p.y;
                        normal_cum[p].z = normal_cum[p].z + T_cum[p] * normal_p.z;
                    }
                    T_cum[p] = T_cum[p] * walk_prod;
                    done_mask |= bit;
                } else {
                    assert(c_stop[p] == -1);
                    assert(T_cum[p] > TRANSMITTANCE_THRESHOLD);
                    batch_replay_mask |= bit;
                    c_stop[p] = c;

                    // Save the accumulator preamble for batch-replay. The
                    // batch-state slot below receives the matching batch-start
                    // T/pix/normal accumulator state.
                    FwdBatchReplayPreambleView<int2> preamble_view(
                        batch_replay_preamble, tile_linear, pixels_per_tile,
                        pix_rank_in_tile);
                    preamble_view.set(last_idx_global[p], n_acc_global[p]);
                    mark_partials_meta_needs_batch_replay(
                        partials_meta, slot, pixels_per_tile);
                }
            }
        }

        // Write cumulative state back to slot c. Pixels that stopped in an
        // earlier batch skip this because batch-replay will propagate their
        // saturated state. Pixels that stop in this batch still write the
        // pre-fold batch-start state so batch-replay can seed from it.
        if constexpr (ForBackward) {
            #pragma unroll
            for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                const uint32_t bit = 1u << p;
                if (((done_mask | batch_replay_mask_before_batch) & bit) != 0u) {
                    continue;
                }
                const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
                const int32_t pix_rank_in_tile =
                    pix_y_in_tile * TILE_SIZE + thread_x;
                FwdBatchSlotView<CDIM, ReturnNormals, scalar_t> cumulative_slot(
                    fwd_batch_state, slot, pixels_per_tile, pix_rank_in_tile);
                cumulative_slot.setT(T_cum[p]);
                #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    cumulative_slot.setFeature(k, pix_cum[p][k]);
                }
                if constexpr (ReturnNormals) {
                    cumulative_slot.setNormal(normal_cum[p]);
                }
            }
        }
    }

    if constexpr (ForBackward) {
        // -- Publish batch-replay handoff -----------------------------------
        // Decoded c_stop == -1 means the pixel never
        // needed gaussian-by-gaussian replay and this kernel writes its outputs.
        #pragma unroll
        for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
            const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
            const int32_t pix_rank_in_tile =
                pix_y_in_tile * TILE_SIZE + thread_x;
            assert(tile_linear <=
                   (INT32_MAX - pix_rank_in_tile) / pixels_per_tile);
            const int32_t compose_idx =
                tile_linear * pixels_per_tile + pix_rank_in_tile;
            compose_c_stop[compose_idx] = encode_compose_c_stop(c_stop[p]);
        }
    }

    // -- Emit final outputs for pixels handled by batch-scan -----------------
    // Batch-replay pixels are finalized after their exact per-gaussian replay.
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if (!pcs[p].inside || c_stop[p] >= 0) {
            continue;
        }
        const int32_t pix_id = pcs[p].pix_id;
        render_alphas[pix_id] = 1.0f - T_cum[p];
        #pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = (backgrounds == nullptr)
                ? pix_cum[p][k]
                : (pix_cum[p][k] + T_cum[p] * static_cast<float>(backgrounds[k]));
        }
        if constexpr (ReturnNormals) {
            render_normals[pix_id * 3 + 0] = normal_cum[p].x;
            render_normals[pix_id * 3 + 1] = normal_cum[p].y;
            render_normals[pix_id * 3 + 2] = normal_cum[p].z;
        }
        if constexpr (ForBackward) {
            last_ids[pix_id] = last_idx_global[p];
            if (sample_counts != nullptr) {
                sample_counts[pix_id] = n_acc_global[p];
            }
        }
    }
}

////////////////////////////////////////////////////////////////
// Batch-replay kernel — one CTA per (tile, batch_id), compact-CTA.
//
// Most batch CTAs exit after a single metadata flag read. Engaging CTAs replay
// exactly the threshold-crossing batch for pixels whose `compose_c_stop`
// matches this batch_id, then overwrite that batch-state slot with the
// post-replay state. The preamble has already been consumed before the
// overwrite, and backward needs the post-saturation state at c_stop.
////////////////////////////////////////////////////////////////
template <
    uint32_t CDIM,
    uint32_t TILE_SIZE,
    uint32_t CTA_SIZE,
    bool ReturnNormals,
    bool UseHitDistance,
    typename scalar_t
>
__global__ void __launch_bounds__(
    CTA_SIZE,
    min_blocks_for_cdim<CDIM, CTA_SIZE>())
rasterize_to_pixels_from_world_3dgs_fwd_batch_replay_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ colors,
    const scalar_t *__restrict__ opacities,
    const scalar_t *__restrict__ backgrounds,
    const bool *__restrict__ masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const scalar_t *__restrict__ viewmats0,
    const scalar_t *__restrict__ viewmats1,
    const scalar_t *__restrict__ Ks,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const float *__restrict__ rays,
    const scalar_t *__restrict__ radial_coeffs,
    const scalar_t *__restrict__ tangential_coeffs,
    const scalar_t *__restrict__ thin_prism_coeffs,
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs,
    const DeviceLidarParamsOpt lidar_device_coeffs,
    const DeviceExternalDistortionParamsOpt external_distortion_device_params,
    const int32_t *__restrict__ tile_offsets,
    const int32_t *__restrict__ flatten_ids,
    const int32_t *__restrict__ batches_per_tile,
    const int32_t *__restrict__ batch_offsets,
    const int32_t *__restrict__ bid_to_slot,
    scalar_t *__restrict__ fwd_batch_state,
    const ushort2 *__restrict__ partials_meta,  // [total_batches, pixels_per_tile]
    const int2 *__restrict__ batch_replay_preamble, // [num_tiles, pixels_per_tile]
    const uint16_t *__restrict__ compose_c_stop,    // [num_tiles, pixels_per_tile]
    scalar_t *__restrict__ render_colors,
    scalar_t *__restrict__ render_alphas,
    scalar_t *__restrict__ render_normals,
    int32_t *__restrict__ last_ids,
    int32_t *__restrict__ sample_counts
) {
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    constexpr int32_t pixels_per_tile    = TILE_SIZE * TILE_SIZE;

    cg::thread_block block = cg::this_thread_block();

    // -- Skip batch slots that batch-scan did not mark for replay ------------
    const uint32_t num_tiles_total = B * C * tile_height * tile_width;
    const int32_t bid = bid_to_slot[block.group_index().x];
    assert(bid >= 0);
    const FwdPartialsMetaView<const ushort2> summary_view(
        partials_meta, bid, pixels_per_tile, 0);
    if (!summary_view.needsBatchReplay()) {
        return;
    }
    const int32_t tile_linear = static_cast<int32_t>(
        find_tile_for_block(
            batch_offsets, num_tiles_total, static_cast<uint32_t>(bid)));
    const int32_t batch_id = bid - batch_offsets[tile_linear];
    const int32_t tile_width_count = static_cast<int32_t>(tile_width);
    const int32_t tile_height_count = static_cast<int32_t>(tile_height);
    assert(tile_width_count > 0 && tile_height_count > 0);
    assert(tile_height_count <= INT32_MAX / tile_width_count);
    const int32_t tiles_per_image = tile_height_count * tile_width_count;
    const int32_t image_index = tile_linear / tiles_per_image;
    const int32_t tile_id = tile_linear - image_index * tiles_per_image;
    const int32_t tile_row = tile_id / tile_width_count;
    const int32_t tile_col = tile_id - tile_row * tile_width_count;
    const uint32_t tid = block.thread_rank();
    const uint32_t thread_x = tid & TILE_MASK;
    const uint32_t thread_y = tid >> TILE_SHIFT;

    // -- Locate tile and move image-local pointers --------------------------
    const int32_t image_width_px = static_cast<int32_t>(image_width);
    const int32_t image_height_px = static_cast<int32_t>(image_height);
    assert(image_width_px > 0 && image_height_px > 0);
    assert(image_height_px <= INT32_MAX / image_width_px);
    const int32_t image_area = image_height_px * image_width_px;
    tile_offsets += image_index * tiles_per_image;
    render_colors += image_index * image_area * CDIM;
    render_alphas += image_index * image_area;
    if constexpr (ReturnNormals) {
        render_normals += image_index * image_area * 3;
    }
    last_ids += image_index * image_area;
    if (sample_counts != nullptr) {
        sample_counts += image_index * image_area;
    }
    if (backgrounds != nullptr) {
        backgrounds += image_index * CDIM;
    }
    if (masks != nullptr) {
        masks += image_index * tiles_per_image;
    }
    if (rays != nullptr) {
        rays += image_index * image_area * 6;
    }
    const RollingShutterParameters rs_params(
        viewmats0 + image_index * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + image_index * 16);
    const uint32_t image_index_u32 = static_cast<uint32_t>(image_index);
    const uint32_t tile_id_u32 = static_cast<uint32_t>(tile_id);
    const uint32_t tile_row_u32 = static_cast<uint32_t>(tile_row);
    const uint32_t tile_col_u32 = static_cast<uint32_t>(tile_col);

    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    // -- Rebuild rays and seed replay lanes ---------------------------------
    PixelCoordsCompact pcs[PIXELS_PER_THREAD];
    vec3 ray_o[PIXELS_PER_THREAD] = {};
    vec3 ray_d[PIXELS_PER_THREAD] = {};
    uint32_t valid_mask = 0u;
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        pcs[p] = compute_pixel_coords_compact<TILE_SIZE, CTA_SIZE>(
            camera_model_type, tile_id_u32, tile_row_u32, tile_col_u32,
            thread_x, thread_y, p,
            image_width, image_height, lidar_device_coeffs);
        if (pcs[p].inside) {
            const WorldRay ray = compute_world_ray<scalar_t>(
                image_index_u32, pcs[p].col, pcs[p].row, pcs[p].pix_id,
                pcs[p].inside,
                rs_params, rays, Ks,
                image_width, image_height,
                camera_model_type, rs_type,
                radial_coeffs, tangential_coeffs, thin_prism_coeffs,
                ftheta_device_coeffs, lidar_device_coeffs,
                external_distortion_device_params);
            if (ray.valid_flag) {
                ray_o[p] = ray.ray_org;
                ray_d[p] = ray.ray_dir;
                valid_mask |= (1u << p);
            }
        }
    }

    const int32_t batch_base = (batch_offsets != nullptr)
        ? batch_offsets[tile_linear]
        : 0;

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (image_index == static_cast<int32_t>(B * C) - 1) &&
            (tile_id == tiles_per_image - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];
    float T_cum[PIXELS_PER_THREAD];
    float pix_cum[PIXELS_PER_THREAD][CDIM] = {};
    vec3 normal_cum[PIXELS_PER_THREAD] = {};
    int32_t last_idx_global[PIXELS_PER_THREAD];
    int32_t n_acc_global[PIXELS_PER_THREAD] = {};
    int32_t c_stop[PIXELS_PER_THREAD];
    uint32_t batch_replay_mask = 0u;
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        T_cum[p] = 1.0f;
        last_idx_global[p] = -1;
        n_acc_global[p] = 0;
        const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
        const int32_t pix_rank_in_tile =
            pix_y_in_tile * TILE_SIZE + thread_x;
        assert(tile_linear <=
               (INT32_MAX - pix_rank_in_tile) / pixels_per_tile);
        const int32_t compose_idx =
            tile_linear * pixels_per_tile + pix_rank_in_tile;
        c_stop[p] = decode_compose_c_stop(
            compose_c_stop[compose_idx]);
        const bool engage =
            c_stop[p] == batch_id &&
            ((valid_mask & (1u << p)) != 0u);
        if (engage) {
            assert(c_stop[p] == batch_id);
            assert((valid_mask & (1u << p)) != 0u);
            batch_replay_mask |= (1u << p);
            const int32_t slot = batch_base + batch_id;
            const FwdBatchSlotView<CDIM, ReturnNormals, scalar_t> slot_view(
                fwd_batch_state, slot, pixels_per_tile, pix_rank_in_tile);
            T_cum[p] = slot_view.T();
            assert(T_cum[p] > TRANSMITTANCE_THRESHOLD);
            assert(T_cum[p] <= 1.0f + 1e-5f);
            #pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_cum[p][k] = slot_view.feature(k);
            }
            if constexpr (ReturnNormals) {
                normal_cum[p] = slot_view.normal();
            }
            const FwdBatchReplayPreambleView<const int2> preamble_view(
                batch_replay_preamble, tile_linear, pixels_per_tile,
                pix_rank_in_tile);
            last_idx_global[p] = preamble_view.last();
            n_acc_global[p] = preamble_view.count();
        }
    }

    extern __shared__ int s[];
    FwdShmemBatch smem =
        fwd_unpack_shmem(reinterpret_cast<int32_t *>(s), CTA_SIZE);

    const bool any_engages =
        __syncthreads_count(batch_replay_mask != 0u ? 1 : 0) > 0;
    assert(any_engages);
    if (!any_engages) {
        // The metadata flag should only be set by a stopping pixel, but keep
        // false positives local to this CTA if the flag source changes later.
        return;
    }

    uint32_t local_done_mask = ALL_DONE & ~batch_replay_mask;
    float transmittance_threshold[PIXELS_PER_THREAD];
    float saturating_T_unused[PIXELS_PER_THREAD];
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        transmittance_threshold[p] = TRANSMITTANCE_THRESHOLD;
        saturating_T_unused[p] = 1.0f;
    }
    assert(batch_id >= 0);
    const uint32_t b = static_cast<uint32_t>(batch_id);

    // -- Replay the threshold-crossing logical batch ------------------------
    process_batch_gaussians_fwd<
        CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance,
        SaturationTPolicy::KeepPreSaturationT,
        scalar_t>(
        b, range_start, range_end, tid, C, N,
        flatten_ids, means, quats, scales, colors, opacities,
        ray_o, ray_d,
        smem.id_batch, smem.xyz_opacity_batch, smem.iscl_rot_batch,
        smem.scale_batch, smem.normal_batch,
        transmittance_threshold,
        T_cum, saturating_T_unused, pix_cum, normal_cum, last_idx_global,
        n_acc_global,
        local_done_mask);

    // -- Emit outputs and patch the backward-visible batch slot --------------
    #pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if ((batch_replay_mask & (1u << p)) == 0u) {
            continue;
        }
        const int32_t pix_id = pcs[p].pix_id;

        render_alphas[pix_id] = 1.0f - T_cum[p];
        #pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] = (backgrounds == nullptr)
                ? pix_cum[p][k]
                : (pix_cum[p][k] + T_cum[p] * static_cast<float>(backgrounds[k]));
        }
        if constexpr (ReturnNormals) {
            render_normals[pix_id * 3 + 0] = normal_cum[p].x;
            render_normals[pix_id * 3 + 1] = normal_cum[p].y;
            render_normals[pix_id * 3 + 2] = normal_cum[p].z;
        }
        last_ids[pix_id] = last_idx_global[p];
        if (sample_counts != nullptr) {
            sample_counts[pix_id] = n_acc_global[p];
        }

        // Persist post-saturation state for bwd consumption. The replay seed
        // in this slot is no longer needed after batch-replay reaches this
        // point, so the slot itself becomes the final state for c_stop.
        const uint32_t pix_y_in_tile = thread_y + p * ROW_STRIDE;
        const int32_t pix_rank_in_tile =
            pix_y_in_tile * TILE_SIZE + thread_x;
        const int32_t slot = batch_base + batch_id;
        FwdBatchSlotView<CDIM, ReturnNormals, scalar_t> slot_view(
            fwd_batch_state, slot, pixels_per_tile, pix_rank_in_tile);
        slot_view.setT(T_cum[p]);
        #pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            slot_view.setFeature(k, pix_cum[p][k]);
        }
        if constexpr (ReturnNormals) {
            slot_view.setNormal(normal_cum[p]);
        }
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, grid_h, grid_w]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [...., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    // external distortion
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, grid_h, grid_w]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    // CSR batch structure (precomputed by caller, shared with bwd)
    const at::Tensor batches_per_tile, // [num_tiles] int32
    const at::Tensor batch_offsets,   // [num_tiles + 1] int32
    const at::Tensor bid_to_slot,     // [total_batches] int32
    const int64_t total_batches,       // scalar; equals batch_offsets[num_tiles]
    const bool fwd_only,
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3]
    at::Tensor fwd_batch_state, // [total_batches, state_dim, pixels_per_tile] fp32
    at::Tensor partials_meta,   // [total_batches, pixels_per_tile, 2] uint16
    at::Tensor batch_replay_preamble,   // [num_tiles, pixels_per_tile, 2] int32
    at::Tensor compose_c_stop,  // [num_tiles, pixels_per_tile] uint16
    at::Tensor priming_state     // [..., C, image_height, image_width] int32
) {
    // Note: quats need to be normalized before passing in.

    // -- Derive runtime dimensions and device-side parameter bundles --------
    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed mode not supported for 3DGUT forward rasterization");

    const uint32_t N = packed ? 0u : static_cast<uint32_t>(means.size(-2));
    // Number of batches: product of the leading dims. Computed this way rather
    // than means.numel()/(N*3) because:
    // - the division is undefined for an empty gaussian set (N == 0)
    // - the leading-dim product matches how the calling op sizes its outputs,
    //   so an empty input renders a clean background image instead of crashing
    const uint32_t B = static_cast<uint32_t>(
        c10::multiply_integers(means.sizes().slice(0, means.dim() - 2)));
    const uint32_t C = static_cast<uint32_t>(viewmats0.size(-3));
    const uint32_t I = B * C;
    const uint32_t tile_height = static_cast<uint32_t>(tile_offsets.size(-2));
    const uint32_t tile_width = static_cast<uint32_t>(tile_offsets.size(-1));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    const int32_t pixels_per_tile = tile_size * tile_size;

    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(
        gsplat::checked_deref(ftheta_coeffs, "ftheta_coeffs"));
    DeviceExternalDistortionParamsOpt external_distortion_device_params =
        cuda::std::nullopt;
    if (external_distortion_params.has_value()) {
        const auto& params = gsplat::checked_deref(
            external_distortion_params.value(), "external_distortion_params");
        CHECK_CONTIGUOUS(params.horizontal_poly);
        CHECK_CONTIGUOUS(params.vertical_poly);
        CHECK_CONTIGUOUS(params.horizontal_poly_inverse);
        CHECK_CONTIGUOUS(params.vertical_poly_inverse);
        external_distortion_device_params =
            extdist::BivariateWindshieldModelDeviceParams(params);
    }

    DeviceLidarParamsOpt lidar_device_coeffs = cuda::std::nullopt;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model == CameraModelType::LIDAR,
                    "If lidar coeffs are given, the camera model must be lidar");
        lidar_device_coeffs = *lidar_coeffs.value();
    } else {
        TORCH_CHECK(camera_model != CameraModelType::LIDAR,
                    "If the sensor isn't lidar, lidar coefficients must not be given");
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK(SupportedChannels::contains(channels),
                "Unsupported number of channels: ", channels,
                " (check GSPLAT_NUM_CHANNELS)");
    TORCH_CHECK(SupportedTileSizes::contains(tile_size),
                "Unsupported tile_size ", tile_size,
                "; supported values are {8, 16}.");

    const bool return_normals = normals.has_value();
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // -- Dispatch the runtime options to the compiled specializations --------
    auto launch_kernel = [&]<
        typename ChannelsT,
        typename TileSizeT,
        typename ReturnNormalsT,
        typename UseHitDistanceT,
        typename FwdOnlyT
    >() {
        constexpr uint32_t CDIM = ChannelsT::value;
        constexpr bool ReturnNormals  = static_cast<bool>(ReturnNormalsT::value);
        constexpr bool UseHitDistance = static_cast<bool>(UseHitDistanceT::value);
        constexpr int64_t StateDim =
            FWD_BATCH_STATE_PIX_OFFSET + CDIM +
            (ReturnNormals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0);

        // -- Validate transient tensors for this specialization --------------
        TORCH_CHECK(fwd_batch_state.dim() == 3,
                    "fwd_batch_state must be 3-D");
        TORCH_CHECK(fwd_batch_state.size(0) == total_batches &&
                        fwd_batch_state.size(1) == StateDim &&
                        fwd_batch_state.size(2) == pixels_per_tile,
                    "fwd_batch_state has wrong shape");
        TORCH_CHECK(
            fwd_batch_state.numel() <= INT32_MAX,
            "ParallelBatch fwd_batch_state exceeds signed 32-bit device "
            "offset range (",
            fwd_batch_state.numel(),
            " elements)");

        TORCH_CHECK(partials_meta.dim() == 3 &&
                        partials_meta.size(0) == total_batches &&
                        partials_meta.size(1) == pixels_per_tile &&
                        partials_meta.size(2) == 2,
                    "partials_meta has wrong shape");
        TORCH_CHECK(
            partials_meta.scalar_type() == at::kUInt16,
            "partials_meta must be uint16");
        TORCH_CHECK(
            partials_meta.numel() <= INT32_MAX,
            "ParallelBatch partials_meta exceeds signed 32-bit device "
            "offset range");

        TORCH_CHECK(
            !fwd_only || !sample_counts.has_value(),
            "ParallelBatch fwd-only mode cannot return sample_counts; "
            "request exact metadata instead");

        if (!fwd_only) {
            TORCH_CHECK(batch_replay_preamble.dim() == 3 &&
                            batch_replay_preamble.size(0) ==
                                static_cast<int64_t>(I) *
                                    static_cast<int64_t>(tile_height) *
                                    static_cast<int64_t>(tile_width) &&
                            batch_replay_preamble.size(1) == pixels_per_tile &&
                            batch_replay_preamble.size(2) == 2,
                        "batch_replay_preamble has wrong shape");
            TORCH_CHECK(
                batch_replay_preamble.scalar_type() == at::kInt,
                "batch_replay_preamble must be int32");
            TORCH_CHECK(
                batch_replay_preamble.numel() <= INT32_MAX,
                "ParallelBatch batch_replay_preamble exceeds signed 32-bit device "
                "offset range");

            TORCH_CHECK(compose_c_stop.dim() == 2 &&
                            compose_c_stop.size(0) ==
                                static_cast<int64_t>(I) *
                                    static_cast<int64_t>(tile_height) *
                                    static_cast<int64_t>(tile_width) &&
                            compose_c_stop.size(1) == pixels_per_tile,
                        "compose_c_stop has wrong shape");
            TORCH_CHECK(
                compose_c_stop.numel() <= INT32_MAX,
                "ParallelBatch compose_c_stop exceeds signed 32-bit device "
                "offset range");
        }

        TORCH_CHECK(
            priming_state.numel() <= INT32_MAX,
            "ParallelBatch priming_state exceeds signed 32-bit device "
            "offset range");

        // Launch the tuned (CTA_SIZE, pixels-per-thread) variant for this tile
        // size. tile_size and fwd_only are resolved at compile time through the
        // dispatch axes below; CtaSizeForTile maps each tile size to its tuned
        // pair:
        //  tile 8  -> CTA 32, 2 px/thread (compact: small shmem, many
        //             co-resident CTAs; wins on training)
        //  tile 16 -> CTA 256, 1 px/thread (one thread per pixel; wins on
        //             high-res fwd-only)
        constexpr uint32_t TILE_SIZE = TileSizeT::value;
        constexpr uint32_t CTA_SIZE = CtaSizeForTile<TILE_SIZE>::value;
        constexpr bool ForBackward = FwdOnlyT::value == 0;
        const dim3 threads = {CTA_SIZE, 1, 1};
        // Shared memory: id_batch + xyz_opacity_batch + iscl_rot_batch +
        // scale_batch + normal_batch — CTA_SIZE entries each.
        const int64_t shmem_size =
            CTA_SIZE *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) +
             sizeof(vec3) + sizeof(vec3));

        // -- Configure dynamic shared memory ----------------------------
        const cudaError_t partials_attr_status = cudaFuncSetAttribute(
            rasterize_to_pixels_from_world_3dgs_fwd_partials_kernel<CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, ForBackward, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size);
        if (partials_attr_status != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size, " bytes), try lowering tile_size."
            );
        }
        if constexpr (ForBackward) {
            const cudaError_t batch_replay_attr_status =
                cudaFuncSetAttribute(
                    rasterize_to_pixels_from_world_3dgs_fwd_batch_replay_kernel<CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, float>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    shmem_size);
            if (batch_replay_attr_status != cudaSuccess) {
                AT_ERROR(
                    "Failed to set max dynamic shmem on batch-replay kernel (",
                    shmem_size, " bytes); try lowering tile_size."
                );
            }
        }

        // ---- Partials pass: 1D grid over total_batches. Skip when
        // total_batches == 0 (degenerate n_isects==0 case) — batch-scan
        // short-circuits on num_batches_this_tile==0 and emits
        // background-only output.
        if (total_batches > 0) {
            const dim3 grid_partials = {
                static_cast<uint32_t>(total_batches), 1, 1};
            rasterize_to_pixels_from_world_3dgs_fwd_partials_kernel<CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, ForBackward, float>
                <<<grid_partials, threads, shmem_size, stream>>>(
                    B, C, N, n_isects,
                    data_ptr_as<const vec3, float>(means),
                    data_ptr_as<const vec4, float>(quats),
                    data_ptr_as<const vec3, float>(scales),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    data_ptr_or_null<const bool>(masks),
                    image_width, image_height, tile_width, tile_height,
                    viewmats0.const_data_ptr<float>(),
                    data_ptr_or_null<const float>(viewmats1),
                    Ks.const_data_ptr<float>(),
                    camera_model, rs_type,
                    data_ptr_or_null<const float>(rays),
                    data_ptr_or_null<const float>(radial_coeffs),
                    data_ptr_or_null<const float>(tangential_coeffs),
                    data_ptr_or_null<const float>(thin_prism_coeffs),
                    ftheta_device_coeffs, lidar_device_coeffs, external_distortion_device_params,
                    tile_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    batch_offsets.const_data_ptr<int32_t>(),
                    data_ptr_or_null<const int32_t>(total_batches > 0, bid_to_slot),
                    priming_state.data_ptr<int32_t>(),
                    data_ptr_or_null<uint16_t>(ForBackward, compose_c_stop),
                    data_ptr_or_null<float>(total_batches > 0, fwd_batch_state),
                    data_ptr_as_or_null<ushort2, uint16_t>(total_batches > 0, partials_meta));
        }

        // ---- Batch-scan pass: one CTA per tile. It folds batch
        // partials independently per pixel and records only the pixels
        // that need exact per-gaussian batch-replay.
        const dim3 grid_batch_scan = {I, tile_height, tile_width};
        // Dispatch only batch-scan on SAFE_MASKED_OUTPUTS so partials and
        // batch-replay keep a single instantiation. unsafe_masked_tile_outputs
        // compiles out batch-scan's masked-tile safe-store.
        auto launch_batch_scan = [&]<bool SAFE_MASKED_OUTPUTS>() {
            rasterize_to_pixels_from_world_3dgs_fwd_batch_scan_kernel<CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, ForBackward, SAFE_MASKED_OUTPUTS, float>
                <<<grid_batch_scan, threads, 0, stream>>>(
                    B, C, N, n_isects,
                    data_ptr_as<const vec3, float>(means),
                    data_ptr_as<const vec4, float>(quats),
                    data_ptr_as<const vec3, float>(scales),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    data_ptr_or_null<const float>(backgrounds),
                    data_ptr_or_null<const bool>(masks),
                    image_width, image_height, tile_width, tile_height,
                    camera_model, lidar_device_coeffs,
                    tile_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    batches_per_tile.const_data_ptr<int32_t>(),
                    batch_offsets.const_data_ptr<int32_t>(),
                    priming_state.data_ptr<int32_t>(),
                    data_ptr_or_null<float>(total_batches > 0, fwd_batch_state),
                    data_ptr_as_or_null<ushort2, uint16_t>(total_batches > 0, partials_meta),
                    data_ptr_as_or_null<int2, int32_t>(ForBackward, batch_replay_preamble),
                    data_ptr_or_null<uint16_t>(ForBackward, compose_c_stop),
                    renders.data_ptr<float>(),
                    alphas.data_ptr<float>(),
                    data_ptr_or_null<float>(normals),
                    data_ptr_or_null<int32_t>(ForBackward, last_ids),
                    data_ptr_or_null<int32_t>(ForBackward, sample_counts));
        };
        if (unsafe_masked_tile_outputs) {
            launch_batch_scan.template operator()<false>();
        } else {
            launch_batch_scan.template operator()<true>();
        }

        // ---- Batch-replay pass: one CTA per batch slot. Most CTAs return
        // after a metadata flag load; engaging CTAs replay only their
        // threshold-crossing batch and patch bwd-visible state.
        if constexpr (ForBackward) {
            if (total_batches > 0) {
                const dim3 grid_batch_replay = {
                    static_cast<uint32_t>(total_batches), 1, 1};
                rasterize_to_pixels_from_world_3dgs_fwd_batch_replay_kernel<CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, float>
                    <<<grid_batch_replay, threads, shmem_size, stream>>>(
                        B, C, N, n_isects,
                        data_ptr_as<const vec3, float>(means),
                        data_ptr_as<const vec4, float>(quats),
                        data_ptr_as<const vec3, float>(scales),
                        colors.const_data_ptr<float>(),
                        opacities.const_data_ptr<float>(),
                        data_ptr_or_null<const float>(backgrounds),
                        data_ptr_or_null<const bool>(masks),
                        image_width, image_height, tile_width, tile_height,
                        viewmats0.const_data_ptr<float>(),
                        data_ptr_or_null<const float>(viewmats1),
                        Ks.const_data_ptr<float>(),
                        camera_model, rs_type,
                        data_ptr_or_null<const float>(rays),
                        data_ptr_or_null<const float>(radial_coeffs),
                        data_ptr_or_null<const float>(tangential_coeffs),
                        data_ptr_or_null<const float>(thin_prism_coeffs),
                        ftheta_device_coeffs, lidar_device_coeffs, external_distortion_device_params,
                        tile_offsets.const_data_ptr<int32_t>(),
                        flatten_ids.const_data_ptr<int32_t>(),
                        batches_per_tile.const_data_ptr<int32_t>(),
                        batch_offsets.const_data_ptr<int32_t>(),
                        data_ptr_or_null<const int32_t>(total_batches > 0, bid_to_slot),
                        data_ptr_or_null<float>(total_batches > 0, fwd_batch_state),
                        data_ptr_as_or_null<const ushort2, uint16_t>(total_batches > 0, partials_meta),
                        data_ptr_as_or_null<const int2, int32_t>(ForBackward, batch_replay_preamble),
                        data_ptr_or_null<const uint16_t>(ForBackward, compose_c_stop),
                        renders.data_ptr<float>(),
                        alphas.data_ptr<float>(),
                        data_ptr_or_null<float>(normals),
                        data_ptr_or_null<int32_t>(ForBackward, last_ids),
                        data_ptr_or_null<int32_t>(ForBackward, sample_counts));
            }
        }
    };
    const bool dispatched = dispatch::dispatch(
        SupportedChannels{channels},
        SupportedTileSizes{static_cast<int>(tile_size)},
        dispatch::IntParam<0, 1>{return_normals  ? 1 : 0},
        dispatch::IntParam<0, 1>{use_hit_distance ? 1 : 0},
        dispatch::IntParam<0, 1>{fwd_only ? 1 : 0},
        std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

} // namespace gsplat

#endif
