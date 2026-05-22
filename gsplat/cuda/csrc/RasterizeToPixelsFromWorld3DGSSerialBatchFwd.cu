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
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "RasterizeCSR.cuh"
#include "RasterizeToPixelsFromWorld3DGS.h"
#include "RasterizeToPixelsFromWorld3DGS.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "TorchUtils.h"
#include "Utils.cuh"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;
using SupportedTileSizes = dispatch::IntParam<8, 16>;
using MaskedOutputSafetyModes = dispatch::IntParam<0, 1>;

// TILE_SIZE and CTA_SIZE are a tuned launch-variant pair. Add a specialization
// here whenever a new supported tile size needs its own CTA size.
template <uint32_t TILE_SIZE>
struct CtaSizeForTile;

template <>
struct CtaSizeForTile<8u> {
    static constexpr uint32_t value = 32u;
};

template <>
struct CtaSizeForTile<16u> {
    static constexpr uint32_t value = 256u;
};

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

// Compact CTA rasterizer for 3DGUT (world-space ray-gaussian evaluation).
// Same architectural pattern as the 3DGS compact CTA:
//   CTA_SIZE threads process a TILE_SIZE x TILE_SIZE tile,
//   each thread handling PIXELS_PER_THREAD pixels in a vertical stride.
// Shared memory per batch: CTA_SIZE * 80B = 2560B (vs 20480B with 256 threads).

// Per-architecture hardware cap on thread blocks per SM. ptxas rejects
// min_blocks_per_sm > HW cap under --warning-as-error.
//   sm_90, sm_100, sm_120: 32 blocks/SM
//   everything else:       16 blocks/SM (covers Ampere/Ada and anything older)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    #define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 32
#else
    #define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 16
#endif

// Range endpoints of the CDIM-indexed schedule. The schedule plateaus at
// `high` blocks/SM for CDIM in [1, MIN_CDIM_FOR_HINT], then linearly descends
// to `low` at MAX_CDIM_FOR_HINT. The plateau covers the camera-inference
// CDIM=4 path so it sustains max occupancy alongside CDIM={1,2,3}.
#define GSPLAT_MIN_CDIM_FOR_HINT 4u
#define GSPLAT_MAX_CDIM_FOR_HINT 24u

// Target min_blocks at the endpoints of the schedule.
//   - CDIM <= MIN_CDIM_FOR_HINT (plateau): push occupancy up to the HW cap,
//              but not above 24 (beyond this the kernel's natural register
//              usage forces ptxas to spill).
//   - CDIM=24: 16. HW cap on pre-sm_90, 25% occ on sm_90+; fits within the
//              kernel's register footprint at full SH + extras.
#define GSPLAT_MIN_BLOCKS_AT_MIN_CDIM \
    (GSPLAT_ARCH_MAX_BLOCKS_PER_SM < 24 ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM : 24)
#define GSPLAT_MIN_BLOCKS_AT_MAX_CDIM 16u

// Per-CDIM occupancy hint for __launch_bounds__ min_blocks_per_sm.
// Plateau at `high` for CDIM in [1, MIN_CDIM_FOR_HINT]; linear descent from
// `high` (at MIN_CDIM_FOR_HINT) to `low` (at MAX_CDIM_FOR_HINT) above. Beyond
// MAX_CDIM_FOR_HINT we emit min_blocks=1, high-channel kernels carry enough
// register pressure that forcing occupancy would spill.
//   cdim_excess     = max(0, CDIM - MIN_CDIM_FOR_HINT)   (saturating sub)
//   blocks_at_cta32 = high - cdim_excess * (high - low) / (max_cdim - min_cdim)
// Collapses to the constant low value on archs where high == low.
//
// The schedule was tuned at CTA_SIZE=32. To use the same kernel at larger
// CTAs, we preserve the threads/SM target (= blocks_at_cta32 * 32) and
// re-derive min_blocks at the actual CTA_SIZE; otherwise asking 16-24
// blocks/SM at CTA=256 yields physically impossible 4096-6144 threads/SM
// and ptxas either rejects (under -Werror) or produces a degenerate spill.

// Here's a table of the different values you can expect per variables and arch
//  CDIM | sm90+ CTA=32 | sm90+ CTA=256 | < sm90 CTA=32 | < sm90 CTA=256
// ----------------------------------------------------------------------
//     1 |      24      |       3       |      16       |       2
//     2 |      24      |       3       |      16       |       2
//     3 |      24      |       3       |      16       |       2
//     4 |      24      |       3       |      16       |       2
//     5 |      24      |       3       |      16       |       2
//     6 |      24      |       3       |      16       |       2
//     7 |      23      |       2       |      16       |       2
//     8 |      23      |       2       |      16       |       2
//     9 |      22      |       2       |      16       |       2
//    10 |      22      |       2       |      16       |       2
//    11 |      22      |       2       |      16       |       2
//    12 |      21      |       2       |      16       |       2
//    13 |      21      |       2       |      16       |       2
//    14 |      20      |       2       |      16       |       2
//    15 |      20      |       2       |      16       |       2
//    16 |      20      |       2       |      16       |       2
//    17 |      19      |       2       |      16       |       2
//    18 |      19      |       2       |      16       |       2
//    19 |      18      |       2       |      16       |       2
//    20 |      18      |       2       |      16       |       2
//    21 |      18      |       2       |      16       |       2
//    22 |      17      |       2       |      16       |       2
//    23 |      17      |       2       |      16       |       2
//    24 |      16      |       2       |      16       |       2
//   >24 |       1      |       1       |       1       |       1

template <uint32_t CDIM, uint32_t CTA_SIZE>
constexpr uint32_t min_blocks_for_cdim() {
    if constexpr (CDIM > GSPLAT_MAX_CDIM_FOR_HINT) {
        return 1;
    } else {
        constexpr uint32_t high = GSPLAT_MIN_BLOCKS_AT_MIN_CDIM;
        constexpr uint32_t low = GSPLAT_MIN_BLOCKS_AT_MAX_CDIM;
        constexpr uint32_t cdim_excess =
            (CDIM > GSPLAT_MIN_CDIM_FOR_HINT)
                ? (CDIM - GSPLAT_MIN_CDIM_FOR_HINT)
                : 0u;
        constexpr uint32_t cdim_span =
            GSPLAT_MAX_CDIM_FOR_HINT - GSPLAT_MIN_CDIM_FOR_HINT;
        constexpr uint32_t block_span = (high >= low) ? (high - low) : 0;
        constexpr uint32_t decrement = (cdim_excess * block_span) / cdim_span;
        constexpr uint32_t blocks_at_cta32 =
            (high > decrement) ? (high - decrement) : low;
        constexpr uint32_t threads_target = blocks_at_cta32 * 32u;
        constexpr uint32_t blocks = threads_target / CTA_SIZE;
        constexpr uint32_t lo_clamped = (blocks == 0u) ? 1u : blocks;
        constexpr uint32_t hi_clamped =
            (lo_clamped > GSPLAT_ARCH_MAX_BLOCKS_PER_SM)
                ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM
                : lo_clamped;
        return hi_clamped;
    }
}

template <
    uint32_t CDIM,
    uint32_t TILE_SIZE,
    uint32_t CTA_SIZE,
    bool ReturnNormals,
    bool UseHitDistance,
    bool SAFE_MASKED_OUTPUTS = true>
__global__ void __launch_bounds__(CTA_SIZE, min_blocks_for_cdim<CDIM, CTA_SIZE>())
rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const float *__restrict__ colors,         // [B, C, N, CDIM] or [nnz, CDIM]
    const float *__restrict__ opacities,      // [B, C, N] or [nnz]
    const float *__restrict__ backgrounds,    // [B, C, CDIM]
    const bool *__restrict__ masks,           // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    // camera model
    const float *__restrict__ viewmats0, // [B, C, 4, 4]
    const float *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const float *__restrict__ Ks,        // [B, C, 3, 3]
    const CameraModelType camera_model_type,
    // unscented transform
    const UnscentedTransformParameters ut_params,
    const ShutterType rs_type,
    const float *__restrict__ rays,                  // [B, C, H, W, 6]
    const float *__restrict__ radial_coeffs,         // [B, C, 6] or [B, C, 4] optional
    const float *__restrict__ tangential_coeffs,     // [B, C, 2] optional
    const float *__restrict__ thin_prism_coeffs,     // [B, C, 4] optional
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    // intersections
    const int32_t *__restrict__ isect_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    float *__restrict__ render_colors,        // [B, C, image_height, image_width, CDIM]
    float *__restrict__ render_alphas,        // [B, C, image_height, image_width, 1]
    float *__restrict__ render_normals,       // [B, C, image_height, image_width, 3] optional
    int32_t *__restrict__ last_ids,           // [B, C, image_height, image_width] optional
    int32_t *__restrict__ sample_counts,      // [B, C, image_height, image_width] optional
    // Optional CSR batch-state persistence (for MixedBatch bwd reuse). See
    // RasterizeCSR.cuh.
    // Storage layout: [total_batches][state_dim][pixels_per_tile] fp32.
    //   - [0]: T (cumulative transmittance after the persist batch)
    //   - [1..1+CDIM): pix_out[CDIM] (cumulative color accumulator)
    //   - [1+CDIM..1+CDIM+3): normal_out[3] when normals are returned
    // Each tile owns batches_per_tile[tile_linear] slots, starting at
    // batch_offsets[tile_linear]. Persist slot c (c in [0, num_batches))
    // corresponds to fwd state after logical batch c, where one logical batch
    // covers pixels_per_tile gaussians (matches bwd's batch unit).
    // batch_offsets_csr and fwd_batch_state are passed null-or-non-null in
    // lockstep by the launcher (gated on `total_batches > 0`); both null ⇔
    // persistence disabled (e.g. `n_isects == 0`).
    const int32_t *__restrict__ batch_offsets_csr, // [num_tiles + 1]
    float *__restrict__ fwd_batch_state // [total_batches, state_dim, pixels_per_tile]
) {
    // FETCH_SIZE = gaussians fetched per cooperative-fetch round (one per
    // thread; sized to the CTA so threads fetch in parallel without idling).
    // LOGICAL_BATCH = gaussians per outer-loop iteration; matches the bwd's
    // batch unit (= pixels_per_tile) so batch-state persist boundaries land
    // at the same gaussian positions the bwd's CSR view expects. Each logical
    // batch is split into FETCHES_PER_BATCH cooperative fetch rounds.
    constexpr uint32_t FETCH_SIZE        = CTA_SIZE;
    constexpr uint32_t PIXELS_PER_THREAD = TILE_SIZE * TILE_SIZE / CTA_SIZE;
    constexpr uint32_t LOGICAL_BATCH     = TILE_SIZE * TILE_SIZE;
    constexpr uint32_t FETCHES_PER_BATCH = LOGICAL_BATCH / FETCH_SIZE;
    constexpr uint32_t ROW_STRIDE        = CTA_SIZE / TILE_SIZE;
    constexpr uint32_t TILE_MASK         = TILE_SIZE - 1;
    constexpr uint32_t TILE_SHIFT        = __builtin_ctz(TILE_SIZE);
    constexpr uint32_t ALL_DONE          = (1u << PIXELS_PER_THREAD) - 1u;
    static_assert(PIXELS_PER_THREAD > 0, "PIXELS_PER_THREAD == 0 - CTA_SIZE must not exceed TILE_SIZE * TILE_SIZE");
    static_assert(LOGICAL_BATCH % FETCH_SIZE == 0, "LOGICAL_BATCH must be a multiple of FETCH_SIZE");
    static_assert(FETCHES_PER_BATCH >= 1, "FETCHES_PER_BATCH must be >= 1");

    const int32_t iid = blockIdx.x;
    const uint32_t grid_width  = gridDim.z;
    const uint32_t grid_height = gridDim.y;

    const uint32_t tile_x = blockIdx.z;
    const uint32_t tile_y = blockIdx.y;
    const int32_t tile_id = blockIdx.y * grid_width + blockIdx.z;

    bool masked_tile = false;
    if (masks != nullptr) {
        masks += iid * grid_height * grid_width;
        masked_tile = !masks[tile_id];
        if constexpr (!SAFE_MASKED_OUTPUTS) {
            if (masked_tile) {
                // Non-safe masked outputs: bwd's rasterize_gradient_bwd_kernel
                // returns on the same mask gate before reading CSR-backed
                // fwd_batch_state, so we intentionally leave it unseeded.
                return;
            }
        }
    }

    const uint32_t tid = threadIdx.x;
    const uint32_t thread_x = tid & TILE_MASK;  // X & 0xF(15) == X % 16
    const uint32_t thread_y = tid >> TILE_SHIFT; // X >> 4 == X / 16

    // Offset pointers to current image
    isect_offsets += iid * grid_height * grid_width;
    render_colors += iid * image_height * image_width * CDIM;
    render_alphas += iid * image_height * image_width;
    if constexpr (ReturnNormals) {
        render_normals += iid * image_height * image_width * 3;
    }
    if (last_ids != nullptr) {
        last_ids += iid * image_height * image_width;
    }
    if (sample_counts != nullptr) {
        sample_counts += iid * image_height * image_width;
    }
    if (backgrounds != nullptr) {
        backgrounds += iid * CDIM;
    }

    // Per-pixel coordinate setup. pc[p] holds (row, col, pix_id,
    // inside) for the p-th pixel slot of this thread. (row, col) live
    // only until ray setup; masked tiles only need pix_id for empty outputs.
    PixelCoords pc[PIXELS_PER_THREAD];
    uint32_t done_mask = 0;
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        pc[p] = compute_pixel_coords(
            camera_model_type, tile_id, tile_y, tile_x, TILE_SIZE,
            thread_y + p * ROW_STRIDE, thread_x, tid + p * CTA_SIZE,
            image_width, image_height, lidar_device_coeffs);
        if (!pc[p].inside) {
            done_mask |= (1u << p);
            continue;
        }
    }

    // When the mask is provided, skip Gaussian rasterization for inactive
    // tiles. The default public behavior writes deterministic empty-render
    // outputs after pix_id is known; expert unsafe mode returned earlier and
    // leaves masked output pixels undefined.
    if constexpr (SAFE_MASKED_OUTPUTS) {
        if (masked_tile) {
#pragma unroll
            for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
                if (pc[p].inside) {
                    render_alphas[pc[p].pix_id] = 0.0f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        render_colors[pc[p].pix_id * CDIM + k] =
                            backgrounds == nullptr ? 0.0f : backgrounds[k];
                    }
                    if (render_normals != nullptr) {
#pragma unroll
                        for (uint32_t k = 0; k < 3; ++k) {
                            render_normals[pc[p].pix_id * 3 + k] = 0.0f;
                        }
                    }
                    last_ids[pc[p].pix_id] = -1;
                    if (sample_counts != nullptr) {
                        sample_counts[pc[p].pix_id] = 0;
                    }
                }
            }
        }
        if (masked_tile) {
            return;
        }
    }

    if (rays != nullptr) {
        rays += iid * image_height * image_width * 6;
    }

    // Rolling shutter parameter (loop-invariant across pixel slots)
    auto rs_params = RollingShutterParameters(
        viewmats0 + iid * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + iid * 16
    );

    vec3 ray_o[PIXELS_PER_THREAD] = {};
    vec3 ray_d[PIXELS_PER_THREAD] = {};
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if (done_mask & (1u << p)) {
            continue;
        }
        WorldRay ray = compute_world_ray<float>(
            iid, pc[p].col, pc[p].row, pc[p].pix_id, /*inside=*/true, rs_params,
            rays, Ks,
            image_width, image_height,
            camera_model_type, rs_type,
            radial_coeffs, tangential_coeffs, thin_prism_coeffs,
            ftheta_device_coeffs, lidar_device_coeffs,
            external_distortion_device_params);

        if (!ray.valid_flag) {
            done_mask |= (1u << p);
        } else {
            ray_o[p] = ray.ray_org;
            ray_d[p] = ray.ray_dir;
        }
    }

    // Gaussian range for this tile
    const int32_t range_start = isect_offsets[tile_id];
    const int32_t range_end =
        (iid == (int32_t)gridDim.x - 1) &&
        (tile_id == (int32_t)(grid_width * grid_height) - 1)
            ? n_isects
            : isect_offsets[tile_id + 1];
    // Logical-batch count matches the bwd's view (= ceil(range / pixels_per_tile)),
    // so persist boundaries align with the CSR-allocated slot positions.
    const uint32_t num_logical_batches =
        (range_end - range_start + LOGICAL_BATCH - 1) / LOGICAL_BATCH;

    // --- Batch-state persistence setup (shared with bwd gradient kernel) -----
    // Compute this tile's base slot in the CSR `fwd_batch_state` buffer. Per
    // the CSR invariant (see `RasterizeCSR.cuh`), the slot `c` for bwd
    // batch c corresponds to fwd state after logical batch c, so c=0 is
    // the front-most batch boundary and c=num_batches-1 is the terminal
    // persistable state. Logical batches advance by LOGICAL_BATCH gaussians,
    // matching the bwd's per-batch gaussian count, so this slot index maps
    // to the same gaussian position the bwd will read from. We precompute
    // per-pixel state write pointers here so the inner batch loop stays tight.
    //
    // batch_offsets_csr and fwd_batch_state are passed null-or-non-null in
    // lockstep by the launcher (gated on `total_batches > 0`); pin the
    // invariant and read either pointer to detect "no persistence". The tile
    // index for CSR matches bwd:
    // tile_linear = iid * grid_height * grid_width + tile_id.
    bool persist_batches = false;
    uint32_t pixels_per_tile = 0;
    // batch_base_slot is the slot index in fwd_batch_state for this tile's
    // c=0 entry (i.e., the front-most batch boundary). Later c entries
    // follow contiguously in the CSR, matching the bwd batch_id.
    int64_t batch_base_slot = 0;
    const uint32_t tile_linear =
        iid * grid_height * grid_width + tile_id;
    pixels_per_tile = TILE_SIZE * TILE_SIZE;
    assert((batch_offsets_csr == nullptr) == (fwd_batch_state == nullptr));
    persist_batches = batch_offsets_csr != nullptr;
    batch_base_slot = persist_batches
        ? static_cast<int64_t>(batch_offsets_csr[tile_linear])
        : 0;
    // Debug parity: the CSR slice for this tile must match the independently
    // recomputed logical-batch count; a mismatch means batch_offsets_csr and
    // num_logical_batches drifted and the kernel would walk past this tile's
    // fwd_batch_state slice.
    if (persist_batches) {
        assert(static_cast<uint32_t>(
                   batch_offsets_csr[tile_linear + 1] -
                   batch_offsets_csr[tile_linear]) == num_logical_batches);
    }
#ifndef NDEBUG
    if (persist_batches) {
        const int64_t batch_end_slot =
            static_cast<int64_t>(batch_offsets_csr[tile_linear + 1]);
        assert(batch_end_slot >= batch_base_slot);
        assert(
            static_cast<uint32_t>(batch_end_slot - batch_base_slot) ==
            num_logical_batches);
    }
#endif
    // Shared memory: FETCH_SIZE (= CTA_SIZE) entries; reused across each
    // logical batch's FETCHES_PER_BATCH cooperative-fetch rounds.
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [FETCH_SIZE]
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[FETCH_SIZE]); // [FETCH_SIZE]
    mat3 *iscl_rot_batch =
        reinterpret_cast<mat3 *>(&xyz_opacity_batch[FETCH_SIZE]); // [FETCH_SIZE]
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&iscl_rot_batch[FETCH_SIZE]); // [FETCH_SIZE]
    vec3 *normal_batch =
        reinterpret_cast<vec3 *>(&scale_batch[FETCH_SIZE]); // [FETCH_SIZE]

    // Per-pixel state
    int32_t cur_idx[PIXELS_PER_THREAD];
    float T[PIXELS_PER_THREAD];
    float transmittance_threshold[PIXELS_PER_THREAD];
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        cur_idx[p] = -1;
        T[p] = 1.0f;
        transmittance_threshold[p] = TRANSMITTANCE_THRESHOLD;
    }
    int32_t n_accumulated[PIXELS_PER_THREAD] = {0};
    float pix_out[PIXELS_PER_THREAD][CDIM] = {0.f};
    vec3 normal_out[PIXELS_PER_THREAD] = {};

    // Per-batch persist of the CURRENT per-pixel cumulative state to CSR
    // slot c is delegated to `persist_batch_state` in
    // `RasterizeToPixelsFromWorld3DGS.cuh`. All threads in the block call
    // it together — one slot row per thread (tr = tid + p * CTA_SIZE),
    // with thread rank tr indexing the pixels_per_tile axis. Threads
    // where `inside == false` write their trivial (T=1, pix_out=0,
    // normal_out=0) state into the slot; the bwd gradient kernel reads
    // the slot unconditionally but ignores out-of-bounds pixels in its
    // downstream gradient walk via the same `inside` check, so the
    // trivial values never propagate. The writes are kept so the memory
    // layout stays fully populated. Caller (this kernel) gates on
    // `persist_batches`.
    //
    // `c=0` corresponds to the front-most batch boundary; `c=num_batches-1`
    // corresponds to the terminal state. The boundary-formula derivation is
    // documented in `RasterizeCSR.cuh`.

#pragma unroll 1
    for (uint32_t lb = 0; lb < num_logical_batches; ++lb) {
        // Each logical batch covers LOGICAL_BATCH (= pixels_per_tile) gaussians,
        // matching the bwd's per-batch unit so persist boundaries align with
        // the CSR slot positions. Internally it issues FETCHES_PER_BATCH
        // cooperative fetch rounds of FETCH_SIZE (= CTA_SIZE) gaussians each
        // — this is what keeps the CTA at one warp.
        const uint32_t logical_batch_start = range_start + LOGICAL_BATCH * lb;
        const bool all_done = process_logical_batch_gaussians<CDIM, LOGICAL_BATCH, FETCH_SIZE, CTA_SIZE, PIXELS_PER_THREAD, true, SaturationTPolicy::KeepPreSaturationT, UseHitDistance, ReturnNormals, float>(
                // CTA scratch and tile range.
                tid, id_batch, xyz_opacity_batch, iscl_rot_batch, scale_batch, normal_batch,
                logical_batch_start, range_end,
                // Gaussian inputs.
                flatten_ids, means, quats, scales, opacities, colors, C, N,
                // Per-pixel rays and accumulation state.
                ray_o, ray_d, ALL_DONE, transmittance_threshold,
                T, T, pix_out, normal_out,
                cur_idx, n_accumulated, done_mask);

        // --- Batch-boundary persist ---------------------------------------
        // After finishing logical batch `lb`, write the current per-pixel
        // state into fwd_batch_state. Since the CSR unit is now one logical
        // batch, the CSR rank is the forward depth-walk index c = lb.
        //
        // `persist_batch_state` is called uniformly across all threads in
        // the block (no divergent control): every thread's T/pix_out/
        // normal_out is valid at this program point since we're outside
        // the per-Gaussian inner loop. This keeps the writes coalesced
        // per CSR row.
        if (persist_batches) {
            persist_batch_state<
                CDIM,
                PIXELS_PER_THREAD,
                CTA_SIZE,
                ReturnNormals>(
                lb,
                batch_base_slot, pixels_per_tile, tid,
                T, pix_out, normal_out, fwd_batch_state);
        }

        // Block-level early exit (the all-done vote was already done
        // inside process_logical_batch_gaussians via cta_sync_count, so
        // no extra `__syncthreads_count` is needed here).
        if (all_done) {
            // Block-level early exit: every pixel has hit T <=
            // TRANSMITTANCE_THRESHOLD (or was never inside). Remaining
            // persist boundaries therefore all reflect the terminal state
            // each thread holds now. Emit them before breaking so the CSR
            // slot array is completely populated — the bwd gradient kernel
            // can then load any slot `c` in [0, num_batches) without needing
            // to know which batches actually executed.
            if (persist_batches) {
                for (uint32_t lbb = lb + 1; lbb < num_logical_batches; ++lbb) {
                    persist_batch_state<
                        CDIM,
                        PIXELS_PER_THREAD,
                        CTA_SIZE,
                        ReturnNormals>(
                        lbb,
                        batch_base_slot, pixels_per_tile, tid,
                        T, pix_out, normal_out, fwd_batch_state);
                }
            }
            break;
        }
    }

    // Write outputs for each in-bounds pixel
#pragma unroll
    for (uint32_t p = 0; p < PIXELS_PER_THREAD; ++p) {
        if (pc[p].inside) {
            render_alphas[pc[p].pix_id] = 1.0f - T[p];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                render_colors[pc[p].pix_id * CDIM + k] =
                    backgrounds == nullptr ? pix_out[p][k]
                                           : (pix_out[p][k] + T[p] * backgrounds[k]);
            }
            if constexpr (ReturnNormals) {
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    render_normals[pc[p].pix_id * 3 + k] = normal_out[p][k];
                }
            }
            if (last_ids != nullptr) {
                last_ids[pc[p].pix_id] = static_cast<int32_t>(cur_idx[p]);
            }
            if (sample_counts != nullptr) {
                sample_counts[pc[p].pix_id] = n_accumulated[p];
            }
        }
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_impl(
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
    const at::Tensor isect_offsets, // [..., C, grid_h, grid_w]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    // CSR batch structure (precomputed by caller, shared with bwd)
    const at::Tensor batches_per_tile, // [num_tiles] int32
    const at::Tensor batch_offsets,   // [num_tiles + 1] int32
    const int64_t total_batches,       // scalar; equals batch_offsets[num_tiles]
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3]
    at::Tensor fwd_batch_state // [total_batches, state_dim, pixels_per_tile] fp32
) {
    // Note: quats need to be normalized before passing in.
    (void)batches_per_tile;  // reserved for future parity checks

    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed mode not supported for 3DGUT forward rasterization");

    const uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    const uint32_t B = means.numel() / (N * 3);       // number of batches
    const uint32_t C = viewmats0.size(-3);            // number of cameras
    const uint32_t I = B * C;                         // number of images
    const uint32_t grid_h = isect_offsets.size(-2);
    const uint32_t grid_w = isect_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);

    TORCH_CHECK(ut_params, "ut_params intrusive_ptr is null");
    TORCH_CHECK(ftheta_coeffs, "ftheta_coeffs intrusive_ptr is null");
    FThetaCameraDistortionDeviceParams ftheta_device_coeffs(*ftheta_coeffs);
    cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params = cuda::std::nullopt;
    if (external_distortion_params.has_value()) {
        TORCH_CHECK(external_distortion_params.value(), "external_distortion_params intrusive_ptr is null");
        external_distortion_device_params = extdist::BivariateWindshieldModelDeviceParams(*external_distortion_params.value());
    }

    cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs = cuda::std::nullopt;
    if (lidar_coeffs.has_value()) {
        TORCH_CHECK(camera_model == CameraModelType::LIDAR, "If lidar sensor coefficients are given, the camera model must be lidar");
        lidar_device_coeffs = *lidar_coeffs.value();
    }
    else
    {
        TORCH_CHECK(camera_model != CameraModelType::LIDAR, "If the sensor isn't lidar, lidar coefficients must not be given");
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(SupportedChannels::contains(channels),
        "Unsupported number of color channels: ", channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h).");

    TORCH_CHECK_VALUE(
        SupportedTileSizes::contains(tile_size),
        "Unsupported tile_size ", tile_size,
        "; supported values are {8, 16}."
    );

    // NOTE: Two (TILE_SIZE, CTA_SIZE) variants are kept because the optimum
    // differs by workload. tile_size=8 (CTA=32, PPT=2) is the compact-CTA
    // path: wins on training (mixed CDIM cameras + lidar) by keeping per-CTA
    // shmem small so many CTAs co-reside per SM. tile_size=16 (CTA=256,
    // PPT=1) is one thread per pixel: wins on render-only workloads where
    // fewer/larger tiles shrink the intersect+sort cost. Especially true with
    // 1080p and 4K resolutions. The kernel body is identical between the two;
    // only the templated constants change. min_blocks_for_cdim is re-derived
    // from CTA_SIZE so the schedule stays valid past CTA=32.
    const int safe_masked_outputs = unsafe_masked_tile_outputs ? 0 : 1;
    const bool return_normals = normals.has_value();
    auto launch_kernel =
        [&]<typename ChannelsT, typename TileSizeT, typename ReturnNormalsT,
            typename UseHitDistanceT, typename SafeMaskedOutputsT>() {
            constexpr uint32_t CDIM = ChannelsT::value;
            constexpr uint32_t TILE_SIZE = TileSizeT::value;
            constexpr uint32_t CTA_SIZE = CtaSizeForTile<TILE_SIZE>::value;
            constexpr bool ReturnNormals = static_cast<bool>(ReturnNormalsT::value);
            constexpr bool UseHitDistance = static_cast<bool>(UseHitDistanceT::value);
            constexpr bool SAFE_MASKED_OUTPUTS = SafeMaskedOutputsT::value != 0;

            const dim3 threads = {CTA_SIZE, 1, 1};
            const dim3 grid = {I, grid_h, grid_w};
            // Shared memory: id_batch + xyz_opacity_batch + iscl_rot_batch + scale_batch + normal_batch
            const int64_t shmem_size =
                CTA_SIZE * (sizeof(int32_t) + sizeof(vec4) + sizeof(mat3) + sizeof(vec3) + sizeof(vec3));

            if (cudaFuncSetAttribute(
                rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel<
                    CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, SAFE_MASKED_OUTPUTS>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size
            ) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size (requested ",
                    shmem_size,
                    " bytes)."
                );
            }

            rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel<
                CDIM, TILE_SIZE, CTA_SIZE, ReturnNormals, UseHitDistance, SAFE_MASKED_OUTPUTS>
                <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                    C, N, n_isects, packed,
                    data_ptr_as<const vec3, float>(means),
                    data_ptr_as<const vec4, float>(quats),
                    data_ptr_as<const vec3, float>(scales),
                    colors.const_data_ptr<float>(),
                    opacities.const_data_ptr<float>(),
                    data_ptr_or_null<const float>(backgrounds),
                    data_ptr_or_null<const bool>(masks),
                    image_width, image_height,
                    // camera model
                    viewmats0.const_data_ptr<float>(),
                    data_ptr_or_null<const float>(viewmats1),
                    Ks.const_data_ptr<float>(), camera_model, *ut_params, rs_type,
                    data_ptr_or_null<const float>(rays),
                    data_ptr_or_null<const float>(radial_coeffs),
                    data_ptr_or_null<const float>(tangential_coeffs),
                    data_ptr_or_null<const float>(thin_prism_coeffs),
                    ftheta_device_coeffs, lidar_device_coeffs, external_distortion_device_params,
                    // intersections
                    isect_offsets.const_data_ptr<int32_t>(),
                    flatten_ids.const_data_ptr<int32_t>(),
                    renders.data_ptr<float>(),
                    alphas.data_ptr<float>(),
                    data_ptr_or_null<float>(normals),
                    data_ptr_or_null<int32_t>(last_ids.defined(), last_ids),
                    data_ptr_or_null<int32_t>(sample_counts),
                    // CSR batch state persistence.
                    data_ptr_or_null<const int32_t>(total_batches > 0, batch_offsets),
                    data_ptr_or_null<float>(total_batches > 0, fwd_batch_state)
                );
        };
    const bool dispatched = dispatch::dispatch(
        SupportedChannels{channels},
        SupportedTileSizes{static_cast<int>(tile_size)},
        dispatch::IntParam<0, 1>{return_normals ? 1 : 0},
        dispatch::IntParam<0, 1>{use_hit_distance ? 1 : 0},
        MaskedOutputSafetyModes{safe_masked_outputs},
        std::move(launch_kernel)
    );
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

void launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel(
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
    const at::Tensor isect_offsets, // [..., C, grid_h, grid_w]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    // CSR batch structure (precomputed by caller, shared with bwd)
    const at::Tensor batches_per_tile, // [num_tiles] int32
    const at::Tensor batch_offsets,   // [num_tiles + 1] int32
    const int64_t total_batches,       // scalar; equals batch_offsets[num_tiles]
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3]
    at::Tensor fwd_batch_state // [total_batches, state_dim, pixels_per_tile] fp32
) {
    launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_impl(
        means, quats, scales, colors, opacities,
        backgrounds, masks,
        image_width, image_height, tile_size,
        viewmats0, viewmats1, Ks,
        camera_model, ut_params, rs_type,
        rays, radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_coeffs, lidar_coeffs, external_distortion_params,
        isect_offsets, flatten_ids, use_hit_distance,
        unsafe_masked_tile_outputs,
        batches_per_tile, batch_offsets, total_batches,
        renders, alphas, last_ids,
        sample_counts, normals, fwd_batch_state
    );
}

} // namespace gsplat

#endif
