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

#pragma once

#include <cuda/std/optional>

#include "Common.h"
#include "Cameras.cuh"
#include "KernelUtils.cuh"
#include "ExternalDistortion.cuh"
#include "Lidars.cuh"
#include "RasterizeCSR.cuh"
#include "Utils.cuh"
#include "Dispatch.h"

namespace gsplat
{
////////////////////////////////////////////////////////////////
// Compact-CTA __launch_bounds__ occupancy hint, shared by the serial- and
// parallel-batch world-space 3DGS forward kernels (both #include this header).
////////////////////////////////////////////////////////////////

// Per-architecture hardware cap on thread blocks per SM. ptxas rejects
// min_blocks_per_sm > HW cap under --warning-as-error.
//   sm_90, sm_100, sm_120: 32 blocks/SM
//   everything else:       16 blocks/SM (covers Ampere/Ada and anything older)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#    define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 32
#else
#    define GSPLAT_ARCH_MAX_BLOCKS_PER_SM 16
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
#define GSPLAT_MIN_BLOCKS_AT_MIN_CDIM (GSPLAT_ARCH_MAX_BLOCKS_PER_SM < 24 ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM : 24)
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

template<uint32_t CDIM, uint32_t CTA_SIZE>
constexpr uint32_t min_blocks_for_cdim()
{
    if constexpr(CDIM > GSPLAT_MAX_CDIM_FOR_HINT)
    {
        return 1;
    }
    else
    {
        constexpr uint32_t high            = GSPLAT_MIN_BLOCKS_AT_MIN_CDIM;
        constexpr uint32_t low             = GSPLAT_MIN_BLOCKS_AT_MAX_CDIM;
        constexpr uint32_t cdim_excess     = (CDIM > GSPLAT_MIN_CDIM_FOR_HINT) ? (CDIM - GSPLAT_MIN_CDIM_FOR_HINT) : 0u;
        constexpr uint32_t cdim_span       = GSPLAT_MAX_CDIM_FOR_HINT - GSPLAT_MIN_CDIM_FOR_HINT;
        constexpr uint32_t block_span      = (high >= low) ? (high - low) : 0;
        constexpr uint32_t decrement       = (cdim_excess * block_span) / cdim_span;
        constexpr uint32_t blocks_at_cta32 = (high > decrement) ? (high - decrement) : low;
        constexpr uint32_t threads_target  = blocks_at_cta32 * 32u;
        constexpr uint32_t blocks          = threads_target / CTA_SIZE;
        constexpr uint32_t lo_clamped      = (blocks == 0u) ? 1u : blocks;
        constexpr uint32_t hi_clamped
            = (lo_clamped > GSPLAT_ARCH_MAX_BLOCKS_PER_SM) ? GSPLAT_ARCH_MAX_BLOCKS_PER_SM : lo_clamped;
        return hi_clamped;
    }
}

#undef GSPLAT_ARCH_MAX_BLOCKS_PER_SM
#undef GSPLAT_MIN_CDIM_FOR_HINT
#undef GSPLAT_MAX_CDIM_FOR_HINT
#undef GSPLAT_MIN_BLOCKS_AT_MIN_CDIM
#undef GSPLAT_MIN_BLOCKS_AT_MAX_CDIM

// TILE_SIZE and CTA_SIZE are a tuned launch-variant pair. Add a specialization
// here whenever a new supported tile size needs its own CTA size.
template<uint32_t TILE_SIZE>
struct CtaSizeForTile;

template<>
struct CtaSizeForTile<8u>
{
    static constexpr uint32_t value = 32u;
};

template<>
struct CtaSizeForTile<16u>
{
    static constexpr uint32_t value = 256u;
};

// Supported forward tile sizes, shared so the serial- and parallel-batch
// launchers select tile_size through the same compile-time dispatch set.
using SupportedTileSizes = dispatch::IntParam<8, 16>;

////////////////////////////////////////////////////////////////
// Reusable building blocks for the world-space 3DGS rasterizer
// kernels (forward and backward).
//
// `compute_pixel_coords` and `compute_world_ray` are shared by
// both the forward and backward kernels. The remaining helpers
// (cooperative_load_fetch_round / process_fetch_round_blend /
// process_logical_batch_gaussians / persist_batch_state, plus
// the cta_sync wrappers they call) are specific to the forward
// kernel's compact-CTA path.
//
// Terminology used by the fwd-side helpers (matches the kernel
// that consumes them):
//   - Fetch round: one cooperative shared-memory refill of
//     FETCH_SIZE (= CTA_SIZE) gaussians; each thread loads its
//     own slot.
//   - Logical batch: TILE_SIZE * TILE_SIZE gaussians, processed
//     as FETCHES_PER_BATCH = LOGICAL_BATCH / FETCH_SIZE rounds.
//     This is the unit the bwd's per-batch view expects, so the
//     fwd's batch-state persist boundaries land at the same
//     gaussian positions.
////////////////////////////////////////////////////////////////

enum class SaturationTPolicy
{
    // Leave T at its pre-saturation value when the threshold is crossed.
    // SerialBatch and batch-replay use this when the post-saturation T is not
    // needed outside the local blend loop.
    KeepPreSaturationT,

    // Store the post-saturation T in T itself. Exact ParallelBatch uses this
    // so batch-scan sees a non-negative walk product and hands the boundary
    // to batch-replay for exact metadata replay.
    StorePostSaturationT,

    // Leave T at its pre-saturation value, but also expose the post-saturation
    // T through saturating_T. Fwd-only ParallelBatch uses this to publish the
    // renderable terminal partial while still priming downstream batches with
    // the saturated chain state.
    KeepPreAndCapturePostSaturationT,
};

// CTA-wide barrier + count: synchronises all threads in the CTA and
// returns the number of threads where `predicate` is true. Behaves like
// `__syncthreads_count` for now; left as a templated stub so a lower-
// overhead warp-only specialisation can drop in for CTA_SIZE_T == 32
// without touching call sites.
template<uint32_t CTA_SIZE_T>
__device__ __forceinline__ int32_t cta_sync_count(bool predicate)
{
    return __syncthreads_count(predicate ? 1 : 0);
}

// Pixel coordinates resolved for a single thread-pixel slot.
//   - row, col: integer pixel position. For OOB threads the values are
//     left at sentinels that make `inside` false (camera path: the
//     natural out-of-range row/col; LIDAR path: row=col=0). Caller MUST
//     check `inside` before treating row/col as in-image coordinates.
//   - pix_id: `row * image_width + col`. For OOB threads this is
//     undefined-but-safe — the camera path may produce an out-of-buffer
//     value; the LIDAR-OOB sentinel produces 0. **Caller MUST gate
//     every load/store at pix_id on `inside`** (or a derived flag like
//     the bwd kernel's `pixel_valid = inside && ray.valid_flag`).
//     For LIDAR-OOB threads `pix_id == 0` is intentional and is safe
//     only because `inside == false` keeps the slot off the write
//     path; without that gate it would race with the active thread
//     covering pixel `(row=0, col=0)`. Removing or weakening any
//     `inside` / `pixel_valid` gate on the consumer side would
//     surface this race.
//   - inside: false for threads that don't map to an in-image pixel
//     (LIDAR idle threads or camera-tile threads off the image edge).
//
// Leaving `pix_id` unclamped for OOB camera threads lets the compiler
// skip a `min()` on every pixel slot. Every read of `pix_id` in both
// the fwd and bwd kernels is gated on `inside` (or a derived flag
// like the bwd's `pixel_valid`); contributors must preserve this
// contract.
struct PixelCoords
{
    uint32_t row;
    uint32_t col;
    int32_t pix_id;
    bool inside;
};

// Resolve the (row, col, pix_id, inside) for a single thread-pixel slot.
//
// Two pixel-mapping conventions are supported and selected by
// `camera_model_type`:
//
//   - LIDAR: lookup in `lidar_device_coeffs->tiles_to_elements_map`
//     keyed by (tile_id, tile_element_id). Threads past the tile's
//     element_count return inside=false (and natural row=col=0).
//   - everything else (camera path): pixel = (tile_row * tile_size +
//     thread_row, tile_col * tile_size + thread_col). `inside` is the
//     bounds check against (image_height, image_width).
//
// Per-slot linearisation of (thread_row, thread_col, tile_element_id)
// is the caller's responsibility — that's the only piece that's
// different between the fwd kernel's compact-CTA layout (PPT pixels
// per thread, 1D threadIdx.x) and the bwd kernel's 2D threadIdx with
// one pixel per thread.
__device__ __forceinline__ PixelCoords compute_pixel_coords(
    const CameraModelType camera_model_type,
    const uint32_t tile_id,
    const uint32_t tile_row,
    const uint32_t tile_col,
    const uint32_t tile_size,
    const uint32_t thread_row,
    const uint32_t thread_col,
    const uint32_t tile_element_id,
    const uint32_t image_width,
    const uint32_t image_height,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> &lidar_device_coeffs
)
{
    PixelCoords out;
    if(camera_model_type == CameraModelType::LIDAR)
    {
        assert(lidar_device_coeffs);
        const int element_start = lidar_device_coeffs->tiles_pack_info[tile_id].x;
        const int element_count = lidar_device_coeffs->tiles_pack_info[tile_id].y;
        const int element_id    = static_cast<int>(tile_element_id);
        if(element_id < element_count)
        {
            out.col = lidar_device_coeffs->tiles_to_elements_map[element_start + element_id].x;
            out.row = lidar_device_coeffs->tiles_to_elements_map[element_start + element_id].y;
            assert(out.row < image_height);
            assert(out.col < image_width);
            out.inside = true;
        }
        else
        {
            out.row    = 0;
            out.col    = 0;
            out.inside = false;
        }
    }
    else
    {
        out.row    = tile_row * tile_size + thread_row;
        out.col    = tile_col * tile_size + thread_col;
        out.inside = (out.row < image_height && out.col < image_width);
    }
    // pix_id is the unclamped natural product. OOB threads (inside=false)
    // carry a value that callers must never read as an index: camera path
    // OOB threads produce a value past the buffer end; LIDAR-OOB threads
    // produce 0 (because the OOB branch above set row=col=0). See the
    // PixelCoords doc for the caller contract.
    out.pix_id = static_cast<int32_t>(out.row * image_width + out.col);
    return out;
}

// Compute the world-space ray for one pixel.
//
// Two paths:
//   - `rays == nullptr` and `inside`: dispatch on `camera_model_type`
//     and `external_distortion_device_params` to the matching camera
//     model and call `element_to_world_ray_shutter_pose(j, i, rs_params)`.
//   - otherwise (explicit rays buffer or `!inside`): when `inside`,
//     load the 6-float ray from `rays[pix_id*6..pix_id*6+6]`; when
//     not inside, return a ray with `valid_flag == false`.
//
// Caller computes `rs_params` once outside any per-pixel loop and
// passes it in — the rolling-shutter pose is the same for all
// pixels of a given thread / image-instance, so hoisting it lets
// the fwd kernel's PPT loop avoid recomputing it per pixel.
//
// Caller is responsible for honouring `ray.valid_flag` after the
// call (e.g., marking the pixel done if the camera model rejected
// the inverse projection).
template<typename scalar_t>
__device__ __forceinline__ WorldRay compute_world_ray(
    const uint32_t iid,
    const uint32_t j,
    const uint32_t i,
    const int32_t pix_id,
    const bool inside,
    const RollingShutterParameters &rs_params,
    const scalar_t *__restrict__ rays,
    const scalar_t *__restrict__ Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const scalar_t *__restrict__ radial_coeffs,
    const scalar_t *__restrict__ tangential_coeffs,
    const scalar_t *__restrict__ thin_prism_coeffs,
    const FThetaCameraDistortionDeviceParams &ftheta_device_coeffs,
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> &lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> &external_distortion_device_params
)
{
    WorldRay ray;
    if(inside && rays == nullptr)
    {
        if(camera_model_type == CameraModelType::PINHOLE)
        {
            if(radial_coeffs == nullptr && tangential_coeffs == nullptr && thin_prism_coeffs == nullptr)
            {
                if(external_distortion_device_params.has_value())
                {
                    using CameraModel = PerfectPinholeCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {{image_width, image_height}, rs_type, *external_distortion_device_params},
                        Ks,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
                else
                {
                    using CameraModel = PerfectPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {{image_width, image_height}, rs_type, {}},
                        Ks,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            }
            else
            {
                if(external_distortion_device_params.has_value())
                {
                    using CameraModel = OpenCVPinholeCameraModel<extdist::BivariateWindshieldModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {{image_width, image_height}, rs_type, *external_distortion_device_params},
                        Ks,
                        radial_coeffs,
                        tangential_coeffs,
                        thin_prism_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
                else
                {
                    using CameraModel = OpenCVPinholeCameraModel<extdist::EmptyExternalDistortionModel>;
                    CameraModel::KernelParameters kernel_params = {
                        {{image_width, image_height}, rs_type, {}},
                        Ks,
                        radial_coeffs,
                        tangential_coeffs,
                        thin_prism_coeffs,
                    };
                    CameraModel camera_model(kernel_params, iid);
                    ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
                }
            }
        }
        else if(camera_model_type == CameraModelType::ORTHO)
        {
            if(external_distortion_device_params.has_value())
            {
                using CameraModel = OrthographicCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, *external_distortion_device_params},
                    Ks,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
            else
            {
                using CameraModel = OrthographicCameraModel<extdist::EmptyExternalDistortionModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, {}},
                    Ks,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
        }
        else if(camera_model_type == CameraModelType::FISHEYE)
        {
            if(external_distortion_device_params.has_value())
            {
                using CameraModel = OpenCVFisheyeCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, *external_distortion_device_params},
                    Ks,
                    radial_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
            else
            {
                using CameraModel = OpenCVFisheyeCameraModel<extdist::EmptyExternalDistortionModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, {}},
                    Ks,
                    radial_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
        }
        else if(camera_model_type == CameraModelType::FTHETA)
        {
            if(external_distortion_device_params.has_value())
            {
                using CameraModel                           = FThetaCameraModel<extdist::BivariateWindshieldModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, *external_distortion_device_params},
                    Ks,
                    ftheta_device_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
            else
            {
                using CameraModel                           = FThetaCameraModel<extdist::EmptyExternalDistortionModel>;
                CameraModel::KernelParameters kernel_params = {
                    {{image_width, image_height}, rs_type, {}},
                    Ks,
                    ftheta_device_coeffs,
                };
                CameraModel camera_model(kernel_params, iid);
                ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
            }
        }
        else if(camera_model_type == CameraModelType::LIDAR)
        {
            using CameraModel = RowOffsetStructuredSpinningLidarModel;
            assert(lidar_device_coeffs);
            CameraModel::KernelParameters kernel_params = {*lidar_device_coeffs};
            CameraModel camera_model(kernel_params, iid);
            ray = camera_model.element_to_world_ray_shutter_pose(j, i, rs_params);
        }
        else
        {
            assert(false);
            ray.valid_flag = false;
        }
    }
    else
    {
        // Explicit rays path — rays may be nullptr for inactive threads
        // when inside == false.
        ray.valid_flag = false;
        if(inside)
        {
            assert(rays != nullptr);
            // TODO: use at least 3x64b loads instead of 6x32b
            ray.ray_org    = {rays[pix_id * 6 + 0], rays[pix_id * 6 + 1], rays[pix_id * 6 + 2]};
            ray.ray_dir    = {rays[pix_id * 6 + 3], rays[pix_id * 6 + 4], rays[pix_id * 6 + 5]};
            ray.valid_flag = true;
        }
    }
    return ray;
}

// Cooperative load of one fetch round.
//
// Each thread loads a single gaussian (its own `tid`-th shared
// slot) when `batch_start + tid < range_end`. The slot count is
// FETCH_SIZE_T; the caller must size the shared-memory arrays to
// at least that many entries.
//
// CTA_SIZE_T is the dispatching block size; FETCH_SIZE_T must
// not exceed it (otherwise some slots would never be loaded).
// When CTA_SIZE_T > FETCH_SIZE_T the excess threads skip the
// load via the per-thread slot-bound check.
//
// Caller is responsible for the post-load sync (cta_sync) before
// any thread reads the shared batches. We don't sync inside so
// that callers performing multiple loads with different ranges
// can amortise the barrier when appropriate.
template<uint32_t FETCH_SIZE_T, uint32_t CTA_SIZE_T, bool ReturnNormals, typename scalar_t>
__device__ __forceinline__ void cooperative_load_fetch_round(
    const uint32_t tid,
    int32_t *__restrict__ id_batch,
    vec4 *__restrict__ xyz_opacity_batch,
    mat3 *__restrict__ iscl_rot_batch,
    vec3 *__restrict__ scale_batch,
    vec3 *__restrict__ normal_batch,
    const uint32_t batch_start,
    const int32_t range_end,
    const int32_t *__restrict__ flatten_ids,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ opacities,
    const uint32_t C,
    const uint32_t N
)
{
    // FETCH_SIZE_T > CTA_SIZE_T would leave shared-memory slots in
    // [CTA_SIZE_T, FETCH_SIZE_T) unloaded; downstream readers would
    // see garbage. Reject at compile time.
    static_assert(FETCH_SIZE_T <= CTA_SIZE_T, "FETCH_SIZE_T must not exceed CTA_SIZE_T");

    // Whether `tid` maps to a valid shared-memory slot. When
    // FETCH_SIZE_T == CTA_SIZE_T no thread is excess and the check
    // is statically true; the `if constexpr` lets nvcc elide it.
    // Without this fold, nvcc emits a redundant ISETP that
    // serialises the predicate chain with `idx < range_end`.
    auto thread_within_fetch_slot = [&]()
    {
        if constexpr(FETCH_SIZE_T == CTA_SIZE_T)
        {
            return true;
        }
        else
        {
            return tid < FETCH_SIZE_T;
        }
    };

    // Each thread fetches 1 gaussian from front to back
    const uint32_t idx = batch_start + tid;
    if(thread_within_fetch_slot() && idx < (uint32_t)range_end)
    {
        // TODO: only support 1 camera for now so it is ok to abuse the index.
        int32_t isect_id       = flatten_ids[idx];
        int32_t isect_bid      = isect_id / (C * N);
        int32_t isect_gid      = isect_id % N;
        id_batch[tid]          = isect_id;
        const vec3 xyz         = means[isect_bid * N + isect_gid];
        const float opac       = opacities[isect_id];
        xyz_opacity_batch[tid] = {xyz.x, xyz.y, xyz.z, opac};

        const vec4 quat = quats[isect_bid * N + isect_gid];
        vec3 scale      = scales[isect_bid * N + isect_gid];

        // Projection kernel culls degenerate Gaussians (zero quaternion,
        // zero scale) by setting radii = 0, preventing them from entering
        // the intersection list. Assert the preconditions here.
        assert(glm::dot(quat, quat) > 0.f);
        assert(scale[0] > 0.f && scale[1] > 0.f && scale[2] > 0.f);

        mat3 R              = quat_to_rotmat(quat);
        mat3 S              = mat3(1.0f / scale[0], 0.f, 0.f, 0.f, 1.0f / scale[1], 0.f, 0.f, 0.f, 1.0f / scale[2]);
        iscl_rot_batch[tid] = S * glm::transpose(R);
        scale_batch[tid]    = scale;

        // Normal = R * (0, 0, 1) = third column of R.
        if constexpr(ReturnNormals)
        {
            normal_batch[tid] = R[2];
        }
    }
}

// Per-thread blend over the gaussians already in shared memory
// for one fetch round.
//
// PIXELS_PER_THREAD_T is the per-thread pixel count (compact CTA:
// TILE_SIZE * TILE_SIZE / CTA_SIZE). Each pixel has its own slot
// in T / pix_out / normal_out / cur_idx / n_accumulated and its
// own bit in done_mask.
//
// CHECK_THRESHOLD enables the per-pixel transmittance early-mark:
// when next_T <= the per-pixel transmittance_threshold[p] supplied by the
// caller, the pixel's bit in done_mask is set and the gaussian's
// contribution is dropped. (In the parallel-batch partials kernel that
// threshold is the priming-tightened bound TRANSMITTANCE_THRESHOLD / T_init.)
// Always instantiated `true` by both the serial-batch and parallel-batch
// forward kernels; `false` is not currently instantiated.
//
// SaturationPolicy controls what survives after a gaussian crosses the
// threshold: keep pre-saturation T, store post-saturation T in T, or keep the
// pre-saturation T while also capturing the post-saturation T in saturating_T.
//
// Caller must run cta_sync between the cooperative load and this
// call so all threads see the loaded shared batches.
template<
    uint32_t CDIM,
    uint32_t PIXELS_PER_THREAD_T,
    bool CHECK_THRESHOLD,
    SaturationTPolicy SaturationPolicy,
    bool UseHitDistance,
    bool ReturnNormals,
    typename scalar_t
>
__device__ __forceinline__ void process_fetch_round_blend(
    const int32_t *__restrict__ id_batch,
    const vec4 *__restrict__ xyz_opacity_batch,
    const mat3 *__restrict__ iscl_rot_batch,
    const vec3 *__restrict__ scale_batch,
    const vec3 *__restrict__ normal_batch,
    const uint32_t batch_start,
    const uint32_t batch_size,
    const scalar_t *__restrict__ colors,
    const vec3 (&ray_o)[PIXELS_PER_THREAD_T],
    const vec3 (&ray_d)[PIXELS_PER_THREAD_T],
    const uint32_t ALL_DONE,
    const float (&transmittance_threshold)[PIXELS_PER_THREAD_T],
    float (&T)[PIXELS_PER_THREAD_T],
    float (&saturating_T)[PIXELS_PER_THREAD_T],
    float (&pix_out)[PIXELS_PER_THREAD_T][CDIM],
    vec3 (&normal_out)[PIXELS_PER_THREAD_T],
    int32_t (&cur_idx)[PIXELS_PER_THREAD_T],
    int32_t (&n_accumulated)[PIXELS_PER_THREAD_T],
    uint32_t &done_mask
)
{
    constexpr bool store_post_saturation_t   = SaturationPolicy == SaturationTPolicy::StorePostSaturationT;
    constexpr bool capture_post_saturation_t = SaturationPolicy == SaturationTPolicy::KeepPreAndCapturePostSaturationT;

    for(uint32_t t = 0; (t < batch_size) && (done_mask != ALL_DONE); ++t)
    {
        const vec4 xyz_opac = xyz_opacity_batch[t];
        const float opac    = xyz_opac[3];
        const vec3 xyz      = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
        const mat3 iscl_rot = iscl_rot_batch[t];
        const vec3 scale    = scale_batch[t];

#pragma unroll
        for(uint32_t p = 0; p < PIXELS_PER_THREAD_T; ++p)
        {
            if(done_mask & (1u << p))
            {
                continue;
            }

            const vec3 gro    = iscl_rot * (ray_o[p] - xyz);
            const vec3 grd    = safe_normalize(iscl_rot * ray_d[p]);
            // hit_t < 0: closest approach is behind the camera origin — skip.
            const float hit_t = -glm::dot(grd, gro);
            if(hit_t < 0.f)
            {
                continue;
            }
            const vec3 gcrod     = glm::cross(grd, gro);
            const float grayDist = glm::dot(gcrod, gcrod);
            const float power    = -0.5f * grayDist;
            float max_response   = __expf(power);
            float alpha          = min(MAX_ALPHA, opac * max_response);

            if(alpha < ALPHA_THRESHOLD)
            {
                continue;
            }

            float hit_distance = 0.0f;
            if constexpr(UseHitDistance)
            {
                const vec3 grds = scale * (grd * hit_t);
                hit_distance    = glm::length(grds);
            }

            const float next_T = T[p] * (1.0f - alpha);
            if constexpr(CHECK_THRESHOLD)
            {
                if(next_T <= transmittance_threshold[p])
                {
                    if constexpr(capture_post_saturation_t)
                    {
                        saturating_T[p] = next_T;
                    }
                    if constexpr(store_post_saturation_t)
                    {
                        T[p] = next_T;
                    }
                    done_mask |= (1u << p);
                    continue;
                }
            }

            int32_t isect_id   = id_batch[t];
            const float vis    = alpha * T[p];
            const float *c_ptr = colors + isect_id * CDIM;

            if constexpr(UseHitDistance)
            {
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    const float value  = (k == CDIM - 1) ? hit_distance : c_ptr[k];
                    pix_out[p][k]     += value * vis;
                }
            }
            else
            {
#pragma unroll
                for(uint32_t k = 0; k < CDIM; ++k)
                {
                    pix_out[p][k] += c_ptr[k] * vis;
                }
            }

            if constexpr(ReturnNormals)
            {
                const vec3 unnormalized_normal   = normal_batch[t];
                const bool flipped               = glm::dot(unnormalized_normal, ray_d[p]) > 0.0f;
                const vec3 unnormalized_flipped  = flipped ? -unnormalized_normal : unnormalized_normal;
                const vec3 normal                = safe_normalize(unnormalized_flipped);
                normal_out[p]                   += normal * vis;
            }

            cur_idx[p] = batch_start + t;
            n_accumulated[p]++;
            T[p] = next_T;
        }
    }
}

// Process one logical batch — wraps FETCHES_PER_BATCH cooperative
// load + blend rounds. Returns true when the per-CTA all-done
// predicate fired at the end of a fetch round (every pixel in the
// CTA hit `done_mask == ALL_DONE`), so the caller can skip remaining
// logical batches without re-doing the vote.
//
// LOGICAL_BATCH_T is the per-batch gaussian count (= TILE_SIZE *
// TILE_SIZE in the consumer kernel); FETCH_SIZE_T is the
// cooperative-load slot count (= CTA_SIZE_T); their ratio is the
// number of fetch rounds inside one logical batch.
//
// No inter-fetch-round sync is added between rounds: with
// CTA_SIZE_T == 32 the whole CTA is one warp and lockstep
// execution covers the gap; with CTA_SIZE_T > 32 the consumer
// kernel uses LOGICAL_BATCH == FETCH_SIZE so FETCHES_PER_BATCH
// is 1 and the question is moot.
template<
    uint32_t CDIM,
    uint32_t LOGICAL_BATCH_T,
    uint32_t FETCH_SIZE_T,
    uint32_t CTA_SIZE_T,
    uint32_t PIXELS_PER_THREAD_T,
    bool CHECK_THRESHOLD,
    SaturationTPolicy SaturationPolicy,
    bool UseHitDistance,
    bool ReturnNormals,
    typename scalar_t
>
__device__ __forceinline__ bool process_logical_batch_gaussians(
    const uint32_t tid,
    int32_t *__restrict__ id_batch,
    vec4 *__restrict__ xyz_opacity_batch,
    mat3 *__restrict__ iscl_rot_batch,
    vec3 *__restrict__ scale_batch,
    vec3 *__restrict__ normal_batch,
    const uint32_t logical_batch_start,
    const int32_t range_end,
    const int32_t *__restrict__ flatten_ids,
    const vec3 *__restrict__ means,
    const vec4 *__restrict__ quats,
    const vec3 *__restrict__ scales,
    const scalar_t *__restrict__ opacities,
    const scalar_t *__restrict__ colors,
    const uint32_t C,
    const uint32_t N,
    const vec3 (&ray_o)[PIXELS_PER_THREAD_T],
    const vec3 (&ray_d)[PIXELS_PER_THREAD_T],
    const uint32_t ALL_DONE,
    const float (&transmittance_threshold)[PIXELS_PER_THREAD_T],
    float (&T)[PIXELS_PER_THREAD_T],
    float (&saturating_T)[PIXELS_PER_THREAD_T],
    float (&pix_out)[PIXELS_PER_THREAD_T][CDIM],
    vec3 (&normal_out)[PIXELS_PER_THREAD_T],
    int32_t (&cur_idx)[PIXELS_PER_THREAD_T],
    int32_t (&n_accumulated)[PIXELS_PER_THREAD_T],
    uint32_t &done_mask
)
{
    static_assert(LOGICAL_BATCH_T % FETCH_SIZE_T == 0, "LOGICAL_BATCH_T must be a multiple of FETCH_SIZE_T");
    constexpr uint32_t FETCHES_PER_BATCH = LOGICAL_BATCH_T / FETCH_SIZE_T;
    static_assert(FETCHES_PER_BATCH >= 1, "FETCHES_PER_BATCH must be >= 1");
    // The multi-fetch path (FETCHES_PER_BATCH > 1) carries no inter-round
    // barrier and is race-free only at warp width. Both forward kernels honor
    // this: tile8 -> CTA_SIZE 32 (FETCHES 2), tile16 -> CTA_SIZE 256 (FETCHES 1).
    static_assert(
        CTA_SIZE_T == 32 || FETCHES_PER_BATCH == 1,
        "CTA_SIZE_T > 32 requires FETCHES_PER_BATCH == 1; the multi-fetch path "
        "relies on warp-synchronous execution (no inter-round barrier)."
    );

#pragma unroll
    for(uint32_t r = 0; r < FETCHES_PER_BATCH; ++r)
    {
        const uint32_t batch_start = logical_batch_start + FETCH_SIZE_T * r;
        // Skip rounds that fall past the tile's gaussian range (last
        // logical batch may be partial). Each thread votes "no fetch"
        // uniformly via the per-thread `idx >= range_end` guard inside
        // cooperative_load_fetch_round; here we only skip the
        // cooperative fetch entirely when the whole round is past
        // range_end.
        if(batch_start >= (uint32_t)range_end)
        {
            break;
        }

        cooperative_load_fetch_round<FETCH_SIZE_T, CTA_SIZE_T, ReturnNormals, scalar_t>(
            tid,
            id_batch,
            xyz_opacity_batch,
            iscl_rot_batch,
            scale_batch,
            normal_batch,
            batch_start,
            range_end,
            flatten_ids,
            means,
            quats,
            scales,
            opacities,
            C,
            N
        );

        cta_sync<CTA_SIZE_T>();

        const uint32_t batch_size = min(FETCH_SIZE_T, (uint32_t)range_end - batch_start);
        process_fetch_round_blend<
            CDIM,
            PIXELS_PER_THREAD_T,
            CHECK_THRESHOLD,
            SaturationPolicy,
            UseHitDistance,
            ReturnNormals,
            scalar_t
        >(id_batch,
          xyz_opacity_batch,
          iscl_rot_batch,
          scale_batch,
          normal_batch,
          batch_start,
          batch_size,
          colors,
          ray_o,
          ray_d,
          ALL_DONE,
          transmittance_threshold,
          T,
          saturating_T,
          pix_out,
          normal_out,
          cur_idx,
          n_accumulated,
          done_mask);

        // CTA-wide early stop: if every pixel in the CTA has crossed
        // the transmittance threshold (`done_mask == ALL_DONE` on every
        // thread), there's no point loading further fetch rounds. The
        // sync that backs `cta_sync_count` doubles as the inter-round
        // synchronisation, so the result is broadcast for free.
        if(cta_sync_count<CTA_SIZE_T>(done_mask == ALL_DONE) >= (int32_t)CTA_SIZE_T)
        {
            return true;
        }
    }
    return false;
}

// Persist the CURRENT per-pixel cumulative state
// (T, pix_out, and optional normal_out) into one batch slot of the CSR
// `fwd_batch_state` buffer. Caller is
// responsible for gating on `fwd_batch_state != nullptr` (no internal
// early-return) and for selecting the correct forward depth-walk batch index
// `c`.
//
// SOA layout: each (slot, state element, thread_row=tid + p*CTA_SIZE_T) writes
// `state_dim = FWD_BATCH_STATE_PIX_OFFSET + CDIM +
// (ReturnNormals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0)`. Pixels are
// fastest-varying so each warp writes a contiguous span for one state element.
//
// `c=0` corresponds to the front-most batch boundary; `c=num_batches-1`
// corresponds to the terminal state. The CSR slot layout these indices
// follow is defined in `RasterizeCSR.cuh`.
template<uint32_t CDIM, uint32_t PIXELS_PER_THREAD_T, uint32_t CTA_SIZE_T, bool ReturnNormals>
__device__ __forceinline__ void persist_batch_state(
    uint32_t c,
    int64_t batch_base_slot,
    uint32_t pixels_per_tile,
    uint32_t tid,
    const float (&T)[PIXELS_PER_THREAD_T],
    const float (&pix_out)[PIXELS_PER_THREAD_T][CDIM],
    const vec3 (&normal_out)[PIXELS_PER_THREAD_T],
    float *__restrict__ fwd_batch_state
)
{
    constexpr uint32_t state_dim
        = FWD_BATCH_STATE_PIX_OFFSET + CDIM + (ReturnNormals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0u);
    const int64_t ppt64 = static_cast<int64_t>(pixels_per_tile);
#pragma unroll
    for(uint32_t p = 0; p < PIXELS_PER_THREAD_T; ++p)
    {
        const uint32_t tr       = tid + p * CTA_SIZE_T;
        const int64_t slot      = batch_base_slot + static_cast<int64_t>(c);
        const int64_t slot_base = slot * static_cast<int64_t>(state_dim) * ppt64;
        const int64_t pix64     = static_cast<int64_t>(tr);
        fwd_batch_state[slot_base + FWD_BATCH_STATE_T_OFFSET * ppt64 + pix64] = T[p];
#pragma unroll
        for(uint32_t k = 0; k < CDIM; ++k)
        {
            fwd_batch_state[slot_base + (FWD_BATCH_STATE_PIX_OFFSET + k) * ppt64 + pix64] = pix_out[p][k];
        }
        if constexpr(ReturnNormals)
        {
            fwd_batch_state[slot_base + (FWD_BATCH_STATE_PIX_OFFSET + CDIM + 0) * ppt64 + pix64] = normal_out[p].x;
            fwd_batch_state[slot_base + (FWD_BATCH_STATE_PIX_OFFSET + CDIM + 1) * ppt64 + pix64] = normal_out[p].y;
            fwd_batch_state[slot_base + (FWD_BATCH_STATE_PIX_OFFSET + CDIM + 2) * ppt64 + pix64] = normal_out[p].z;
        }
    }
}
} // namespace gsplat
