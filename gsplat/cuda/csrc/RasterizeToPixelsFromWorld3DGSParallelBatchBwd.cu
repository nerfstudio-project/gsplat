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
#include <ATen/Functions.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda/std/optional>

#include "Common.h"
#include "ExternalDistortion.cuh"
#include "RasterizeCSR.cuh"
#include "RasterizeToPixelsFromWorld3DGS.h"
#include "RasterizeToPixelsFromWorld3DGS.cuh"
#include "Utils.cuh"
#include "Cameras.cuh"
#include "Lidars.cuh"
#include "TorchUtils.h"
#include "Dispatch.h"

namespace gsplat {

using SupportedChannels = dispatch::IntParam<GSPLAT_NUM_CHANNELS>;

namespace cg = cooperative_groups;

// Pad CDIM to an odd stride when it would be even, so that the per-thread
// STORE `rgbs_batch[tr * stride + k] = colors[...]` in bwd writes to distinct
// 32-bit banks across a warp. With a stride S, the 32 lanes in a warp land
// in banks (tr * S) mod 32 for tr in [0,32); that hits all 32 banks iff
// gcd(S, 32) == 1, i.e. S is odd. The inner loop READ from a fixed `t` and
// varying `k` is a broadcast-per-lane and is not the conflict site.
// sizeof(float) == 4 → 32-bit banks; static_assert below pins that assumption.
template <uint32_t CDIM>
constexpr uint32_t cdim_smem_stride() {
    static_assert(sizeof(float) == 4, "bank layout assumes 32-bit banks");
    return (CDIM % 2 == 0) ? CDIM + 1 : CDIM;
}

// ---------------------------------------------------------------------------
// Single-kernel batch-parallel backward pass.
//
// fwd persists per-batch cumulative state into `fwd_batch_state` at every
// batch boundary; bwd derives the per-batch starting accumulators via a
// single CDIM dot + vec3 dot per thread in
// its preamble, then runs the per-gaussian gradient walk. See
// `RasterizeCSR.cuh` for the tensor layout and boundary formula.
//
// Math (derived from the bwd recurrence in the hot loop below):
//   T_start_c          = fwd_batch_state[slot=c, T_OFFSET, pix]
//                      = T_fwd at fwd-batch c (depth-walk index from front)
//   render_accum_dot_c = dot(v_render_c,
//                            pix_out_final - pix_out_fwd_at_boundary_c)
//   normal_accum_dot_c = dot(v_render_n,
//                            normal_out_final - normal_out_fwd_at_boundary_c)
//   where `_final` is the c=num_batches-1 slot (terminal fwd state,
//   deepest batch) — the CSR-slot semantics are documented in
//   `RasterizeCSR.cuh`.
//
// Grid: 1D {total_batches, 1, 1}. Batches are independent — no sequential
// dependency — exposing ample work to the GPU scheduler and eliminating the
// tail latency on tiles with many Gaussians.
// ---------------------------------------------------------------------------

// Device kernel that computes per-tile batch counts. This was previously
// file-scoped to this TU; it is now reachable by the fwd persist path via
// the `compute_batch_csr` host helper below. Kept `static` because only
// this TU launches it directly.
static __global__ void compute_batches_per_tile_kernel(
    const int32_t *__restrict__ tile_offsets,
    const uint32_t num_tiles,
    const uint32_t n_isects,
    const int32_t pixels_per_tile,
    int32_t *__restrict__ batches_per_tile
) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tiles) {
        return;
    }
    const int32_t start = tile_offsets[t];
    const int32_t end = (t + 1 < num_tiles)
        ? tile_offsets[t + 1]
        : static_cast<int32_t>(n_isects);
    const int32_t range = end - start;
    if (range <= 0) {
        batches_per_tile[t] = 0;
        return;
    }
    assert(pixels_per_tile > 0);
    const int32_t num_batches =
        (range + pixels_per_tile - 1) / pixels_per_tile;
    batches_per_tile[t] = num_batches;
}

// Host helper: see declaration in `RasterizeCSR.cuh`. Launches the
// device kernel above, builds the [num_tiles+1] CSR offsets via cumsum+cat,
// then reads back `total_batches` with a blocking `.item<int32_t>()`. Shared
// by fwd (persist buffer sizing) and bwd (gradient-kernel grid sizing).
std::tuple<at::Tensor, at::Tensor, int64_t> compute_batch_csr(
    const at::Tensor &tile_offsets,
    int64_t n_isects,
    uint32_t num_tiles,
    int32_t pixels_per_tile,
    at::TensorOptions dummy_options
) {
    auto int_opts = dummy_options.dtype(at::kInt);
    auto batches_per_tile_t =
        at::empty({static_cast<int64_t>(num_tiles)}, int_opts);
    if (num_tiles > 0) {
        const uint32_t threads_per_block = 256;
        const uint32_t blocks =
            (num_tiles + threads_per_block - 1) / threads_per_block;
        compute_batches_per_tile_kernel
            <<<blocks, threads_per_block, 0,
               at::cuda::getCurrentCUDAStream()>>>(
                tile_offsets.const_data_ptr<int32_t>(),
                num_tiles,
                static_cast<uint32_t>(n_isects),
                pixels_per_tile,
                batches_per_tile_t.data_ptr<int32_t>());
    }
    // batch_offsets[0..num_tiles]: exclusive prefix sum with a trailing sum.
    auto batch_offsets_tail = at::cumsum(batches_per_tile_t, 0, at::kInt);
    auto batch_offsets_t = at::cat(
        {at::zeros({1}, int_opts), batch_offsets_tail});
    const int64_t total_batches = num_tiles == 0
        ? int64_t{0}
        : static_cast<int64_t>(
              batch_offsets_t[static_cast<int64_t>(num_tiles)]
                  .item<int32_t>());
    return std::make_tuple(batches_per_tile_t, batch_offsets_t, total_batches);
}

// --- Helpers for CSR-packed batch state ---
//
// Each tile contributes `batches_per_tile[t]` batch-slots to the flat
// fwd_batch_state buffer, with `batch_offsets[t]` giving the starting slot
// of tile t. `batch_offsets[num_tiles]` is the total number of batch-slots
// and equals the CTA count for the gradient kernel. The flat layout
// avoids the per-tile padding (`num_tiles x max_batches`) that would
// otherwise create a dense-scene OOM vector driven by the worst tile.

// Ray-generation helper (shared with the fwd kernel) lives in
// `RasterizeToPixelsFromWorld3DGS.cuh` as `compute_world_ray`.

// Pixel-coordinate resolution helper (shared with the fwd kernel) lives in
// `RasterizeToPixelsFromWorld3DGS.cuh` as `compute_pixel_coords`.

// Gradient pass. Per-CTA: (1) batch_id decoded from grid x via binary
// search on batch_offsets, (2) initial state materialised from the
// fwd-persisted `fwd_batch_state` tensor via one CDIM dot + one vec3 dot
// per thread, (3) one front-aligned depth batch is walked, (4) v_rays is accumulated
// via atomicAdd because multiple batches contribute per pixel.
//
// Grid: 1D {total_batches, 1, 1}; each CTA's (tile_linear, batch_id) is decoded
// via a binary search on batch_offsets in the preamble.
template <
    uint32_t CDIM,
    bool ReturnNormals,
    bool UseHitDistance,
    typename scalar_t
>
__global__ void rasterize_gradient_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // fwd inputs
    const vec3 *__restrict__ means,           // [B, N, 3]
    const vec4 *__restrict__ quats,           // [B, N, 4]
    const vec3 *__restrict__ scales,          // [B, N, 3]
    const scalar_t *__restrict__ colors,      // [B, C, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [B, C, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [B, C, CDIM]
    const bool *__restrict__ masks,           // [B, C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // camera model
    const scalar_t *__restrict__ viewmats0, // [B, C, 4, 4]
    const scalar_t *__restrict__ viewmats1, // [B, C, 4, 4] optional for rolling shutter
    const scalar_t *__restrict__ Ks,        // [B, C, 3, 3]
    const CameraModelType camera_model_type,
    const ShutterType rs_type,
    const scalar_t *__restrict__ rays,              // [B, C, H, W, 6]
    const scalar_t *__restrict__ radial_coeffs,     // [B, C, 6] or [B, C, 4] optional
    const scalar_t *__restrict__ tangential_coeffs, // [B, C, 2] optional
    const scalar_t *__restrict__ thin_prism_coeffs, // [B, C, 4] optional
    const FThetaCameraDistortionDeviceParams ftheta_device_coeffs, // shared parameters for all cameras
    const cuda::std::optional<RowOffsetStructuredSpinningLidarModelParametersExtDevice> lidar_device_coeffs,
    const cuda::std::optional<extdist::BivariateWindshieldModelDeviceParams> external_distortion_device_params,
    // intersections
    const int32_t *__restrict__ tile_offsets, // [B, C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [B, C, image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [B, C, image_height, image_width]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [B, C, image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [B, C, image_height, image_width, 1]
    const scalar_t
        *__restrict__ v_render_normals, // [B, C, image_height, image_width, 3] optional
    // grad inputs
    vec3 *__restrict__ v_means,        // [B, N, 3]
    vec4 *__restrict__ v_quats,        // [B, N, 4]
    vec3 *__restrict__ v_scales,       // [B, N, 3]
    scalar_t *__restrict__ v_colors,   // [B, C, N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [B, C, N] or [nnz]
    scalar_t *__restrict__ v_rays,     // [B, C, image_height, image_width, 6]
    // Fwd-persisted per-batch cumulative state (CSR-packed; see
    // `RasterizeCSR.cuh`). Within a slot, state elements are ordered as
    // [T, pix_out[CDIM], optional normal_out[3]] and pixels are fastest-varying. Slot
    // c=num_batches-1 is the terminal fwd state for MixedBatch and for
    // non-saturating ParallelBatch pixels. Saturating ParallelBatch pixels use
    // the c_stop slot as the terminal reference after batch-replay overwrites
    // it with post-saturation state; batch CTAs with batch_id > c_stop skip
    // the pixel because it never reached those gaussians.
    const scalar_t *__restrict__ fwd_batch_state, // [total_batches, state_dim, pixels_per_tile]
    const int32_t *__restrict__ batch_offsets,    // [num_tiles + 1]
    const uint16_t *__restrict__ compose_c_stop   // [num_tiles, pixels_per_tile] or null
)
{
    // ---- Preamble: CSR batch decode, pixel map, ray gen, fwd-state load ----
    // The batch-start accumulators are derived from `fwd_batch_state` further
    // down via one CDIM dot + one vec3 dot per thread.
    auto block = cg::this_thread_block();
    const uint32_t num_tiles_total = (B * C) * tile_height * tile_width;
    const int32_t bid = static_cast<int32_t>(block.group_index().x);
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
    const uint32_t tr = block.thread_rank();
    const uint32_t block_size = block.size();

    const int32_t image_width_px = static_cast<int32_t>(image_width);
    const int32_t image_height_px = static_cast<int32_t>(image_height);
    assert(image_width_px > 0 && image_height_px > 0);
    assert(image_height_px <= INT32_MAX / image_width_px);
    const int32_t image_area = image_height_px * image_width_px;
    tile_offsets += image_index * tiles_per_image;
    render_alphas += image_index * image_area;
    last_ids += image_index * image_area;
    v_render_colors += image_index * image_area * CDIM;
    v_render_alphas += image_index * image_area;
    if constexpr (ReturnNormals) {
        v_render_normals += image_index * image_area * 3;
    }
    if (backgrounds != nullptr) {
        backgrounds += image_index * CDIM;
    }
    if (masks != nullptr) {
        masks += image_index * tiles_per_image;
    }
    if(rays != nullptr) {
        rays += image_index * image_area * 6;
    }
    if (v_rays != nullptr) {
        v_rays += image_index * image_area * 6;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end =
        (image_index == static_cast<int32_t>(B * C) - 1) &&
            (tile_id == tiles_per_image - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const int32_t pixels_per_tile = tile_size * tile_size;
    // Grid sized to total_batches (CSR); every launched block has a valid slot.

    // Pixel mapping (camera vs lidar) — shared helper.
    const PixelCoords pc = compute_pixel_coords(
        camera_model_type,
        static_cast<uint32_t>(tile_id),
        static_cast<uint32_t>(tile_row),
        static_cast<uint32_t>(tile_col),
        tile_size,
        threadIdx.y, threadIdx.x, tr,
        image_width, image_height, lidar_device_coeffs);
    const uint32_t i = pc.row;
    const uint32_t j = pc.col;
    const bool inside = pc.inside;
    const int32_t pix_id = pc.pix_id;

    // Ray generation — shared with the fwd kernel via compute_world_ray
    // in RasterizeToPixelsFromWorld3DGS.cuh.
    auto rs_params = RollingShutterParameters(
        viewmats0 + image_index * 16,
        viewmats1 == nullptr ? nullptr : viewmats1 + image_index * 16);
    WorldRay ray = compute_world_ray<scalar_t>(
        static_cast<uint32_t>(image_index), j, i, pix_id, inside, rs_params,
        rays, Ks,
        image_width, image_height,
        camera_model_type, rs_type,
        radial_coeffs, tangential_coeffs, thin_prism_coeffs,
        ftheta_device_coeffs, lidar_device_coeffs,
        external_distortion_device_params);
    const vec3 ray_d = ray.ray_dir;
    const vec3 ray_o = ray.ray_org;
    const bool pix_valid_raw = inside && ray.valid_flag;
    const bool has_compose_stop = compose_c_stop != nullptr;
    const int32_t thread_rank = static_cast<int32_t>(tr);
    assert(tile_linear <=
           (INT32_MAX - thread_rank) / pixels_per_tile);
    const int32_t compose_idx =
        tile_linear * pixels_per_tile + thread_rank;
    const int32_t c_stop =
        (has_compose_stop && inside)
            ? decode_compose_c_stop(compose_c_stop[compose_idx])
            : decode_compose_c_stop(COMPOSE_C_STOP_NONE);
    assert(
        !has_compose_stop || !inside || ray.valid_flag ||
        c_stop == decode_compose_c_stop(COMPOSE_C_STOP_INVALID_RAY));
    const bool sat_flag = c_stop >= 0;
    const bool pixel_valid =
        pix_valid_raw &&
        !(sat_flag && batch_id > c_stop);
    const int32_t bin_final = pixel_valid ? last_ids[pix_id] : 0;

    // No lane in this CTA can contribute a gradient when every pixel is
    // invalid, out of tile, or already past its c_stop batch. This sync also
    // serves as the pre-cooperative-load CTA sync below.
    const bool any_pixel_valid = __syncthreads_or(pixel_valid);
    if (!any_pixel_valid) {
        return;
    }

    const float T_final = pixel_valid ? 1.0f - render_alphas[pix_id] : 1.0f;

    // ---- Per-pixel gradients (same as original kernel) ----
    // These are loaded BEFORE the batch-state materialisation below so the
    // CDIM / vec3 dots can be fused in the same arithmetic region without
    // a second pass over v_render_c / v_render_n.
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = pixel_valid ? v_render_colors[pix_id * CDIM + k] : 0.f;
    }
    const float v_render_a = pixel_valid ? v_render_alphas[pix_id] : 0.f;

    vec3 v_render_n = vec3(0.f);
    if constexpr (ReturnNormals) {
        if (pixel_valid) {
            v_render_n.x = v_render_normals[pix_id * 3 + 0];
            v_render_n.y = v_render_normals[pix_id * 3 + 1];
            v_render_n.z = v_render_normals[pix_id * 3 + 2];
        }
    }

    // ---- Materialise batch-start state from `fwd_batch_state` ----------------
    // The monolithic bwd recurrence satisfies, at the start of bwd batch c:
    //   T_start_c          = T_fwd at fwd-batch c (depth-walk index from front)
    //   render_accum_dot_c = dot(v_render_c, pix_out_final - pix_out_boundary_c)
    //   normal_accum_dot_c = dot(v_render_n, normal_out_final - normal_out_boundary_c)
    // where slot c=num_batches-1 IS the terminal fwd state (deepest batch;
    // see `RasterizeCSR.cuh`).
    //
    // CSR slot layout: `[slot, state_element, pix]`, with state elements
    // ordered as `[T, pix_out[CDIM], optional normal_out[3]]`.
    //   - This CTA's boundary slot: `bid` (=batch_offsets[tile]+batch_id).
    //   - The terminal slot for this tile: `batch_offsets[tile_linear+1]-1`
    //     (c=num_batches-1, deepest batch).
    //
    // Slot IDs are int32 CSR offsets. Keep logical offsets signed and narrow
    // until the final pointer add inside FwdBatchSlotView.
    const int32_t terminal_slot = batch_offsets[tile_linear + 1] - 1;
    const FwdBatchSlotView<CDIM, ReturnNormals, const scalar_t>
        boundary_slot_view(
            fwd_batch_state,
            bid,
            pixels_per_tile,
            thread_rank);

    const int32_t final_slot = (sat_flag && pixel_valid)
        ? batch_offsets[tile_linear] + c_stop
        : terminal_slot;
    const FwdBatchSlotView<CDIM, ReturnNormals, const scalar_t>
        final_slot_view(
            fwd_batch_state,
            final_slot,
            pixels_per_tile,
            thread_rank);
    const bool final_matches_boundary = pixel_valid &&
        (final_slot == bid);

    // T starts from the fwd state after this batch. For the saturating
    // boundary, batch-replay has already overwritten the c_stop slot with the
    // post-replay state that backward needs here.
    float T = 1.0f;
    if (pixel_valid) {
        T = boundary_slot_view.T();
    }

    // render_accum_dot = dot(v_render_c, pix_out_final - pix_out_boundary)
    // Streaming per-k load keeps the register footprint scalar (one `delta`
    // in flight at a time) rather than materialising pix_out_boundary[CDIM]
    // as live registers — critical for the kernel's already-tight register budget.
    float render_accum_dot = 0.f;
    if (pixel_valid && !final_matches_boundary) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            const float pix_final_k = final_slot_view.feature(k);
            const float pix_boundary_k = boundary_slot_view.feature(k);
            const float delta_k = pix_final_k - pix_boundary_k;
            render_accum_dot += delta_k * v_render_c[k];
        }
    }

    // normal_accum_dot = dot(v_render_n, normal_final - normal_boundary).
    // Guarded by the constexpr flag so the normal-slot loads compile away when
    // normals are disabled.
    float normal_accum_dot = 0.f;
    if constexpr (ReturnNormals) {
        if (pixel_valid && !final_matches_boundary) {
            const vec3 normal_final = final_slot_view.normal();
            const vec3 normal_boundary = boundary_slot_view.normal();
            const vec3 normal_delta = normal_final - normal_boundary;
            normal_accum_dot = normal_delta.x * v_render_n.x
                             + normal_delta.y * v_render_n.y
                             + normal_delta.z * v_render_n.z;
        }
    }
    float background_render_dot = 0.f;
    if (pixel_valid && backgrounds != nullptr) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            background_render_dot += backgrounds[k] * v_render_c[k];
        }
    }
    const float v_alpha_ind_coeff = v_render_a - background_render_dot;

    vec3 v_ray_o = {0.f, 0.f, 0.f};
    vec3 v_ray_d = {0.f, 0.f, 0.f};

    extern __shared__ int s[];  // id_batch prefix + xyz_opacity + scale + quat + rgbs
    int32_t *id_batch = (int32_t *)s;
    vec4 *xyz_opacity_batch =
        reinterpret_cast<vec4 *>(&id_batch[block_size]);
    vec3 *scale_batch =
        reinterpret_cast<vec3 *>(&xyz_opacity_batch[block_size]);
    vec4 *quat_batch =
        reinterpret_cast<vec4 *>(&scale_batch[block_size]);
    float *rgbs_batch = (float *)&quat_batch[block_size];

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    // ---- Batch walk: this CTA owns one front-aligned depth batch ----
    {
        const int32_t batch_start =
            range_start + static_cast<int32_t>(block_size) *
                batch_id;
        const int32_t batch_end = min(
            range_end - 1,
            batch_start + static_cast<int32_t>(block_size) - 1);
        const int32_t batch_size = batch_end + 1 - batch_start;
        const int32_t idx = batch_end - static_cast<int32_t>(tr);

        // Threads with tr >= batch_size would otherwise alias into the
        // previous front-side batch for the partial deepest batch.
        if (idx >= batch_start) {
            // TODO: only support 1 camera for now so it is ok to abuse the index.
            int32_t isect_id = flatten_ids[idx]; // flatten index in [B * C * N] or [nnz]
            int32_t isect_bid = isect_id / (C * N);   // intersection batch index
            // int32_t isect_cid = (isect_id / N) % C; // intersection camera index
            int32_t isect_gid = isect_id % N;          // intersection gaussian index
            id_batch[tr] = isect_id;
            const vec3 xyz = means[isect_bid * N + isect_gid];
            const float opac = opacities[isect_id];
            xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
            scale_batch[tr] = scales[isect_bid * N + isect_gid];
            quat_batch[tr] = quats[isect_bid * N + isect_gid];
            assert(glm::dot(quat_batch[tr], quat_batch[tr]) > 0.f);
            assert(scale_batch[tr][0] > 0.f && scale_batch[tr][1] > 0.f && scale_batch[tr][2] > 0.f);
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * cdim_smem_stride<CDIM>() + k] = colors[isect_id * CDIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = pixel_valid;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            float vis;

            mat3 R, S;
            vec3 xyz;
            vec3 scale;
            vec4 quat;
            mat3 Mt;
            vec3 o_minus_mu, gro, grd, grd_n, gcrod;
            float grayDist, power;
            // Per-pixel hit-distance state. Three values:
            // - `hit_distance`: world-frame length of `grds` (per-pixel)
            // - `hit_t`:        whitened parametric closest-point distance
            // - `grds`:         scale-weighted whitened ray direction
            //
            // Held in registers (not in shmem `rgbs_batch`) because they
            // depend on ray_o/ray_d (per-pixel), whereas `rgbs_batch` is
            // per-Gaussian (shared across all pixels in the tile).
            //
            // Computed once in the alpha-recompute block below and reused by:
            // - the per-pixel rgb-render-dot path (hit_distance only)
            // - the hit-distance VJP block       (all three)
            float hit_distance = 0.f;
            float hit_t = 0.f;
            vec3 grds = vec3(0.f);
            if (valid) {
                const vec4 xyz_opac = xyz_opacity_batch[t];
                opac = xyz_opac[3];
                xyz = {xyz_opac[0], xyz_opac[1], xyz_opac[2]};
                scale = scale_batch[t];
                quat = quat_batch[t];

                R = quat_to_rotmat(quat);
                S = mat3(
                    1.0f / scale[0],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[1],
                    0.f,
                    0.f,
                    0.f,
                    1.0f / scale[2]
                );
                // Match fwd's Mt construction expression form (S * Rᵀ vs
                // (R * S)ᵀ are mathematically equal for diagonal S, but
                // compilers may pick different FFMA fusions per source
                // form). Today nvcc emits identical SASS for both — this
                // is forward-defensive: a future compiler / surrounding-
                // code change could break the bit-equivalence we get
                // today, and source-form match keeps it stable. Softer
                // guarantee than `safe_normalize`'s explicit-intrinsic
                // pinning, but no intrinsics exist for matrix
                // construction expressions.
                Mt = S * glm::transpose(R);
                o_minus_mu = ray_o - xyz;
                gro = Mt * o_minus_mu;
                grd = Mt * ray_d;
                grd_n = safe_normalize(grd);
                // hit_t < 0 → closest approach behind camera; skip this gaussian.
                // Declared in outer scope so the hit-distance VJP block can reuse it.
                hit_t = -glm::dot(grd_n, gro);
                if (hit_t < 0.f) {
                    valid = false;
                }
                else {
                    gcrod = glm::cross(grd_n, gro);
                    grayDist = glm::dot(gcrod, gcrod);
                    power = -0.5f * grayDist;

                    vis = __expf(power);
                    alpha = min(MAX_ALPHA, opac * vis);
                    // grayDist = dot(gcrod, gcrod) is a sum of three squares, so
                    // grayDist >= 0 and therefore power = -0.5 * grayDist <= 0
                    // under any IEEE-754 evaluation order on finite inputs.
                    // Assert the invariant and use the same skip predicate as
                    // fwd. The assert also fires on NaN power, which would
                    // itself indicate an upstream numerics bug feeding NaN
                    // into gcrod.
                    assert(power <= 0.f);
                    if (alpha < ALPHA_THRESHOLD) {
                        valid = false;
                    }

                    if constexpr (UseHitDistance) {
                        grds = scale * (grd_n * hit_t);
                        hit_distance = glm::length(grds);
                    }
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_mean_local = {0.f, 0.f, 0.f};
            vec3 v_scale_local = {0.f, 0.f, 0.f};
            vec4 v_quat_local = {0.f, 0.f, 0.f, 0.f};
            float v_opacity_local = 0.f;

            // initialize everything to 0, only set if the lane is valid
            vec3 normal = {0.f, 0.f, 0.f};
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / fmaxf(MIN_ONE_MINUS_ALPHA, 1.0f - alpha);
                T *= ra;
                // Per-Gaussian color VJP: v_rgb_local[k] = fac * v_render_c[k].
                //
                // Last-channel special case when UseHitDistance is on:
                // - fwd substitutes per-pixel hit_distance for colors[g,CDIM-1]
                //   in pix_out, so colors[..., CDIM-1] is structurally absent
                //   from the rendered output.
                // - Therefore d(loss)/d(colors[..., CDIM-1]) = 0.
                // - Stop the loop one channel short so the zero-initialized
                //   v_rgb_local[CDIM-1] stays 0.
                // - The hit_distance VJP below pulls the depth-channel
                //   gradient straight from v_render_c[CDIM-1], not from this
                //   slot.
                const float fac = alpha * T;
                constexpr uint32_t kRgbWriteCount = UseHitDistance ? CDIM - 1u : CDIM;
#pragma unroll
                for (uint32_t k = 0; k < kRgbWriteCount; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }

                // Precompute normal if needed. Used by:
                // - v_alpha computation (this block)
                // - normal-gradient VJP block below
                //
                // Hoist `flipped` and `unnormalized_flipped` to outer scope so
                // the gradient block reuses the values without recomputing
                // R[2] / the dot / the flip.
                bool flipped = false;
                vec3 unnormalized_flipped = vec3(0.f);
                if constexpr (ReturnNormals) {
                    // Recompute normal from forward pass
                    // normal = R * (0, 0, 1) = R[:, 2] (third column)
                    const vec3 unnormalized_normal = R[2];

                    // Direction resolution: flip if facing away from ray
                    flipped = glm::dot(unnormalized_normal, ray_d) > 0.0f;
                    unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal;

                    // Normalize
                    normal = safe_normalize(unnormalized_flipped);
                }
                
                // contribution from this pixel
                float rgb_render_dot = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    // For the last channel with UseHitDistance, use the per-pixel
                    // hit_distance instead of per-Gaussian rgbs_batch (which is
                    // shared memory and would race across pixels in the tile).
                    const float rgb_k = (UseHitDistance && k == CDIM - 1)
                        ? hit_distance
                        : rgbs_batch[t * cdim_smem_stride<CDIM>() + k];
                    rgb_render_dot += rgb_k * v_render_c[k];
                }

                // Per-Gaussian projection of the rendered-normal grad onto the
                // Gaussian's normal vector — used in the product-rule term for
                // v_alpha below.
                float normal_render_dot = 0.f;
                if constexpr (ReturnNormals) {
                    normal_render_dot = glm::dot(normal, v_render_n);
                }
                
                // Add contribution from normals to v_alpha (product rule term)
                float v_alpha = rgb_render_dot * T - render_accum_dot * ra
                              + T_final * ra * v_alpha_ind_coeff;
                // Forward: render_normals += normal * vis (where vis = alpha * T)
                // So v_alpha_normals = dot(normal * T - normal_accum * ra, v_render_n)
                if constexpr (ReturnNormals) {
                    v_alpha += normal_render_dot * T - normal_accum_dot * ra;
                }

                // Add contribution from hit distance (if enabled)
                vec3 v_grd_n_hit = vec3(0.f);
                vec3 v_gro_hit = vec3(0.f);
                if constexpr (UseHitDistance) {
                    // Depth-channel gradient for the hit_distance VJP.
                    // v_rgb_local[CDIM-1] was left at 0 above (fwd replaces
                    // that color channel with hit_distance, so the color
                    // VJP must be 0). Compute v_depth directly from
                    // v_render_c instead of reading the zeroed slot.
                    const float v_depth = fac * v_render_c[CDIM - 1];

                    // hit_t / grds / hit_distance were computed in the
                    // alpha-recompute block above and hoisted to outer scope;
                    // reuse them here to avoid recomputation.

                    // Backward through length(grds)
                    vec3 v_grds = vec3(0.f);
                    if (hit_distance > 1e-8f) {
                        v_grds = (grds / hit_distance) * v_depth;
                    }

                    // Backward through grds = scale * grd_n * hit_t (element-wise multiply)
                    // d/d(hit_t): scale * grd_n
                    const float v_hit_t = glm::dot(scale * grd_n, v_grds);
                    // d/d(grd_n): scale * hit_t (element-wise)
                    v_grd_n_hit = (scale * hit_t) * v_grds;  // element-wise
                    // d/d(scale): grd_n * hit_t (element-wise)
                    v_scale_local += (grd_n * hit_t) * v_grds;  // element-wise

                    // Backward through hit_t = dot(grd_n, -gro)
                    v_grd_n_hit += -gro * v_hit_t;
                    v_gro_hit = -grd_n * v_hit_t;
                }

                if (opac * vis <= MAX_ALPHA) {
                    const float v_vis = opac * v_alpha;
                    float v_gradDist = -0.5f * vis * v_vis;
                    vec3 v_gcrod = 2.0f * v_gradDist * gcrod;
                    vec3 v_grd_n = -glm::cross(v_gcrod, gro) + v_grd_n_hit;
                    vec3 v_gro = glm::cross(v_gcrod, grd_n) + v_gro_hit;
                    vec3 v_grd = safe_normalize_bw(grd, v_grd_n);
                    mat3 v_Mt = glm::outerProduct(v_grd, ray_d) + 
                        glm::outerProduct(v_gro, o_minus_mu);
                    vec3 v_o_minus_mu = glm::transpose(Mt) * v_gro;

                    v_mean_local += -v_o_minus_mu;
                    
                    // Compute ray gradients
                    // From o_minus_mu = ray_o - xyz, we get:
                    v_ray_o += v_o_minus_mu;
                    // From grd = Mt * ray_d, we get:
                    v_ray_d += glm::transpose(Mt) * v_grd;

                    quat_scale_to_preci_half_vjp(
                        quat, scale, R, glm::transpose(v_Mt), v_quat_local, v_scale_local
                    );
                    v_opacity_local = vis * v_alpha;
                    
                    // Compute normal gradient contribution (if computing normals)
                    // Note: normal was precomputed above for v_alpha contribution
                    if constexpr (ReturnNormals) {
                        // Compute gradient contribution
                        // Forward: render_normals += normal * fac (where fac = alpha * T)
                        const vec3 v_normal_local = v_render_n * fac;
                        
                        // Forward: normal = safe_normalize(unnormalized_flipped)
                        // unnormalized_flipped was computed in the v_alpha
                        // precompute block above and reused here.
                        const vec3 v_unnormalized_flipped = safe_normalize_bw(unnormalized_flipped, v_normal_local);

                        // Forward: unnormalized_flipped = flipped ? -unnormalized_normal : unnormalized_normal
                        const vec3 v_unnormalized = flipped ? -v_unnormalized_flipped : v_unnormalized_flipped;

                        // Forward: R[2][:] = unnormalized_normal
                        const mat3 v_R = mat3(vec3(0.f, 0.f, 0.f), vec3(0.f, 0.f, 0.f), v_unnormalized);

                        // backward through R = quat_to_rotmat(quat)
                        quat_to_rotmat_vjp(quat, v_R, v_quat_local);
                    }
                }

                render_accum_dot += rgb_render_dot * fac;
                
                // Accumulate normal contribution (for product rule in next iterations).
                if constexpr (ReturnNormals) {
                    normal_accum_dot += normal_render_dot * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_mean_local, warp);
            warpSum(v_scale_local, warp);
            warpSum(v_quat_local, warp);
            warpSum(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t isect_id = id_batch[t]; // flatten index in [B * C * N] or [nnz]
                int32_t isect_bid = isect_id / (C * N);   // intersection batch index
                // int32_t isect_cid = (isect_id / N) % C;   // intersection camera index
                int32_t isect_gid = isect_id % N;         // intersection gaussian index
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * isect_id;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_mean_ptr = (float *)(v_means) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_mean_ptr, v_mean_local.x);
                gpuAtomicAdd(v_mean_ptr + 1, v_mean_local.y);
                gpuAtomicAdd(v_mean_ptr + 2, v_mean_local.z);

                float *v_scale_ptr = (float *)(v_scales) + 3 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_scale_ptr, v_scale_local.x);
                gpuAtomicAdd(v_scale_ptr + 1, v_scale_local.y);
                gpuAtomicAdd(v_scale_ptr + 2, v_scale_local.z);

                float *v_quat_ptr = (float *)(v_quats) + 4 * (isect_bid * N + isect_gid);
                gpuAtomicAdd(v_quat_ptr, v_quat_local.x);
                gpuAtomicAdd(v_quat_ptr + 1, v_quat_local.y);
                gpuAtomicAdd(v_quat_ptr + 2, v_quat_local.z);
                gpuAtomicAdd(v_quat_ptr + 3, v_quat_local.w);

                gpuAtomicAdd(v_opacities + isect_id, v_opacity_local);
            }
        }
    }

    // v_rays: atomicAdd since multiple batches contribute per pixel.
    if (v_rays != nullptr && pixel_valid) {
        float *vr = (float *)(v_rays) + 6 * pix_id;
        gpuAtomicAdd(vr + 0, v_ray_o.x);
        gpuAtomicAdd(vr + 1, v_ray_o.y);
        gpuAtomicAdd(vr + 2, v_ray_o.z);
        gpuAtomicAdd(vr + 3, v_ray_d.x);
        gpuAtomicAdd(vr + 4, v_ray_d.y);
        gpuAtomicAdd(vr + 5, v_ray_d.z);
    }
}

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // camera
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    const CameraModelType camera_model,
    // uncented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets,    // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,     // [n_isects]
    const bool use_hit_distance,
    // forward outputs
    const at::Tensor render_alphas,   // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,        // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., C, image_height, image_width, 1]
    const at::optional<at::Tensor> v_render_normals, // [..., C, image_height, image_width, 3]
    // CSR batch structure precomputed by the forward pass
    const at::Tensor batches_per_tile,  // [num_tiles] int32
    const at::Tensor batch_offsets,    // [num_tiles + 1] int32
    const int64_t total_batches,        // scalar, equals batch_offsets[num_tiles]
    // Per-batch cumulative state persisted by the fwd kernel at every
    // batch boundary. The bwd kernel reads it directly and
    // derives the per-batch starting accumulators in its preamble — no
    // separate state-scan or prefix-scan pass needed.
    const at::Tensor fwd_batch_state,  // [total_batches, state_dim, pixels_per_tile] fp32
    // ParallelBatch-only saturation handoff. MixedBatch passes an undefined
    // tensor and uses the terminal fwd_batch_state slot as before.
    const at::Tensor compose_c_stop,    // [num_tiles, pixels_per_tile] uint16
    // outputs
    at::Tensor v_means,      // [..., N, 3]
    at::Tensor v_quats,      // [..., N, 4]
    at::Tensor v_scales,     // [..., N, 3]
    at::Tensor v_colors,     // [..., C, N, channels] or [nnz, channels]
    at::Tensor v_opacities,  // [..., C, N] or [nnz]
    at::optional<at::Tensor> v_rays // [..., C, image_height, image_width, 6]
) {
    bool packed = opacities.dim() == 1;
    TORCH_CHECK(!packed, "packed opacities are not supported in the 3DGS world-space backward");
    (void)batches_per_tile;  // only used for the shape check below

    uint32_t N = packed ? 0 : means.size(-2);   // number of gaussians
    uint32_t B = means.numel() / (N * 3);       // number of batches
    uint32_t C = viewmats0.size(-3);            // number of cameras
    uint32_t I = B * C;                         // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    dim3 threads = {tile_size, tile_size, 1};
    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.

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
    } else {
        TORCH_CHECK(camera_model != CameraModelType::LIDAR, "If the sensor isn't lidar, lidar coefficients must not be given");
    }

    const int32_t channels = colors.size(-1);
    TORCH_CHECK_VALUE(SupportedChannels::contains(channels),
        "Unsupported number of color channels: ", channels,
        ". To add support, rebuild gsplat with this channel count included "
        "in -DGSPLAT_NUM_CHANNELS=... (see gsplat/cuda/csrc/Config.h).");

    const int32_t pixels_per_tile = tile_size * tile_size;
    const uint32_t num_tiles = I * tile_height * tile_width;

    // CSR batch structure was computed in the forward pass by compute_batch_csr,
    // declared in RasterizeCSR.cuh and implemented in this TU with
    // compute_batches_per_tile_kernel, then threaded through save_for_backward.
    // We consume the precomputed tensors directly, avoiding the kernel launch +
    // cumsum + blocking `.item<int32_t>()` readback that used to live here on
    // every bwd.
    TORCH_CHECK(batches_per_tile.numel() == static_cast<int64_t>(num_tiles),
                "batches_per_tile has wrong size");
    TORCH_CHECK(batch_offsets.numel() == static_cast<int64_t>(num_tiles) + 1,
                "batch_offsets has wrong size");

    const bool return_normals = v_render_normals.has_value();
    auto launch_kernel = [&]<typename ChannelsT,
                            typename ReturnNormalsT,
                            typename UseHitDistanceT>() {
        constexpr uint32_t CDIM = ChannelsT::value;
        constexpr bool ReturnNormals = static_cast<bool>(ReturnNormalsT::value);
        constexpr bool UseHitDistance = static_cast<bool>(UseHitDistanceT::value);
        int64_t shmem_size =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec4) + sizeof(vec3) + sizeof(vec4) + sizeof(float) * cdim_smem_stride<CDIM>());

        // Shape parity check for the fwd-persisted state. fwd allocates with
        // shape `[total_batches, state_dim, pixels_per_tile]` and fills every
        // boundary slot; early-exit padding fills only the remaining slots with
        // terminal state, so bwd can read every slot. Check each dim — matching
        // numel alone would accept a transposed tensor with the same total
        // element count but wrong stride, silently corrupting bwd's reads.
        const int64_t state_dim =
            FWD_BATCH_STATE_PIX_OFFSET + CDIM +
            (ReturnNormals ? FWD_BATCH_STATE_NORMAL_EXTRA : 0);
        TORCH_CHECK(fwd_batch_state.dim() == 3,
                    "fwd_batch_state must be 3-D (got ",
                    fwd_batch_state.dim(), "-D)");
        TORCH_CHECK(fwd_batch_state.size(0) == total_batches &&
                        fwd_batch_state.size(1) == state_dim &&
                        fwd_batch_state.size(2) == pixels_per_tile,
                    "fwd_batch_state has wrong shape (got [",
                    fwd_batch_state.size(0), ", ", fwd_batch_state.size(1),
                    ", ", fwd_batch_state.size(2), "], expected [",
                    total_batches, ", ", state_dim, ", ", pixels_per_tile, "])");
        TORCH_CHECK(
            fwd_batch_state.numel() <= INT32_MAX,
            "ParallelBatch fwd_batch_state exceeds signed 32-bit device "
            "offset range (",
            fwd_batch_state.numel(),
            " elements)");
        const bool use_compose_stop = compose_c_stop.defined();
        if (use_compose_stop) {
            TORCH_CHECK(compose_c_stop.dim() == 2,
                        "compose_c_stop must be 2-D");
            TORCH_CHECK(
                compose_c_stop.scalar_type() == at::kUInt16,
                "compose_c_stop must be uint16");
            TORCH_CHECK(
                compose_c_stop.size(0) == static_cast<int64_t>(num_tiles) &&
                    compose_c_stop.size(1) == pixels_per_tile,
                "compose_c_stop has wrong shape");
            TORCH_CHECK(
                compose_c_stop.numel() <= INT32_MAX,
                "ParallelBatch compose_c_stop exceeds signed 32-bit device "
                "range");
        }

        // ---- Gradient kernel: 1D grid over CSR batch-slots ----
        // The preamble loads per-batch boundary and terminal state from
        // `fwd_batch_state` and materialises the starting accumulators via
        // one CDIM dot + one vec3 dot per thread, at the cost of one extra
        // CSR slot read per batch (the terminal slot).
        dim3 grad_grid = {static_cast<uint32_t>(total_batches), 1, 1};
        if (cudaFuncSetAttribute(
                rasterize_gradient_bwd_kernel<CDIM, ReturnNormals, UseHitDistance, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size) != cudaSuccess) {
            AT_ERROR("Failed to set shmem for gradient kernel (",
                     shmem_size, " bytes).");
        }
        rasterize_gradient_bwd_kernel<CDIM, ReturnNormals, UseHitDistance, float>
            <<<grad_grid, threads, shmem_size,
               at::cuda::getCurrentCUDAStream()>>>(
                B, C, N, n_isects,
                data_ptr_as<const vec3, float>(means),
                data_ptr_as<const vec4, float>(quats),
                data_ptr_as<const vec3, float>(scales),
                colors.const_data_ptr<float>(),
                opacities.const_data_ptr<float>(),
                data_ptr_or_null<const float>(backgrounds),
                data_ptr_or_null<const bool>(masks),
                image_width, image_height, tile_size,
                tile_width, tile_height,
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
                render_alphas.const_data_ptr<float>(),
                last_ids.const_data_ptr<int32_t>(),
                v_render_colors.const_data_ptr<float>(),
                v_render_alphas.const_data_ptr<float>(),
                data_ptr_or_null<const float>(v_render_normals),
                data_ptr_as<vec3, float>(v_means),
                data_ptr_as<vec4, float>(v_quats),
                data_ptr_as<vec3, float>(v_scales),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                data_ptr_or_null<float>(v_rays),
                fwd_batch_state.const_data_ptr<float>(),
                batch_offsets.const_data_ptr<int32_t>(),
                data_ptr_or_null<const uint16_t>(use_compose_stop, compose_c_stop));
    };
    const bool dispatched = dispatch::dispatch(
        SupportedChannels{channels},
        dispatch::IntParam<0, 1>{return_normals ? 1 : 0},
        dispatch::IntParam<0, 1>{use_hit_distance ? 1 : 0},
        std::move(launch_kernel));
    TORCH_CHECK(dispatched, "dispatch failed: no matching compile-time instantiation for runtime parameters");
}

} // namespace gsplat

#endif
