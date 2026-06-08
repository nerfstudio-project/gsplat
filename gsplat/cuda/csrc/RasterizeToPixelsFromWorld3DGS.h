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

#include <cstdint>

#include "Cameras.h"
#include "Common.h"
#include "Lidars.h"
#include "ExternalDistortion.h"

namespace at {
class Tensor;
}

namespace gsplat {

void launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N]  or [nnz]
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
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays, // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
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
    at::Tensor fwd_batch_state // [total_batches, state_dim, pixels_per_tile] fp32, persisted cumulative state for bwd reuse
);

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means,     // [..., N, 3]
    const at::Tensor quats,     // [..., N, 4]
    const at::Tensor scales,    // [..., N, 3]
    const at::Tensor colors,    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., C, N]  or [nnz]
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
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays, // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    const bool unsafe_masked_tile_outputs,
    // CSR batch structure (precomputed by caller, shared with bwd)
    const at::Tensor batches_per_tile, // [num_tiles] int32
    const at::Tensor batch_offsets,   // [num_tiles + 1] int32
    const at::Tensor bid_to_slot,     // [total_batches] int32
    const int64_t total_batches,       // scalar; equals batch_offsets[num_tiles]
    bool fwd_only, // skip exact debug/backward metadata and batch-replay
    // outputs
    at::Tensor renders, // [..., C, image_height, image_width, channels]
    at::Tensor alphas,  // [..., C, image_height, image_width]
    at::Tensor last_ids, // [..., C, image_height, image_width]
    at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width]
    at::optional<at::Tensor> normals, // [..., C, image_height, image_width, 3]
    at::Tensor fwd_batch_state, // [total_batches, state_dim, pixels_per_tile] fp32
    at::Tensor partials_meta, // [total_batches, pixels_per_tile, 2] uint16
    at::Tensor batch_replay_preamble, // [num_tiles, pixels_per_tile, 2] int32
    at::Tensor compose_c_stop, // [num_tiles, pixels_per_tile] uint16
    at::Tensor priming_state // [..., C, H, W] int32, temporary ParallelBatch fwd chain
);

void launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means,  // [..., N, 3]
    const at::Tensor quats,  // [..., N, 4]
    const at::Tensor scales, // [..., N, 3]
    const at::Tensor colors,                    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., C, N] or [nnz]
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
    // unscented transform
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays, // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs, // shared parameters for all cameras
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    // intersections
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool use_hit_distance,
    // forward outputs
    const at::Tensor render_alphas, // [..., C, image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., C, image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., C, image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., C, image_height, image_width, 1]
    const at::optional<at::Tensor> v_render_normals, // [..., C, image_height, image_width, 3]
    // CSR batch structure (precomputed by forward)
    const at::Tensor batches_per_tile, // [num_tiles] int32
    const at::Tensor batch_offsets,   // [num_tiles + 1] int32
    const int64_t total_batches,       // scalar; equals batch_offsets[num_tiles]
    // Per-batch cumulative (T, pix_out, normal_out) persisted by the fwd pass.
    const at::Tensor fwd_batch_state, // [total_batches, state_dim, pixels_per_tile] fp32
    // ParallelBatch-only saturation handoff. Undefined tensor keeps MixedBatch
    // on the legacy terminal-slot path without placeholder allocations.
    const at::Tensor compose_c_stop,  // [num_tiles, pixels_per_tile] uint16
    // outputs
    at::Tensor v_means,      // [..., N, 3]
    at::Tensor v_quats,      // [..., N, 4]
    at::Tensor v_scales,     // [..., N, 3]
    at::Tensor v_colors,     // [..., C, N, 3] or [nnz, 3]
    at::Tensor v_opacities,  // [..., C, N] or [nnz]
    at::optional<at::Tensor> v_rays // [..., C, image_height, image_width, 6]
) ;

} // namespace gsplat
