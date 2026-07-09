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
#include <tuple>

#include <ATen/core/Tensor.h>

#include "Cameras.h"
#include "Common.h"
#include "Config.h"
#include "Lidars.h"
#include "ExternalDistortion.h"
#include "TorchUtils.h"

namespace at
{
class Tensor;
}

namespace gsplat
{
#define FILTER_INV_SQUARE_2DGS 2.0f

// Public outputs of rasterize_to_pixels_3dgs (the forward-internal last_ids is
// dropped).
struct RasterizeToPixels3DGSResult
{
    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor means2d_absgrad;
};

RasterizeToPixels3DGSResult rasterize_to_pixels_3dgs(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &isect_offsets,
    const at::Tensor &flatten_ids,
    bool packed,
    bool absgrad
);

// Public outputs of rasterize_to_pixels_2dgs (excludes the internal
// last_ids / median_ids that the forward kernel additionally produces).
struct RasterizeToPixels2DGSResult
{
    at::Tensor renders;
    at::Tensor alphas;
    at::Tensor render_normals;
    at::Tensor render_distort;
    at::Tensor render_median;
    at::Tensor means2d_absgrad;
};

RasterizeToPixels2DGSResult rasterize_to_pixels_2dgs(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &colors,
    const at::Tensor &opacities,
    const at::Tensor &normals,
    const at::Tensor &densify,
    const at::optional<at::Tensor> &backgrounds,
    const at::optional<at::Tensor> &masks,
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    bool packed,
    bool absgrad,
    bool distloss
);

/////////////////////////////////////////////////
// rasterize_to_pixels_3dgs
/////////////////////////////////////////////////

void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor isect_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,   // [n_isects]
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids // [..., image_height, image_width]
);

void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities                  // [..., N] or [nnz]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_sparse
/////////////////////////////////////////////////

// Sparse 3DGS rasterizer. Shares the dense per-gaussian blending math but
// touches only active tiles and writes only the requested pixels, packed in
// original-pixel order ([P, ...]). Consumes the layout produced by
// build_sparse_tile_layout (active_tiles / tile_pixel_mask / tile_pixel_cumsum /
// pixel_map) and the intersections from intersect_tile_sparse (tile_offsets /
// flatten_ids).

void launch_rasterize_to_pixels_sparse_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // sparse layout
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    // outputs
    at::Tensor renders, // [P, channels]
    at::Tensor alphas,  // [P, 1]
    at::Tensor last_ids // [P]
);

void launch_rasterize_to_pixels_sparse_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // sparse layout
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    // forward outputs
    const at::Tensor render_alphas, // [P, 1]
    const at::Tensor last_ids,      // [P]
    // gradients of outputs
    const at::Tensor v_render_colors, // [P, channels]
    const at::Tensor v_render_alphas, // [P, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, channels] or [nnz, channels]
    at::Tensor v_opacities                  // [..., N] or [nnz]
);

/////////////////////////////////////////////////
// rasterize_to_indices_3dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_3dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2]
    const at::Tensor conics,    // [..., N, 3]
    const at::Tensor opacities, // [..., N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor> chunk_starts, // [..., image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [..., image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

void launch_rasterize_num_contributing_gaussians_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    at::Tensor num_contributing,   // [..., image_height, image_width]
    at::Tensor alphas              // [..., image_height, image_width]
);

// Sparse counterpart: packed [P] outputs, consuming the sparse layout.
void launch_rasterize_num_contributing_gaussians_sparse_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    at::Tensor num_contributing,        // [P]
    at::Tensor alphas                   // [P]
);

void launch_rasterize_contributing_gaussian_ids_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t max_num_contributing,
    const at::Tensor tile_offsets,  // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,   // [n_isects]
    at::Tensor contributing_ids,    // [..., image_height, image_width, K]
    at::Tensor contributing_weights // [..., image_height, image_width, K]
);

// Sparse counterpart: packed [P, K] outputs, consuming the sparse layout.
void launch_rasterize_contributing_gaussian_ids_sparse_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t max_num_contributing,
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    at::Tensor contributing_ids,        // [P, K]
    at::Tensor contributing_weights     // [P, K]
);

void launch_rasterize_top_contributing_gaussian_ids_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t num_depth_samples,
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    at::Tensor top_ids,            // [..., image_height, image_width, K]
    at::Tensor top_weights         // [..., image_height, image_width, K]
);

// Sparse counterpart: packed [P, num_depth_samples] outputs.
void launch_rasterize_top_contributing_gaussian_ids_sparse_kernel(
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities, // [..., N] or [nnz]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t num_depth_samples,
    const at::Tensor active_tiles,      // [AT]
    const at::Tensor tile_offsets,      // [AT + 1]
    const at::Tensor flatten_ids,       // [n_isects]
    const at::Tensor tile_pixel_mask,   // [AT, words]
    const at::Tensor tile_pixel_cumsum, // [AT]
    const at::Tensor pixel_map,         // [P]
    at::Tensor top_ids,                 // [P, num_depth_samples]
    at::Tensor top_weights              // [P, num_depth_samples]
);

/////////////////////////////////////////////////
// rasterize_to_pixels_2dgs
/////////////////////////////////////////////////

void launch_rasterize_to_pixels_2dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., N]  or [nnz]
    const at::Tensor normals,                   // [..., N, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width, 1]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_distort, // [..., image_height, image_width, 1]
    at::Tensor render_median,  // [..., image_height, image_width, 1]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids      // [..., image_height, image_width]
);
void launch_rasterize_to_pixels_2dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [..., image_height, image_width, CDIM]
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors,  // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    at::Tensor v_normals,                   // [..., N, 3] or [nnz, 3]
    at::Tensor v_densify                    // [..., N, 2] or [nnz, 2]
);

/////////////////////////////////////////////////
// rasterize_to_indices_2dgs
/////////////////////////////////////////////////

void launch_rasterize_to_indices_2dgs_kernel(
    const uint32_t range_start,
    const uint32_t range_end,        // iteration steps
    const at::Tensor transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3]
    const at::Tensor opacities,      // [..., N]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // helper for double pass
    const at::optional<at::Tensor> chunk_starts, // [..., image_height, image_width]
    // outputs
    at::optional<at::Tensor> chunk_cnts,   // [..., image_height, image_width]
    at::optional<at::Tensor> gaussian_ids, // [n_elems]
    at::optional<at::Tensor> pixel_ids     // [n_elems]
);

///////////////////////////////////////////////////
// rasterize_to_pixels_from_world_3dgs
///////////////////////////////////////////////////

struct RasterizeToPixelsFromWorld3DGSFwdResult
{
    at::Tensor renders;
    at::Tensor alphas;
    // Defined when exact metadata is requested or when a forward implementation
    // needs it internally.
    at::Tensor last_ids;
    // Persisted forward batch state consumed by batch-parallel backward.
    at::Tensor batches_per_tile;
    at::Tensor batch_offsets;
    at::Tensor fwd_batch_state;
    // Defined only for ParallelBatch. MixedBatch uses the serial forward path,
    // while ParallelBatch stores the per-pixel saturation handoff batch here.
    at::Tensor compose_c_stop;
    at::Tensor priming_state;
};

// Internal C++ forward entry point. When `fwd_only` is false, the caller is
// asking for exact traversal metadata for backward/debugging. ParallelBatch
// still allocates transient batch state in fwd-only mode because its compose
// pass depends on it, but that mode must not expose exact metadata.
RasterizeToPixelsFromWorld3DGSFwdResult rasterize_to_pixels_from_world_3dgs_fwd(
    const at::Tensor means,                     // [..., N, 3]
    const at::Tensor quats,                     // [..., N, 4]
    const at::Tensor scales,                    // [..., N, 3]
    const at::Tensor colors,                    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor opacities,                 // [..., C, N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> masks,       // [..., C, tile_height, tile_width]
    uint32_t image_width,
    uint32_t image_height,
    uint32_t tile_size,
    const at::Tensor viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor Ks,                      // [..., C, 3, 3]
    CameraModelType camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    ShutterType rs_type,
    const at::optional<at::Tensor> rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    bool use_hit_distance,
    RendererConfig renderer_config,
    bool fwd_only,
    bool return_last_ids,
    const at::optional<at::Tensor> sample_counts, // [..., C, image_height, image_width] optional
    const at::optional<at::Tensor> normals,       // [..., C, image_height, image_width, 3] optional output tensor
    bool unsafe_masked_tile_outputs
);

// Public op result type. The single dispatcher entry below returns this and the
// pybind binding TU resolves its symbol through this header. last_ids is
// optional because the caller can suppress it (return_last_ids=false).
struct RasterizeToPixelsFromWorld3DGSResult
{
    at::Tensor renders;
    at::Tensor alphas;
    at::optional<at::Tensor> last_ids;
    at::optional<at::Tensor> sample_counts;
    at::optional<at::Tensor> normals;
};

template<>
struct TorchArgDef<RasterizeToPixelsFromWorld3DGSResult>
{
    static auto to(const RasterizeToPixelsFromWorld3DGSResult &r)
    {
        return to_torch_args(r.renders, r.alphas, r.last_ids, r.sample_counts, r.normals);
    }

    template<class TT>
    static RasterizeToPixelsFromWorld3DGSResult from(TT &&t)
    {
        return {
            .renders       = std::get<0>(std::forward<TT>(t)),
            .alphas        = std::get<1>(std::forward<TT>(t)),
            .last_ids      = std::get<2>(std::forward<TT>(t)),
            .sample_counts = std::get<3>(std::forward<TT>(t)),
            .normals       = std::get<4>(std::forward<TT>(t)),
        };
    }
};

// Dispatcher entry for rasterize_to_pixels_from_world_3dgs, registered for both
// the CUDA and AutogradCUDA keys. It runs the state-light forward directly when
// no tensor input requires grad (or under no-grad / inference); otherwise it
// routes through the C++ custom autograd Function so backward is owned by the
// extension. Defined in Rasterization.cpp; bound in ext.cpp.
RasterizeToPixelsFromWorld3DGSResult rasterize_to_pixels_from_world_3dgs(
    const at::Tensor &means,                     // [..., N, 3]
    const at::Tensor &quats,                     // [..., N, 4]
    const at::Tensor &scales,                    // [..., N, 3]
    const at::Tensor &colors,                    // [..., C, N, channels] or [nnz, channels]
    const at::Tensor &opacities,                 // [..., C, N] or [nnz]
    const at::optional<at::Tensor> &backgrounds, // [..., C, channels]
    const at::optional<at::Tensor> &masks,       // [..., C, tile_height, tile_width]
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    const at::Tensor &viewmats0,               // [..., C, 4, 4]
    const at::optional<at::Tensor> &viewmats1, // [..., C, 4, 4] optional for rolling shutter
    const at::Tensor &Ks,                      // [..., C, 3, 3]
    int64_t camera_model,
    const c10::intrusive_ptr<UnscentedTransformParameters> &ut_params,
    int64_t rs_type,
    const at::optional<at::Tensor> &rays,              // [..., C, H, W, 6]
    const at::optional<at::Tensor> &radial_coeffs,     // [..., C, 6] or [..., C, 4] optional
    const at::optional<at::Tensor> &tangential_coeffs, // [..., C, 2] optional
    const at::optional<at::Tensor> &thin_prism_coeffs, // [..., C, 4] optional
    const c10::intrusive_ptr<FThetaCameraDistortionParameters> &ftheta_coeffs,
    const at::optional<c10::intrusive_ptr<RowOffsetStructuredSpinningLidarModelParametersExt>> &lidar_coeffs,
    const at::optional<c10::intrusive_ptr<extdist::BivariateWindshieldModelParameters>> &external_distortion_params,
    const at::Tensor &tile_offsets, // [..., C, tile_height, tile_width]
    const at::Tensor &flatten_ids,  // [n_isects]
    bool return_sample_counts,
    bool use_hit_distance,
    bool return_normals,
    int64_t renderer_config,
    bool return_last_ids,
    bool unsafe_masked_tile_outputs
);
} // namespace gsplat
