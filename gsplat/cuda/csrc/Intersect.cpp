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

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "MathUtils.h"
#include "Common.h"    // where all the macros are defined
#include "Config.h"
#include "Intersect.h" // where the launch function is declared

namespace gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor &means2d,                           // [..., N, 2] or [nnz, 2]
    const at::Tensor &radii,                             // [..., N, 2] or [nnz, 2]
    const at::Tensor &depths,                            // [..., N] or [nnz]
    const at::optional<at::Tensor> &conics,              // [..., N, 3] or [nnz, 3] 
    const at::optional<at::Tensor> &opacities,           // [..., N] or [nnz]        
    const at::optional<at::Tensor> &image_ids,           // [nnz]
    const at::optional<at::Tensor> &gaussian_ids,        // [nnz]
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (conics.has_value()) { CHECK_INPUT(conics.value()); }
    if (opacities.has_value()) { CHECK_INPUT(opacities.value()); }

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    // packed-mode `offsets` (below) reduce a 1-D [nnz] tiles_per_gauss and
    // collapse to a single [0, total] segment, so a per-image segmented sort
    // would index past the 2-entry offsets buffer. Reject until packed offsets
    // are built per image.
    TORCH_CHECK(
        !(packed && segmented),
        "segmented sort is not supported for packed inputs"
    );

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits = bits_for_count(n_tiles);
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=", I, ", n_tiles=", n_tiles, ").");

    // first pass: compute number of tiles per gaussian
    // TODO: This first pass can be avoided, see todo comment in intersect_tile_lidar_kernel.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;
    if (n_elements) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics, // at::optional, AccuTile when provided, AABB fallback otherwise
            opacities, // at::optional
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        // Explicit int64 to match the kernel's int64 read of cum_tiles_per_gauss.
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
        if (segmented) {
            // offsets in the isect_ids and flatten_ids
            offsets = at::cumsum(
                at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong
            );
            offsets = at::cat(
                {at::tensor({0}, opt.dtype(at::kLong)),
                offsets}
            );
        }
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (n_isects) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics, // at::optional, AccuTile when provided, AABB fallback otherwise
            opacities, // at::optional
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort) {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        if (segmented) {
            segmented_radix_sort_double_buffer(
                n_isects,
                I,
                image_n_bits,
                tile_n_bits,
                offsets,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        } else {
            radix_sort_double_buffer(
                n_isects,
                image_n_bits,
                tile_n_bits,
                isect_ids,
                flatten_ids,
                isect_ids_sorted, 
                flatten_ids_sorted
            );
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const int64_t I,
    const bool sort,
    const bool segmented
) {
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(false, "intersect_tile_lidar requires GSPLAT_BUILD_3DGUT=1");
    return {};
#else
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    // packed-mode `offsets` (below) reduce a 1-D [nnz] tiles_per_gauss and
    // collapse to a single [0, total] segment, so a per-image segmented sort
    // would index past the 2-entry offsets buffer. Reject until packed offsets
    // are built per image.
    TORCH_CHECK(
        !(packed && segmented),
        "segmented sort is not supported for packed inputs"
    );

    uint32_t n_tiles = lidar->n_bins_azimuth*lidar->n_bins_elevation;
    // the number of bits needed to encode the image id and tile id
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits = bits_for_count(n_tiles);
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile_lidar: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=", I, ", n_tiles=", n_tiles, ").");

    // first pass: compute number of tiles per gaussian
    // TODO: This first pass can be avoided, see todo comment in intersect_tile_lidar_kernel.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;
    if (n_elements)
    {
        launch_intersect_tile_lidar_kernel(
            // inputs
            lidar,
            means2d,
            radii,
            depths,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            tiles_per_gauss,
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
        if (segmented)
        {
            // offsets in the isect_ids and flatten_ids
            offsets = at::cumsum(
                at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong
            );
            offsets = at::cat(
                {at::tensor({0}, opt.dtype(at::kLong)),
                offsets}
            );
        }
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (n_isects) {
        TORCH_CHECK(cum_tiles_per_gauss.scalar_type() == at::ScalarType::Long,
                    "cum_tiles_per_gauss must be int64 for lidar kernel");
        TORCH_CHECK(cum_tiles_per_gauss.is_contiguous(),
                    "cum_tiles_per_gauss must be contiguous");

        launch_intersect_tile_lidar_kernel(
            // inputs
            lidar,
            means2d,
            radii,
            depths,
            packed ? image_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            I,
            cum_tiles_per_gauss,
            // outputs
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids)
        );
    }

    // optionally sort the Gaussians by isect_ids
    if (n_isects && sort)
    {
        at::Tensor isect_ids_sorted = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        if (segmented)
        {
            segmented_radix_sort_double_buffer(
                n_isects,
                I,
                image_n_bits,
                tile_n_bits,
                offsets,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        }
        else
        {
            radix_sort_double_buffer(
                n_isects,
                image_n_bits,
                tile_n_bits,
                isect_ids,
                flatten_ids,
                isect_ids_sorted,
                flatten_ids_sorted
            );
        }
        return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
#endif // !GSPLAT_BUILD_3DGUT
}

at::Tensor intersect_offset(
    const at::Tensor &isect_ids, // [n_isects]
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    auto opt = isect_ids.options();
    at::Tensor offsets = at::empty(
        {I, tile_height, tile_width}, opt.dtype(at::kInt)
    );
    launch_intersect_offset_kernel(
        isect_ids, I, tile_width, tile_height, offsets
    );
    return offsets;
}

std::tuple<at::Tensor, at::Tensor> intersect_tile_sparse(
    const at::Tensor &means2d,                 // [I, N, 2] or [nnz, 2]
    const at::Tensor &radii,                   // [I, N, 2] or [nnz, 2]
    const at::Tensor &depths,                  // [I, N] or [nnz]
    const at::optional<at::Tensor> &image_ids, // [nnz] (packed mode)
    const at::Tensor &tile_mask,    // [I, tile_height, tile_width] bool
    const at::Tensor &active_tiles, // [num_active_tiles] int32, ascending
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    CHECK_INPUT(tile_mask);
    CHECK_INPUT(active_tiles);
    TORCH_CHECK(tile_mask.scalar_type() == at::kBool, "tile_mask must be bool");
    TORCH_CHECK(
        active_tiles.scalar_type() == at::kInt, "active_tiles must be int32"
    );

    auto opt = depths.options();
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            image_ids.has_value(),
            "image_ids is required when means2d is packed ([nnz, 2])."
        );
        CHECK_INPUT(image_ids.value());
    }

    const uint32_t n_tiles = tile_width * tile_height;
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits = bits_for_count(n_tiles);
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile_sparse: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=",
        I,
        ", n_tiles=",
        n_tiles,
        ")."
    );

    // The enumeration kernel reads image_ids as int64; callers may pass int32
    // (e.g. fvdb camera ids), so widen once here (packed mode only).
    at::optional<at::Tensor> image_ids_i64;
    if (packed) {
        image_ids_i64 = image_ids.value().to(at::kLong).contiguous();
    }

    const int64_t n_elements = means2d.numel() / 2;

    // First pass: per-gaussian count, restricted to active tiles by tile_mask.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects = 0;
    at::Tensor cum_tiles_per_gauss;
    if (n_elements) {
        launch_intersect_tile_kernel(
            means2d,
            radii,
            depths,
            c10::nullopt, // conics  -> AABB path
            c10::nullopt, // opacities
            packed ? image_ids_i64 : c10::nullopt,
            c10::nullopt, // gaussian_ids (unused by the kernel)
            I,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt, // flatten_ids
            tile_mask
        );
        cum_tiles_per_gauss =
            at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    }

    // Second pass: emit (isect_id, flatten_id) for the active-tile
    // intersections.
    at::Tensor isect_ids = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if (n_isects) {
        launch_intersect_tile_kernel(
            means2d,
            radii,
            depths,
            c10::nullopt,
            c10::nullopt,
            packed ? image_ids_i64 : c10::nullopt,
            c10::nullopt,
            I,
            tile_size,
            tile_width,
            tile_height,
            cum_tiles_per_gauss,
            c10::nullopt, // tiles_per_gauss
            at::optional<at::Tensor>(isect_ids),
            at::optional<at::Tensor>(flatten_ids),
            tile_mask
        );
    }

    // Sort by (image, tile, depth); reuse the dense radix sort.
    at::Tensor isect_ids_sorted = isect_ids;
    at::Tensor flatten_ids_sorted = flatten_ids;
    if (n_isects) {
        isect_ids_sorted = at::empty_like(isect_ids);
        flatten_ids_sorted = at::empty_like(flatten_ids);
        radix_sort_double_buffer(
            n_isects,
            image_n_bits,
            tile_n_bits,
            isect_ids,
            flatten_ids,
            isect_ids_sorted,
            flatten_ids_sorted
        );
    }

    // Compacted per-active-tile offsets ([num_active_tiles + 1]).
    const int64_t num_active_tiles = active_tiles.size(0);
    at::Tensor tile_offsets = at::empty(
        {num_active_tiles + 1}, active_tiles.options().dtype(at::kInt)
    );
    launch_intersect_offset_sparse_kernel(
        isect_ids_sorted, active_tiles, tile_width, tile_height, tile_offsets
    );

    return std::make_tuple(tile_offsets, flatten_ids_sorted);
}

void register_intersect_cuda_impl(torch::Library &m) {
    m.impl("intersect_tile", &intersect_tile);
    m.impl("intersect_tile_lidar", &intersect_tile_lidar);
    m.impl("intersect_offset", &intersect_offset);
    m.impl("intersect_tile_sparse", &intersect_tile_sparse);
}

} // namespace gsplat
