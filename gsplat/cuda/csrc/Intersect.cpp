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
#include "TorchUtils.h" // to_torch_op, TorchArgDef

namespace gsplat {

namespace {

// Validates intersect_tile inputs. Each checked assumption is a precondition
// of the tile-intersection kernel.
void check_intersect_tile_inputs(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::optional<at::Tensor> &conics,
    const at::optional<at::Tensor> &opacities,
    const at::optional<at::Tensor> &image_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    bool packed
) {
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (conics.has_value()) { CHECK_INPUT(conics.value()); }
    if (opacities.has_value()) { CHECK_INPUT(opacities.value()); }

    if (packed) {
        int64_t nnz = means2d.size(0);
        TORCH_CHECK(means2d.size(1) == 2,
                    "means2d must be [nnz, 2], got ", means2d.sizes());
        TORCH_CHECK(radii.dim() == 2 && radii.size(0) == nnz && radii.size(1) == 2,
                    "radii must be [nnz, 2], got ", radii.sizes());
        TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz,
                    "depths must be [nnz], got ", depths.sizes());
        if (conics.has_value()) {
            TORCH_CHECK(conics->dim() == 2 && conics->size(0) == nnz && conics->size(1) == 3,
                        "conics must be [nnz, 3], got ", conics->sizes());
        }
        if (opacities.has_value()) {
            TORCH_CHECK(opacities->dim() == 1 && opacities->size(0) == nnz,
                        "opacities must be [nnz], got ", opacities->sizes());
        }
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    } else {
        TORCH_CHECK(means2d.dim() >= 2 && means2d.size(-1) == 2,
                    "means2d must be [..., N, 2], got ", means2d.sizes());
        auto lead = means2d.sizes().slice(0, means2d.dim() - 1);
        TORCH_CHECK(radii.sizes() == means2d.sizes(),
                    "radii must be [..., N, 2] matching means2d, got ", radii.sizes());
        TORCH_CHECK(depths.sizes() == lead,
                    "depths must be [..., N], got ", depths.sizes());
        if (conics.has_value()) {
            TORCH_CHECK(conics->dim() == means2d.dim() && conics->size(-1) == 3 &&
                            conics->sizes().slice(0, conics->dim() - 1) == lead,
                        "conics must be [..., N, 3], got ", conics->sizes());
        }
        if (opacities.has_value()) {
            TORCH_CHECK(opacities->sizes() == lead,
                        "opacities must be [..., N], got ", opacities->sizes());
        }
    }
}

#if GSPLAT_BUILD_3DGUT
// Validates intersect_tile_lidar inputs. Each checked assumption is a
// precondition of the lidar tile-intersection kernel.
void check_intersect_tile_lidar_inputs(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::optional<at::Tensor> &image_ids,
    const at::optional<at::Tensor> &gaussian_ids,
    bool packed
) {
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    if (packed) {
        int64_t nnz = means2d.size(0);
        TORCH_CHECK(means2d.size(1) == 2,
                    "means2d must be [nnz, 2], got ", means2d.sizes());
        TORCH_CHECK(radii.dim() == 2 && radii.size(0) == nnz && radii.size(1) == 2,
                    "radii must be [nnz, 2], got ", radii.sizes());
        TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz,
                    "depths must be [nnz], got ", depths.sizes());
        TORCH_CHECK(
            image_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, image_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(image_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    } else {
        TORCH_CHECK(means2d.dim() >= 2 && means2d.size(-1) == 2,
                    "means2d must be [..., N, 2], got ", means2d.sizes());
        auto lead = means2d.sizes().slice(0, means2d.dim() - 1);
        TORCH_CHECK(radii.sizes() == means2d.sizes(),
                    "radii must be [..., N, 2] matching means2d, got ", radii.sizes());
        TORCH_CHECK(depths.sizes() == lead,
                    "depths must be [..., N], got ", depths.sizes());
    }
}
#endif

} // namespace

TileIntersectResult intersect_tile(
    const at::Tensor &means2d,                           // [..., N, 2] or [nnz, 2]
    const at::Tensor &radii,                             // [..., N, 2] or [nnz, 2]
    const at::Tensor &depths,                            // [..., N] or [nnz]
    const at::optional<at::Tensor> &conics,              // [..., N, 3] or [nnz, 3] 
    const at::optional<at::Tensor> &opacities,           // [..., N] or [nnz]        
    const at::optional<at::Tensor> &image_ids,           // [nnz]
    const at::optional<at::Tensor> &gaussian_ids,        // [nnz]
    std::optional<int64_t> n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
) {
    DEVICE_GUARD(means2d);

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    check_intersect_tile_inputs(
        means2d, radii, depths, conics, opacities, image_ids, gaussian_ids,
        packed
    );

    // Flattened image count. For the non-packed [..., N, 2] layout it is the
    // product of the leading image dims; the packed [nnz, 2] layout flattens
    // those dims away, so the caller must supply it.
    int64_t I;
    if (packed) {
        TORCH_CHECK(n_images.has_value(),
                    "n_images is required when means2d is packed ([nnz, 2]).");
        I = n_images.value();
    } else {
        I = c10::multiply_integers(means2d.sizes().slice(0, means2d.dim() - 2));
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
        return {.tiles_per_gauss = tiles_per_gauss,
                .isect_ids = isect_ids_sorted, .flatten_ids = flatten_ids_sorted};
    } else {
        return {.tiles_per_gauss = tiles_per_gauss,
                .isect_ids = isect_ids, .flatten_ids = flatten_ids};
    }
}

TileIntersectResult intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    std::optional<int64_t> n_images,
    const bool sort,
    const bool segmented
) {
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(false, "intersect_tile_lidar requires GSPLAT_BUILD_3DGUT=1");
    return {};
#else
    DEVICE_GUARD(means2d);

    auto opt = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    check_intersect_tile_lidar_inputs(
        means2d, radii, depths, image_ids, gaussian_ids, packed
    );

    // Flattened image count. For the non-packed [..., N, 2] layout it is the
    // product of the leading image dims; the packed [nnz, 2] layout flattens
    // those dims away, so the caller must supply it.
    int64_t I;
    if (packed) {
        TORCH_CHECK(n_images.has_value(),
                    "n_images is required when means2d is packed ([nnz, 2]).");
        I = n_images.value();
    } else {
        I = c10::multiply_integers(means2d.sizes().slice(0, means2d.dim() - 2));
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
        return {.tiles_per_gauss = tiles_per_gauss,
                .isect_ids = isect_ids_sorted, .flatten_ids = flatten_ids_sorted};
    } else {
        return {.tiles_per_gauss = tiles_per_gauss,
                .isect_ids = isect_ids, .flatten_ids = flatten_ids};
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
    // (e.g. external camera ids), so widen once here (packed mode only).
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

static int64_t num_words_per_tile_bitmask(int64_t tile_size) {
    return (tile_size * tile_size + 63) / 64;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
build_sparse_tile_layout(
    const at::Tensor &pixels,    // [P, 2] int, (row, col)
    const at::Tensor &image_ids, // [P] int
    int64_t n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
) {
    DEVICE_GUARD(pixels);
    CHECK_INPUT(pixels);
    CHECK_INPUT(image_ids);
    TORCH_CHECK(pixels.dim() == 2 && pixels.size(1) == 2, "pixels must be [P, 2]");

    const int64_t P = pixels.size(0);
    const int64_t n_tiles = tile_width * tile_height;
    const int64_t words = num_words_per_tile_bitmask(tile_size);
    const auto dev = pixels.device();
    const auto i64 = at::TensorOptions().device(dev).dtype(at::kLong);
    const auto i32 = at::TensorOptions().device(dev).dtype(at::kInt);
    const auto bln = at::TensorOptions().device(dev).dtype(at::kBool);

    if (P == 0 || n_images == 0) {
        // Degenerate shapes (note tile_pixel_cumsum is [1] here).
        return std::make_tuple(
            at::empty({0}, i32),
            at::zeros({n_images, tile_height, tile_width}, bln),
            at::empty({0, words}, i64),
            at::zeros({1}, i64),
            at::empty({0}, i64)
        );
    }

    // (image, tile) ids are packed into the high 32 bits of the int64 sort key,
    // so the dense tile id must fit below 2^31 to keep the key positive.
    TORCH_CHECK(
        n_images * n_tiles < (int64_t(1) << 31),
        "build_sparse_tile_layout: n_images * n_tiles (",
        n_images * n_tiles,
        ") must be < 2^31 to pack the tile id into the sort key."
    );

    const auto pix = pixels.to(at::kLong);
    const auto rows = pix.select(1, 0);
    const auto cols = pix.select(1, 1);
    const auto img = image_ids.to(at::kLong).reshape({P});

    // Dense tile id and raster-order position within the tile, per pixel.
    const auto tile_id = img.mul(n_tiles)
                             .add(rows.floor_divide(tile_size).mul(tile_width))
                             .add(cols.floor_divide(tile_size));
    const auto pix_in_tile = rows.remainder(tile_size)
                                 .mul(tile_size)
                                 .add(cols.remainder(tile_size));

    // Per-tile activity mask over the dense [n_images, TH, TW] grid.
    auto active_tile_mask = at::zeros({n_images * n_tiles}, bln);
    active_tile_mask.index_fill_(0, tile_id, 1);
    active_tile_mask = active_tile_mask.view({n_images, tile_height, tile_width});

    // Sort pixels by (tile_id, in-tile position). pixel_map is the argsort; keys
    // are unique under the no-duplicate precondition, so the order is stable.
    const auto key = tile_id.bitwise_left_shift(32).bitwise_or(pix_in_tile);
    const auto sorted = at::sort(
        key, /*stable=*/c10::optional<bool>(true), /*dim=*/-1, /*descending=*/false
    );
    const auto sorted_key = std::get<0>(sorted);
    const auto pixel_map = std::get<1>(sorted);
    const auto tile_id_sorted = sorted_key.bitwise_right_shift(32);
    const auto pix_in_tile_sorted =
        pix_in_tile.index_select(0, pixel_map).contiguous();

    // Run-length encode the sorted tile ids -> active tiles + per-tile counts.
    const auto uc = at::unique_consecutive(
        tile_id_sorted, /*return_inverse=*/false, /*return_counts=*/true
    );
    const auto active_tiles = std::get<0>(uc).to(at::kInt);
    const auto counts = std::get<2>(uc);
    const auto tile_pixel_cumsum = counts.cumsum(0); // inclusive

    // Per-active-tile raster-order bitmask.
    const int64_t AT = active_tiles.size(0);
    auto tile_pixel_mask = at::zeros({AT, words}, i64);
    launch_build_tile_bitmask_kernel(
        pix_in_tile_sorted,
        tile_pixel_cumsum,
        static_cast<uint32_t>(words),
        tile_pixel_mask
    );

    return std::make_tuple(
        active_tiles,
        active_tile_mask,
        tile_pixel_mask,
        tile_pixel_cumsum,
        pixel_map
    );
}

void register_intersect_cuda_impl(torch::Library &m) {
    m.impl("intersect_tile", to_torch_op<&intersect_tile>);
    m.impl("intersect_tile_lidar", to_torch_op<&intersect_tile_lidar>);
    m.impl("intersect_offset", to_torch_op<&intersect_offset>);
    m.impl("intersect_tile_sparse", &intersect_tile_sparse);
    m.impl("build_sparse_tile_layout", &build_sparse_tile_layout);
}

} // namespace gsplat
