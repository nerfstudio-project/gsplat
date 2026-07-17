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
#include <limits>
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "MathUtils.h"
#include "Common.h" // where all the macros are defined
#include "GSplatBuildConfig.h"
#include "Intersect.h" // where the launch function is declared
#include "IntersectMTConfig.h"
#include "IntersectMacroTile.h"
#include "TorchUtils.h" // to_torch_op, TorchArgDef

namespace gsplat
{
namespace
{
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
    )
    {
        CHECK_INPUT(means2d);
        CHECK_INPUT(radii);
        CHECK_INPUT(depths);
        if(conics.has_value())
        {
            CHECK_INPUT(conics.value());
        }
        if(opacities.has_value())
        {
            CHECK_INPUT(opacities.value());
        }

        if(packed)
        {
            int64_t nnz = means2d.size(0);
            TORCH_CHECK(means2d.size(1) == 2, "means2d must be [nnz, 2], got ", means2d.sizes());
            TORCH_CHECK(
                radii.dim() == 2 && radii.size(0) == nnz && radii.size(1) == 2,
                "radii must be [nnz, 2], got ",
                radii.sizes()
            );
            TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz, "depths must be [nnz], got ", depths.sizes());
            if(conics.has_value())
            {
                TORCH_CHECK(
                    conics->dim() == 2 && conics->size(0) == nnz && conics->size(1) == 3,
                    "conics must be [nnz, 3], got ",
                    conics->sizes()
                );
            }
            if(opacities.has_value())
            {
                TORCH_CHECK(
                    opacities->dim() == 1 && opacities->size(0) == nnz,
                    "opacities must be [nnz], got ",
                    opacities->sizes()
                );
            }
            TORCH_CHECK(
                image_ids.has_value() && gaussian_ids.has_value(),
                "When packed is set, image_ids and gaussian_ids must be provided."
            );
            CHECK_INPUT(image_ids.value());
            CHECK_INPUT(gaussian_ids.value());
        }
        else
        {
            TORCH_CHECK(
                means2d.dim() >= 2 && means2d.size(-1) == 2, "means2d must be [..., N, 2], got ", means2d.sizes()
            );
            auto lead = means2d.sizes().slice(0, means2d.dim() - 1);
            TORCH_CHECK(
                radii.sizes() == means2d.sizes(), "radii must be [..., N, 2] matching means2d, got ", radii.sizes()
            );
            TORCH_CHECK(depths.sizes() == lead, "depths must be [..., N], got ", depths.sizes());
            if(conics.has_value())
            {
                TORCH_CHECK(
                    conics->dim() == means2d.dim()
                        && conics->size(-1) == 3
                        && conics->sizes().slice(0, conics->dim() - 1) == lead,
                    "conics must be [..., N, 3], got ",
                    conics->sizes()
                );
            }
            if(opacities.has_value())
            {
                TORCH_CHECK(opacities->sizes() == lead, "opacities must be [..., N], got ", opacities->sizes());
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
    )
    {
        CHECK_INPUT(means2d);
        CHECK_INPUT(radii);
        CHECK_INPUT(depths);
        if(packed)
        {
            int64_t nnz = means2d.size(0);
            TORCH_CHECK(means2d.size(1) == 2, "means2d must be [nnz, 2], got ", means2d.sizes());
            TORCH_CHECK(
                radii.dim() == 2 && radii.size(0) == nnz && radii.size(1) == 2,
                "radii must be [nnz, 2], got ",
                radii.sizes()
            );
            TORCH_CHECK(depths.dim() == 1 && depths.size(0) == nnz, "depths must be [nnz], got ", depths.sizes());
            TORCH_CHECK(
                image_ids.has_value() && gaussian_ids.has_value(),
                "When packed is set, image_ids and gaussian_ids must be provided."
            );
            CHECK_INPUT(image_ids.value());
            CHECK_INPUT(gaussian_ids.value());
        }
        else
        {
            TORCH_CHECK(
                means2d.dim() >= 2 && means2d.size(-1) == 2, "means2d must be [..., N, 2], got ", means2d.sizes()
            );
            auto lead = means2d.sizes().slice(0, means2d.dim() - 1);
            TORCH_CHECK(
                radii.sizes() == means2d.sizes(), "radii must be [..., N, 2] matching means2d, got ", radii.sizes()
            );
            TORCH_CHECK(depths.sizes() == lead, "depths must be [..., N], got ", depths.sizes());
        }
    }
#endif
} // namespace

TileIntersectResult intersect_tile(
    const at::Tensor &means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor &radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor &depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> &conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> &opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> &image_ids,    // [nnz]
    const at::optional<at::Tensor> &gaussian_ids, // [nnz]
    const std::optional<int64_t> &n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
)
{
    DEVICE_GUARD(means2d);

    auto opt            = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed         = means2d.dim() == 2;
    check_intersect_tile_inputs(means2d, radii, depths, conics, opacities, image_ids, gaussian_ids, packed);

    // Flattened image count. For the non-packed [..., N, 2] layout it is the
    // product of the leading image dims; the packed [nnz, 2] layout flattens
    // those dims away, so the caller must supply it.
    int64_t I;
    if(packed)
    {
        TORCH_CHECK(n_images.has_value(), "n_images is required when means2d is packed ([nnz, 2]).");
        I = n_images.value();
    }
    else
    {
        I = c10::multiply_integers(means2d.sizes().slice(0, means2d.dim() - 2));
    }

    // packed-mode `offsets` (below) reduce a 1-D [nnz] tiles_per_gauss and
    // collapse to a single [0, total] segment, so a per-image segmented sort
    // would index past the 2-entry offsets buffer. Reject until packed offsets
    // are built per image.
    TORCH_CHECK(!(packed && segmented), "segmented sort is not supported for packed inputs");

    uint32_t n_tiles            = tile_width * tile_height;
    // the number of bits needed to encode the image id and tile id
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=",
        I,
        ", n_tiles=",
        n_tiles,
        ")."
    );

    // first pass: compute number of tiles per gaussian
    // TODO: This first pass can be avoided, see todo comment in intersect_tile_lidar_kernel.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;
    if(n_elements)
    {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics,    // at::optional, AccuTile when provided, AABB fallback otherwise
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
        n_isects            = cum_tiles_per_gauss[-1].item<int64_t>();
        if(segmented)
        {
            // offsets in the isect_ids and flatten_ids
            offsets = at::cumsum(at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong);
            offsets = at::cat({at::tensor({0}, opt.dtype(at::kLong)), offsets});
        }
    }
    else
    {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if(n_isects)
    {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            conics,    // at::optional, AccuTile when provided, AABB fallback otherwise
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
    if(n_isects && sort)
    {
        at::Tensor isect_ids_sorted   = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        if(segmented)
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
                n_isects, image_n_bits, tile_n_bits, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted
            );
        }
        return {.tiles_per_gauss = tiles_per_gauss, .isect_ids = isect_ids_sorted, .flatten_ids = flatten_ids_sorted};
    }
    else
    {
        return {.tiles_per_gauss = tiles_per_gauss, .isect_ids = isect_ids, .flatten_ids = flatten_ids};
    }
}

TileIntersectResult intersect_tile_privateuseone(
    const at::Tensor &means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor &radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor &depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> &conics,       // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> &opacities,    // [..., N] or [nnz]
    const at::optional<at::Tensor> &image_ids,    // [nnz]
    const at::optional<at::Tensor> &gaussian_ids, // [nnz]
    const std::optional<int64_t> &n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height,
    bool sort,
    bool segmented
)
{
    DEVICE_GUARD(means2d);
    bool packed = means2d.dim() == 2;
    check_intersect_tile_inputs(means2d, radii, depths, conics, opacities, image_ids, gaussian_ids, packed);
    if(packed)
    {
        TORCH_CHECK(n_images.has_value(), "n_images is required when means2d is packed ([nnz, 2]).");
    }
    TORCH_CHECK(!segmented, "segmented mGPU intersect_tile is not supported");

    int64_t I = packed ? n_images.value() : c10::multiply_integers(means2d.sizes().slice(0, means2d.dim() - 2));

    uint32_t n_tiles            = tile_width * tile_height;
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=",
        I,
        ", n_tiles=",
        n_tiles,
        ")."
    );

    return intersect_tile_kernels_privateuseone(
        means2d,
        radii,
        depths,
        conics,
        opacities,
        packed ? image_ids : c10::nullopt,
        packed ? gaussian_ids : c10::nullopt,
        I,
        tile_size,
        tile_width,
        tile_height,
        sort
    );
}

TileIntersectResult intersect_tile_lidar(
    const c10::intrusive_ptr<gsplat::RowOffsetStructuredSpinningLidarModelParametersExt> &lidar,
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const std::optional<int64_t> &n_images,
    const bool sort,
    const bool segmented
)
{
#if !GSPLAT_BUILD_3DGUT
    TORCH_CHECK(false, "intersect_tile_lidar requires GSPLAT_BUILD_3DGUT=1");
    return {};
#else
    DEVICE_GUARD(means2d);

    auto opt            = depths.options();
    uint32_t n_elements = means2d.numel() / 2;
    bool packed         = means2d.dim() == 2;
    check_intersect_tile_lidar_inputs(means2d, radii, depths, image_ids, gaussian_ids, packed);

    // Flattened image count. For the non-packed [..., N, 2] layout it is the
    // product of the leading image dims; the packed [nnz, 2] layout flattens
    // those dims away, so the caller must supply it.
    int64_t I;
    if(packed)
    {
        TORCH_CHECK(n_images.has_value(), "n_images is required when means2d is packed ([nnz, 2]).");
        I = n_images.value();
    }
    else
    {
        I = c10::multiply_integers(means2d.sizes().slice(0, means2d.dim() - 2));
    }

    // packed-mode `offsets` (below) reduce a 1-D [nnz] tiles_per_gauss and
    // collapse to a single [0, total] segment, so a per-image segmented sort
    // would index past the 2-entry offsets buffer. Reject until packed offsets
    // are built per image.
    TORCH_CHECK(!(packed && segmented), "segmented sort is not supported for packed inputs");

    uint32_t n_tiles            = lidar->n_bins_azimuth * lidar->n_bins_elevation;
    // the number of bits needed to encode the image id and tile id
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
    // the first 32 bits are used for the image id and tile id altogether, so
    // check if we have enough bits for them.
    TORCH_CHECK(
        image_n_bits + tile_n_bits <= 32,
        "intersect_tile_lidar: (image, tile) id packing needs ",
        image_n_bits + tile_n_bits,
        " bits but only 32 are available (I=",
        I,
        ", n_tiles=",
        n_tiles,
        ")."
    );

    // first pass: compute number of tiles per gaussian
    // TODO: This first pass can be avoided, see todo comment in intersect_tile_lidar_kernel.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    at::Tensor offsets;
    if(n_elements)
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
        n_isects            = cum_tiles_per_gauss[-1].item<int64_t>();
        if(segmented)
        {
            // offsets in the isect_ids and flatten_ids
            offsets = at::cumsum(at::sum(tiles_per_gauss, -1).view({-1}), 0, at::kLong);
            offsets = at::cat({at::tensor({0}, opt.dtype(at::kLong)), offsets});
        }
    }
    else
    {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if(n_isects)
    {
        TORCH_CHECK(
            cum_tiles_per_gauss.scalar_type() == at::ScalarType::Long,
            "cum_tiles_per_gauss must be int64 for lidar kernel"
        );
        TORCH_CHECK(cum_tiles_per_gauss.is_contiguous(), "cum_tiles_per_gauss must be contiguous");

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
    if(n_isects && sort)
    {
        at::Tensor isect_ids_sorted   = at::empty_like(isect_ids);
        at::Tensor flatten_ids_sorted = at::empty_like(flatten_ids);
        if(segmented)
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
                n_isects, image_n_bits, tile_n_bits, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted
            );
        }
        return {.tiles_per_gauss = tiles_per_gauss, .isect_ids = isect_ids_sorted, .flatten_ids = flatten_ids_sorted};
    }
    else
    {
        return {.tiles_per_gauss = tiles_per_gauss, .isect_ids = isect_ids, .flatten_ids = flatten_ids};
    }
#endif // !GSPLAT_BUILD_3DGUT
}

at::Tensor intersect_offset(
    const at::Tensor &isect_ids, // [n_isects]
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
)
{
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    auto opt           = isect_ids.options();
    at::Tensor offsets = at::empty({I, tile_height, tile_width}, opt.dtype(at::kInt));
    launch_intersect_offset_kernel(isect_ids, I, tile_width, tile_height, offsets);
    return offsets;
}

std::tuple<at::Tensor, at::Tensor> intersect_tile_sparse(
    const at::Tensor &means2d,                 // [I, N, 2] or [nnz, 2]
    const at::Tensor &radii,                   // [I, N, 2] or [nnz, 2]
    const at::Tensor &depths,                  // [I, N] or [nnz]
    const at::optional<at::Tensor> &image_ids, // [nnz] (packed mode)
    const at::Tensor &tile_mask,               // [I, tile_height, tile_width] bool
    const at::Tensor &active_tiles,            // [num_active_tiles] int32, ascending
    int64_t I,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
)
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    CHECK_INPUT(tile_mask);
    CHECK_INPUT(active_tiles);
    TORCH_CHECK(tile_mask.scalar_type() == at::kBool, "tile_mask must be bool");
    TORCH_CHECK(active_tiles.scalar_type() == at::kInt, "active_tiles must be int32");

    auto opt    = depths.options();
    bool packed = means2d.dim() == 2;
    if(packed)
    {
        TORCH_CHECK(image_ids.has_value(), "image_ids is required when means2d is packed ([nnz, 2]).");
        CHECK_INPUT(image_ids.value());
    }

    const uint32_t n_tiles      = tile_width * tile_height;
    const uint32_t image_n_bits = bits_for_count(I);
    const uint32_t tile_n_bits  = bits_for_count(n_tiles);
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
    if(packed)
    {
        image_ids_i64 = image_ids.value().to(at::kLong).contiguous();
    }

    const int64_t n_elements = means2d.numel() / 2;

    // First pass: per-gaussian count, restricted to active tiles by tile_mask.
    at::Tensor tiles_per_gauss = at::empty_like(depths, opt.dtype(at::kInt));
    int64_t n_isects           = 0;
    at::Tensor cum_tiles_per_gauss;
    if(n_elements)
    {
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
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0, at::kLong);
        n_isects            = cum_tiles_per_gauss[-1].item<int64_t>();
    }

    // Second pass: emit (isect_id, flatten_id) for the active-tile
    // intersections.
    at::Tensor isect_ids   = at::empty({n_isects}, opt.dtype(at::kLong));
    at::Tensor flatten_ids = at::empty({n_isects}, opt.dtype(at::kInt));
    if(n_isects)
    {
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
    at::Tensor isect_ids_sorted   = isect_ids;
    at::Tensor flatten_ids_sorted = flatten_ids;
    if(n_isects)
    {
        isect_ids_sorted   = at::empty_like(isect_ids);
        flatten_ids_sorted = at::empty_like(flatten_ids);
        radix_sort_double_buffer(
            n_isects, image_n_bits, tile_n_bits, isect_ids, flatten_ids, isect_ids_sorted, flatten_ids_sorted
        );
    }

    // Compacted per-active-tile offsets ([num_active_tiles + 1]).
    const int64_t num_active_tiles = active_tiles.size(0);
    at::Tensor tile_offsets        = at::empty({num_active_tiles + 1}, active_tiles.options().dtype(at::kInt));
    launch_intersect_offset_sparse_kernel(isect_ids_sorted, active_tiles, tile_width, tile_height, tile_offsets);

    return std::make_tuple(tile_offsets, flatten_ids_sorted);
}

std::tuple<at::Tensor, at::Tensor> intersect_tile_macro_binning(
    const at::Tensor &means2d,
    const at::Tensor &radii,
    const at::Tensor &depths,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    int64_t tile_size,
    int64_t tile_cols,
    int64_t tile_rows
)
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);

    TORCH_CHECK(means2d.dim() == 3 && means2d.size(2) == 2, "means2d must have shape [I, N, 2], got ", means2d.sizes());
    TORCH_CHECK(means2d.scalar_type() == at::kFloat, "means2d must be float32, got ", means2d.scalar_type());
    const int64_t I = means2d.size(0);
    const int64_t N = means2d.size(1);
    TORCH_CHECK(N <= std::numeric_limits<int32_t>::max(), "N must fit in int32, got ", N);
    TORCH_CHECK(
        radii.dim() == 3
            && radii.size(0) == I
            && radii.size(1) == N
            && radii.size(2) == 2
            && radii.scalar_type() == at::kInt,
        "radii must be int32 [I, N, 2], got ",
        radii.sizes(),
        " ",
        radii.scalar_type()
    );
    TORCH_CHECK(
        depths.dim() == 2 && depths.size(0) == I && depths.size(1) == N && depths.scalar_type() == at::kFloat,
        "depths must be float32 [I, N], got ",
        depths.sizes(),
        " ",
        depths.scalar_type()
    );
    TORCH_CHECK(
        conics.dim() == 3
            && conics.size(0) == I
            && conics.size(1) == N
            && conics.size(2) == 3
            && conics.scalar_type() == at::kFloat,
        "conics must be float32 [I, N, 3], got ",
        conics.sizes(),
        " ",
        conics.scalar_type()
    );
    TORCH_CHECK(
        opacities.dim() == 2
            && opacities.size(0) == I
            && opacities.size(1) == N
            && opacities.scalar_type() == at::kFloat,
        "opacities must be float32 [I, N], got ",
        opacities.sizes(),
        " ",
        opacities.scalar_type()
    );
    TORCH_CHECK(tile_size == 8 || tile_size == 16, "tile_size must be 8 or 16, got ", tile_size);

    const auto opts_i32      = means2d.options().dtype(at::kInt);
    at::Tensor isect_offsets = at::zeros({I, tile_rows, tile_cols}, opts_i32);
    at::Tensor flatten_ids   = at::empty({0}, opts_i32);

    const int64_t n_tiles = tile_rows * tile_cols;
    if(I == 0 || N == 0 || n_tiles == 0)
    {
        return std::make_tuple(isect_offsets, flatten_ids);
    }

    const int32_t macro_tile_cols = static_cast<int32_t>((tile_cols + MACRO_TILE_WIDTH - 1) / MACRO_TILE_WIDTH);
    const int32_t macro_tile_rows = static_cast<int32_t>((tile_rows + MACRO_TILE_HEIGHT - 1) / MACRO_TILE_HEIGHT);
    const int32_t n_macro_tiles   = macro_tile_rows * macro_tile_cols;
    TORCH_CHECK(
        I * static_cast<int64_t>(N) <= std::numeric_limits<int32_t>::max(),
        "macro-tile flatten ids must fit in int32, got I * N = ",
        I * static_cast<int64_t>(N)
    );
    TORCH_CHECK(
        I * static_cast<int64_t>(n_macro_tiles) <= std::numeric_limits<int32_t>::max(),
        "macro-tile segment count must fit in int32, got I * n_macro_tiles = ",
        I * static_cast<int64_t>(n_macro_tiles)
    );
    TORCH_CHECK(
        I * n_tiles <= std::numeric_limits<int32_t>::max(),
        "render-tile count must fit in int32, got I * n_tiles = ",
        I * n_tiles
    );
    const int32_t total_macro_tiles = static_cast<int32_t>(I) * n_macro_tiles;

    // This stage implements the 3DGS EWA path. 3DGUT will use the AABB mode
    // once the macro-tile path is extended there.
    const IntersectionType intersection_type = IntersectionType::ELLIPSE;

    at::Tensor mt_gauss_counts = at::zeros({I, n_macro_tiles}, opts_i32);
    launch_mt_binning_count(
        intersection_type,
        means2d,
        radii,
        conics,
        opacities,
        static_cast<int32_t>(tile_size),
        macro_tile_cols,
        macro_tile_rows,
        mt_gauss_counts
    );

    at::Tensor mt_gauss_offsets = at::zeros({total_macro_tiles + 1}, opts_i32);
    {
        at::Tensor mt_gauss_offsets_tail = mt_gauss_offsets.slice(0, 1, total_macro_tiles + 1);
        at::cumsum_out(mt_gauss_offsets_tail, mt_gauss_counts.view({total_macro_tiles}), 0, at::kInt);
    }
    const int64_t n_macro_isects = mt_gauss_offsets[total_macro_tiles].item<int64_t>();
    if(n_macro_isects == 0)
    {
        return std::make_tuple(isect_offsets, flatten_ids);
    }

    at::Tensor mt_gauss_write_cursor = at::zeros({I, n_macro_tiles}, opts_i32);
    at::Tensor mt_depth_keys         = at::empty({n_macro_isects}, opts_i32);
    at::Tensor mt_gauss_ids          = at::empty({n_macro_isects}, opts_i32);
    launch_mt_binning_fill(
        intersection_type,
        means2d,
        radii,
        depths,
        conics,
        opacities,
        static_cast<int32_t>(tile_size),
        macro_tile_cols,
        macro_tile_rows,
        mt_gauss_offsets,
        mt_gauss_write_cursor,
        mt_depth_keys,
        mt_gauss_ids
    );

    at::Tensor mt_tmp_depth_keys   = at::empty({n_macro_isects}, opts_i32);
    at::Tensor mt_tmp_gauss_ids    = at::empty({n_macro_isects}, opts_i32);
    at::Tensor mt_gauss_ids_sorted = launch_mt_segmented_sort(
        n_macro_isects,
        total_macro_tiles,
        mt_gauss_offsets,
        mt_depth_keys,
        mt_gauss_ids,
        mt_tmp_depth_keys,
        mt_tmp_gauss_ids
    );

    at::Tensor mt_gauss_batch_offsets = at::zeros({total_macro_tiles + 1}, opts_i32);
    launch_mt_gauss_batch_offsets(mt_gauss_offsets, mt_gauss_batch_offsets, total_macro_tiles);
    const int32_t total_gauss_batches = mt_gauss_batch_offsets[total_macro_tiles].item<int32_t>();
    if(total_gauss_batches == 0)
    {
        return std::make_tuple(isect_offsets, flatten_ids);
    }

    const int64_t n_batch_rt_slots = static_cast<int64_t>(total_gauss_batches) * (MACRO_TILE_WIDTH * MACRO_TILE_HEIGHT);
    const int64_t n_bitmask_words  = n_batch_rt_slots * RT_MASKS_PER_GAUSS_BATCH;
    at::Tensor rt_bitmasks         = at::empty({n_bitmask_words}, opts_i32);
    launch_rt_gauss_batch_intersection_masks(
        intersection_type,
        total_gauss_batches,
        total_macro_tiles,
        n_macro_tiles,
        macro_tile_cols,
        static_cast<int32_t>(tile_size),
        means2d,
        conics,
        opacities,
        mt_gauss_offsets,
        mt_gauss_ids_sorted,
        mt_gauss_batch_offsets,
        rt_bitmasks
    );

    at::Tensor rt_batch_gauss_offsets = at::empty({n_batch_rt_slots}, opts_i32);
    at::Tensor rt_gauss_counts        = at::zeros({I * n_tiles}, opts_i32);
    launch_rt_count_gauss_and_prefix_offsets(
        total_macro_tiles,
        n_macro_tiles,
        macro_tile_cols,
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tile_cols),
        static_cast<int32_t>(tile_rows),
        mt_gauss_batch_offsets,
        rt_bitmasks,
        rt_batch_gauss_offsets,
        rt_gauss_counts
    );

    at::Tensor rt_gauss_offsets_full = at::zeros({I * n_tiles + 1}, opts_i32);
    {
        at::Tensor rt_gauss_offsets_tail = rt_gauss_offsets_full.slice(0, 1, I * n_tiles + 1);
        at::cumsum_out(rt_gauss_offsets_tail, rt_gauss_counts, 0, at::kInt);
    }
    const int64_t n_rt_isects = rt_gauss_offsets_full[I * n_tiles].item<int64_t>();

    isect_offsets = rt_gauss_offsets_full.slice(0, 0, I * n_tiles).view({I, tile_rows, tile_cols}).contiguous();
    if(n_rt_isects == 0)
    {
        return std::make_tuple(isect_offsets, flatten_ids);
    }

    flatten_ids = at::empty({n_rt_isects}, opts_i32);
    launch_rt_binning_fill(
        total_gauss_batches,
        total_macro_tiles,
        n_macro_tiles,
        macro_tile_cols,
        static_cast<int32_t>(tile_size),
        static_cast<int32_t>(tile_cols),
        static_cast<int32_t>(tile_rows),
        mt_gauss_offsets,
        mt_gauss_ids_sorted,
        mt_gauss_batch_offsets,
        rt_gauss_offsets_full,
        rt_batch_gauss_offsets,
        rt_bitmasks,
        flatten_ids
    );

    return std::make_tuple(isect_offsets, flatten_ids);
}

static int64_t num_words_per_tile_bitmask(int64_t tile_size)
{
    return (tile_size * tile_size + 63) / 64;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> build_sparse_tile_layout(
    const at::Tensor &pixels,    // [P, 2] int, (row, col)
    const at::Tensor &image_ids, // [P] int
    int64_t n_images,
    int64_t tile_size,
    int64_t tile_width,
    int64_t tile_height
)
{
    DEVICE_GUARD(pixels);
    CHECK_INPUT(pixels);
    CHECK_INPUT(image_ids);
    TORCH_CHECK(pixels.dim() == 2 && pixels.size(1) == 2, "pixels must be [P, 2]");

    const int64_t P       = pixels.size(0);
    const int64_t n_tiles = tile_width * tile_height;
    const int64_t words   = num_words_per_tile_bitmask(tile_size);
    const auto dev        = pixels.device();
    const auto i64        = at::TensorOptions().device(dev).dtype(at::kLong);
    const auto u64        = at::TensorOptions().device(dev).dtype(at::kUInt64);
    const auto i32        = at::TensorOptions().device(dev).dtype(at::kInt);
    const auto bln        = at::TensorOptions().device(dev).dtype(at::kBool);

    if(P == 0 || n_images == 0)
    {
        // Degenerate shapes (note tile_pixel_cumsum is [1] here).
        return std::make_tuple(
            at::empty({0}, i32),
            at::zeros({n_images, tile_height, tile_width}, bln),
            at::empty({0, words}, u64),
            at::zeros({1}, i64),
            at::empty({0}, i64)
        );
    }

    // The dense tile id must fit so the tightly-packed key stays a positive
    // int64 (end_bit below is well under 63 given this bound).
    TORCH_CHECK(
        n_images * n_tiles < (int64_t(1) << 31),
        "build_sparse_tile_layout: n_images * n_tiles (",
        n_images * n_tiles,
        ") must be < 2^31."
    );

    // Bits to index values in [0, count): ceil(log2(count)), 0 for count <= 1.
    const auto nbits = [](int64_t count) -> int
    {
        int b = 0;
        for(int64_t m = (count > 1) ? count - 1 : 0; m > 0; m >>= 1)
        {
            ++b;
        }
        return b;
    };
    // Tight key packing: tile id above pos_bits, in-tile position below. end_bit
    // bounds the radix sort to the meaningful key width (vs a full 64-bit sort).
    const int pos_bits = nbits(tile_size * tile_size);
    const int end_bit  = pos_bits + nbits(n_images * n_tiles);

    // Per-pixel sort key in one fused kernel. (.to(kInt) is a no-op for the
    // common int32 inputs.)
    const auto pixels_i    = pixels.to(at::kInt).contiguous();
    const auto image_ids_i = image_ids.to(at::kInt).reshape({P}).contiguous();
    auto key               = at::empty({P}, i64);
    launch_compute_tile_keys_kernel(pixels_i, image_ids_i, n_tiles, tile_size, tile_width, pos_bits, key);

    // Bit-bounded radix sort of (key, pixel index). Keys are unique under the
    // no-duplicate precondition, so the argsort (pixel_map) is deterministic.
    auto pixel_index = at::arange(P, i64);
    auto sorted_key  = at::empty({P}, i64);
    auto pixel_map   = at::empty({P}, i64);
    launch_sort_tile_keys(P, end_bit, key, pixel_index, sorted_key, pixel_map);

    // Run-length encode the sorted tile ids -> active tiles + per-tile counts.
    const auto tile_id_sorted = sorted_key.bitwise_right_shift(pos_bits);
    const auto uc           = at::unique_consecutive(tile_id_sorted, /*return_inverse=*/false, /*return_counts=*/true);
    const auto active_tiles = std::get<0>(uc); // int64 dense tile ids
    const auto counts       = std::get<2>(uc);
    const auto tile_pixel_cumsum = counts.cumsum(0); // inclusive

    // Per-tile activity mask, scattered from the (few) active tile ids.
    auto active_tile_mask = at::zeros({n_images * n_tiles}, bln);
    active_tile_mask.index_fill_(0, active_tiles, 1);
    active_tile_mask = active_tile_mask.view({n_images, tile_height, tile_width});

    // Per-active-tile raster-order bitmask (reads the in-tile position from the
    // low pos_bits of each sorted key).
    const int64_t AT     = active_tiles.size(0);
    auto tile_pixel_mask = at::zeros({AT, words}, u64);
    launch_build_tile_bitmask_kernel(
        sorted_key, tile_pixel_cumsum, static_cast<uint32_t>(words), pos_bits, tile_pixel_mask
    );

    return std::make_tuple(active_tiles.to(at::kInt), active_tile_mask, tile_pixel_mask, tile_pixel_cumsum, pixel_map);
}

at::Tensor intersect_offset_privateuseone(
    const at::Tensor &isect_ids, // [n_isects]
    int64_t I,
    int64_t tile_width,
    int64_t tile_height
)
{
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    auto opt           = isect_ids.options();
    at::Tensor offsets = at::empty({I, tile_height, tile_width}, opt.dtype(at::kInt));
    launch_intersect_offset_kernels(isect_ids, I, tile_width, tile_height, offsets);
    return offsets;
}

void register_intersect_cuda_impl(torch::Library &m)
{
    m.impl("intersect_tile", to_torch_op<&intersect_tile>);
    m.impl("intersect_tile_lidar", to_torch_op<&intersect_tile_lidar>);
    m.impl("intersect_offset", to_torch_op<&intersect_offset>);
    m.impl("intersect_tile_sparse", &intersect_tile_sparse);
    m.impl("intersect_tile_macro_binning", &intersect_tile_macro_binning);
    m.impl("build_sparse_tile_layout", &build_sparse_tile_layout);
}

void register_intersect_privateuseone_impl(torch::Library &m)
{
    m.impl("intersect_tile", to_torch_op<&intersect_tile_privateuseone>);
    m.impl("intersect_offset", to_torch_op<&intersect_offset_privateuseone>);
}
} // namespace gsplat
