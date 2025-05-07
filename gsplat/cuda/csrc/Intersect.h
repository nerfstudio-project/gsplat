#pragma once

#include <cstdint>

namespace at {
class Tensor;
}

namespace gsplat {

void launch_intersect_tile_kernel(
    // inputs
    const at::Tensor means2d,                    // [..., N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [..., N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [..., N] or [nnz]
    const at::optional<at::Tensor> image_ids,    // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t I,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const at::optional<at::Tensor> cum_tiles_per_gauss, // [..., N] or [nnz]
    // outputs
    at::optional<at::Tensor> tiles_per_gauss, // [..., N] or [nnz]
    at::optional<at::Tensor> isect_ids,       // [n_isects]
    at::optional<at::Tensor> flatten_ids      // [n_isects]
);

void launch_intersect_offset_kernel(
    // inputs
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t I,
    const uint32_t tile_width,
    const uint32_t tile_height,
    // outputs
    at::Tensor offsets // [..., tile_height, tile_width]
);

void radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);

void segmented_radix_sort_double_buffer(
    const int64_t n_isects,
    const uint32_t n_segments,
    const uint32_t image_n_bits,
    const uint32_t tile_n_bits,
    const at::Tensor offsets,
    at::Tensor isect_ids,
    at::Tensor flatten_ids,
    at::Tensor isect_ids_sorted,
    at::Tensor flatten_ids_sorted
);

} // namespace gsplat