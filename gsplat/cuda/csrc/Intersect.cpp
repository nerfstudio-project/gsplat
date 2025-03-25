#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"    // where all the macros are defined
#include "Intersect.h" // where the launch function is declared
#include "Ops.h"       // a collection of all gsplat operators

namespace gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor> intersect_tile(
    const at::Tensor means2d,                    // [C, N, 2] or [nnz, 2]
    const at::Tensor radii,                      // [C, N, 2] or [nnz, 2]
    const at::Tensor depths,                     // [C, N] or [nnz]
    const at::optional<at::Tensor> camera_ids,   // [nnz]
    const at::optional<at::Tensor> gaussian_ids, // [nnz]
    const uint32_t C,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const bool sort
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(radii);
    CHECK_INPUT(depths);

    uint32_t n_elements = means2d.numel() / 2;
    bool packed = means2d.dim() == 2;
    if (packed) {
        TORCH_CHECK(
            camera_ids.has_value() && gaussian_ids.has_value(),
            "When packed is set, camera_ids and gaussian_ids must be provided."
        );
        CHECK_INPUT(camera_ids.value());
        CHECK_INPUT(gaussian_ids.value());
    }

    uint32_t n_tiles = tile_width * tile_height;
    // the number of bits needed to encode the camera id and tile id
    // Note: std::bit_width requires C++20
    // uint32_t tile_n_bits = std::bit_width(n_tiles);
    // uint32_t cam_n_bits = std::bit_width(C);
    uint32_t tile_n_bits = (uint32_t)floor(log2(n_tiles)) + 1;
    uint32_t cam_n_bits = (uint32_t)floor(log2(C)) + 1;
    // the first 32 bits are used for the camera id and tile id altogether, so
    // check if we have enough bits for them.
    assert(tile_n_bits + cam_n_bits <= 32);

    // first pass: compute number of tiles per gaussian
    at::Tensor tiles_per_gauss =
        at::empty_like(depths, depths.options().dtype(at::kInt));
    int64_t n_isects;
    at::Tensor cum_tiles_per_gauss;
    if (n_elements) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            C,
            tile_size,
            tile_width,
            tile_height,
            c10::nullopt, // cum_tiles_per_gauss
            // outputs
            at::optional<at::Tensor>(tiles_per_gauss),
            c10::nullopt, // isect_ids
            c10::nullopt  // flatten_ids
        );
        cum_tiles_per_gauss = at::cumsum(tiles_per_gauss.view({-1}), 0);
        n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    } else {
        n_isects = 0;
    }

    // second pass: compute isect_ids and flatten_ids as a packed tensor
    at::Tensor isect_ids =
        at::empty({n_isects}, depths.options().dtype(at::kLong));
    at::Tensor flatten_ids =
        at::empty({n_isects}, depths.options().dtype(at::kInt));
    if (n_isects) {
        launch_intersect_tile_kernel(
            // inputs
            means2d,
            radii,
            depths,
            packed ? camera_ids : c10::nullopt,
            packed ? gaussian_ids : c10::nullopt,
            C,
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
        radix_sort_double_buffer(
            n_isects,
            tile_n_bits,
            cam_n_bits,
            isect_ids,
            flatten_ids,
            isect_ids_sorted,
            flatten_ids_sorted
        );
        return std::make_tuple(
            tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted
        );
    } else {
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
}

at::Tensor intersect_offset(
    const at::Tensor isect_ids, // [n_isects]
    const uint32_t C,
    const uint32_t tile_width,
    const uint32_t tile_height
) {
    DEVICE_GUARD(isect_ids);
    CHECK_INPUT(isect_ids);

    at::Tensor offsets = at::empty(
        {C, tile_height, tile_width}, isect_ids.options().dtype(at::kInt)
    );
    launch_intersect_offset_kernel(
        isect_ids, C, tile_width, tile_height, offsets
    );
    return offsets;
}

// at::Tensor spherical_harmonics_fwd(
//     const uint32_t degrees_to_use,
//     const at::Tensor dirs,              // [..., 3]
//     const at::Tensor coeffs,            // [..., K, 3]
//     const at::optional<at::Tensor> masks // [...]
// ) {
//     DEVICE_GUARD(dirs);
//     CHECK_INPUT(dirs);
//     CHECK_INPUT(coeffs);
//     if (masks.has_value()) {
//         CHECK_INPUT(masks.value());
//     }
//     TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
//     TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");

//     at::Tensor colors = at::empty_like(dirs); // [..., 3]

//     launch_spherical_harmonics_fwd_kernel(
//         degrees_to_use,
//         dirs,
//         coeffs,
//         masks,
//         colors
//     );
//     return colors; // [..., 3]
// }

// std::tuple<at::Tensor, at::Tensor> spherical_harmonics_bwd(
//     const uint32_t K,
//     const uint32_t degrees_to_use,
//     const at::Tensor dirs,               // [..., 3]
//     const at::Tensor coeffs,             // [..., K, 3]
//     const at::optional<at::Tensor> masks, // [...]
//     const at::Tensor v_colors,           // [..., 3]
//     bool compute_v_dirs
// ) {
//     DEVICE_GUARD(dirs);
//     CHECK_INPUT(dirs);
//     CHECK_INPUT(coeffs);
//     CHECK_INPUT(v_colors);
//     if (masks.has_value()) {
//         CHECK_INPUT(masks.value());
//     }
//     TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension
//     3"); TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension
//     3"); TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
//     const uint32_t N = dirs.numel() / 3;

//     at::Tensor v_coeffs = at::zeros_like(coeffs);
//     at::Tensor v_dirs;
//     if (compute_v_dirs) {
//         v_dirs = at::zeros_like(dirs);
//     }

//     at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
//     uint32_t n_elements = N;
//     uint32_t shmem_size = 0;
//     launch_spherical_harmonics_bwd_kernel(
//         degrees_to_use,
//         dirs,
//         coeffs,
//         masks,
//         v_colors,
//         v_coeffs,
//         v_dirs.defined() ? at::optional<at::Tensor>(v_dirs) : c10::nullopt
//     );
//     return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
// }

} // namespace gsplat
