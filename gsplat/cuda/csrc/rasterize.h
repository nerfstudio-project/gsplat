#pragma once
#include <torch/extension.h>
#include <tuple>

std::
    tuple<
        torch::Tensor, // output image
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // tile_bins
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // gaussian_ids_unsorted
        torch::Tensor, // isect_ids_sorted
        torch::Tensor  // isect_ids_unsorted
        >
    nd_rasterize_forward_tensor(
        const torch::Tensor &xys,
        const torch::Tensor &depths,
        const torch::Tensor &radii,
        const torch::Tensor &conics,
        const torch::Tensor &num_tiles_hit,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &background
    );

std::
    tuple<
        torch::Tensor, // output image
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // tile_bins
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // gaussian_ids_unsorted
        torch::Tensor, // isect_ids_sorted
        torch::Tensor  // isect_ids_unsorted
        >
    rasterize_forward_tensor(
        const torch::Tensor &xys,
        const torch::Tensor &depths,
        const torch::Tensor &radii,
        const torch::Tensor &conics,
        const torch::Tensor &num_tiles_hit,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &background
    );

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    nd_rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output // dL_dout_color
    );

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const unsigned img_height,
        const unsigned img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &background,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output // dL_dout_color
    );
