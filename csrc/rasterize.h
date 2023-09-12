#pragma once
#include <torch/extension.h>
#include <tuple>

std::
    tuple<
        torch::Tensor, // output image
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor,  // tile_bins
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // gaussian_ids_unsorted 
        torch::Tensor, // isect_ids_sorted 
        torch::Tensor // isect_ids_unsorted 
        >
    rasterize_forward_tensor(
        const torch::Tensor &xys,
        const torch::Tensor &depths,
        const torch::Tensor &radii,
        const torch::Tensor &conics,
        const torch::Tensor &num_tiles_hit,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const int img_height,
        const int img_width
    );

int render_gaussians_forward(
    const int num_points,
    const int channels,
    const float *means3d,
    const float *scales,
    const float glob_scale,
    const float *quats,
    const float *colors,
    const float *opacity,
    const float *view_matrix,
    const float *proj_matrix,
    const int img_height,
    const int img_width,
    const float fx,
    const float fy,
    float *out_img,
    float *radii,
    float *final_Ts,
    int *final_idx,
    int *gaussian_ids_sorted,
    int *tile_bins,
    float *xy,
    float *conics
);

std::
    tuple<
        torch::Tensor, // dL_dxy
        torch::Tensor, // dL_dconic
        torch::Tensor, // dL_dcolors
        torch::Tensor  // dL_dopacity
        >
    rasterize_backward_tensor(
        const int img_height,
        const int img_width,
        const torch::Tensor &gaussians_ids_sorted,
        const torch::Tensor &tile_bins,
        const torch::Tensor &xys,
        const torch::Tensor &conics,
        const torch::Tensor &colors,
        const torch::Tensor &opacities,
        const torch::Tensor &final_Ts,
        const torch::Tensor &final_idx,
        const torch::Tensor &v_output // dL_dout_color
    );
