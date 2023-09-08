#pragma once
#include <torch/extension.h>
#include <tuple>

std::
    tuple<
        int,
        torch::Tensor, // output image
        torch::Tensor, // ouptut depth
        torch::Tensor, // final_Ts
        torch::Tensor, // final_idx
        torch::Tensor, // gaussian_ids_sorted
        torch::Tensor, // tile_bins
        torch::Tensor, // xy
        torch::Tensor  // conics
        >
    rasterize_forward_tensor(
        const torch::Tensor &means3d,
        const torch::Tensor &scales,
        const float glob_scale,
        const torch::Tensor &rotations_quat,
        const torch::Tensor &colors,
        const torch::Tensor &opacity,
        const torch::Tensor &view_matrix,
        const torch::Tensor &proj_matrix,
        const int img_height,
        const int img_width,
        const float fx,
        const float fy
    );

int rasterize_forward_impl(
    const int num_points,
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
    float *out_radii,
    float *final_Ts,
    int *final_idx,
    int *gaussian_ids_sorted,
    int *tile_bins,
    float *xy,
    float *conics
);

std::
    tuple<
        int,
        torch::Tensor, // dL_dmeans2D also referred to as dL_dxys
        torch::Tensor, // dL_dcolors
        torch::Tensor, // dL_dopacity
        torch::Tensor, // dL_dmeans3D
        torch::Tensor, // dL_dcov3D
        torch::Tensor, // dL_dscales
        torch::Tensor  // dL_drotations_quat
        >
    rasterize_backward_tensor(
        const torch::Tensor &means3D,
        const torch::Tensor &radii,
        const torch::Tensor &colors,
        const torch::Tensor &scales,
        const torch::Tensor &rotations_quat,
        const float glob_scale,
        const torch::Tensor &view_matrix,
        const torch::Tensor &proj_matrix,
        const torch::Tensor &v_output,
        const int img_height,
        const int img_width,
        const float fx,
        const float fy,
        const torch::Tensor gaussians_ids_sorted,
        const torch::Tensor tile_bins,
        const torch::Tensor xy,
        const torch::Tensor conics,
        const torch::Tensor opacities,
        const torch::Tensor final_Ts,
        const torch::Tensor final_idx

    );