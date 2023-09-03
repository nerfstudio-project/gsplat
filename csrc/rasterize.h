#pragma once
#include <torch/extension.h>
#include <tuple>

std::tuple<
    int,
    torch::Tensor, // output image
    torch::Tensor  // ouptut depth
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
    const float fy);

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
    float *out_radii);
