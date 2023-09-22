#include "cuda_runtime.h"
#include "forward.cuh"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_forward_tensor(const int num_pts, torch::Tensor A);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_forward_tensor(
    const int num_points,
    torch::Tensor means3d,
    torch::Tensor scales,
    const float glob_scale,
    torch::Tensor quats,
    torch::Tensor viewmat,
    torch::Tensor projmat,
    const float fx,
    const float fy,
    const float img_height,
    const float img_width,
    const std::tuple<int, int, int> tile_bounds
);

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
project_gaussians_backward_tensor(
    const int num_points,
    torch::Tensor means3d,
    torch::Tensor scales,
    const float glob_scale,
    torch::Tensor quats,
    torch::Tensor viewmat,
    torch::Tensor projmat,
    const float fx,
    const float fy,
    const float img_height,
    const float img_width,
    torch::Tensor cov3d,
    torch::Tensor radii,
    torch::Tensor conics,
    torch::Tensor v_xy,
    torch::Tensor v_conic
);
