#include "cuda_runtime.h"
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

std::
    tuple<
        torch::Tensor, // output conics
        torch::Tensor  // ouptut radii
        >
    compute_cov2d_bounds_forward_tensor(const int num_pts, torch::Tensor A);