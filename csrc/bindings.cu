#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <tuple>
#include <cuda_runtime_api.h>
#include "helpers.cuh"
#include "bindings.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace cg = cooperative_groups;


template <typename scalar_t>
__global__ void compute_cov2d_bounds_kernel_forward(
    const int num_pts,
    const scalar_t * __restrict__ A,
    scalar_t * __restrict__ conics,
    scalar_t * __restrict__ radii
){   
    unsigned row = cg::this_grid().thread_rank(); // same as threadIdx.x + blockIdx.x * blockDim.x;
    if (row>=num_pts){return;}
    int index = row * 3;

    float3 conic;
    float radius;
    float3 cov2d{(float)A[index], (float)A[index+1], (float)A[index+2]};
    compute_cov2d_bounds(cov2d,conic,radius);
    
    conics[index] = conic.x;
    conics[index+1] = conic.y;
    conics[index+2] = conic.z;
    radii[row]=radius;
}


std::tuple<
    torch::Tensor, // output conics
    torch::Tensor // ouptut radii
    >
compute_cov2d_bounds_cu_forward(
    int num_pts,
    torch::Tensor A
){

    torch::Tensor conics = torch::zeros({num_pts, A.size(1)}, A.options().dtype(torch::kFloat32));
    torch::Tensor radii = torch::zeros({num_pts, 1}, A.options().dtype(torch::kFloat32));

    const int threads = 256;
    int blocks = (num_pts + threads - 1) / threads;
    // instantiate kernel
    AT_DISPATCH_FLOATING_TYPES(A.type(), "compute_cov2d_bounds_cu_forward", 
    ([&] {
        compute_cov2d_bounds_kernel_forward<scalar_t><<<blocks, threads>>>(
            num_pts,
            A.contiguous().data_ptr<scalar_t>(),
            conics.contiguous().data_ptr<scalar_t>(),
            radii.contiguous().data_ptr<scalar_t>()
        );
    })
    );
    return std::make_tuple(conics, radii);
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor // output radii
    >
compute_cov2d_bounds_forward(
    const int num_pts,
    torch::Tensor A)
{
    CHECK_INPUT(A);
    return compute_cov2d_bounds_cu_forward(num_pts, A);
}