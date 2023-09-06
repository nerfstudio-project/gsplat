#include "backward.cuh"
#include "bindings.h"
#include "helpers.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <tuple>

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void compute_cov2d_bounds_forward_kernel(
    const int num_pts,
    const scalar_t *__restrict__ A,
    scalar_t *__restrict__ conics,
    scalar_t *__restrict__ radii
) {
    unsigned row = cg::this_grid().thread_rank(
    ); // same as threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= num_pts) {
        return;
    }
    int index = row * 3;

    float3 conic;
    float radius;
    float3 cov2d{(float)A[index], (float)A[index + 1], (float)A[index + 2]};
    compute_cov2d_bounds(cov2d, conic, radius);

    conics[index] = conic.x;
    conics[index + 1] = conic.y;
    conics[index + 2] = conic.z;
    radii[row] = radius;
}

std::tuple<
    torch::Tensor, // output conics
    torch::Tensor> // output radii
compute_cov2d_bounds_forward_tensor(const int num_pts, torch::Tensor A) {
    CHECK_INPUT(A);

    torch::Tensor conics =
        torch::zeros({num_pts, A.size(1)}, A.options().dtype(torch::kFloat32));
    torch::Tensor radii =
        torch::zeros({num_pts, 1}, A.options().dtype(torch::kFloat32));

    int blocks = (num_pts + N_THREADS - 1) / N_THREADS;
    // instantiate kernel
    AT_DISPATCH_FLOATING_TYPES(
        A.type(), "compute_cov2d_bounds_cu_forward", ([&] {
            compute_cov2d_bounds_forward_kernel<scalar_t>
                <<<blocks, N_THREADS>>>(
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
    const std::tuple<int, int> img_size,
    const std::tuple<int, int, int> tile_bounds
) {
    const auto num_cov3d = num_points * 6;

    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);

    dim3 tile_bounds_dim3;
    tile_bounds_dim3.x = std::get<0>(tile_bounds);
    tile_bounds_dim3.y = std::get<1>(tile_bounds);
    tile_bounds_dim3.z = std::get<2>(tile_bounds);

    // Triangular covariance.
    torch::Tensor cov3d_d =
        torch::zeros({num_cov3d, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor xys_d =
        torch::zeros({num_points, 2}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor depths_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor radii_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));
    torch::Tensor conics_d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor num_tiles_hit_d =
        torch::zeros({num_points}, means3d.options().dtype(torch::kInt32));

    project_gaussians_forward_impl(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        fx,
        fy,
        img_size_dim3,
        tile_bounds_dim3,
        // Outputs.
        cov3d_d.contiguous().data_ptr<float>(),
        (float2 *)xys_d.contiguous().data_ptr<float>(),
        depths_d.contiguous().data_ptr<float>(),
        radii_d.contiguous().data_ptr<int>(),
        (float3 *)conics_d.contiguous().data_ptr<float>(),
        num_tiles_hit_d.contiguous().data_ptr<int32_t>()
    );

    return std::make_tuple(
        cov3d_d, xys_d, depths_d, radii_d, conics_d, num_tiles_hit_d
    );
}

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
    const std::tuple<int, int> img_size,
    torch::Tensor cov3d,
    torch::Tensor radii,
    torch::Tensor conics,
    torch::Tensor v_xy,
    torch::Tensor v_conic
) {
    dim3 img_size_dim3;
    img_size_dim3.x = std::get<0>(img_size);
    img_size_dim3.y = std::get<1>(img_size);

    const auto num_cov3d = num_points * 6;

    // Triangular covariance.
    torch::Tensor v_cov2d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_cov3d =
        torch::zeros({num_points, 6}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_mean3d =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_scale =
        torch::zeros({num_points, 3}, means3d.options().dtype(torch::kFloat32));
    torch::Tensor v_quat =
        torch::zeros({num_points, 4}, means3d.options().dtype(torch::kFloat32));

    project_gaussians_backward_impl(
        num_points,
        (float3 *)means3d.contiguous().data_ptr<float>(),
        (float3 *)scales.contiguous().data_ptr<float>(),
        glob_scale,
        (float4 *)quats.contiguous().data_ptr<float>(),
        viewmat.contiguous().data_ptr<float>(),
        projmat.contiguous().data_ptr<float>(),
        fx,
        fy,
        img_size_dim3,
        cov3d.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int32_t>(),
        (float3 *)conics.contiguous().data_ptr<float>(),
        (float2 *)v_xy.contiguous().data_ptr<float>(),
        (float3 *)v_conic.contiguous().data_ptr<float>(),
        // Outputs.
        (float3 *)v_cov2d.contiguous().data_ptr<float>(),
        v_cov3d.contiguous().data_ptr<float>(),
        (float3 *)v_mean3d.contiguous().data_ptr<float>(),
        (float3 *)v_scale.contiguous().data_ptr<float>(),
        (float4 *)v_quat.contiguous().data_ptr<float>()
    );

    return std::make_tuple(v_cov2d, v_cov3d, v_mean3d, v_scale, v_quat);
}
