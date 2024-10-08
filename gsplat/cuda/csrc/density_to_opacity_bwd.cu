#include "bindings.h"
#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include "tetra.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename T>
__global__ void density_to_opacity_bwd_kernel(
    const uint32_t N,
    const T *__restrict__ densities, // [N]
    const T *__restrict__ rays_o, // [N, 3]
    const T *__restrict__ rays_d, // [N, 3]
    const T *__restrict__ means,  // [N, 3]
    const T *__restrict__ precisions, // [N, 3, 3]
    // gradients
    const T *__restrict__ v_opacities, // [N]
    // output
    T *__restrict__ v_densities, // [N]
    T *__restrict__ v_means, // [N, 3]
    T *__restrict__ v_precisions // [N, 3, 3]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers
    means += idx * 3;
    precisions += idx * 9;
    rays_o += idx * 3;
    rays_d += idx * 3;
    densities += idx;
    v_opacities += idx;

    v_densities += idx;
    v_means += idx * 3;
    v_precisions += idx * 9;

    vec3<T> v_mean(0.0f);
    mat3<T> v_precision(0.0f);
    T v_density = 0.0f;
    integral_opacity_vjp(
        densities[0],
        glm::make_vec3(rays_o),
        glm::make_vec3(rays_d),
        glm::make_vec3(means),
        glm::transpose(glm::make_mat3(precisions)), // glm is column-major
        v_opacities[0],
        // output
        v_mean,
        v_precision,
        v_density
    );

    v_densities[0] = v_density;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        v_means[i] = v_mean[i];
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            v_precisions[i * 3 + j] = v_precision[j][i]; // glm is column-major
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> density_to_opacity_bwd_tensor(
    const torch::Tensor &densities,
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &means,
    const torch::Tensor &precisions,
    const torch::Tensor &v_opacities
) {
    GSPLAT_DEVICE_GUARD(densities);
    GSPLAT_CHECK_INPUT(densities);
    GSPLAT_CHECK_INPUT(rays_o);
    GSPLAT_CHECK_INPUT(rays_d);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(precisions);
    GSPLAT_CHECK_INPUT(v_opacities);

    uint32_t N = means.size(0);

    torch::Tensor v_densities = torch::empty({N}, densities.options());
    torch::Tensor v_means = torch::empty({N, 3}, means.options());
    torch::Tensor v_precisions = torch::empty({N, 3, 3}, precisions.options());

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        density_to_opacity_bwd_kernel<float>
            <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                GSPLAT_N_THREADS,
                0,
                stream>>>(
                N,
                densities.data_ptr<float>(),
                rays_o.data_ptr<float>(),
                rays_d.data_ptr<float>(),
                means.data_ptr<float>(),
                precisions.data_ptr<float>(),
                v_opacities.data_ptr<float>(),
                v_densities.data_ptr<float>(),
                v_means.data_ptr<float>(),
                v_precisions.data_ptr<float>()
            );
    }
    return std::make_tuple(v_densities, v_means, v_precisions);
}

} // namespace gsplat