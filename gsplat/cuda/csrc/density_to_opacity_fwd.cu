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
__global__ void density_to_opacity_fwd_kernel(
    const uint32_t N,
    const T *__restrict__ densities, // [N]
    const T *__restrict__ rays_o, // [N, 3]
    const T *__restrict__ rays_d, // [N, 3]
    const T *__restrict__ means,  // [N, 3]
    const T *__restrict__ precisions, // [N, 3, 3]
    T *opacities // [N]
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

    bool verbose = false;
    // if (idx == 10003) {
    //     verbose = true;
    // }

    T opacity = integral_opacity(
        densities[0],
        glm::make_vec3(rays_o),
        glm::make_vec3(rays_d),
        glm::make_vec3(means),
        glm::transpose(glm::make_mat3(precisions)),
        verbose
    );

    opacities[idx] = opacity;
}

torch::Tensor density_to_opacity_fwd_tensor(
    const torch::Tensor &densities,
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &means,
    const torch::Tensor &precisions
) {
    GSPLAT_DEVICE_GUARD(densities);
    GSPLAT_CHECK_INPUT(densities);
    GSPLAT_CHECK_INPUT(rays_o);
    GSPLAT_CHECK_INPUT(rays_d);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(precisions);

    uint32_t N = means.size(0);

    torch::Tensor opacities = torch::empty({N}, densities.options());

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        density_to_opacity_fwd_kernel<float>
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
                opacities.data_ptr<float>()
            );
    }
    return opacities;
}

} // namespace gsplat