#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;


/****************************************************************************
 * Perspective Projection Backward Pass
 ****************************************************************************/

__global__ void
persp_proj_bwd_kernel(const uint32_t C, const uint32_t N,
                      const float *__restrict__ means,  // [C, N, 3]
                      const float *__restrict__ covars, // [C, N, 3, 3]
                      const float *__restrict__ Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height,
                      const float *__restrict__ v_means2d,  // [C, N, 2]
                      const float *__restrict__ v_covars2d, // [C, N, 2, 2]
                      float *__restrict__ v_means,          // [C, N, 3]
                      float *__restrict__ v_covars          // [C, N, 3, 3]
) {
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    // const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    glm::mat3 v_covar(0.f);
    glm::vec3 v_mean(0.f);
    persp_proj_vjp<float>(glm::make_vec3(means), glm::make_mat3(covars), fx, fy, cx, cy, width,
                          height, glm::transpose(glm::make_mat2(v_covars2d)),
                          glm::make_vec2(v_means2d), v_mean, v_covar);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_bwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height,
                      const torch::Tensor &v_means2d, // [C, N, 2]
                      const torch::Tensor &v_covars2d // [C, N, 2, 2]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_covars2d);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor v_means = torch::empty({C, N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        persp_proj_bwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            Ks.data_ptr<float>(), width, height, v_means2d.data_ptr<float>(),
            v_covars2d.data_ptr<float>(), v_means.data_ptr<float>(),
            v_covars.data_ptr<float>());
    }
    return std::make_tuple(v_means, v_covars);
}