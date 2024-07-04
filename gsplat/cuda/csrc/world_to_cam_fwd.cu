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
 * World to Camera Transformation Forward Pass
 ****************************************************************************/

__global__ void world_to_cam_fwd_kernel(const uint32_t C, const uint32_t N,
                                        const float *__restrict__ means,    // [N, 3]
                                        const float *__restrict__ covars,   // [N, 3, 3]
                                        const float *__restrict__ viewmats, // [C, 4, 4]
                                        float *__restrict__ means_c,        // [C, N, 3]
                                        float *__restrict__ covars_c // [C, N, 3, 3]
) { // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    covars += gid * 9;
    viewmats += cid * 16;

    // glm is column-major but input is row-major
    glm::mat3 R = glm::mat3(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    glm::vec3 t = glm::vec3(viewmats[3], viewmats[7], viewmats[11]);

    if (means_c != nullptr) {
        glm::vec3 mean_c;
        pos_world_to_cam(R, t, glm::make_vec3(means), mean_c);
        means_c += idx * 3;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            means_c[i] = mean_c[i];
        }
    }

    // write to outputs: glm is column-major but we want row-major
    if (covars_c != nullptr) {
        glm::mat3 covar_c;
        covar_world_to_cam(R, glm::make_mat3(covars), covar_c);
        covars_c += idx * 9;
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 3; i++) { // rows
            PRAGMA_UNROLL
            for (uint32_t j = 0; j < 3; j++) { // cols
                covars_c[i * 3 + j] = covar_c[j][i];
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor>
world_to_cam_fwd_tensor(const torch::Tensor &means,   // [N, 3]
                        const torch::Tensor &covars,  // [N, 3, 3]
                        const torch::Tensor &viewmats // [C, 4, 4]
) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);

    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor means_c = torch::empty({C, N, 3}, means.options());
    torch::Tensor covars_c = torch::empty({C, N, 3, 3}, means.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_fwd_kernel<<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0,
                                  stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(), means_c.data_ptr<float>(),
            covars_c.data_ptr<float>());
    }
    return std::make_tuple(means_c, covars_c);
}
