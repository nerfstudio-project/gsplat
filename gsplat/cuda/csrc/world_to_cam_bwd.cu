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
 * World to Camera Transformation Backward Pass
 ****************************************************************************/

template <typename T>
__global__ void
world_to_cam_bwd_kernel(const uint32_t C, const uint32_t N,
                        const T *__restrict__ means,      // [N, 3]
                        const T *__restrict__ covars,     // [N, 3, 3]
                        const T *__restrict__ viewmats,   // [C, 4, 4]
                        const T *__restrict__ v_means_c,  // [C, N, 3]
                        const T *__restrict__ v_covars_c, // [C, N, 3, 3]
                        T *__restrict__ v_means,          // [N, 3]
                        T *__restrict__ v_covars,         // [N, 3, 3]
                        T *__restrict__ v_viewmats        // [C, 4, 4]
) {
    // parallelize over C * N.
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
    mat3<T> R = mat3<T>(viewmats[0], viewmats[4], viewmats[8], // 1st column
                            viewmats[1], viewmats[5], viewmats[9], // 2nd column
                            viewmats[2], viewmats[6], viewmats[10] // 3rd column
    );
    vec3<T> t = vec3<T>(viewmats[3], viewmats[7], viewmats[11]);

    vec3<T> v_mean(0.f);
    mat3<T> v_covar(0.f);
    mat3<T> v_R(0.f);
    vec3<T> v_t(0.f);

    if (v_means_c != nullptr) {
        vec3<T> v_mean_c = glm::make_vec3(v_means_c + idx * 3);
        pos_world_to_cam_vjp<T>(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    }
    if (v_covars_c != nullptr) {
        mat3<T> v_covar_c = glm::transpose(glm::make_mat3(v_covars_c + idx * 9));
        covar_world_to_cam_vjp<T>(R, glm::make_mat3(covars), v_covar_c, v_R, v_covar);
    }

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += gid * 3;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                atomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += gid * 9;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    atomicAdd(v_covars + i * 3 + j, v_covar[j][i]);
                }
            }
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += cid * 16;
            PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    atomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                atomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_cam_bwd_tensor(const torch::Tensor &means,                    // [N, 3]
                        const torch::Tensor &covars,                   // [N, 3, 3]
                        const torch::Tensor &viewmats,                 // [C, 4, 4]
                        const at::optional<torch::Tensor> &v_means_c,  // [C, N, 3]
                        const at::optional<torch::Tensor> &v_covars_c, // [C, N, 3, 3]
                        const bool means_requires_grad, const bool covars_requires_grad,
                        const bool viewmats_requires_grad) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(viewmats);
    if (v_means_c.has_value()) {
        CHECK_INPUT(v_means_c.value());
    }
    if (v_covars_c.has_value()) {
        CHECK_INPUT(v_covars_c.value());
    }
    uint32_t N = means.size(0);
    uint32_t C = viewmats.size(0);

    torch::Tensor v_means, v_covars, v_viewmats;
    if (means_requires_grad) {
        v_means = torch::zeros({N, 3}, means.options());
    }
    if (covars_requires_grad) {
        v_covars = torch::zeros({N, 3, 3}, means.options());
    }
    if (viewmats_requires_grad) {
        v_viewmats = torch::zeros({C, 4, 4}, means.options());
    }

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        world_to_cam_bwd_kernel<float><<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
            C, N, means.data_ptr<float>(), covars.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            v_means_c.has_value() ? v_means_c.value().data_ptr<float>() : nullptr,
            v_covars_c.has_value() ? v_covars_c.value().data_ptr<float>() : nullptr,
            means_requires_grad ? v_means.data_ptr<float>() : nullptr,
            covars_requires_grad ? v_covars.data_ptr<float>() : nullptr,
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_means, v_covars, v_viewmats);
}