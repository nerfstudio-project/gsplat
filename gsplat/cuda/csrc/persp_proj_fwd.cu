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
 * Perspective Projection Forward Pass
 ****************************************************************************/

template <typename T>
__global__ void persp_proj_fwd_kernel(const uint32_t C, const uint32_t N,
                                      const T *__restrict__ means,  // [C, N, 3]
                                      const T *__restrict__ covars, // [C, N, 3, 3]
                                      const T *__restrict__ Ks,     // [C, 3, 3]
                                      const uint32_t width, const uint32_t height,
                                      T *__restrict__ means2d, // [C, N, 2]
                                      T *__restrict__ covars2d, // [C, N, 2, 2]
                                      T *__restrict__ ray_planes, // [C, N, 2]
                                      T *__restrict__ normals // [C, N, 3]
) {
    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

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
    Ks += cid * 9;
    means2d += idx * 2;
    covars2d += idx * 4;
    ray_planes += idx * 2;
    normals += idx * 3;

    OpT fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat2<OpT> covar2d(0.f);
    vec2<OpT> mean2d(0.f);
    vec2<OpT> ray_plane(0.f);
    vec3<OpT> normal(0.f);
    const vec3<OpT> mean = glm::make_vec3(means);
    const mat3<OpT> covar = glm::make_mat3(covars);
    persp_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d, ray_plane, normal);

    // write to outputs: glm is column-major but we want row-major
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) { // rows
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = T(covar2d[j][i]);
        }
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        means2d[i] = T(mean2d[i]);
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        ray_planes[i] = T(ray_plane[i]);
    }
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < 2; i++) {
        normals[i] = T(normal[i]);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
persp_proj_fwd_tensor(const torch::Tensor &means,  // [C, N, 3]
                      const torch::Tensor &covars, // [C, N, 3, 3]
                      const torch::Tensor &Ks,     // [C, 3, 3]
                      const uint32_t width, const uint32_t height) {
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(covars);
    CHECK_INPUT(Ks);

    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    torch::Tensor means2d = torch::empty({C, N, 2}, means.options());
    torch::Tensor covars2d = torch::empty({C, N, 2, 2}, covars.options());
    torch::Tensor ray_plane = torch::empty({C, N, 2}, covars.options());
    torch::Tensor normal = torch::empty({C, N, 3}, covars.options());

    if (C && N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, means.scalar_type(),
            "persp_proj_fwd", [&]() {
                persp_proj_fwd_kernel<scalar_t>
                    <<<(C * N + N_THREADS - 1) / N_THREADS, N_THREADS, 0, stream>>>(
                        C, N, means.data_ptr<scalar_t>(), covars.data_ptr<scalar_t>(),
                        Ks.data_ptr<scalar_t>(), width, height,
                        means2d.data_ptr<scalar_t>(), covars2d.data_ptr<scalar_t>(),
                        ray_plane.data_ptr<scalar_t>(), normal.data_ptr<scalar_t>());
            });
    }
    return std::make_tuple(means2d, covars2d);
}
