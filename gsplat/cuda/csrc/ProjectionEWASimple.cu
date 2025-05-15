#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void projection_ewa_simple_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,  // [C, N, 3]
    const scalar_t *__restrict__ covars, // [C, N, 3, 3]
    const scalar_t *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    scalar_t *__restrict__ means2d, // [C, N, 2]
    scalar_t *__restrict__ covars2d // [C, N, 2, 2]
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
    Ks += cid * 9;
    means2d += idx * 2;
    covars2d += idx * 4;

    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat2 covar2d(0.f);
    vec2 mean2d(0.f);
    const vec3 mean = glm::make_vec3(means);
    const mat3 covar = glm::make_mat3(covars);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d);
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d);
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean, covar, fx, fy, cx, cy, width, height, covar2d, mean2d
        );
        break;
    }

// write to outputs: glm is column-major but we want row-major
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) { // rows
#pragma unroll
        for (uint32_t j = 0; j < 2; j++) { // cols
            covars2d[i * 2 + j] = covar2d[j][i];
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 2; i++) {
        means2d[i] = mean2d[i];
    }
}

void launch_projection_ewa_simple_fwd_kernel(
    // inputs
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    // outputs
    at::Tensor means2d, // [C, N, 2]
    at::Tensor covars2d // [C, N, 2, 2]
) {
    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_simple_fwd_kernel",
        [&]() {
            projection_ewa_simple_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    width,
                    height,
                    camera_model,
                    means2d.data_ptr<scalar_t>(),
                    covars2d.data_ptr<scalar_t>()
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_ewa_simple_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,  // [C, N, 3]
    const scalar_t *__restrict__ covars, // [C, N, 3, 3]
    const scalar_t *__restrict__ Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const scalar_t *__restrict__ v_means2d,  // [C, N, 2]
    const scalar_t *__restrict__ v_covars2d, // [C, N, 2, 2]
    scalar_t *__restrict__ v_means,          // [C, N, 3]
    scalar_t *__restrict__ v_covars          // [C, N, 3, 3]
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
    mat3 v_covar(0.f);
    vec3 v_mean(0.f);
    const vec3 mean = glm::make_vec3(means);
    const mat3 covar = glm::make_mat3(covars);
    const vec2 v_mean2d = glm::make_vec2(v_means2d);
    const mat2 v_covar2d = glm::make_mat2(v_covars2d);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean,
            covar,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            glm::transpose(v_covar2d),
            v_mean2d,
            v_mean,
            v_covar
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean,
            covar,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            glm::transpose(v_covar2d),
            v_mean2d,
            v_mean,
            v_covar
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean,
            covar,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            glm::transpose(v_covar2d),
            v_mean2d,
            v_mean,
            v_covar
        );
        break;
    }

// write to outputs: glm is column-major but we want row-major
#pragma unroll
    for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = v_covar[j][i];
        }
    }

#pragma unroll
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = v_mean[i];
    }
}

void launch_projection_ewa_simple_bwd_kernel(
    // inputs
    const at::Tensor means,  // [C, N, 3]
    const at::Tensor covars, // [C, N, 3, 3]
    const at::Tensor Ks,     // [C, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const CameraModelType camera_model,
    const at::Tensor v_means2d,  // [C, N, 2]
    const at::Tensor v_covars2d, // [C, N, 2, 2]
    // outputs
    at::Tensor v_means, // [C, N, 3]
    at::Tensor v_covars // [C, N, 3, 3]
) {
    uint32_t C = means.size(0);
    uint32_t N = means.size(1);

    int64_t n_elements = C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_simple_fwd_kernel",
        [&]() {
            projection_ewa_simple_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    width,
                    height,
                    camera_model,
                    v_means2d.data_ptr<scalar_t>(),
                    v_covars2d.data_ptr<scalar_t>(),
                    v_means.data_ptr<scalar_t>(),
                    v_covars.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat
