#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "QuatScaleToCovar.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void quat_scale_to_covar_preci_fwd_kernel(
    const uint32_t N,
    const scalar_t *__restrict__ quats,  // [N, 4]
    const scalar_t *__restrict__ scales, // [N, 3]
    const bool triu,
    // outputs
    scalar_t *__restrict__ covars, // [N, 3, 3] or [N, 6]
    scalar_t *__restrict__ precis  // [N, 3, 3] or [N, 6]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    quats += idx * 4;
    scales += idx * 3;

    // compute the matrices
    mat3 covar, preci;
    const vec4 quat = glm::make_vec4(quats);
    const vec3 scale = glm::make_vec3(scales);
    quat_scale_to_covar_preci(
        quat, scale, covars ? &covar : nullptr, precis ? &preci : nullptr
    );

    // write to outputs: glm is column-major but we want row-major
    if (covars != nullptr) {
        if (triu) {
            covars += idx * 6;
            covars[0] = covar[0][0];
            covars[1] = covar[0][1];
            covars[2] = covar[0][2];
            covars[3] = covar[1][1];
            covars[4] = covar[1][2];
            covars[5] = covar[2][2];
        } else {
            covars += idx * 9;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    covars[i * 3 + j] = covar[j][i];
                }
            }
        }
    }
    if (precis != nullptr) {
        if (triu) {
            precis += idx * 6;
            precis[0] = preci[0][0];
            precis[1] = preci[0][1];
            precis[2] = preci[0][2];
            precis[3] = preci[1][1];
            precis[4] = preci[1][2];
            precis[5] = preci[2][2];
        } else {
            precis += idx * 9;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    precis[i * 3 + j] = preci[j][i];
                }
            }
        }
    }
}

void launch_quat_scale_to_covar_preci_fwd_kernel(
    // inputs
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    // outputs
    at::optional<at::Tensor> covars, // [N, 3, 3] or [N, 6]
    at::optional<at::Tensor> precis  // [N, 3, 3] or [N, 6]
) {

    uint32_t N = quats.size(0);

    int64_t n_elements = N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        quats.scalar_type(),
        "quat_scale_to_covar_preci_fwd_kernel",
        [&]() {
            quat_scale_to_covar_preci_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    quats.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    triu,
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    precis.has_value() ? precis.value().data_ptr<scalar_t>()
                                       : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void quat_scale_to_covar_preci_bwd_kernel(
    const uint32_t N,
    // fwd inputs
    const scalar_t *__restrict__ quats,  // [N, 4]
    const scalar_t *__restrict__ scales, // [N, 3]
    const bool triu,
    // grad outputs
    const scalar_t *__restrict__ v_covars, // [N, 3, 3] or [N, 6]
    const scalar_t *__restrict__ v_precis, // [N, 3, 3] or [N, 6]
    // grad inputs
    scalar_t *__restrict__ v_quats, // [N, 4]
    scalar_t *__restrict__ v_scales // [N, 3]
) {
    // parallelize over N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    // shift pointers to the current gaussian
    v_scales += idx * 3;
    v_quats += idx * 4;

    vec4 quat = glm::make_vec4(quats + idx * 4);
    vec3 scale = glm::make_vec3(scales + idx * 3);
    mat3 rotmat = quat_to_rotmat(quat);

    vec4 v_quat(0.f);
    vec3 v_scale(0.f);
    if (v_covars != nullptr) {
        // glm is column-major, input is row-major
        mat3 v_covar;
        if (triu) {
            v_covars += idx * 6;
            v_covar = mat3(
                v_covars[0],
                v_covars[1] * .5f,
                v_covars[2] * .5f,
                v_covars[1] * .5f,
                v_covars[3],
                v_covars[4] * .5f,
                v_covars[2] * .5f,
                v_covars[4] * .5f,
                v_covars[5]
            );
        } else {
            v_covars += idx * 9;
            mat3 v_covar_cast = glm::make_mat3(v_covars);
            v_covar = glm::transpose(v_covar_cast);
        }
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
    }
    if (v_precis != nullptr) {
        // glm is column-major, input is row-major
        mat3 v_preci;
        if (triu) {
            v_precis += idx * 6;
            v_preci = mat3(
                v_precis[0],
                v_precis[1] * .5f,
                v_precis[2] * .5f,
                v_precis[1] * .5f,
                v_precis[3],
                v_precis[4] * .5f,
                v_precis[2] * .5f,
                v_precis[4] * .5f,
                v_precis[5]
            );
        } else {
            v_precis += idx * 9;
            mat3 v_precis_cast = glm::make_mat3(v_precis);
            v_preci = glm::transpose(v_precis_cast);
        }
        quat_scale_to_preci_vjp(quat, scale, rotmat, v_preci, v_quat, v_scale);
    }

// write out results
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        v_scales[k] = v_scale[k];
    }
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
        v_quats[k] = v_quat[k];
    }
}

void launch_quat_scale_to_covar_preci_bwd_kernel(
    // inputs
    const at::Tensor quats,  // [N, 4]
    const at::Tensor scales, // [N, 3]
    const bool triu,
    const at::optional<at::Tensor> v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<at::Tensor> v_precis, // [N, 3, 3] or [N, 6]
    // outputs
    at::Tensor v_quats, // [N, 4]
    at::Tensor v_scales // [N, 3]
) {
    uint32_t N = quats.size(0);

    int64_t n_elements = N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }
    if (!v_covars.has_value() && !v_precis.has_value()) {
        // skip the kernel launch if there are no gradients to propagate
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        quats.scalar_type(),
        "quat_scale_to_covar_preci_bwd_kernel",
        [&]() {
            quat_scale_to_covar_preci_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    quats.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    triu,
                    v_covars.has_value() ? v_covars.value().data_ptr<scalar_t>()
                                         : nullptr,
                    v_precis.has_value() ? v_precis.value().data_ptr<scalar_t>()
                                         : nullptr,
                    v_quats.data_ptr<scalar_t>(),
                    v_scales.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat
