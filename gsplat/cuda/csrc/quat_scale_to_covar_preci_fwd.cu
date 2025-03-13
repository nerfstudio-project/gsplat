#include "bindings.h"
#include "quaternion.cuh"

#include <cooperative_groups.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Quat-Scale to Covariance and Precision Forward Pass
 ****************************************************************************/


__global__ void quat_scale_to_covar_preci_fwd_kernel(
    const uint32_t N,
    const float *__restrict__ quats,  // [N, 4]
    const float *__restrict__ scales, // [N, 3]
    const bool triu,
    // outputs
    float *__restrict__ covars, // [N, 3, 3] or [N, 6]
    float *__restrict__ precis  // [N, 3, 3] or [N, 6]
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
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
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
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) { // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    precis[i * 3 + j] = preci[j][i];
                }
            }
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_fwd_tensor(
    const torch::Tensor &quats,  // [N, 4]
    const torch::Tensor &scales, // [N, 3]
    const bool compute_covar,
    const bool compute_preci,
    const bool triu
) {
    GSPLAT_DEVICE_GUARD(quats);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);

    uint32_t N = quats.size(0);

    torch::Tensor covars, precis;
    if (compute_covar) {
        if (triu) {
            covars = torch::empty({N, 6}, quats.options());
        } else {
            covars = torch::empty({N, 3, 3}, quats.options());
        }
    }
    if (compute_preci) {
        if (triu) {
            precis = torch::empty({N, 6}, quats.options());
        } else {
            precis = torch::empty({N, 3, 3}, quats.options());
        }
    }

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                quat_scale_to_covar_preci_fwd_kernel<<<
                    (N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                    GSPLAT_N_THREADS,
                    0,
                    stream>>>(
                    N,
                    quats.data_ptr<float>(),
                    scales.data_ptr<float>(),
                    triu,
                    compute_covar ? covars.data_ptr<float>() : nullptr,
                    compute_preci ? precis.data_ptr<float>() : nullptr
                );
    }
    return std::make_tuple(covars, precis);
}

} // namespace gsplat