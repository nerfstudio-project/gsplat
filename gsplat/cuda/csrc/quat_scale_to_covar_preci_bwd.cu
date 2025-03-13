#include "bindings.h"
#include "quaternion.cuh"

#include <cooperative_groups.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Quat-Scale to Covariance and Precision Backward Pass
 ****************************************************************************/


__global__ void quat_scale_to_covar_preci_bwd_kernel(
    const uint32_t N,
    // fwd inputs
    const float *__restrict__ quats,  // [N, 4]
    const float *__restrict__ scales, // [N, 3]
    // grad outputs
    const float *__restrict__ v_covars, // [N, 3, 3] or [N, 6]
    const float *__restrict__ v_precis, // [N, 3, 3] or [N, 6]
    const bool triu,
    // grad inputs
    float *__restrict__ v_scales, // [N, 3]
    float *__restrict__ v_quats   // [N, 4]
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
        quat_scale_to_covar_vjp(
            quat, scale, rotmat, v_covar, v_quat, v_scale
        );
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
        quat_scale_to_preci_vjp(
            quat, scale, rotmat, v_preci, v_quat, v_scale
        );
    }

    // write out results
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < 3; ++k) {
        v_scales[k] = v_scale[k];
    }
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < 4; ++k) {
        v_quats[k] = v_quat[k];
    }
}

std::tuple<torch::Tensor, torch::Tensor> quat_scale_to_covar_preci_bwd_tensor(
    const torch::Tensor &quats,                  // [N, 4]
    const torch::Tensor &scales,                 // [N, 3]
    const at::optional<torch::Tensor> &v_covars, // [N, 3, 3] or [N, 6]
    const at::optional<torch::Tensor> &v_precis, // [N, 3, 3] or [N, 6]
    const bool triu
) {
    GSPLAT_DEVICE_GUARD(quats);
    GSPLAT_CHECK_INPUT(quats);
    GSPLAT_CHECK_INPUT(scales);
    if (v_covars.has_value()) {
        GSPLAT_CHECK_INPUT(v_covars.value());
    }
    if (v_precis.has_value()) {
        GSPLAT_CHECK_INPUT(v_precis.value());
    }

    uint32_t N = quats.size(0);

    torch::Tensor v_scales = torch::empty_like(scales);
    torch::Tensor v_quats = torch::empty_like(quats);

    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
                quat_scale_to_covar_preci_bwd_kernel
                    <<<(N + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        N,
                        quats.data_ptr<float>(),
                        scales.data_ptr<float>(),
                        v_covars.has_value()
                            ? v_covars.value().data_ptr<float>()
                            : nullptr,
                        v_precis.has_value()
                            ? v_precis.value().data_ptr<float>()
                            : nullptr,
                        triu,
                        v_scales.data_ptr<float>(),
                        v_quats.data_ptr<float>()
                    );
    }

    return std::make_tuple(v_quats, v_scales);
}

} // namespace gsplat