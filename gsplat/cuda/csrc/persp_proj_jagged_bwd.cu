#include "bindings.h"
#include "helpers.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

/****************************************************************************
 * Perspective Projection Backward Pass (Jagged Version)
 ****************************************************************************/

template <typename T>
__global__ void persp_proj_jagged_bwd_kernel(
    const uint32_t B,
    const int64_t nnz,
    const int64_t *__restrict__ g_sizes,  // [B]
    const int64_t *__restrict__ c_sizes,  // [B]
    const int64_t *__restrict__ g_indptr, // [B] start indices
    const int64_t *__restrict__ c_indptr, // [B] start indices
    const int64_t *__restrict__ n_indptr, // [B] start indices
    const T *__restrict__ means,          // [ggz, 3]
    const T *__restrict__ covars,         // [ggz, 3, 3]
    const T *__restrict__ Ks,             // [ccz, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const T *__restrict__ v_means2d,  // [nnz, 2]
    const T *__restrict__ v_covars2d, // [nnz, 2, 2]
    T *__restrict__ v_means,          // [ggz, 3]
    T *__restrict__ v_covars          // [ggz, 3, 3]
) {

    // For now we'll upcast float16 and bfloat16 to float32
    using OpT = typename OpType<T>::type;

    // parallelize over nnz.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }

    // TODO: too many global memory accesses.
    int64_t bid =
        bin_search(n_indptr, B, static_cast<int64_t>(idx)); // batch id
    int64_t idx_local = idx - n_indptr[bid]; // local elem idx within Ci * Ni
    int64_t cid_local = idx_local / g_sizes[bid]; // local camera id within Ci
    int64_t gid_local = idx_local % g_sizes[bid]; // local gaussian id within Ni
    int64_t cid = cid_local + c_indptr[bid];      // camera id
    int64_t gid = gid_local + g_indptr[bid];      // gaussian id

    // shift pointers to the current camera and gaussian
    means += idx * 3;
    covars += idx * 9;
    v_means += idx * 3;
    v_covars += idx * 9;
    Ks += cid * 9;
    v_means2d += idx * 2;
    v_covars2d += idx * 4;

    OpT fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3<OpT> v_covar(0.f);
    vec3<OpT> v_mean(0.f);
    const vec3<OpT> mean = glm::make_vec3(means);
    const mat3<OpT> covar = glm::make_mat3(covars);
    const vec2<OpT> v_mean2d = glm::make_vec2(v_means2d);
    const mat2<OpT> v_covar2d = glm::make_mat2(v_covars2d);
    persp_proj_vjp<OpT>(
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

    // write to outputs: glm is column-major but we want row-major
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) { // rows
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t j = 0; j < 3; j++) { // cols
            v_covars[i * 3 + j] = T(v_covar[j][i]);
        }
    }

    GSPLAT_PRAGMA_UNROLL
    for (uint32_t i = 0; i < 3; i++) {
        v_means[i] = T(v_mean[i]);
    }
}

std::tuple<torch::Tensor, torch::Tensor> persp_proj_jagged_bwd_tensor(
    const torch::Tensor &g_sizes, // [B] gaussian sizes
    const torch::Tensor &means,   // [ggz, 3]
    const torch::Tensor &covars,  // [ggz, 3, 3]
    const torch::Tensor &c_sizes, // [B] camera sizes
    const torch::Tensor &Ks,      // [ccz, 3, 3]
    const uint32_t width,
    const uint32_t height,
    const torch::Tensor &v_means2d, // [nnz, 2]
    const torch::Tensor &v_covars2d // [nnz, 2, 2]
) {
    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(g_sizes);
    GSPLAT_CHECK_INPUT(means);
    GSPLAT_CHECK_INPUT(covars);
    GSPLAT_CHECK_INPUT(c_sizes);
    GSPLAT_CHECK_INPUT(Ks);
    GSPLAT_CHECK_INPUT(v_means2d);
    GSPLAT_CHECK_INPUT(v_covars2d);

    // TODO: use inclusive sum
    uint32_t B = g_sizes.size(0);
    int64_t nnz = v_means2d.size(0);
    torch::Tensor c_indptr = torch::cumsum(c_sizes, 0, torch::kInt64) - c_sizes;
    torch::Tensor g_indptr = torch::cumsum(g_sizes, 0, torch::kInt64) - g_sizes;
    torch::Tensor n_sizes = c_sizes * g_sizes; // element size = Ci * Ni
    torch::Tensor n_indptr = torch::cumsum(n_sizes, 0, torch::kInt64) - n_sizes;

    torch::Tensor v_means = torch::zeros_like(means);
    torch::Tensor v_covars = torch::zeros_like(covars);

    if (nnz) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            v_means.scalar_type(),
            "persp_proj_jagged_bwd",
            [&]() {
                persp_proj_jagged_bwd_kernel<scalar_t>
                    <<<(nnz + GSPLAT_N_THREADS - 1) / GSPLAT_N_THREADS,
                       GSPLAT_N_THREADS,
                       0,
                       stream>>>(
                        B,
                        nnz,
                        g_sizes.data_ptr<int64_t>(),
                        c_sizes.data_ptr<int64_t>(),
                        g_indptr.data_ptr<int64_t>(),
                        c_indptr.data_ptr<int64_t>(),
                        n_indptr.data_ptr<int64_t>(),
                        means.data_ptr<scalar_t>(),
                        covars.data_ptr<scalar_t>(),
                        Ks.data_ptr<scalar_t>(),
                        width,
                        height,
                        v_means2d.data_ptr<scalar_t>(),
                        v_covars2d.data_ptr<scalar_t>(),
                        v_means.data_ptr<scalar_t>(),
                        v_covars.data_ptr<scalar_t>()
                    );
            }
        );
    }
    return std::make_tuple(v_means, v_covars);
}

} // namespace gsplat