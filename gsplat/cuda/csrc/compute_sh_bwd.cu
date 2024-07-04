#include "bindings.h"
#include "spherical_harmonics.cuh"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;



__global__ void compute_sh_bwd_kernel(const uint32_t N, const uint32_t K,
                                      const uint32_t degrees_to_use,
                                      const float3 *__restrict__ dirs,    // [N, 3]
                                      const float *__restrict__ coeffs,   // [N, K, 3]
                                      const bool *__restrict__ masks,     // [N]
                                      const float *__restrict__ v_colors, // [N, 3
                                      float *__restrict__ v_coeffs,       // [N, K, 3]
                                      float *__restrict__ v_dirs // [N, 3] optional
) {
    // parallelize over N * 3
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N * 3) {
        return;
    }
    uint32_t elem_id = idx / 3;
    uint32_t c = idx % 3; // color channel
    if (masks != nullptr && !masks[elem_id]) {
        return;
    }

    float3 v_dir = {0.f, 0.f, 0.f};
    sh_coeffs_to_color_fast_vjp(degrees_to_use, c, dirs[elem_id],
                                coeffs + elem_id * K * 3, v_colors + elem_id * 3,
                                v_coeffs + elem_id * K * 3,
                                v_dirs == nullptr ? nullptr : &v_dir);
    if (v_dirs != nullptr) {
        atomicAdd(v_dirs + elem_id * 3, v_dir.x);
        atomicAdd(v_dirs + elem_id * 3 + 1, v_dir.y);
        atomicAdd(v_dirs + elem_id * 3 + 2, v_dir.z);
    }
}

std::tuple<torch::Tensor, torch::Tensor>
compute_sh_bwd_tensor(const uint32_t K, const uint32_t degrees_to_use,
                      torch::Tensor &dirs,               // [..., 3]
                      torch::Tensor &coeffs,             // [..., K, 3]
                      at::optional<torch::Tensor> masks, // [...]
                      torch::Tensor &v_colors,           // [..., 3]
                      bool compute_v_dirs) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    CHECK_INPUT(v_colors);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(v_colors.size(-1) == 3, "v_colors must have last dimension 3");
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t N = dirs.numel() / 3;

    torch::Tensor v_coeffs = torch::zeros_like(coeffs);
    torch::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = torch::zeros_like(dirs);
    }
    if (N) {
        compute_sh_bwd_kernel<<<(N * 3 + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
            N, K, degrees_to_use, (float3 *)dirs.data_ptr<float>(),
            coeffs.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            v_colors.data_ptr<float>(), v_coeffs.data_ptr<float>(),
            compute_v_dirs ? v_dirs.data_ptr<float>() : nullptr);
    }
    return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
}
