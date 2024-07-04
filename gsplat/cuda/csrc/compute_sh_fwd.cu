#include "bindings.h"
#include "spherical_harmonics.cuh"


#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;


__global__ void compute_sh_fwd_kernel(const uint32_t N, const uint32_t K,
                                      const uint32_t degrees_to_use,
                                      const float3 *__restrict__ dirs,  // [N, 3]
                                      const float *__restrict__ coeffs, // [N, K, 3]
                                      const bool *__restrict__ masks,   // [N]
                                      float *__restrict__ colors        // [N, 3]
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
    sh_coeffs_to_color_fast(degrees_to_use, c, dirs[elem_id], coeffs + elem_id * K * 3,
                            colors + elem_id * 3);
}


torch::Tensor compute_sh_fwd_tensor(const uint32_t degrees_to_use,
                                    torch::Tensor &dirs,              // [..., 3]
                                    torch::Tensor &coeffs,            // [..., K, 3]
                                    at::optional<torch::Tensor> masks // [...]
) {
    DEVICE_GUARD(dirs);
    CHECK_INPUT(dirs);
    CHECK_INPUT(coeffs);
    if (masks.has_value()) {
        CHECK_INPUT(masks.value());
    }
    TORCH_CHECK(coeffs.size(-1) == 3, "coeffs must have last dimension 3");
    TORCH_CHECK(dirs.size(-1) == 3, "dirs must have last dimension 3");
    const uint32_t K = coeffs.size(-2);
    const uint32_t N = dirs.numel() / 3;
    torch::Tensor colors = torch::empty_like(dirs); // [..., 3]
    // parallelize over N * 3
    if (N) {
        compute_sh_fwd_kernel<<<(N * 3 + N_THREADS - 1) / N_THREADS, N_THREADS>>>(
            N, K, degrees_to_use, (float3 *)dirs.data_ptr<float>(),
            coeffs.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            colors.data_ptr<float>());
    }
    return colors; // [..., 3]
}

