#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>

#include "Common.h"
#include "Relocation.h"

namespace gsplat {

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
template <typename scalar_t>
__global__ void relocation_kernel(
    int N,
    scalar_t *opacities,
    scalar_t *scales,
    int *ratios,
    scalar_t *binoms,
    int n_max,
    scalar_t *new_opacities,
    scalar_t *new_scales
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N)
        return;

    int n_idx = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    new_opacities[idx] = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term = (pow(-1.0f, k) / sqrt(static_cast<float>(k + 1))) *
                         pow(new_opacities[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacities[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}

void launch_relocation_kernel(
    // inputs
    at::Tensor opacities, // [N]
    at::Tensor scales,    // [N, 3]
    at::Tensor ratios,    // [N]
    at::Tensor binoms,    // [n_max, n_max]
    const int n_max,
    // outputs
    at::Tensor new_opacities, // [N]
    at::Tensor new_scales     // [N, 3]
) {
    uint32_t N = opacities.size(0);

    int64_t n_elements = N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        opacities.scalar_type(),
        "relocation_kernel",
        [&]() {
            relocation_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    N,
                    opacities.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    ratios.data_ptr<int>(),
                    binoms.data_ptr<scalar_t>(),
                    n_max,
                    new_opacities.data_ptr<scalar_t>(),
                    new_scales.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat
