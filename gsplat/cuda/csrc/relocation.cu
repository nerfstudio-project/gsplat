#include "bindings.h"

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation_kernel(
    int P,
    float *opacities,
    float *scales,
    int *ratios,
    float *binoms,
    int n_max,
    float *new_opacities,
    float *new_scales
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P)
        return;

    int n_idx = ratios[idx];
    float denom_sum = 0.0f;

    // compute new opacity
    new_opacities[idx] = 1.0f - powf(1.0f - opacities[idx], 1.0f / n_idx);

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term =
                (pow(-1, k) / sqrt(k + 1)) * pow(new_opacities[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (opacities[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * scales[idx * 3 + i];
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    torch::Tensor &opacities,
    torch::Tensor &scales,
    torch::Tensor &ratios,
    torch::Tensor &binoms,
    const int n_max
) {
    DEVICE_GUARD(opacities);
    CHECK_INPUT(opacities);
    CHECK_INPUT(scales);
    CHECK_INPUT(ratios);
    CHECK_INPUT(binoms);
    torch::Tensor new_opacities = torch::empty_like(opacities);
    torch::Tensor new_scales = torch::empty_like(scales);

    const int P = opacities.size(0);
    assert(P != 0);
    int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    compute_relocation_kernel<<<grid, block>>>(
        P,
        opacities.data_ptr<float>(),
        scales.data_ptr<float>(),
        ratios.data_ptr<int>(),
        binoms.data_ptr<float>(),
        n_max,
        new_opacities.data_ptr<float>(),
        new_scales.data_ptr<float>()
    );
    return std::make_tuple(new_opacities, new_scales);
}
