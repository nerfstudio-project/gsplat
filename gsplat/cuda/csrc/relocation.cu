#include "bindings.h"

// Equation (9) in "3D Gaussian Splatting as Markov Chain Monte Carlo"
__global__ void compute_relocation_kernel(
    int P,
    float *old_opacities,
    float *old_scales,
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
    new_opacities[idx] = 1.0f - powf(1.0f - old_opacities[idx], 1.0f / n_idx);

    // compute new scale
    for (int i = 1; i <= n_idx; ++i) {
        for (int k = 0; k <= (i - 1); ++k) {
            float bin_coeff = binoms[(i - 1) * n_max + k];
            float term =
                (pow(-1, k) / sqrt(k + 1)) * pow(new_opacities[idx], k + 1);
            denom_sum += (bin_coeff * term);
        }
    }
    float coeff = (old_opacities[idx] / denom_sum);
    for (int i = 0; i < 3; ++i)
        new_scales[idx * 3 + i] = coeff * old_scales[idx * 3 + i];
}

std::tuple<torch::Tensor, torch::Tensor> compute_relocation_tensor(
    torch::Tensor &old_opacities,
    torch::Tensor &old_scales,
    torch::Tensor &ratios,
    torch::Tensor &binoms,
    const int n_max
) {
    const int P = old_opacities.size(0);
    assert(P != 0);

    torch::Tensor final_opacities =
        torch::full({P}, 0, old_opacities.options().dtype(torch::kFloat32));
    torch::Tensor final_scales =
        torch::full({3 * P}, 0, old_scales.options().dtype(torch::kFloat32));

    int num_blocks = (P + 255) / 256;
    dim3 block(256, 1, 1);
    dim3 grid(num_blocks, 1, 1);
    compute_relocation_kernel<<<grid, block>>>(
        P,
        old_opacities.data_ptr<float>(),
        old_scales.data_ptr<float>(),
        ratios.data_ptr<int>(),
        binoms.data_ptr<float>(),
        n_max,
        final_opacities.data_ptr<float>(),
        final_scales.data_ptr<float>()
    );
    return std::make_tuple(final_opacities, final_scales);
}
