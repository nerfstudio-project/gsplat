#include <ATen/Dispatch.h> // AT_DISPATCH_XXX
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream
#include <cooperative_groups.h>

#include "Null.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void adam_kernel(
    const uint32_t N,
    const uint32_t D,
    scalar_t *__restrict__ param,
    const scalar_t *__restrict__ param_grad,
    scalar_t *__restrict__ exp_avg,
    scalar_t *__restrict__ exp_avg_sq,
    const bool *valid,
    const scalar_t *__restrict__ step_per_gaussian,
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    auto p_idx = cg::this_grid().thread_rank();
    const uint32_t g_idx = p_idx / D;

    if (g_idx >= N)
        return;
    if (valid != nullptr && !valid[g_idx])
        return;

    float register_param_grad = param_grad[p_idx];
    float register_exp_avg = exp_avg[p_idx];
    float register_exp_avg_sq = exp_avg_sq[p_idx];

    // Update moments
    register_exp_avg =
        b1 * register_exp_avg + (1.0f - b1) * register_param_grad;
    register_exp_avg_sq = b2 * register_exp_avg_sq + (1.0f - b2) *
                                                     register_param_grad *
                                                     register_param_grad;
    // Bias correction
    float bias_correction1 = 1.0f - powf(b1, step_per_gaussian[g_idx]);
    float bias_correction2 = 1.0f - powf(b2, step_per_gaussian[g_idx]);

    float m_hat = register_exp_avg / bias_correction1;
    float v_hat = register_exp_avg_sq / bias_correction2;

    // Parameter update
    float step = -lr * m_hat / (sqrtf(v_hat) + eps);

    // Apply update
    param[p_idx] += step;
    exp_avg[p_idx] = register_exp_avg;
    exp_avg_sq[p_idx] = register_exp_avg_sq;
}

template <typename scalar_t>
__global__ void increment_step_kernel(
    scalar_t *__restrict__ step_per_gaussian,
    const bool* valid,
    int N
) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= N) return;
    if (valid == nullptr || valid[g_idx]) {
        step_per_gaussian[g_idx]++;
    }
}


void launch_adam_kernel(
    at::Tensor &param,                    // [N, ...]
    const at::Tensor &param_grad,         // [N, ...]
    at::Tensor &exp_avg,                  // [N, ...]
    at::Tensor &exp_avg_sq,               // [N, ...]
    const at::optional<at::Tensor> valid, // [N]
    at::Tensor &step_per_gaussian,        // [N]
    const float lr,
    const float b1,
    const float b2,
    const float eps
) {
    const uint32_t N = param.size(0);
    const uint32_t D = param.numel() / N;
    if (N == 0 || D == 0) {
        return;
    }

    int64_t shmem_size = 0;
    // Step 1: increment step counter
    dim3 threads_step(256);
    dim3 grid_step((N + threads_step.x - 1) / threads_step.x);
    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "increment_step_kernel", [&]() {
        increment_step_kernel<scalar_t>
            <<<grid_step, threads_step, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            step_per_gaussian.data_ptr<scalar_t>(),
            valid.value().data_ptr<bool>(),
            N
        );
    });
    // Step 2: Launch Adam kernel
    // parallel over [N, ...]
    int64_t n_elements = N * D;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);

    AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_kernel", [&]() {
        adam_kernel<scalar_t>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                N,
                D,
                param.data_ptr<scalar_t>(),
                param_grad.data_ptr<scalar_t>(),
                exp_avg.data_ptr<scalar_t>(),
                exp_avg_sq.data_ptr<scalar_t>(),
                valid.has_value() ? valid.value().data_ptr<bool>() : nullptr,
                step_per_gaussian.data_ptr<scalar_t>(),
                lr,
                b1,
                b2,
                eps
            );
    });
}

} // namespace gsplat
