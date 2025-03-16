#include <ATen/Dispatch.h> // AT_DISPATCH_XXX
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h> // at::cuda::getCurrentCUDAStream

#include "Null.h"
// #include "Utils.cuh" // optionally include some shared utility functions

namespace gsplat {

// Define the CUDA kernel
template <typename scalar_t>
__global__ void null_kernel(
    const int64_t n_elements,
    const scalar_t *input,
    scalar_t *output,
    scalar_t unused
) {
    // do nothing here
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elements) {
        return;
    }
    output[idx] = input[idx];
}

// Kernel launching (kernel<<<...>>>) has to be compiled by nvcc so it has to be
// defined in a .cu file.
//
// Yet we want to keep this `launch_xxx` function lightweight
// without too much torch CPU-side logic, to avoid heavy torch dependencies
// (e.g., torch/extension.h) in the .cu file. This is very helpful for reducing
// compilation time.
//
// The complete CPU operator should be defined in a .cpp file. with
// .h file as the bridge to call this CUDA kernel launching function.
void launch_null_kernel(const at::Tensor input, at::Tensor output) {
    int64_t n_elements = input.numel();
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "null_kernel",
        [&]() {
            scalar_t unused = 0;
            null_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    n_elements,
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    unused
                );
        }
    );
}

} // namespace gsplat
