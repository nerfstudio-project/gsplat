#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "kernels.cuh"

#define N_THREADS 256

#define CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                  \
    CHECK_CUDA(x);                                                      \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)


torch::Tensor func(const torch::Tensor input) {
    DEVICE_GUARD(input);
    CHECK_INPUT(input);

    uint32_t N = input.size(0);
    torch::Tensor output = torch::empty({N}, input.options());

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    launch_func(N, N_THREADS, stream);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
