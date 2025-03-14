#pragma once

#include "common.h"

#define CUDA_CHECK_THROW(x) \
	do { \
		cudaError_t _result = x; \
		if (_result != cudaSuccess) \
			throw std::runtime_error{fmt::format(FILE_LINE " " #x " failed: {}", cudaGetErrorString(_result))}; \
	} while(0)
#define CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                  \
    CHECK_CUDA(x);                                                      \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

// // lambda kernel launch
// template <typename K, typename T, typename ... Types>
// inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
//     if (n_elements <= 0) {
//         return;
//     }
//     kernel<<<n_blocks_linear(n_elements), N_THREADS, shmem_size, stream>>>(n_elements, args...);
// }