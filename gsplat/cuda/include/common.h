#pragma once

#include <cstdint>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>

//
// Some Macros.
//
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
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...) do {                                       \
    size_t temp_storage_bytes = 0;                                          \
    func(nullptr, temp_storage_bytes, __VA_ARGS__);                         \
    auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();    \
    auto temp_storage = caching_allocator.allocate(temp_storage_bytes);     \
    func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);              \
  } while (false)


//
// Convenience typedefs for CUDA types
//
using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat2 = glm::mat<2, 2, float>;
using mat3 = glm::mat<3, 3, float>;
using mat4 = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;


//
// Legacy Camera Types
//
enum CameraModelType
{
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
};


// 
// Constants
//
constexpr uint32_t N_THREADS = 256;


//
// Helper functions
// https://github.com/NVlabs/tiny-cuda-nn
template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
constexpr __host__ __device__ uint32_t n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads);
}


