#pragma once

#include <cstdint>
#include <algorithm>
#include <glm/gtc/type_ptr.hpp>

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
