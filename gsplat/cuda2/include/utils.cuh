#ifndef GSPLAT_CUDA_UTILS_CUH
#define GSPLAT_CUDA_UTILS_CUH

#include "types.cuh"

namespace gsplat {

__device__ float inverse(const mat2 M, mat2 &Minv);

__device__ void inverse_vjp(const float Minv, const float v_Minv, float &v_M);

__device__ float add_blur(const float eps2d, mat2 &covar, float &compensation);

__device__ void add_blur_vjp(
    const float eps2d,
    const mat2 conic_blur,
    const float compensation,
    const float v_compensation,
    mat2 &v_covar
);

} // namespace gsplat

#endif // GSPLAT_CUDA_UTILS_CUH
