// Compatibility shim for CUDA 12.x toolkits (< CCCL 3.0).
// gsplat sources call ::cuda::ceil_div<T>(a, b), an API introduced with
// CCCL 3.0 (CUDA 13). Force-include this file on older toolkits:
//   NVCC_FLAGS="-include /path/to/CudaCompat.h"
#pragma once

#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 13)

#include <cuda_runtime.h>

namespace cuda {
template <typename T>
__host__ __device__ constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}
}  // namespace cuda

#endif  // __CUDACC__ && CUDA < 13
