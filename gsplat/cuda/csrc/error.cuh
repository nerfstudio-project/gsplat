#pragma once

#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_SAFE_CALL(err) cuda_safe_call(err, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR cuda_check_error(__FILE__, __LINE__)
#define CUDA_STREAM_CHECK_ERROR(stream) cuda_check_error(__FILE__, __LINE__, stream)

__device__ __host__ inline void cuda_safe_call(const cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    const char* err_msg = cudaGetErrorString(err);
    printf("\033[0;31mCUDA Error: %s at %s:%d\n\033[0m", err_msg, file, line);
  } else {
    printf("CUDA success;\n");
  }
}

inline void cuda_check_error(const char* file, int line, cudaStream_t stream = NULL) {
  if (stream == NULL) {
    cuda_safe_call(cudaDeviceSynchronize(), file, line);
  } else {
    cuda_safe_call(cudaStreamSynchronize(stream), file, line);
  }
}