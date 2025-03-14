#pragma once
#include <cstdint>
#include <cuda_runtime.h>

void launch_func(uint32_t N, uint32_t N_THREADS, cudaStream_t stream);