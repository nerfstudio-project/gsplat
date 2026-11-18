#pragma once

#ifdef USE_ROCM
#    include <ATen/hip/Atomic.cuh>
#else
#    include <ATen/cuda/Atomic.cuh>
#endif
