#pragma once

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>

#ifndef USE_ROCM
#    include <cooperative_groups.h>
#    include <cub/cub.cuh>
#    include <c10/cuda/CUDAGuard.h>
#    include "ATenAtomic.cuh"
#else
#    include <hip/hip_cooperative_groups.h>
#    include <c10/hip/HIPGuard.h>
#    include "ATenAtomic.cuh"
#    include <hipcub/block/block_reduce.hpp>
#    include <hipcub/block/block_scan.hpp>
#    include <rocprim/warp/warp_reduce.hpp>
#endif

namespace gsplat {

#ifdef USE_ROCM
// ROCm: hipcub equivalent of the CUB_WRAPPER defined in Common.h for CUDA builds.
#define cub hipcub
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        auto res = func(nullptr, temp_storage_bytes, __VA_ARGS__);             \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        res = func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);       \
        assert(res == hipSuccess);                                             \
    } while (false)
#endif
} // namespace gsplat
