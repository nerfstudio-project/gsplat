// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// torch_rocm_compat.h — compatibility shim for PyTorch ROCm builds whose
// headers omit the `at::cuda::*` and `c10::cuda::*` masquerade aliases.
//
// Background:
//   The HIP-translated source files in this repo reference symbols by the
//   "logical CUDA name" (e.g. `at::cuda::getCurrentCUDAStream`,
//   `at::cuda::OptionalCUDAGuard`, `c10::cuda::CUDACachingAllocator::get`)
//   — that's the convention PyTorch's hipify tool emits and that PyTorch
//   2.13+ exposes on ROCm.
//
//   PyTorch 2.8 release on ROCm only ships the explicitly-named HIP
//   masquerade symbols (`c10::hip::HIPStreamMasqueradingAsCUDA`,
//   `c10::hip::OptionalHIPGuardMasqueradingAsCUDA`,
//   `c10::hip::HIPCachingAllocatorMasqueradingAsCUDA`, etc.) under
//   `ATen/hip/impl/` and does NOT publish the `at::cuda::*` /
//   `c10::cuda::CUDACachingAllocator::*` aliases — so our sources fail to
//   compile with errors like
//     `no member named 'getCurrentCUDAStream' in namespace 'at::cuda'`
//     `'OptionalCUDAGuard' in namespace 'at::cuda' does not name a type`
//     `no member named 'CUDACachingAllocator' in namespace 'c10::cuda'`.
//
//   This header is force-included (via `-include`, see _rocm_flags() in
//   build.py) on every ROCm TU and reintroduces the missing aliases.
//
//   Guarded with TORCH_VERSION to no-op on PyTorch 2.13+, where the
//   aliases already exist (defining them again would be a redeclaration).

#pragma once

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)

#include <torch/csrc/api/include/torch/version.h>

// Apply only when the alias macros aren't already provided by PyTorch.
// PyTorch 2.13+ ships them; 2.8.x does not.
#if !defined(TORCH_VERSION_MAJOR) || \
    (TORCH_VERSION_MAJOR < 2) || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 13)

#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace at {
namespace cuda {

// at::cuda::getCurrentCUDAStream — forward to the HIP masquerade.
inline ::c10::hip::HIPStreamMasqueradingAsCUDA
getCurrentCUDAStream(int8_t device_index = -1) {
    return ::c10::hip::getCurrentHIPStreamMasqueradingAsCUDA(device_index);
}

// at::cuda::OptionalCUDAGuard — type alias to the HIP masquerade guard.
using OptionalCUDAGuard = ::c10::hip::OptionalHIPGuardMasqueradingAsCUDA;

} // namespace cuda
} // namespace at

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {

// c10::cuda::CUDACachingAllocator::get() — forward to the HIP masquerade.
// Returns the HIPAllocator pointer, which has the same interface
// (raw_alloc / raw_delete / etc.) that gsplat's RAII allocator wrappers use.
inline auto* get() {
    return ::c10::hip::HIPCachingAllocatorMasqueradingAsCUDA::get();
}

} // namespace CUDACachingAllocator
} // namespace cuda
} // namespace c10

#endif // PyTorch < 2.13
#endif // USE_ROCM
