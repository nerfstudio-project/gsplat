/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Windows-only torch ABI shim.
//
// The 3DGUT ParallelBatch backward rasterizer
// (RasterizeToPixelsFromWorld3DGSParallelBatchBwd.cu) calls
// `at::cuda::cub::radix_sort_pairs<int64_t, int32_t>`, an inline wrapper in
// <ATen/cuda/cub.h> that forwards to the out-of-line template
// `at::cuda::cub::detail::radix_sort_pairs_impl<int64_t, 4>`. PyTorch declares
// that template in cub.h but, per that header's own note, does NOT define it
// there: "If you get a link error, you need to add an explicit instantiation
// for your types in cub.cu." On Linux every torch_cuda symbol is visible by
// default, so the wheel's own instantiation satisfies the reference; the
// Windows cu128 torch wheel does not export the `radix_sort_pairs_impl`
// instantiations from torch_cuda.dll's import lib, producing
// `LNK2019: unresolved external symbol radix_sort_pairs_impl<__int64,4>`.
//
// We cannot simply re-instantiate torch's own definition from
// <ATen/cuda/cub-RadixSortPairs.cuh>: that header manually re-wraps CUB into
// the `at_cuda_detail::cub` namespace via CUB_NS_PREFIX, which clashes with
// CUB's own `#pragma once` headers under CUDA 12.9 (CUB's `Debug` ends up in
// `::cub` while util_device.cuh references `at_cuda_detail::cub::Debug` ->
// "namespace at_cuda_detail::cub has no member Debug"). Instead we provide our
// own definition of the exact symbol, declared (and thus name-mangled) by
// torch's declaration-only <ATen/cuda/cub.h>, implemented against plain
// (unwrapped) CUB DeviceRadixSort. The function body mirrors torch's
// cub-RadixSortPairs.cuh. Gated to Windows + 3DGUT so it neither duplicates
// torch's exported symbol on Linux nor compiles when the only caller is
// excluded.

#include "Config.h"

#if defined(_WIN32) && GSPLAT_BUILD_3DGUT

#include <ATen/cuda/cub.h>   // declares detail::radix_sort_pairs_impl + OpaqueType (no CUB internals)
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <cub/device/device_radix_sort.cuh>  // plain (unwrapped) CUB

#include <limits>

namespace at::cuda::cub::detail {

template <typename key_t, int value_size>
void radix_sort_pairs_impl(
    const key_t *keys_in, key_t *keys_out,
    const OpaqueType<value_size> *values_in, OpaqueType<value_size> *values_out,
    int64_t n, bool descending, int64_t begin_bit, int64_t end_bit) {
  TORCH_CHECK(n <= std::numeric_limits<int>::max(),
      "cub sort does not support sorting more than INT_MAX elements");
  using value_t = OpaqueType<value_size>;

  auto stream = c10::cuda::getCurrentCUDAStream();
  auto allocator = c10::cuda::CUDACachingAllocator::get();

  // Two-phase CUB pattern: first query temp-storage size, then run.
  size_t temp_storage_bytes = 0;
  if (descending) {
    ::cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
        static_cast<int>(n), static_cast<int>(begin_bit),
        static_cast<int>(end_bit), stream);
  } else {
    ::cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes, keys_in, keys_out, values_in, values_out,
        static_cast<int>(n), static_cast<int>(begin_bit),
        static_cast<int>(end_bit), stream);
  }

  auto temp_storage = allocator->allocate(temp_storage_bytes);
  if (descending) {
    ::cub::DeviceRadixSort::SortPairsDescending(
        temp_storage.get(), temp_storage_bytes, keys_in, keys_out,
        values_in, values_out, static_cast<int>(n),
        static_cast<int>(begin_bit), static_cast<int>(end_bit), stream);
  } else {
    ::cub::DeviceRadixSort::SortPairs(
        temp_storage.get(), temp_storage_bytes, keys_in, keys_out,
        values_in, values_out, static_cast<int>(n),
        static_cast<int>(begin_bit), static_cast<int>(end_bit), stream);
  }
}

// Explicit instantiation for the one (key_t, value_size) pair the 3DGUT
// ParallelBatch backward path uses (int64_t keys, 4-byte / int32_t values).
template void radix_sort_pairs_impl<int64_t, 4>(
    const int64_t *, int64_t *,
    const OpaqueType<4> *, OpaqueType<4> *,
    int64_t, bool, int64_t, int64_t);

} // namespace at::cuda::cub::detail

#endif // _WIN32 && GSPLAT_BUILD_3DGUT
