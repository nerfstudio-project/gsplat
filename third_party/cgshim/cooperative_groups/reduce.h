// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
// HIP shim for <cooperative_groups/reduce.h>.
//
// ROCm's cooperative_groups header (hip/hip_cooperative_groups.h) does NOT
// provide cg::reduce, cg::plus<T>, cg::greater<T>, or cg::labeled_partition.
// CUDA users porting to HIP hit this when their device code does:
//
//     #include <cooperative_groups/reduce.h>
//     namespace cg = cooperative_groups;
//     ...
//     val = cg::reduce(warp, val, cg::plus<float>());
//     auto group = cg::labeled_partition(warp, label);
//
// This shim provides the same API surface, implemented via __shfl_xor butterfly
// reductions on AMD wavefronts. It is wave-size-agnostic — it uses each tile's
// runtime size(), not a hard-coded 32.
//
// labeled_partition: ROCm has no equivalent. We return a trivial 1-thread tile
// (thread_rank()==0, size()==1). User code that does
//   warp_group = cg::labeled_partition(warp, gid);
//   warpSum(v, warp_group);                       // -> no-op via reduce
//   if (warp_group.thread_rank() == 0) atomicAdd(...);  // -> always true
// then becomes "every thread does its own atomicAdd". Correctness is preserved
// (the global atomic accumulates the partial sums); the warp-aggregation
// optimization is lost, which the Optimizer phase can revisit per arch.

#pragma once

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

namespace cooperative_groups {

template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct greater {
    __device__ __forceinline__ T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
struct less {
    __device__ __forceinline__ T operator()(T a, T b) const { return a < b ? a : b; }
};

// Generic warp/tile reduction via shfl_xor butterfly. Works for any tile that
// is a power-of-two slice of a wavefront (e.g. tiled_partition<32>).
// For tiles of size 1 (our labeled_partition fallback), the loop body never
// executes and val is returned unchanged.
template <typename TileT, typename T, typename Op>
__device__ __forceinline__ T reduce(const TileT& tile, T val, Op op) {
    const unsigned int n = tile.size();
    for (unsigned int offset = n >> 1; offset > 0; offset >>= 1) {
        T other = __shfl_xor(val, offset, n);
        val = op(val, other);
    }
    return val;
}

// Trivial 1-thread group used as the labeled_partition fallback on ROCm.
struct trivial_partition {
    __device__ __forceinline__ unsigned int thread_rank() const { return 0u; }
    __device__ __forceinline__ unsigned int size() const { return 1u; }
    __device__ __forceinline__ void sync() const { }
    __device__ __forceinline__ unsigned int meta_group_rank() const { return 0u; }
    __device__ __forceinline__ unsigned int meta_group_size() const { return 1u; }
};

template <typename ParentT, typename LabelT>
__device__ __forceinline__ trivial_partition labeled_partition(const ParentT& /*parent*/, LabelT /*label*/) {
    return trivial_partition{};
}

}  // namespace cooperative_groups
