// ROCm shim for cooperative_groups/reduce.h
// Provides cg::reduce(), cg::greater<T>, cg::plus<T> for HIP/ROCm 7.0+
//
// Supported tile sizes: 8, 16, 32 (Wave32 — RDNA), 64 (Wave64 — CDNA/GCN)
// Drop this file into your include path so that:
//   #include <cooperative_groups/reduce.h>
// resolves to this shim instead of the missing CUDA header.
//
// CORRECTNESS NOTE (fixed): the reduction below MUST use `shfl_xor`
// (a symmetric butterfly network), not `shfl_down`. CUDA's
// `cooperative_groups::reduce(group, val, op)` is an *all-reduce*: every
// lane in `group` receives the fully-combined result. A `shfl_down`-based
// tree only computes the correct combined value for lane 0 (each step reads
// from `lane + offset`; for calling lanes >= tile_size/2 that source lane is
// out of range, so `shfl_down` returns the calling lane's own unchanged
// value instead of a real partner, silently corrupting the reduction for
// every lane except a small neighborhood around lane 0). This went
// undetected for `cg::plus` (addition/atomicAdd-style warp reductions,
// `warpSum()` in Utils.hip.h) because every caller of those only reads the
// result from `group.thread_rank() == 0`. It was NOT undetected for
// `cg::greater` in `RasterizeToPixelsFromWorld3DGSParallelBatchBwd.hip`,
// where `warp_bin_final = cg::reduce(warp, bin_final, cg::greater<int>())`
// is read by *every* lane to size its own per-pixel gaussian-processing
// loop: with the broken `shfl_down` version, ~31/32 lanes computed a wrong
// (too-small) `warp_bin_final`, causing them to skip earlier gaussians in
// the tile and drop those gaussians' gradient contributions entirely —
// reproduced and confirmed via a standalone kernel (see translator_report.md)
// showing >90% of lanes returning the wrong max. `shfl_xor` has no
// out-of-range reads for any power-of-two offset <= tile_size/2 (lane i XOR
// offset is always in [0, tile_size)), so every lane converges to the same,
// correct, fully-combined value, matching CUDA's `cg::reduce` semantics
// for both commutative ops used here (`plus`, `greater`).

#pragma once

#if defined(__HIP__) || defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>

namespace cooperative_groups {

template <typename T>
struct plus {
    __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct greater {
    __device__ T operator()(T a, T b) const { return a > b ? a : b; }
};

struct labeled_partition_identity {
    __device__ int thread_rank() const { return 0; }
};

template <typename Group, typename Label>
__device__ labeled_partition_identity labeled_partition(const Group&, Label) {
    return {};
}

template <typename T, typename Op>
__device__ T reduce(const labeled_partition_identity&, T val, Op) {
    return val;
}
// tile<64> — Wave64 (CDNA gfx90a/gfx942, GCN gfx906/gfx908)
template <typename T, typename Op>
__device__ T reduce(const cooperative_groups::thread_block_tile<64>& group, T val, Op op) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        T other = group.shfl_xor(val, offset);
        val = op(val, other);
    }
    return val;
}

// tile<32> — Wave32 (RDNA gfx10xx/gfx11xx/gfx12xx)
template <typename T, typename Op>
__device__ T reduce(const cooperative_groups::thread_block_tile<32>& group, T val, Op op) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = group.shfl_xor(val, offset);
        val = op(val, other);
    }
    return val;
}

// tile<16>
template <typename T, typename Op>
__device__ T reduce(const cooperative_groups::thread_block_tile<16>& group, T val, Op op) {
    for (int offset = 8; offset > 0; offset >>= 1) {
        T other = group.shfl_xor(val, offset);
        val = op(val, other);
    }
    return val;
}

// tile<8>
template <typename T, typename Op>
__device__ T reduce(const cooperative_groups::thread_block_tile<8>& group, T val, Op op) {
    for (int offset = 4; offset > 0; offset >>= 1) {
        T other = group.shfl_xor(val, offset);
        val = op(val, other);
    }
    return val;
}

} // namespace cooperative_groups
#endif // __HIP__
