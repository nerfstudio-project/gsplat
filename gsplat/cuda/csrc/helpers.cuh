#ifndef GSPLAT_CUDA_HELPERS_H
#define GSPLAT_CUDA_HELPERS_H

#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define PRAGMA_UNROLL _Pragma("unroll")

namespace cg = cooperative_groups;

template <uint32_t DIM, class T, class WarpT>
inline __device__ void warpSum(T *val, WarpT &warp) {
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < DIM; i++) {
        val[i] = cg::reduce(warp, val[i], cg::plus<T>());
    }
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec4<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
    val.w = cg::reduce(warp, val.w, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec3<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
    val.z = cg::reduce(warp, val.z, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(vec2<ScalarT> &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<ScalarT>());
    val.y = cg::reduce(warp, val.y, cg::plus<ScalarT>());
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat4<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
    warpSum(val[3], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat3<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
    warpSum(val[2], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpSum(mat2<ScalarT> &val, WarpT &warp) {
    warpSum(val[0], warp);
    warpSum(val[1], warp);
}

template <class WarpT, class ScalarT>
inline __device__ void warpMax(ScalarT &val, WarpT &warp) {
    val = cg::reduce(warp, val, cg::greater<ScalarT>());
}

__device__ __forceinline__ float atomicMax(float *address, float val) {
    int *address_as_i = reinterpret_cast<int *>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(reinterpret_cast<int *>(address), assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

#endif // GSPLAT_CUDA_HELPERS_H