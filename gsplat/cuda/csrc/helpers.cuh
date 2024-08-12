#ifndef GSPLAT_CUDA_HELPERS_H
#define GSPLAT_CUDA_HELPERS_H

#include "types.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>

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

///////////////////
// template <uint32_t DIM, class T, class WarpT>
// inline __device__ void warpSum(T *val, WarpT &warp) {
//     PRAGMA_UNROLL
//     for (uint32_t i = 0; i < DIM; i++) {
//         val[i] = cg::reduce(warp, val[i], cg::plus<T>());
//     }
// }

template <class WarpT>
inline __device__ void warpSum(float3 &val, WarpT &warp) {
    val.x = cg::reduce(warp, val.x, cg::plus<float>());
    val.y = cg::reduce(warp, val.y, cg::plus<float>());
    val.z = cg::reduce(warp, val.z, cg::plus<float>());
}

template <typename T>
inline __device__ vec3<T> cross_product(vec3<T> a, vec3<T> b) {
    return vec3<T>(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x
    );
}

template <typename T> __forceinline__ __device__ T f3_sum(vec3<T> a) {
    return a.x + a.y + a.z;
}

template <typename T> __forceinline__ __device__ T f2_norm2(vec2<T> a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ float3 operator*(float a, float3 b) {
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ __forceinline__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ __forceinline__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float f2_norm2(float2 a) {
    // Technically L2 norm squared
    return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ float f3_sum(float3 a) { return a.x + a.y + a.z; }

__device__ __forceinline__ float3 cross_product(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x
    );
}
#endif // GSPLAT_CUDA_HELPERS_H