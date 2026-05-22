/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_fp16.h>

// Optional: set to 1 to enforce LD128/LD64 via inline PTX (single 128/64-bit load)
#ifndef GSPLAT_ENFORCE_LD128_LD64_PTX
#define GSPLAT_ENFORCE_LD128_LD64_PTX 0
#endif

namespace gsplat {

// Load 8 halfs at ptr into out[0..7] as float using a single 128-bit load (LD128).
// ptr must be 16-byte aligned for correct behavior.
__device__ __forceinline__ void load_8_halves_ld128(const __half* ptr, float out[8]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(ptr)
                 : "memory");
    out[0] = __half2float(__ushort_as_half(static_cast<uint16_t>(r0)));
    out[1] = __half2float(__ushort_as_half(static_cast<uint16_t>(r0 >> 16)));
    out[2] = __half2float(__ushort_as_half(static_cast<uint16_t>(r1)));
    out[3] = __half2float(__ushort_as_half(static_cast<uint16_t>(r1 >> 16)));
    out[4] = __half2float(__ushort_as_half(static_cast<uint16_t>(r2)));
    out[5] = __half2float(__ushort_as_half(static_cast<uint16_t>(r2 >> 16)));
    out[6] = __half2float(__ushort_as_half(static_cast<uint16_t>(r3)));
    out[7] = __half2float(__ushort_as_half(static_cast<uint16_t>(r3 >> 16)));
#else
    const uint4 q = __ldg(reinterpret_cast<const uint4*>(ptr));
    out[0] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.x)));
    out[1] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.x >> 16)));
    out[2] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.y)));
    out[3] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.y >> 16)));
    out[4] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.z)));
    out[5] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.z >> 16)));
    out[6] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.w)));
    out[7] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.w >> 16)));
#endif
}

// Load 4 halfs at ptr into out[0..3] as float using a single 64-bit load (LD64).
// ptr must be 8-byte aligned.
__device__ __forceinline__ void load_4_halves_ld64(const __half* ptr, float out[4]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "l"(ptr)
                 : "memory");
    out[0] = __half2float(__ushort_as_half(static_cast<uint16_t>(r0)));
    out[1] = __half2float(__ushort_as_half(static_cast<uint16_t>(r0 >> 16)));
    out[2] = __half2float(__ushort_as_half(static_cast<uint16_t>(r1)));
    out[3] = __half2float(__ushort_as_half(static_cast<uint16_t>(r1 >> 16)));
#else
    const uint2 q = __ldg(reinterpret_cast<const uint2*>(ptr));
    out[0] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.x)));
    out[1] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.x >> 16)));
    out[2] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.y)));
    out[3] = __half2float(__ushort_as_half(static_cast<uint16_t>(q.y >> 16)));
#endif
}

// Same as above but keep values in half (saves registers; use for fp16 load->accumulate->write).
__device__ __forceinline__ void load_8_halves_ld128_to_half(const __half* ptr, __half out[8]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(ptr)
                 : "memory");
    out[0] = __ushort_as_half(static_cast<uint16_t>(r0));
    out[1] = __ushort_as_half(static_cast<uint16_t>(r0 >> 16));
    out[2] = __ushort_as_half(static_cast<uint16_t>(r1));
    out[3] = __ushort_as_half(static_cast<uint16_t>(r1 >> 16));
    out[4] = __ushort_as_half(static_cast<uint16_t>(r2));
    out[5] = __ushort_as_half(static_cast<uint16_t>(r2 >> 16));
    out[6] = __ushort_as_half(static_cast<uint16_t>(r3));
    out[7] = __ushort_as_half(static_cast<uint16_t>(r3 >> 16));
#else
    const uint4 q = __ldg(reinterpret_cast<const uint4*>(ptr));
    out[0] = __ushort_as_half(static_cast<uint16_t>(q.x));
    out[1] = __ushort_as_half(static_cast<uint16_t>(q.x >> 16));
    out[2] = __ushort_as_half(static_cast<uint16_t>(q.y));
    out[3] = __ushort_as_half(static_cast<uint16_t>(q.y >> 16));
    out[4] = __ushort_as_half(static_cast<uint16_t>(q.z));
    out[5] = __ushort_as_half(static_cast<uint16_t>(q.z >> 16));
    out[6] = __ushort_as_half(static_cast<uint16_t>(q.w));
    out[7] = __ushort_as_half(static_cast<uint16_t>(q.w >> 16));
#endif
}

__device__ __forceinline__ void load_4_halves_ld64_to_half(const __half* ptr, __half out[4]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "l"(ptr)
                 : "memory");
    out[0] = __ushort_as_half(static_cast<uint16_t>(r0));
    out[1] = __ushort_as_half(static_cast<uint16_t>(r0 >> 16));
    out[2] = __ushort_as_half(static_cast<uint16_t>(r1));
    out[3] = __ushort_as_half(static_cast<uint16_t>(r1 >> 16));
#else
    const uint2 q = __ldg(reinterpret_cast<const uint2*>(ptr));
    out[0] = __ushort_as_half(static_cast<uint16_t>(q.x));
    out[1] = __ushort_as_half(static_cast<uint16_t>(q.x >> 16));
    out[2] = __ushort_as_half(static_cast<uint16_t>(q.y));
    out[3] = __ushort_as_half(static_cast<uint16_t>(q.y >> 16));
#endif
}

// Load 8 halfs (LD128) into 4 x __half2 (2 halfs per __half2). Uses one 128-bit load.
__device__ __forceinline__ void load_8_halves_ld128_to_half2(const __half* ptr, __half2 out[4]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1, r2, r3;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "l"(ptr)
                 : "memory");
    out[0] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r0)), __ushort_as_half(static_cast<uint16_t>(r0 >> 16)));
    out[1] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r1)), __ushort_as_half(static_cast<uint16_t>(r1 >> 16)));
    out[2] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r2)), __ushort_as_half(static_cast<uint16_t>(r2 >> 16)));
    out[3] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r3)), __ushort_as_half(static_cast<uint16_t>(r3 >> 16)));
#else
    const uint4 q = __ldg(reinterpret_cast<const uint4*>(ptr));
    out[0] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.x)), __ushort_as_half(static_cast<uint16_t>(q.x >> 16)));
    out[1] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.y)), __ushort_as_half(static_cast<uint16_t>(q.y >> 16)));
    out[2] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.z)), __ushort_as_half(static_cast<uint16_t>(q.z >> 16)));
    out[3] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.w)), __ushort_as_half(static_cast<uint16_t>(q.w >> 16)));
#endif
}

// Load 4 halfs (LD64) into 2 x __half2. Uses one 64-bit load.
__device__ __forceinline__ void load_4_halves_ld64_to_half2(const __half* ptr, __half2 out[2]) {
#if GSPLAT_ENFORCE_LD128_LD64_PTX
    uint32_t r0, r1;
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(r0), "=r"(r1)
                 : "l"(ptr)
                 : "memory");
    out[0] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r0)), __ushort_as_half(static_cast<uint16_t>(r0 >> 16)));
    out[1] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(r1)), __ushort_as_half(static_cast<uint16_t>(r1 >> 16)));
#else
    const uint2 q = __ldg(reinterpret_cast<const uint2*>(ptr));
    out[0] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.x)), __ushort_as_half(static_cast<uint16_t>(q.x >> 16)));
    out[1] = __halves2half2(__ushort_as_half(static_cast<uint16_t>(q.y)), __ushort_as_half(static_cast<uint16_t>(q.y >> 16)));
#endif
}

// Accumulate in half2: a += b * vis (both lanes). Single __hfma2 for fused mul-add.
__device__ __forceinline__ void acc_add_half2(__half2& a, const __half2 b, float vis) {
    a = __hfma2(b, __float2half2_rn(vis), a);
}

// Accumulator add: half path keeps registers in fp16.
__device__ __forceinline__ void acc_add_half(__half& a, __half h, float vis) {
    a = __hfma(h, __float2half_rn(vis), a);
}
__device__ __forceinline__ void acc_add_half(__half& a, float v, float vis) {
    a = __hfma(__float2half_rn(v), __float2half_rn(vis), a);
}
__device__ __forceinline__ void acc_add_float(float& a, float v, float vis) {
    a = fmaf(v, vis, a);
}
__device__ __forceinline__ void acc_add_float(float& a, __half h, float vis) {
    a = fmaf(__half2float(h), vis, a);
}

}  // namespace gsplat
