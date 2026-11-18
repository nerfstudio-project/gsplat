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

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Type-punning helper: reinterpret the bits of v as type T and assign to u.
template<typename T, typename U, typename V>
__device__ __forceinline__ void AssignAs(U &u, const V &v)
{
    reinterpret_cast<T &>(u) = reinterpret_cast<const T &>(v);
}

#if defined(USE_ROCM) || defined(__HIP_PLATFORM_AMD__)
// ---------------------------------------------------------------------------
// ROCm/HIP portable fallbacks for NVIDIA-specific inline PTX helpers below.
// AMD GCN has no equivalent of the PTX lop3/ex2.approx.f16x2/prmt/cvt.pack
// instructions, so these reproduce the exact bit-level behavior in plain C++.
// The CUDA build keeps the faster PTX paths (see the #else branch).
// ---------------------------------------------------------------------------

// Round-to-nearest (ties away from zero) __half2 pair to the M bits of mantissa
template<uint M, bool MASK_OUT_OUTPUT = true>
__device__ __forceinline__ __half2 RoundToNearest(const __half2 &v)
{
    constexpr uint MANTISSA_MASK = 0x03FF03FF;
    constexpr uint TRUNC_MASK    = ~(((1u << (10 - M)) - 1) * 0x10001u);
    const uint vn                = reinterpret_cast<const uint &>(v) & ~MANTISSA_MASK;
    const __half2 vb             = reinterpret_cast<const __half2 &>(vn);
    const __half2 vs             = __float2half2_rn((1 << (10 - M)) / 2048.0f);
    const __half2 v_r            = __hfma2(vb, vs, v);
    const uint vi_r
        = MASK_OUT_OUTPUT ? (reinterpret_cast<const uint &>(v_r) & TRUNC_MASK) : reinterpret_cast<const uint &>(v_r);
    return reinterpret_cast<const __half2 &>(vi_r);
}

// Packed half2 comparison masks. CUDA's __h*2_mask intrinsics return a uint32
// where the low 16 bits are all-ones when the low-half comparison is true and
// the high 16 bits are all-ones when the high-half comparison is true (0 else).
// Callers rely on bit 0 (low half) and bit 31 (high half); __h2if consumes the
// full 0xFFFF/0xFFFF0000 pattern as a per-bit select mask. HIP ships the __half2
// comparison forms (returning __half2 of 1.0/0.0) but not these packed-uint
// variants, so reconstruct them from the scalar comparisons.
__device__ __forceinline__ uint32_t __hlt2_mask(const __half2 &a, const __half2 &b)
{
    const uint32_t lo = __hlt(__low2half(a), __low2half(b)) ? 0x0000FFFFu : 0u;
    const uint32_t hi = __hlt(__high2half(a), __high2half(b)) ? 0xFFFF0000u : 0u;
    return lo | hi;
}

__device__ __forceinline__ uint32_t __hle2_mask(const __half2 &a, const __half2 &b)
{
    const uint32_t lo = __hle(__low2half(a), __low2half(b)) ? 0x0000FFFFu : 0u;
    const uint32_t hi = __hle(__high2half(a), __high2half(b)) ? 0xFFFF0000u : 0u;
    return lo | hi;
}

// Unordered >= (true if either operand is NaN), matching CUDA __hgeu2_mask.
__device__ __forceinline__ uint32_t __hgeu2_mask(const __half2 &a, const __half2 &b)
{
    const uint32_t lo = __hgeu(__low2half(a), __low2half(b)) ? 0x0000FFFFu : 0u;
    const uint32_t hi = __hgeu(__high2half(a), __high2half(b)) ? 0xFFFF0000u : 0u;
    return lo | hi;
}

// Per-lane min/max on __half2 (HIP provides scalar __hmin/__hmax only).
__device__ __forceinline__ __half2 __hmin2(const __half2 &a, const __half2 &b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)), __hmin(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ __half2 __hmax2(const __half2 &a, const __half2 &b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)), __hmax(__high2half(a), __high2half(b)));
}

// Streaming (cache-non-temporal) store. CUDA's __stcs bypasses the L1 for
// write-once data; GCN has no direct equivalent exposed here, so fall back to a
// plain store — correct, just without the cache-streaming hint.
template<typename T>
__device__ __forceinline__ void __stcs(T *ptr, const T &value)
{
    *ptr = value;
}

// Bitwise select: result = (cond & a) | (~cond & b), per PTX lop3 immLut 0xCA.
template<typename T>
__device__ __forceinline__ T __select(const uint32_t &cond, const T &a, const T &b)
{
    static_assert(sizeof(T) == 4, "Select only works on 32-bit types");
    const uint32_t ua   = reinterpret_cast<const uint32_t &>(a);
    const uint32_t ub   = reinterpret_cast<const uint32_t &>(b);
    const uint32_t rval = (cond & ua) | (~cond & ub);
    return reinterpret_cast<const T &>(rval);
}

__device__ __forceinline__ __half2 __h2_copy_sign(const __half2 &v, const __half2 &sign)
{
    return __select(0x80008000u, sign, v);
}

__device__ __forceinline__ __half __h_copy_sign(const __half &v, const __half &sign)
{
    return __low2half(
        __h2_copy_sign(__halves2half2(v, __float2half_rn(0.0f)), __halves2half2(sign, __float2half_rn(0.0f)))
    );
}

// Packed half2 exp2 (approximate on CUDA via PTX; per-lane exp2f on ROCm).
__forceinline__ __device__ __half2 __h2exp2_approx(__half2 v)
{
    const float2 f = __half22float2(v);
    return __floats2half2_rn(exp2f(f.x), exp2f(f.y));
}

// Bitwise select on packed half2: result = (cond & a) | (~cond & b).
__device__ __forceinline__ __half2 __h2if(const uint32_t &cond, const __half2 &a, const __half2 &b)
{
    const uint32_t ua   = reinterpret_cast<const uint32_t &>(a);
    const uint32_t ub   = reinterpret_cast<const uint32_t &>(b);
    const uint32_t rval = (cond & ua) | (~cond & ub);
    return reinterpret_cast<const __half2 &>(rval);
}

// PTX prmt.b32 default (index) mode: build a 4-byte result by selecting bytes
// from the 8-byte little-endian source {src1:src0}. Each 4-bit selector in
// mask picks a source byte (low 3 bits); bit 3 requests sign-fill (0x00/0xFF).
__forceinline__ __device__ uint32_t __prmt_idx(uint32_t src0, uint32_t src1, uint32_t mask)
{
    const uint64_t src = (static_cast<uint64_t>(src1) << 32) | static_cast<uint64_t>(src0);
    uint32_t rval      = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const uint32_t sel = (mask >> (i * 4)) & 0xF;
        const uint32_t idx = sel & 0x7;
        uint32_t byte      = static_cast<uint32_t>((src >> (idx * 8)) & 0xFF);
        if (sel & 0x8)
            byte = (byte & 0x80) ? 0xFF : 0x00;
        rval |= byte << (i * 8);
    }
    return rval;
}

// Convert 4 unsigned uint8 values packed in a uint32 to 4 fp16 values.
__forceinline__ __device__ void __uint8x4_to_half4(half *dst, uint32_t src)
{
    constexpr int FP16_BASE_LOG2 = 10;
    constexpr float FP16_BASE_FP = 1 << FP16_BASE_LOG2;
    constexpr int FP16_BASE      = (15 + FP16_BASE_LOG2) << 10; // 1024.0f in fp16.

    union
    {
        uint32_t i;
        half2 f;
    } dp[2];

    dp[0].i = __prmt_idx(src, FP16_BASE, 0x5150);
    dp[1].i = __prmt_idx(src, FP16_BASE, 0x5352);
    dp[0].f = __hsub2(dp[0].f, __float2half2_rn(FP16_BASE_FP));
    dp[1].f = __hsub2(dp[1].f, __float2half2_rn(FP16_BASE_FP));
    AssignAs<uint2>(dst[0], dp);
}

// Convert 2 unsigned uint8 values (low 16 bits of src) to half2.
__forceinline__ __device__ __half2 __uint8x2_to_half2(uint32_t src)
{
    constexpr int FP16_BASE_LOG2 = 10;
    constexpr float FP16_BASE_FP = 1 << FP16_BASE_LOG2;
    constexpr int FP16_BASE      = (15 + FP16_BASE_LOG2) << 10;

    union
    {
        uint32_t i;
        half2 f;
    } dp;

    dp.i = __prmt_idx(src, FP16_BASE, 0x5150);
    dp.f = __hsub2(dp.f, __float2half2_rn(FP16_BASE_FP));
    return dp.f;
}

// Round-to-nearest-int, saturate to u8, pack 4 lanes. Byte order matches the
// PTX cvt.pack.sat.u8.s32.b32 sequence used in the CUDA path: {a1,a0,a3,a2}.
__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(float a0, float a1, float a2, float a3)
{
    auto sat = [](float f) -> uint32_t {
        int v = static_cast<int>(rintf(f));
        v     = v < 0 ? 0 : (v > 255 ? 255 : v);
        return static_cast<uint32_t>(v);
    };
    // Natural byte order a0..a3 (low..high). The CUDA PTX path feeds a3,a2,a1,a0
    // in reverse so cvt.pack shifts them into this same layout.
    return sat(a0) | (sat(a1) << 8) | (sat(a2) << 16) | (sat(a3) << 24);
}

__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(const float4 &a)
{
    return __cvt_pack_sat_u8_f32(a.x, a.y, a.z, a.w);
}

#else // CUDA path (NVIDIA PTX intrinsics)

// Round-to-nearest (ties away from zero) __half2 pair to the M bits of mantissa
template<uint M, bool MASK_OUT_OUTPUT = true>
__device__ __forceinline__ __half2 RoundToNearest(const __half2 &v)
{
    constexpr uint MANTISSA_MASK = 0x03FF03FF;
    constexpr uint TRUNC_MASK    = ~(((1u << (10 - M)) - 1) * 0x10001u);
    // get rounding offset scale s*2^e (input float with mantissa zeroed out)
    const uint vn                = reinterpret_cast<const uint &>(v) & ~MANTISSA_MASK;
    const __half2 vb             = reinterpret_cast<const __half2 &>(vn);
    // get rounding offset
    const __half2 vs             = __float2half2_rn((1 << (10 - M)) / 2048.0f);
    // add to float a scaled rounding offset
    const __half2 v_r            = __hfma2(vb, vs, v);
    // return masked __half2
    const uint vi_r
        = MASK_OUT_OUTPUT ? (reinterpret_cast<const uint &>(v_r) & TRUNC_MASK) : reinterpret_cast<const uint &>(v_r);
    return reinterpret_cast<const __half2 &>(vi_r);
}

template<typename T>
__device__ __forceinline__ T __select(const uint32_t &cond, const T &a, const T &b)
{
    static_assert(sizeof(T) == 4, "Select only works on 32-bit types");
    int rval;
    asm volatile("lop3.b32 %0, %1, %2, %3, 202;"
                 : "=r"(rval)
                 : "r"(reinterpret_cast<const uint32_t &>(cond)),
                   "r"(reinterpret_cast<const uint32_t &>(a)),
                   "r"(reinterpret_cast<const uint32_t &>(b)));
    return reinterpret_cast<const T &>(rval);
}

__device__ __forceinline__ __half2 __h2_copy_sign(const __half2 &v, const __half2 &sign)
{
    return __select(0x80008000u, sign, v);
}

__device__ __forceinline__ __half __h_copy_sign(const __half &v, const __half &sign)
{
    return __low2half(
        __h2_copy_sign(__halves2half2(v, __float2half_rn(0.0f)), __halves2half2(sign, __float2half_rn(0.0f)))
    );
}

// Approximate exp2 on packed half2 via PTX ex2.approx.f16x2.
__forceinline__ __device__ __half2 __h2exp2_approx(__half2 v)
{
    uint32_t rval;
    asm volatile("ex2.approx.f16x2 %0, %1;\n" : "=r"(rval) : "r"(reinterpret_cast<const uint32_t &>(v)));
    return reinterpret_cast<const half2 &>(rval);
}

// Bitwise select on packed half2: result = cond ? a : b (per-bit, via lop3 truth table 202).
__device__ __forceinline__ __half2 __h2if(const uint32_t &cond, const __half2 &a, const __half2 &b)
{
    uint32_t rval;
    asm("lop3.b32 %0, %1, %2, %3, 202;"
        : "=r"(rval)
        : "r"(cond), "r"(reinterpret_cast<const uint32_t &>(a)), "r"(reinterpret_cast<const uint32_t &>(b)));
    return reinterpret_cast<const __half2 &>(rval);
}

// PTX prmt.b32 — byte-permute instruction.
__forceinline__ __device__ uint32_t __prmt_idx(uint32_t src0, uint32_t src1, uint32_t mask)
{
    uint32_t rval;
    asm volatile("prmt.b32 %0, %1, %2, %3;\n" : "=r"(rval) : "r"(src0), "r"(src1), "r"(mask));
    return rval;
}

// Convert 4 unsigned uint8 values packed in a uint32 to 4 fp16 values.
// Uses prmt.b32 to construct fp16 bit patterns directly in registers.
__forceinline__ __device__ void __uint8x4_to_half4(half *dst, uint32_t src)
{
    constexpr int FP16_BASE_LOG2 = 10;
    constexpr float FP16_BASE_FP = 1 << FP16_BASE_LOG2;
    constexpr int FP16_BASE      = (15 + FP16_BASE_LOG2) << 10; // 1024.0f in fp16.

    union
    {
        uint32_t i;
        half2 f;
    } dp[2];

    dp[0].i = __prmt_idx(src, FP16_BASE, 0x5150);
    dp[1].i = __prmt_idx(src, FP16_BASE, 0x5352);
    dp[0].f = __hsub2(dp[0].f, __float2half2_rn(FP16_BASE_FP));
    dp[1].f = __hsub2(dp[1].f, __float2half2_rn(FP16_BASE_FP));
    AssignAs<uint2>(dst[0], dp);
}

// Convert 2 unsigned uint8 values (low 16 bits of src) to half2.
// Same technique as __uint8x4_to_half4 but only the lower two bytes.
__forceinline__ __device__ __half2 __uint8x2_to_half2(uint32_t src)
{
    constexpr int FP16_BASE_LOG2 = 10;
    constexpr float FP16_BASE_FP = 1 << FP16_BASE_LOG2;
    constexpr int FP16_BASE      = (15 + FP16_BASE_LOG2) << 10;

    union
    {
        uint32_t i;
        half2 f;
    } dp;

    dp.i = __prmt_idx(src, FP16_BASE, 0x5150);
    dp.f = __hsub2(dp.f, __float2half2_rn(FP16_BASE_FP));
    return dp.f;
}

__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(float a0, float a1, float a2, float a3)
{
    uint rval;
    asm volatile(
        "{.reg .u32 tmp, i0, i1;\n"
        "cvt.rni.s32.f32 i0, %1;\n"
        "cvt.rni.s32.f32 i1, %2;\n"
        "cvt.pack.sat.u8.s32.b32 tmp, i0, i1, 0;\n"
        "cvt.rni.s32.f32 i0, %3;\n"
        "cvt.rni.s32.f32 i1, %4;\n"
        "cvt.pack.sat.u8.s32.b32 %0, i0, i1, tmp;}\n"
        : "=r"(rval)
        : "f"(a3), "f"(a2), "f"(a1), "f"(a0)
    );
    return rval;
}

__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(const float4 &a)
{
    return __cvt_pack_sat_u8_f32(a.x, a.y, a.z, a.w);
}

#endif // USE_ROCM / __HIP_PLATFORM_AMD__
