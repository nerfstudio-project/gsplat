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
