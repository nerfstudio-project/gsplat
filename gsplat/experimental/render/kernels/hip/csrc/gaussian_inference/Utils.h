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

#include <hip/hip_fp16.h>
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
    const uint32_t abits = reinterpret_cast<const uint32_t &>(a);
    const uint32_t bbits = reinterpret_cast<const uint32_t &>(b);
    const uint32_t rval  = (cond & abits) | (~cond & bbits);
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

__forceinline__ __device__ __half2 __h2exp2_approx(__half2 v)
{
    return __floats2half2_rn(exp2f(__low2float(v)), exp2f(__high2float(v)));
}

__forceinline__ __device__ uint32_t __prmt_idx(uint32_t src0, uint32_t src1, uint32_t mask)
{
    uint32_t rval = 0;
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        const uint32_t sel = (mask >> (4 * i)) & 0xFu;
        uint32_t byte      = 0;
        if(sel < 4)
        {
            byte = (src0 >> (8 * sel)) & 0xFFu;
        }
        else if(sel < 8)
        {
            byte = (src1 >> (8 * (sel - 4))) & 0xFFu;
        }
        rval |= byte << (8 * i);
    }
    return rval;
}

__forceinline__ __device__ void __uint8x4_to_half4(half *dst, uint32_t src)
{
    dst[0] = __float2half_rn(static_cast<float>(src & 0xFFu));
    dst[1] = __float2half_rn(static_cast<float>((src >> 8) & 0xFFu));
    dst[2] = __float2half_rn(static_cast<float>((src >> 16) & 0xFFu));
    dst[3] = __float2half_rn(static_cast<float>((src >> 24) & 0xFFu));
}

__forceinline__ __device__ __half2 __uint8x2_to_half2(uint32_t src)
{
    return __floats2half2_rn(static_cast<float>(src & 0xFFu), static_cast<float>((src >> 8) & 0xFFu));
}


__device__ __forceinline__ __half2 __h2if(const uint32_t &cond, const __half2 &a, const __half2 &b)
{
    return __select(cond, a, b);
}
__device__ __forceinline__ uint32_t __half2_mask(bool lo, bool hi)
{
    return (lo ? 0x0000FFFFu : 0u) | (hi ? 0xFFFF0000u : 0u);
}

__device__ __forceinline__ uint32_t __hlt2_mask(const __half2 &a, const __half2 &b)
{
    return __half2_mask(__low2float(a) < __low2float(b), __high2float(a) < __high2float(b));
}

__device__ __forceinline__ uint32_t __hle2_mask(const __half2 &a, const __half2 &b)
{
    return __half2_mask(__low2float(a) <= __low2float(b), __high2float(a) <= __high2float(b));
}

__device__ __forceinline__ uint32_t __hgeu2_mask(const __half2 &a, const __half2 &b)
{
    return __half2_mask(__low2float(a) >= __low2float(b), __high2float(a) >= __high2float(b));
}

__device__ __forceinline__ __half2 __hmax2(const __half2 &a, const __half2 &b)
{
    return __floats2half2_rn(fmaxf(__low2float(a), __low2float(b)), fmaxf(__high2float(a), __high2float(b)));
}

__device__ __forceinline__ __half2 __hmin2(const __half2 &a, const __half2 &b)
{
    return __floats2half2_rn(fminf(__low2float(a), __low2float(b)), fminf(__high2float(a), __high2float(b)));
}

__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(float a0, float a1, float a2, float a3)
{
    auto pack_one = [] __device__ (float x) -> uint32_t {
        int v = __float2int_rn(x);
        v = max(0, min(255, v));
        return static_cast<uint32_t>(v);
    };
    return pack_one(a0) | (pack_one(a1) << 8) | (pack_one(a2) << 16) | (pack_one(a3) << 24);
}

__forceinline__ __device__ uint __cvt_pack_sat_u8_f32(const float4 &a)
{
    return __cvt_pack_sat_u8_f32(a.x, a.y, a.z, a.w);
}
