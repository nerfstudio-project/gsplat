/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <glm/gtc/type_ptr.hpp>

namespace glm
{
// GLM's pointer helpers use memcpy. In CUDA device code that can lower to byte
// loads, so provide restricted float pointer overloads for the hot gsplat paths.
// Use aligned float4 device loads only when the particular pointer is 16-byte
// aligned; otherwise fall back to scalar indexing so odd storage offsets remain
// safe on device.
GLM_FUNC_QUALIFIER vec<2, float, defaultp> make_vec2(const float *__restrict__ ptr)
{
    return vec<2, float, defaultp>(ptr[0], ptr[1]);
}

GLM_FUNC_QUALIFIER vec<3, float, defaultp> make_vec3(const float *__restrict__ ptr)
{
    return vec<3, float, defaultp>(ptr[0], ptr[1], ptr[2]);
}

GLM_FUNC_QUALIFIER vec<4, float, defaultp> make_vec4(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 v                  = ptr4[0];
        return vec<4, float, defaultp>(v.x, v.y, v.z, v.w);
    }
#endif
    return vec<4, float, defaultp>(ptr[0], ptr[1], ptr[2], ptr[3]);
}

GLM_FUNC_QUALIFIER mat<2, 2, float, defaultp> make_mat2(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 v                  = ptr4[0];
        return mat<2, 2, float, defaultp>(v.x, v.y, v.z, v.w);
    }
#endif
    return mat<2, 2, float, defaultp>(ptr[0], ptr[1], ptr[2], ptr[3]);
}

GLM_FUNC_QUALIFIER mat<3, 3, float, defaultp> make_mat3(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 c0c1x              = ptr4[0];
        const float4 c1yzc2xy           = ptr4[1];
        return mat<3, 3, float, defaultp>(
            c0c1x.x, c0c1x.y, c0c1x.z, c0c1x.w, c1yzc2xy.x, c1yzc2xy.y, c1yzc2xy.z, c1yzc2xy.w, ptr[8]
        );
    }
#endif
    return mat<3, 3, float, defaultp>(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8]);
}

GLM_FUNC_QUALIFIER mat<3, 4, float, defaultp> make_mat3x4(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 c0                 = ptr4[0];
        const float4 c1                 = ptr4[1];
        const float4 c2                 = ptr4[2];
        return mat<3, 4, float, defaultp>(c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w);
    }
#endif
    return mat<3, 4, float, defaultp>(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11]
    );
}

GLM_FUNC_QUALIFIER mat<4, 3, float, defaultp> make_mat4x3(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 c0c1x              = ptr4[0];
        const float4 c1yzc2xy           = ptr4[1];
        const float4 c2zc3              = ptr4[2];
        return mat<4, 3, float, defaultp>(
            c0c1x.x,
            c0c1x.y,
            c0c1x.z,
            c0c1x.w,
            c1yzc2xy.x,
            c1yzc2xy.y,
            c1yzc2xy.z,
            c1yzc2xy.w,
            c2zc3.x,
            c2zc3.y,
            c2zc3.z,
            c2zc3.w
        );
    }
#endif
    return mat<4, 3, float, defaultp>(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9], ptr[10], ptr[11]
    );
}

GLM_FUNC_QUALIFIER mat<4, 4, float, defaultp> make_mat4x4(const float *__restrict__ ptr)
{
#if defined(__CUDA_ARCH__)
    if((reinterpret_cast<std::uintptr_t>(ptr) & 0xFu) == 0u)
    {
        const float4 *__restrict__ ptr4 = reinterpret_cast<const float4 *>(ptr);
        const float4 c0                 = ptr4[0];
        const float4 c1                 = ptr4[1];
        const float4 c2                 = ptr4[2];
        const float4 c3                 = ptr4[3];
        return mat<4, 4, float, defaultp>(
            c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w, c2.x, c2.y, c2.z, c2.w, c3.x, c3.y, c3.z, c3.w
        );
    }
#endif
    return mat<4, 4, float, defaultp>(
        ptr[0],
        ptr[1],
        ptr[2],
        ptr[3],
        ptr[4],
        ptr[5],
        ptr[6],
        ptr[7],
        ptr[8],
        ptr[9],
        ptr[10],
        ptr[11],
        ptr[12],
        ptr[13],
        ptr[14],
        ptr[15]
    );
}

GLM_FUNC_QUALIFIER mat<4, 4, float, defaultp> make_mat4(const float *__restrict__ ptr)
{
    return make_mat4x4(ptr);
}
} // namespace glm
