/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <ATen/DeviceGuard.h>
#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <glm/geometric.hpp>
#include <glm/matrix.hpp>
#include <glm/gtx/matrix_operation.hpp>
#include <hip/hip_runtime.h>

#if defined(__HIPCC__)
#define hipFuncSetAttribute(...) hipSuccess
#endif

namespace gsplat
{
//
// Some Macros.
//
#define CHECK_DEVICE(x)     TORCH_CHECK(x.is_cuda() || x.is_privateuseone(), #x " must be a CUDA or PrivateUse1 tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_DEVICE(x);   \
    CHECK_CONTIGUOUS(x)
// Kernels index raw dense storage; a strided (non-sparse) layout is required.
#define CHECK_DENSE(x)     TORCH_CHECK(x.layout() == c10::kStrided, #x " must be a dense tensor")
#define DEVICE_GUARD(_ten) const at::OptionalDeviceGuard device_guard(device_of(_ten));

// Host/device qualifier for helpers shared between host (tests, host-side
// setup) and device code. Expands to nothing under a non-CUDA compiler.
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#    define GSPLAT_HOST_DEVICE __host__ __device__
#else
#    define GSPLAT_HOST_DEVICE
#endif

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.hip.h#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                    \
    do                                                                            \
    {                                                                             \
        size_t temp_storage_bytes = 0;                                            \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                           \
        auto &caching_allocator = *::c10::hip::HIPCachingAllocator::get();      \
        auto temp_storage       = caching_allocator.allocate(temp_storage_bytes); \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                \
    } while(false)

//
// Convenience typedefs for CUDA types
//
using vec2   = glm::vec<2, float>;
using vec3   = glm::vec<3, float>;
using vec4   = glm::vec<4, float>;
using mat2   = glm::mat<2, 2, float>;
using mat3   = glm::mat<3, 3, float>;
using mat4   = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;
using mat2x3 = glm::mat<2, 3, float>;

namespace detail
{
template<int C, int R, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::vec<R, T, Q> gsplat_matvec(const glm::mat<C, R, T, Q> &m, const glm::vec<C, T, Q> &v)
{
    glm::vec<R, T, Q> out(static_cast<T>(0));
    for(int c = 0; c < C; ++c)
    {
        out += m[c] * v[c];
    }
    return out;
}

template<int CA, int RA, int CB, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::mat<CB, RA, T, Q> gsplat_matmul(
    const glm::mat<CA, RA, T, Q> &a,
    const glm::mat<CB, CA, T, Q> &b
)
{
    glm::mat<CB, RA, T, Q> out(static_cast<T>(1));
    for(int c = 0; c < CB; ++c)
    {
        out[c] = gsplat_matvec(a, b[c]);
    }
    return out;
}

template<int C, int R, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::mat<R, C, T, Q> gsplat_transpose(const glm::mat<C, R, T, Q> &m)
{
    glm::mat<R, C, T, Q> out(static_cast<T>(1));
    for(int c = 0; c < C; ++c)
    {
        for(int r = 0; r < R; ++r)
        {
            out[r][c] = m[c][r];
        }
    }
    return out;
}

template<int C, int R, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::mat<R, C, T, Q> gsplat_outer_product(const glm::vec<C, T, Q> &cvec, const glm::vec<R, T, Q> &rvec)
{
    glm::mat<R, C, T, Q> out(static_cast<T>(1));
    for(int c = 0; c < R; ++c)
    {
        out[c] = cvec * rvec[c];
    }
    return out;
}

template<int N, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline T gsplat_dot(const glm::vec<N, T, Q> &a, const glm::vec<N, T, Q> &b)
{
    T out = static_cast<T>(0);
    for(int i = 0; i < N; ++i)
    {
        out += a[i] * b[i];
    }
    return out;
}



template<int C, int R, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::mat<C, R, T, Q> gsplat_matadd(const glm::mat<C, R, T, Q> &a, const glm::mat<C, R, T, Q> &b)
{
    glm::mat<C, R, T, Q> out(static_cast<T>(1));
    for(int c = 0; c < C; ++c)
    {
        out[c] = a[c] + b[c];
    }
    return out;
}template<int N, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline T gsplat_length(const glm::vec<N, T, Q> &v)
{
    return sqrtf(gsplat_dot(v, v));
}template<int C, int R, typename T, glm::qualifier Q>
GSPLAT_HOST_DEVICE inline glm::mat<C, R, T, Q> gsplat_scale_mat(const glm::mat<C, R, T, Q> &m, T s)
{
    glm::mat<C, R, T, Q> out(static_cast<T>(1));
    for(int c = 0; c < C; ++c)
    {
        out[c] = m[c] * s;
    }
    return out;
}
}

using detail::gsplat_dot;
using detail::gsplat_length;
using detail::gsplat_matadd;
using detail::gsplat_matmul;
using detail::gsplat_matvec;
using detail::gsplat_outer_product;
using detail::gsplat_scale_mat;
using detail::gsplat_transpose;
//
// Legacy Camera Types
//
enum CameraModelType
{
    PINHOLE = 0,
    ORTHO   = 1,
    FISHEYE = 2,
    FTHETA  = 3,
    LIDAR   = 4,
};

enum RendererConfig
{
    MIXED_BATCH    = 0,
    PARALLEL_BATCH = 1,
};

#define N_THREADS_PACKED 256

// CUDA caps grid.y (and grid.z) at 65535; only grid.x reaches 2^31 - 1. Kernels
// that map a batch (or batch-camera) dimension onto grid.y must reject launches
// that would exceed this.
constexpr uint32_t kMaxCudaGridDimY = 65535;

#define ALPHA_THRESHOLD         (1.f / 255.f)
// GAUSSIAN_EXTEND determines where the gaussian is truncated in standard deviations."
#define GAUSSIAN_EXTEND         3.33f
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// without getting the transmittance too small for numerical stability of
// the backward pass.
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA               0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f

// Floor for the antialiased compensation factor (sqrt(det_orig / det_blur)).
// Prevents compensation from reaching zero for extremely small Gaussians.
#define MIN_COMPENSATION 0.005f

// Floor for (1 - alpha) when computing 1/(1-alpha) in backward rasterization.
// Prevents gradient explosion when alpha approaches 1.0.
#define MIN_ONE_MINUS_ALPHA 1e-6f
} // namespace gsplat
