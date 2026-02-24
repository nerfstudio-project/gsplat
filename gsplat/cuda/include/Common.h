/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

#include <algorithm>
#include <cstdint>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>

namespace gsplat {

//
// Some Macros.
//
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
    CHECK_CUDA(x);                                                             \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                                     \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...)                                                 \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

//
// Convenience typedefs for CUDA types
//
using vec2 = glm::vec<2, float>;
using vec3 = glm::vec<3, float>;
using vec4 = glm::vec<4, float>;
using mat2 = glm::mat<2, 2, float>;
using mat3 = glm::mat<3, 3, float>;
using mat4 = glm::mat<4, 4, float>;
using mat3x2 = glm::mat<3, 2, float>;

//
// Legacy Camera Types
//
enum CameraModelType {
    PINHOLE = 0,
    ORTHO = 1,
    FISHEYE = 2,
    FTHETA = 3,
    LIDAR = 4,
};

#define N_THREADS_PACKED 256
#define ALPHA_THRESHOLD (1.f / 255.f)
// MAX_ALPHA and TRANSMITTANCE_THRESHOLD are chosen so that the equivalent of
// a maximal opacity Gaussian has to be rasterized twice to reach the threshold,
// without getting the transmittance too small for numerical stability of
// the backward pass.
// i.e. TRANSMITTANCE_THRESHOLD = (1 - MAX_ALPHA)^2
#define MAX_ALPHA 0.99f
#define TRANSMITTANCE_THRESHOLD 1e-4f

#define MAX_KERNEL_DENSITY_CUTOFF 0.0113

#ifdef __CUDACC__
#   define GSPLAT_NOINLINE __noinline__
#else
#   define GSPLAT_NOINLINE
#endif

} // namespace gsplat
