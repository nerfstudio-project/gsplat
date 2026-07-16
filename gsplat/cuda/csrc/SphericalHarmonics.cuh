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

#include <ATen/cuda/Atomic.cuh>

#include "Common.h"

namespace gsplat
{
__device__ __forceinline__ vec3 view_direction_from_world_to_camera(
    const float *__restrict__ mean, const float *__restrict__ viewmat, const float *__restrict__ viewmat_rs = nullptr
)
{
    const float tx = viewmat[3];
    const float ty = viewmat[7];
    const float tz = viewmat[11];
    vec3 camera_offset(
        viewmat[0] * tx + viewmat[4] * ty + viewmat[8] * tz,
        viewmat[1] * tx + viewmat[5] * ty + viewmat[9] * tz,
        viewmat[2] * tx + viewmat[6] * ty + viewmat[10] * tz
    );
    if(viewmat_rs != nullptr)
    {
        const float tx_rs  = viewmat_rs[3];
        const float ty_rs  = viewmat_rs[7];
        const float tz_rs  = viewmat_rs[11];
        camera_offset     += vec3(
            viewmat_rs[0] * tx_rs + viewmat_rs[4] * ty_rs + viewmat_rs[8] * tz_rs,
            viewmat_rs[1] * tx_rs + viewmat_rs[5] * ty_rs + viewmat_rs[9] * tz_rs,
            viewmat_rs[2] * tx_rs + viewmat_rs[6] * ty_rs + viewmat_rs[10] * tz_rs
        );
        camera_offset *= 0.5f;
    }
    return vec3(mean[0], mean[1], mean[2]) + camera_offset;
}

__device__ __forceinline__ void accumulate_viewmat_vjp(
    const vec3 &v_dir, const float scale, const float *__restrict__ viewmat, float *__restrict__ v_viewmat
)
{
    if(viewmat == nullptr || v_viewmat == nullptr)
    {
        return;
    }

    const float v[3] = {scale * v_dir.x, scale * v_dir.y, scale * v_dir.z};
#pragma unroll
    for(int row = 0; row < 3; ++row)
    {
        const float t = viewmat[row * 4 + 3];
#pragma unroll
        for(int col = 0; col < 3; ++col)
        {
            atomicAdd_system(v_viewmat + row * 4 + col, t * v[col]);
        }
        atomicAdd_system(
            v_viewmat + row * 4 + 3, viewmat[row * 4] * v[0] + viewmat[row * 4 + 1] * v[1] + viewmat[row * 4 + 2] * v[2]
        );
    }
}

__device__ __forceinline__ void accumulate_view_direction_vjp(
    const vec3 &v_dir,
    const float *__restrict__ viewmat,
    float *__restrict__ v_mean,
    float *__restrict__ v_viewmat,
    const float *__restrict__ viewmat_rs = nullptr,
    float *__restrict__ v_viewmat_rs     = nullptr
)
{
    if(v_mean != nullptr)
    {
        gpuAtomicAdd(v_mean, v_dir.x);
        gpuAtomicAdd(v_mean + 1, v_dir.y);
        gpuAtomicAdd(v_mean + 2, v_dir.z);
    }
    const float viewmat_scale = viewmat_rs == nullptr ? 1.f : 0.5f;
    accumulate_viewmat_vjp(v_dir, viewmat_scale, viewmat, v_viewmat);
    accumulate_viewmat_vjp(v_dir, 0.5f, viewmat_rs, v_viewmat_rs);
}
} // namespace gsplat
