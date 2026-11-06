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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "Common.h"

namespace gsplat
{
namespace sh_cg = cooperative_groups;

__device__ __forceinline__ bool reduce_view_direction_channels(const int64_t elem_id, vec3 &v_dir)
{
    auto active_threads = sh_cg::coalesced_threads();
    auto elem_threads   = sh_cg::labeled_partition(active_threads, static_cast<int>(elem_id));
    v_dir.x             = sh_cg::reduce(elem_threads, v_dir.x, sh_cg::plus<float>());
    v_dir.y             = sh_cg::reduce(elem_threads, v_dir.y, sh_cg::plus<float>());
    v_dir.z             = sh_cg::reduce(elem_threads, v_dir.z, sh_cg::plus<float>());
    return elem_threads.thread_rank() == 0;
}

__device__ __forceinline__ vec3 camera_offset_from_world_to_camera(
    const float *__restrict__ viewmat, const float *__restrict__ viewmat_rs = nullptr
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
    return camera_offset;
}

__device__ __forceinline__ vec3
    view_direction_from_world_to_camera(const float *__restrict__ mean, const float *__restrict__ viewmat)
{
    return vec3(mean[0], mean[1], mean[2]) + camera_offset_from_world_to_camera(viewmat);
}

__device__ __forceinline__ vec3
    view_direction_from_camera_offset(const float *__restrict__ mean, const float *__restrict__ camera_offset)
{
    return vec3(mean[0] + camera_offset[0], mean[1] + camera_offset[1], mean[2] + camera_offset[2]);
}

__device__ __forceinline__ vec3 view_direction_from_camera_data(
    const float *__restrict__ mean, const float *__restrict__ viewmat, const float *__restrict__ camera_offset
)
{
    return camera_offset == nullptr ? view_direction_from_world_to_camera(mean, viewmat)
                                    : view_direction_from_camera_offset(mean, camera_offset);
}

__device__ __forceinline__ vec3 view_direction_from_camera_data(
    const float *__restrict__ mean,
    const float *__restrict__ viewmats,
    const float *__restrict__ camera_offsets,
    const int64_t view_id
)
{
    return view_direction_from_camera_data(
        mean, viewmats + view_id * 16, camera_offsets == nullptr ? nullptr : camera_offsets + view_id * 3
    );
}
} // namespace gsplat
