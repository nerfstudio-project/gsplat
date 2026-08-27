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

#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

// Multi-GPU communication for the classic 3DGS rasterization pipeline.
//
// The single-GPU and distributed paths share one orchestrator (`rasterization_3dgs`).
// Distribution is injected at exactly two seams: cameras are all-gathered before
// projection, and the projected Gaussians are all-to-all scattered to the
// camera-owning ranks before tiling. These two seam hooks own all of that; the
// orchestrator just calls them when a process group name is supplied.
namespace gsplat
{
// Cameras and per-rank sizes produced at the gather seam (before projection).
struct DistributedCameraGather
{
    at::Tensor viewmats;          // [global_C, 4, 4] cameras from all ranks
    at::Tensor Ks;                // [global_C, 3, 3]
    std::vector<int64_t> N_world; // per-rank Gaussian counts
    std::vector<int64_t> C_world; // per-rank camera counts
    int64_t global_C;             // sum(C_world)
};

// Projected per-Gaussian tensors exchanged at the scatter seam. In packed mode
// the index tensors are carried too; dense mode leaves them undefined.
struct DistributedProjection
{
    at::Tensor radii;
    at::Tensor means2d;
    at::Tensor depths;
    at::Tensor conics;
    at::Tensor opacities;
    at::Tensor features;     // packed [nnz, ch] / dense [C, N, ch]; may be undefined
    at::Tensor batch_ids;    // packed only
    at::Tensor camera_ids;   // packed only
    at::Tensor gaussian_ids; // packed only
};

// Seam A. All-gather the per-rank Gaussian/camera counts and the camera tensors
// so every rank can project its local Gaussians against all cameras. Each rank
// must render the same number of cameras.
DistributedCameraGather gather_cameras_for_distributed(
    const at::Tensor &viewmats,
    const at::Tensor &Ks,
    int64_t local_N,
    int64_t local_C,
    int64_t world_size,
    const std::string &process_group_name
);

// Seam B. All-to-all scatter the projected Gaussians to the ranks that own the
// cameras they are visible to. Remaps camera_ids global->local and gaussian_ids
// local->global (packed), or rebuilds the local camera-major layout (dense).
// The returned tensors carry the slice this rank rasterizes.
DistributedProjection scatter_projection_for_distributed(
    bool packed,
    const DistributedProjection &projection,
    const std::vector<int64_t> &C_world,
    const std::vector<int64_t> &N_world,
    int64_t local_C,
    int64_t local_N,
    int64_t global_C,
    int64_t world_size,
    const std::string &process_group_name
);
} // namespace gsplat
