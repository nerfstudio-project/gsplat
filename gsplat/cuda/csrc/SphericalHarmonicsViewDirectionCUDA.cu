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

#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/ops/empty.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/zeros.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>

#include "SphericalHarmonics.h"
#include "SphericalHarmonics.cuh"
#include "Utils.cuh"

namespace gsplat
{
namespace cg = cooperative_groups;

namespace
{
    constexpr unsigned int kShReductionThreads     = 256;
    constexpr int64_t kShReductionItemsPerThread   = 8;
    constexpr int64_t kShReductionTileSize         = kShReductionThreads * kShReductionItemsPerThread;
    constexpr int64_t kShMinParallelReductionTiles = 4;

    __global__ __launch_bounds__(kShReductionThreads) void precompute_camera_offsets(
        const int64_t num_viewmats,
        const float *__restrict__ viewmats,
        const float *__restrict__ viewmats_rs,
        float *__restrict__ camera_offsets
    )
    {
        const int64_t image_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(image_id >= num_viewmats)
        {
            return;
        }

        const vec3 offset = camera_offset_from_world_to_camera(
            viewmats + image_id * 16, viewmats_rs == nullptr ? nullptr : viewmats_rs + image_id * 16
        );
        float *output = camera_offsets + image_id * 3;
        output[0]     = offset.x;
        output[1]     = offset.y;
        output[2]     = offset.z;
    }

    struct ViewDirectionGradient
    {
        float x;
        float y;
        float z;
    };

    struct AddViewDirectionGradients
    {
        __device__ ViewDirectionGradient
            operator()(const ViewDirectionGradient &lhs, const ViewDirectionGradient &rhs) const
        {
            return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
        }
    };

    template<ViewmatGradientUpdate Update>
    __device__ __forceinline__ void update_viewmat_gradient(float *__restrict__ output, const float value)
    {
        if constexpr(Update == ViewmatGradientUpdate::SystemAtomicAdd)
        {
            atomicAdd_system(output, value);
        }
        else
        {
            *output = value;
        }
    }

    template<ViewmatGradientUpdate Update>
    __device__ __forceinline__ void write_viewmat_vjp(
        const ViewDirectionGradient &v_dir,
        const float scale,
        const float *__restrict__ viewmat,
        float *__restrict__ v_viewmat
    )
    {
        const float vx = scale * v_dir.x;
        const float vy = scale * v_dir.y;
        const float vz = scale * v_dir.z;
#pragma unroll
        for(int row = 0; row < 3; ++row)
        {
            const float t = viewmat[row * 4 + 3];
            update_viewmat_gradient<Update>(v_viewmat + row * 4, t * vx);
            update_viewmat_gradient<Update>(v_viewmat + row * 4 + 1, t * vy);
            update_viewmat_gradient<Update>(v_viewmat + row * 4 + 2, t * vz);
            update_viewmat_gradient<Update>(
                v_viewmat + row * 4 + 3, viewmat[row * 4] * vx + viewmat[row * 4 + 1] * vy + viewmat[row * 4 + 2] * vz
            );
        }
    }

    template<ViewmatGradientUpdate Update>
    __global__ __launch_bounds__(kShReductionThreads) void reduce_unpacked_view_direction_gradients_to_viewmats(
        const int64_t N,
        const int64_t gaussian_offset,
        const int64_t gaussian_count,
        const float viewmat_scale,
        const float *__restrict__ viewmats,
        const float *__restrict__ viewmats_rs,
        const float *__restrict__ v_viewdirs,
        float *__restrict__ v_viewmats,
        float *__restrict__ v_viewmats_rs
    )
    {
        const int64_t image_id = blockIdx.x;
        ViewDirectionGradient local{0.f, 0.f, 0.f};
        for(int64_t local_gaussian_id  = threadIdx.x; local_gaussian_id < gaussian_count;
            local_gaussian_id         += blockDim.x)
        {
            const int64_t gaussian_id  = gaussian_offset + local_gaussian_id;
            const float *v_dir         = v_viewdirs + (image_id * N + gaussian_id) * 3;
            local.x                   += v_dir[0];
            local.y                   += v_dir[1];
            local.z                   += v_dir[2];
        }

        using BlockReduce = cub::BlockReduce<ViewDirectionGradient, kShReductionThreads>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        const ViewDirectionGradient reduced = BlockReduce(temp_storage).Reduce(local, AddViewDirectionGradients{});

        if(threadIdx.x == 0)
        {
            if(v_viewmats != nullptr)
            {
                write_viewmat_vjp<Update>(reduced, viewmat_scale, viewmats + image_id * 16, v_viewmats + image_id * 16);
            }
            if(v_viewmats_rs != nullptr)
            {
                write_viewmat_vjp<Update>(reduced, 0.5f, viewmats_rs + image_id * 16, v_viewmats_rs + image_id * 16);
            }
        }
    }

    __global__ __launch_bounds__(kShReductionThreads) void reduce_unpacked_view_direction_gradients_to_partials(
        const int64_t N,
        const int64_t gaussian_offset,
        const int64_t gaussian_count,
        const int64_t tiles_per_view,
        const float *__restrict__ v_viewdirs,
        float *__restrict__ partials
    )
    {
        const int64_t view_tile_id = blockIdx.x;
        const int64_t view_id      = view_tile_id / tiles_per_view;
        const int64_t tile_id      = view_tile_id % tiles_per_view;
        const int64_t tile_begin   = tile_id * kShReductionTileSize;
        const int64_t tile_limit   = tile_begin + kShReductionTileSize;
        const int64_t tile_end     = tile_limit < gaussian_count ? tile_limit : gaussian_count;

        ViewDirectionGradient local{0.f, 0.f, 0.f};
        for(int64_t local_gaussian_id  = tile_begin + threadIdx.x; local_gaussian_id < tile_end;
            local_gaussian_id         += blockDim.x)
        {
            const int64_t gaussian_id  = gaussian_offset + local_gaussian_id;
            const float *v_dir         = v_viewdirs + (view_id * N + gaussian_id) * 3;
            local.x                   += v_dir[0];
            local.y                   += v_dir[1];
            local.z                   += v_dir[2];
        }

        using BlockReduce = cub::BlockReduce<ViewDirectionGradient, kShReductionThreads>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        const ViewDirectionGradient reduced = BlockReduce(temp_storage).Reduce(local, AddViewDirectionGradients{});

        if(threadIdx.x == 0)
        {
            float *output = partials + view_tile_id * 3;
            output[0]     = reduced.x;
            output[1]     = reduced.y;
            output[2]     = reduced.z;
        }
    }

    __global__ __launch_bounds__(kShReductionThreads) void reduce_packed_view_direction_gradients(
        const int64_t E,
        const int64_t C,
        const int64_t *__restrict__ batch_ids,
        const int64_t *__restrict__ camera_ids,
        const float *__restrict__ v_viewdirs,
        float *__restrict__ reduced_v_viewdirs
    )
    {
        const int64_t elem_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(elem_id >= E)
        {
            return;
        }

        const int64_t image_id = batch_ids[elem_id] * C + camera_ids[elem_id];
        const float *v_dir     = v_viewdirs + elem_id * 3;
        ViewDirectionGradient local{v_dir[0], v_dir[1], v_dir[2]};
        auto active_threads = cg::coalesced_threads();
        auto image_threads  = cg::labeled_partition(active_threads, static_cast<int>(image_id));
        local.x             = cg::reduce(image_threads, local.x, cg::plus<float>());
        local.y             = cg::reduce(image_threads, local.y, cg::plus<float>());
        local.z             = cg::reduce(image_threads, local.z, cg::plus<float>());

        if(image_threads.thread_rank() == 0)
        {
            float *reduced_v_dir = reduced_v_viewdirs + image_id * 3;
            gpuAtomicAdd(reduced_v_dir, local.x);
            gpuAtomicAdd(reduced_v_dir + 1, local.y);
            gpuAtomicAdd(reduced_v_dir + 2, local.z);
        }
    }

    __global__ __launch_bounds__(kShReductionThreads) void view_direction_gradients_to_viewmats(
        const int64_t num_viewmats,
        const float viewmat_scale,
        const float *__restrict__ viewmats,
        const float *__restrict__ viewmats_rs,
        const float *__restrict__ reduced_v_viewdirs,
        float *__restrict__ v_viewmats,
        float *__restrict__ v_viewmats_rs
    )
    {
        const int64_t image_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(image_id >= num_viewmats)
        {
            return;
        }

        const float *v_dir = reduced_v_viewdirs + image_id * 3;
        const ViewDirectionGradient gradient{v_dir[0], v_dir[1], v_dir[2]};
        if(v_viewmats != nullptr)
        {
            write_viewmat_vjp<ViewmatGradientUpdate::Assign>(
                gradient, viewmat_scale, viewmats + image_id * 16, v_viewmats + image_id * 16
            );
        }
        if(v_viewmats_rs != nullptr)
        {
            write_viewmat_vjp<ViewmatGradientUpdate::Assign>(
                gradient, 0.5f, viewmats_rs + image_id * 16, v_viewmats_rs + image_id * 16
            );
        }
    }

    template<ViewmatGradientUpdate Update>
    void launch_unpacked_spherical_harmonics_view_direction_vjp_reduction(
        const int64_t N,
        const int64_t gaussian_offset,
        const int64_t gaussian_count,
        const float viewmat_scale,
        const at::Tensor &viewmats,
        const float *__restrict__ viewmats_rs,
        const at::Tensor &v_viewdirs,
        float *__restrict__ v_viewmats,
        float *__restrict__ v_viewmats_rs,
        const at::TensorOptions &partial_options,
        const c10::cuda::CUDAStream stream
    )
    {
        const int64_t num_viewmats   = viewmats.numel() / 16;
        const int64_t tiles_per_view = ::cuda::ceil_div<int64_t>(gaussian_count, kShReductionTileSize);

        // Below four tiles, the extra allocation and kernel launch cost more
        // than the parallelism they expose. Larger views use unique CTA
        // partial slots, avoiding atomics while allowing multiple SMs to
        // reduce one view.
        if(tiles_per_view >= kShMinParallelReductionTiles)
        {
            at::Tensor partials;
            if constexpr(Update == ViewmatGradientUpdate::SystemAtomicAdd)
            {
                // v_viewdirs is a PrivateUse1 tensor in this path, so allocate
                // stream-local CUDA scratch directly and expose only a tensor view.
                const size_t num_bytes
                    = static_cast<size_t>(num_viewmats) * static_cast<size_t>(tiles_per_view) * 3 * sizeof(float);
                void *data = nullptr;
                C10_CUDA_CHECK(cudaMallocAsync(&data, num_bytes, stream));
                partials = at::from_blob(
                    data,
                    {num_viewmats, tiles_per_view, 3},
                    [stream](void *ptr) { C10_CUDA_CHECK_WARN(cudaFreeAsync(ptr, stream)); },
                    at::TensorOptions().dtype(at::kFloat).device(stream.device())
                );
            }
            else
            {
                partials = at::empty({num_viewmats, tiles_per_view, 3}, partial_options);
            }
            const unsigned int partial_blocks = static_cast<unsigned int>(num_viewmats * tiles_per_view);
            reduce_unpacked_view_direction_gradients_to_partials<<<partial_blocks, kShReductionThreads, 0, stream>>>(
                N,
                gaussian_offset,
                gaussian_count,
                tiles_per_view,
                v_viewdirs.const_data_ptr<float>(),
                partials.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            reduce_unpacked_view_direction_gradients_to_viewmats<Update>
                <<<num_viewmats, kShReductionThreads, 0, stream>>>(
                    tiles_per_view,
                    0,
                    tiles_per_view,
                    viewmat_scale,
                    viewmats.const_data_ptr<float>(),
                    viewmats_rs,
                    partials.const_data_ptr<float>(),
                    v_viewmats,
                    v_viewmats_rs
                );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }

        reduce_unpacked_view_direction_gradients_to_viewmats<Update><<<num_viewmats, kShReductionThreads, 0, stream>>>(
            N,
            gaussian_offset,
            gaussian_count,
            viewmat_scale,
            viewmats.const_data_ptr<float>(),
            viewmats_rs,
            v_viewdirs.const_data_ptr<float>(),
            v_viewmats,
            v_viewmats_rs
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
} // namespace

at::Tensor precompute_spherical_harmonics_camera_offsets(
    const at::Tensor viewmats, const at::optional<at::Tensor> viewmats_rs
)
{
    auto stream                = at::cuda::getCurrentCUDAStream();
    const int64_t num_viewmats = viewmats.numel() / 16;
    at::Tensor camera_offsets  = at::empty({num_viewmats, 3}, viewmats.options());
    if(num_viewmats == 0)
    {
        return camera_offsets;
    }

    const unsigned int blocks = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(num_viewmats, kShReductionThreads));
    precompute_camera_offsets<<<blocks, kShReductionThreads, 0, stream>>>(
        num_viewmats,
        viewmats.const_data_ptr<float>(),
        viewmats_rs.has_value() ? viewmats_rs.value().const_data_ptr<float>() : nullptr,
        camera_offsets.data_ptr<float>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return camera_offsets;
}

template<ViewmatGradientUpdate Update>
void launch_spherical_harmonics_view_direction_vjp_reduction(
    const int64_t N,
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const at::Tensor viewmats,
    const at::Tensor v_viewdirs,
    at::optional<at::Tensor> v_viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    at::optional<at::Tensor> v_viewmats_rs
)
{
    auto stream = at::cuda::getCurrentCUDAStream();

    if constexpr(Update == ViewmatGradientUpdate::SystemAtomicAdd)
    {
        launch_unpacked_spherical_harmonics_view_direction_vjp_reduction<Update>(
            N,
            gaussian_offset,
            gaussian_count,
            1.f,
            viewmats,
            nullptr,
            v_viewdirs,
            v_viewmats.value().data_ptr<float>(),
            nullptr,
            at::TensorOptions{},
            stream
        );
    }
    else
    {
        const int64_t num_viewmats = viewmats.numel() / 16;
        const int64_t C            = viewmats.size(-3);
        const int64_t E            = v_viewdirs.numel() / 3;
        const float viewmat_scale  = viewmats_rs.has_value() ? 0.5f : 1.f;

        if(batch_ids.has_value())
        {
            at::Tensor reduced_v_viewdirs = at::zeros({num_viewmats, 3}, v_viewdirs.options());
            const unsigned int reduction_blocks
                = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(E, kShReductionThreads));
            reduce_packed_view_direction_gradients<<<reduction_blocks, kShReductionThreads, 0, stream>>>(
                E,
                C,
                batch_ids.value().const_data_ptr<int64_t>(),
                camera_ids.value().const_data_ptr<int64_t>(),
                v_viewdirs.const_data_ptr<float>(),
                reduced_v_viewdirs.data_ptr<float>()
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            const unsigned int viewmat_blocks
                = static_cast<unsigned int>(::cuda::ceil_div<int64_t>(num_viewmats, kShReductionThreads));
            view_direction_gradients_to_viewmats<<<viewmat_blocks, kShReductionThreads, 0, stream>>>(
                num_viewmats,
                viewmat_scale,
                viewmats.const_data_ptr<float>(),
                viewmats_rs.has_value() ? viewmats_rs.value().const_data_ptr<float>() : nullptr,
                reduced_v_viewdirs.const_data_ptr<float>(),
                v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>() : nullptr,
                v_viewmats_rs.has_value() ? v_viewmats_rs.value().data_ptr<float>() : nullptr
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }

        launch_unpacked_spherical_harmonics_view_direction_vjp_reduction<Update>(
            N,
            0,
            N,
            viewmat_scale,
            viewmats,
            viewmats_rs.has_value() ? viewmats_rs.value().const_data_ptr<float>() : nullptr,
            v_viewdirs,
            v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>() : nullptr,
            v_viewmats_rs.has_value() ? v_viewmats_rs.value().data_ptr<float>() : nullptr,
            v_viewdirs.options(),
            stream
        );
    }
}

template void launch_spherical_harmonics_view_direction_vjp_reduction<ViewmatGradientUpdate::Assign>(
    const int64_t N,
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const at::Tensor viewmats,
    const at::Tensor v_viewdirs,
    at::optional<at::Tensor> v_viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    at::optional<at::Tensor> v_viewmats_rs
);

template void launch_spherical_harmonics_view_direction_vjp_reduction<ViewmatGradientUpdate::SystemAtomicAdd>(
    const int64_t N,
    const int64_t gaussian_offset,
    const int64_t gaussian_count,
    const at::Tensor viewmats,
    const at::Tensor v_viewdirs,
    at::optional<at::Tensor> v_viewmats,
    const at::optional<at::Tensor> viewmats_rs,
    const at::optional<at::Tensor> batch_ids,
    const at::optional<at::Tensor> camera_ids,
    at::optional<at::Tensor> v_viewmats_rs
);
} // namespace gsplat
