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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>

#include "env_map_sample.cuh"

constexpr int kThreadsPerBlock = 256;

namespace
{
// One thread per ray i in [0, n): sample equirectangular texture into out (N,3).
template<typename scalar_t>
__global__ void sample_env_map_equirect_fwd_kernel(
    int64_t n,
    const scalar_t *__restrict__ rays_d,
    const scalar_t *__restrict__ textures,
    int H,
    int W,
    scalar_t *__restrict__ out
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    gsplat::scene::env_map_equirect_fwd_device(i, rays_d, textures, H, W, out);
}

// One thread per ray: scatter grad_out (N,3) into grad_textures via atomicAdd.
template<typename scalar_t>
__global__ void sample_env_map_equirect_bwd_kernel(
    int64_t n,
    const scalar_t *__restrict__ rays_d,
    int H,
    int W,
    const scalar_t *__restrict__ grad_out,
    scalar_t *__restrict__ grad_textures
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    gsplat::scene::env_map_equirect_bwd_device(i, rays_d, H, W, grad_out, grad_textures);
}

// One thread per ray i in [0, n): sample cubemap texture into out (N,3).
template<typename scalar_t>
__global__ void sample_env_map_cubemap_fwd_kernel(
    int64_t n,
    const scalar_t *__restrict__ rays_d,
    const scalar_t *__restrict__ textures,
    int H,
    int W,
    scalar_t *__restrict__ out
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    gsplat::scene::env_map_cubemap_fwd_device(i, rays_d, textures, H, W, out);
}

// One thread per ray: scatter grad_out (N,3) into grad_textures via atomicAdd.
template<typename scalar_t>
__global__ void sample_env_map_cubemap_bwd_kernel(
    int64_t n,
    const scalar_t *__restrict__ rays_d,
    int H,
    int W,
    const scalar_t *__restrict__ grad_out,
    scalar_t *__restrict__ grad_textures
)
{
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(i >= n)
    {
        return;
    }
    gsplat::scene::env_map_cubemap_bwd_device(i, rays_d, H, W, grad_out, grad_textures);
}

// Launch equirectangular forward sampling on the current CUDA stream.
// Inputs: rays_d (N,3) unit directions and textures (1,H,W,3).
// Output: out (N,3) raw (pre-activation) radiance.
template<typename scalar_t>
void launch_sample_env_map_equirect_fwd(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out)
{
    const int64_t n = rays_d.size(0);
    if(n == 0)
    {
        return;
    }
    const int H               = static_cast<int>(textures.size(1));
    const int W               = static_cast<int>(textures.size(2));
    const int threads         = kThreadsPerBlock;
    const int blocks          = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rays_d.device().index());
    sample_env_map_equirect_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, rays_d.data_ptr<scalar_t>(), textures.data_ptr<scalar_t>(), H, W, out.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for equirectangular sampling.
// Inputs: rays_d (N,3), textures (1,H,W,3) for shape, and grad_out (N,3).
// Output: grad_textures (1,H,W,3), accumulated with atomicAdd (must be zeroed).
template<typename scalar_t>
void launch_sample_env_map_equirect_bwd(
    const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
)
{
    const int64_t n = rays_d.size(0);
    if(n == 0)
    {
        return;
    }
    const int H               = static_cast<int>(textures.size(1));
    const int W               = static_cast<int>(textures.size(2));
    const int threads         = kThreadsPerBlock;
    const int blocks          = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rays_d.device().index());
    sample_env_map_equirect_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, rays_d.data_ptr<scalar_t>(), H, W, grad_out.data_ptr<scalar_t>(), grad_textures.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch cubemap forward sampling on the current CUDA stream.
// Inputs: rays_d (N,3) unit directions and textures (1,6,H,W,3), W == H.
// Output: out (N,3) raw (pre-activation) radiance.
template<typename scalar_t>
void launch_sample_env_map_cubemap_fwd(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out)
{
    const int64_t n = rays_d.size(0);
    if(n == 0)
    {
        return;
    }
    const int H               = static_cast<int>(textures.size(2));
    const int W               = static_cast<int>(textures.size(3));
    const int threads         = kThreadsPerBlock;
    const int blocks          = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rays_d.device().index());
    sample_env_map_cubemap_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, rays_d.data_ptr<scalar_t>(), textures.data_ptr<scalar_t>(), H, W, out.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Launch the VJP for cubemap sampling.
// Inputs: rays_d (N,3), textures (1,6,H,W,3) for shape, and grad_out (N,3).
// Output: grad_textures (1,6,H,W,3), accumulated with atomicAdd (must be zeroed).
template<typename scalar_t>
void launch_sample_env_map_cubemap_bwd(
    const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
)
{
    const int64_t n = rays_d.size(0);
    if(n == 0)
    {
        return;
    }
    const int H               = static_cast<int>(textures.size(2));
    const int W               = static_cast<int>(textures.size(3));
    const int threads         = kThreadsPerBlock;
    const int blocks          = static_cast<int>((n + threads - 1) / threads);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream(rays_d.device().index());
    sample_env_map_cubemap_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        n, rays_d.data_ptr<scalar_t>(), H, W, grad_out.data_ptr<scalar_t>(), grad_textures.data_ptr<scalar_t>()
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

namespace gsplat
{
namespace scene
{
    // Public CUDA entrypoint for equirectangular env-map sampling.
    // Inputs: rays_d (N,3) unit directions and textures (1,H,W,3).
    // Output: out (N,3) raw (pre-activation) radiance.
    void sample_env_map_equirect_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out)
    {
        // Inputs are float32-only (enforced by CHECK_FLOAT32 in the bindings), so
        // dispatch directly on float instead of instantiating a dead float64 path.
        launch_sample_env_map_equirect_fwd<float>(rays_d, textures, out);
    }

    // Public CUDA entrypoint for the VJP of equirectangular env-map sampling.
    // Inputs: rays_d (N,3), textures (1,H,W,3), and grad_out (N,3).
    // Output: grad_textures (1,H,W,3), zero-initialized by the caller.
    void sample_env_map_equirect_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    )
    {
        // Inputs are float32-only (enforced by CHECK_FLOAT32 in the bindings), so
        // dispatch directly on float instead of instantiating a dead float64 path.
        launch_sample_env_map_equirect_bwd<float>(rays_d, textures, grad_out, grad_textures);
    }

    // Public CUDA entrypoint for cubemap env-map sampling.
    // Inputs: rays_d (N,3) unit directions and textures (1,6,H,W,3), W == H.
    // Output: out (N,3) raw (pre-activation) radiance.
    void sample_env_map_cubemap_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out)
    {
        // Inputs are float32-only (enforced by CHECK_FLOAT32 in the bindings), so
        // dispatch directly on float instead of instantiating a dead float64 path.
        launch_sample_env_map_cubemap_fwd<float>(rays_d, textures, out);
    }

    // Public CUDA entrypoint for the VJP of cubemap env-map sampling.
    // Inputs: rays_d (N,3), textures (1,6,H,W,3), and grad_out (N,3).
    // Output: grad_textures (1,6,H,W,3), zero-initialized by the caller.
    void sample_env_map_cubemap_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    )
    {
        // Inputs are float32-only (enforced by CHECK_FLOAT32 in the bindings), so
        // dispatch directly on float instead of instantiating a dead float64 path.
        launch_sample_env_map_cubemap_bwd<float>(rays_d, textures, grad_out, grad_textures);
    }
} // namespace scene
} // namespace gsplat
