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

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "gaussian_scene_pack.cuh"

// Extern declarations of the env-map sampling CUDA launchers. These are defined
// in csrc/env_map_sample.cu; that header carries __device__ code, so it is not
// included directly into this host translation unit.
namespace gsplat
{
namespace scene
{
    void sample_env_map_equirect_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out);
    void sample_env_map_equirect_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    );
    void sample_env_map_cubemap_fwd_cuda(const at::Tensor &rays_d, const at::Tensor &textures, at::Tensor &out);
    void sample_env_map_cubemap_bwd_cuda(
        const at::Tensor &rays_d, const at::Tensor &textures, const at::Tensor &grad_out, at::Tensor &grad_textures
    );
} // namespace scene
} // namespace gsplat

#define CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x)     TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_INPUT(x)    \
    CHECK_CUDA_TENSOR(x); \
    CHECK_CONTIGUOUS(x);  \
    CHECK_FLOAT32(x)

std::tuple<at::Tensor, at::Tensor, at::Tensor> pack_gaussian_inference_scene(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    int64_t sh_degree,
    int64_t sh_compression_mode
)
{
    return gsplat::scene::pack_gaussian_inference_scene_cuda(
        means, quats, scales, opacities, colors, sh_degree, sh_compression_mode
    );
}

// -----------------------------------------------------------------------------
// Environment-map background sampling (equirectangular / cubemap). Allocating
// torch-facing wrappers around the CUDA launchers. rays_d is assumed to be
// unit-normalized by the caller.
// -----------------------------------------------------------------------------

static void check_env_map_rays(const at::Tensor &rays_d)
{
    CHECK_INPUT(rays_d);
    TORCH_CHECK(rays_d.dim() == 2 && rays_d.size(1) == 3, "rays_d must be (N, 3)");
}

torch::Tensor sample_env_map_equirect_fwd(const torch::Tensor &rays_d, const torch::Tensor &textures)
{
    // Set the active CUDA device to the inputs' device (parity with the sibling
    // pack op); required for correct allocation/launch under multi-GPU-in-process.
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(rays_d));
    check_env_map_rays(rays_d);
    CHECK_INPUT(textures);
    TORCH_CHECK(textures.dim() == 4 && textures.size(0) == 1 && textures.size(3) == 3, "textures must be (1, H, W, 3)");
    TORCH_CHECK(rays_d.device() == textures.device(), "rays_d and textures must be on the same device");
    const int64_t n = rays_d.size(0);
    auto out        = torch::empty({n, 3}, rays_d.options());
    gsplat::scene::sample_env_map_equirect_fwd_cuda(rays_d, textures, out);
    return out;
}

torch::Tensor sample_env_map_equirect_bwd(
    const torch::Tensor &rays_d, const torch::Tensor &textures, const torch::Tensor &grad_out
)
{
    // Set the active CUDA device to the inputs' device (parity with the sibling
    // pack op); required for correct allocation/launch under multi-GPU-in-process.
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(rays_d));
    check_env_map_rays(rays_d);
    CHECK_INPUT(textures);
    CHECK_INPUT(grad_out);
    TORCH_CHECK(textures.dim() == 4 && textures.size(0) == 1 && textures.size(3) == 3, "textures must be (1, H, W, 3)");
    TORCH_CHECK(
        grad_out.dim() == 2 && grad_out.size(0) == rays_d.size(0) && grad_out.size(1) == 3, "grad_out must be (N, 3)"
    );
    TORCH_CHECK(rays_d.device() == textures.device(), "rays_d and textures must be on the same device");
    TORCH_CHECK(grad_out.device() == textures.device(), "grad_out and textures must be on the same device");
    // Zero-initialized: the backward kernel accumulates with atomicAdd.
    auto grad_textures = torch::zeros_like(textures);
    gsplat::scene::sample_env_map_equirect_bwd_cuda(rays_d, textures, grad_out, grad_textures);
    return grad_textures;
}

torch::Tensor sample_env_map_cubemap_fwd(const torch::Tensor &rays_d, const torch::Tensor &textures)
{
    // Set the active CUDA device to the inputs' device (parity with the sibling
    // pack op); required for correct allocation/launch under multi-GPU-in-process.
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(rays_d));
    check_env_map_rays(rays_d);
    CHECK_INPUT(textures);
    TORCH_CHECK(
        textures.dim() == 5 && textures.size(0) == 1 && textures.size(1) == 6 && textures.size(4) == 3,
        "textures must be (1, 6, H, W, 3)"
    );
    TORCH_CHECK(textures.size(2) == textures.size(3), "cubemap textures require W == H");
    TORCH_CHECK(rays_d.device() == textures.device(), "rays_d and textures must be on the same device");
    const int64_t n = rays_d.size(0);
    auto out        = torch::empty({n, 3}, rays_d.options());
    gsplat::scene::sample_env_map_cubemap_fwd_cuda(rays_d, textures, out);
    return out;
}

torch::Tensor sample_env_map_cubemap_bwd(
    const torch::Tensor &rays_d, const torch::Tensor &textures, const torch::Tensor &grad_out
)
{
    // Set the active CUDA device to the inputs' device (parity with the sibling
    // pack op); required for correct allocation/launch under multi-GPU-in-process.
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(rays_d));
    check_env_map_rays(rays_d);
    CHECK_INPUT(textures);
    CHECK_INPUT(grad_out);
    TORCH_CHECK(
        textures.dim() == 5 && textures.size(0) == 1 && textures.size(1) == 6 && textures.size(4) == 3,
        "textures must be (1, 6, H, W, 3)"
    );
    TORCH_CHECK(textures.size(2) == textures.size(3), "cubemap textures require W == H");
    TORCH_CHECK(
        grad_out.dim() == 2 && grad_out.size(0) == rays_d.size(0) && grad_out.size(1) == 3, "grad_out must be (N, 3)"
    );
    TORCH_CHECK(rays_d.device() == textures.device(), "rays_d and textures must be on the same device");
    TORCH_CHECK(grad_out.device() == textures.device(), "grad_out and textures must be on the same device");
    // Zero-initialized: the backward kernel accumulates with atomicAdd.
    auto grad_textures = torch::zeros_like(textures);
    gsplat::scene::sample_env_map_cubemap_bwd_cuda(rays_d, textures, grad_out, grad_textures);
    return grad_textures;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "pack_gaussian_inference_scene",
        &pack_gaussian_inference_scene,
        "Pack activated Gaussian splat tensors into the inference viewer-internal layout.\n\n"
        "Args:\n"
        "    means      [N, 3] float32 CUDA\n"
        "    quats      [N, 4] float32 CUDA  (unit quaternions in wxyz order)\n"
        "    scales     [N, 3] float32 CUDA  (positive, after exp())\n"
        "    opacities  [N]    float32 CUDA  (in [0,1], after sigmoid())\n"
        "    colors     [N, 3] or [N, K, 3] float32 CUDA\n"
        "    sh_degree        int  (-1=RGB, 0-3=SH)\n"
        "    sh_compression_mode  int  (0=none, 1=32b, 2=16b)\n\n"
        "Returns:\n"
        "    (means_planar [3,N] float32,\n"
        "     qso_packed [N,8] float16,\n"
        "     colors_packed [N,4] fp16 RGB, [N,K,3] fp32 SH0-2,\n"
        "                   [N,16,3] fp16 SH3 none, [N,48] fp32 SH3 32b,\n"
        "                   or [N,48] fp16 SH3 16b)"
    );
    m.def(
        "sample_env_map_equirect_fwd",
        &sample_env_map_equirect_fwd,
        "Sample an equirectangular env-map texture (1,H,W,3) for unit rays_d (N,3) -> out (N,3), CUDA. "
        "Matches F.grid_sample(bilinear, align_corners=False, border) with antimeridian wrap."
    );
    m.def(
        "sample_env_map_equirect_bwd",
        &sample_env_map_equirect_bwd,
        "Backward for sample_env_map_equirect: grad_out (N,3) -> grad_textures (1,H,W,3), CUDA"
    );
    m.def(
        "sample_env_map_cubemap_fwd",
        &sample_env_map_cubemap_fwd,
        "Sample a cubemap env-map texture (1,6,H,W,3), W==H, for unit rays_d (N,3) -> out (N,3), CUDA. "
        "Uses OpenGL dominant-axis face routing (0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z)."
    );
    m.def(
        "sample_env_map_cubemap_bwd",
        &sample_env_map_cubemap_bwd,
        "Backward for sample_env_map_cubemap: grad_out (N,3) -> grad_textures (1,6,H,W,3), CUDA"
    );
}
