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
#include "csrc/gaussian_scene_pack.cuh"

std::tuple<at::Tensor, at::Tensor, at::Tensor> pack_gaussian_inference_scene(
    const at::Tensor &means,
    const at::Tensor &quats,
    const at::Tensor &scales,
    const at::Tensor &opacities,
    const at::Tensor &colors,
    int64_t sh_degree,
    int64_t sh_compression_mode
) {
    return gsplat::scene::pack_gaussian_inference_scene_cuda(
        means, quats, scales, opacities, colors, sh_degree, sh_compression_mode
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
