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

#include "csrc/gaussian_inference/GaussianRenderInferenceScene.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer>(m, "GaussianInferenceRenderer")
        .def(py::init<const at::Tensor&, const at::Tensor&, const at::Tensor&, int64_t, int64_t>(),
             py::arg("means_planar"), py::arg("qso_packed"), py::arg("colors_packed"),
             py::arg("sh_degree"), py::arg("sh_compression_mode"))
        .def("render", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::render,
             py::arg("means_planar"), py::arg("qso_packed"), py::arg("colors_packed"),
             py::arg("viewmat"), py::arg("K"),
             py::arg("width"), py::arg("height"),
             py::arg("tile_size"),
             py::arg("near_plane"), py::arg("far_plane"),
             py::arg("radius_clip"), py::arg("eps2d"),
             py::arg("sh_degree"), py::arg("sh_compression_mode"),
             py::arg("background"), py::arg("out_rgbt"))
        .def("release", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::release)
        .def("num_gaussians", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::numGaussians)
        .def("is_released", &gsplat::gaussian_render_inference_scene::GaussianInferenceRenderer::isReleased);
}

TORCH_LIBRARY(experimental, m) {
    m.def("gaussian_render_inference_only(Tensor means_planar, Tensor qso_packed, Tensor colors_packed, Tensor viewmat, Tensor K, int width, int height, int sh_degree, int tile_size, float near_plane, float far_plane, float radius_clip, float eps2d, int sh_compression_mode, Tensor? background, *, Tensor(a!)? out_renders=None, Tensor(b!)? out_alphas=None) -> (Tensor renders, Tensor alphas)");
}

TORCH_LIBRARY_IMPL(experimental, CUDA, m) {
    m.impl("gaussian_render_inference_only", &gsplat::gaussian_render_inference_scene::gaussian_render_inference_only);
}

// Explicitly suppress AutogradCUDA for gaussian_render_inference_only: it is
// inference-only and has no backward kernel. Without this, PyTorch ≥ 2.x
// auto-registers an AutogradCUDA entry that wraps outputs in a grad_fn even
// when no backward is available, which confuses callers that check grad_fn is None.
TORCH_LIBRARY_IMPL(experimental, Autograd, m) {
    m.impl("gaussian_render_inference_only", torch::CppFunction::makeFallthrough());
}
