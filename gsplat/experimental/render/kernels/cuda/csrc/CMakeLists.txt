# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gsplat_add_python_extension(
    TARGET experimental_gaussian_render_inference_scene_cuda
    SOURCES
        ext.cpp
        gaussian_inference/GaussianRenderInferenceScene.cu
        gaussian_inference/IntersectCommon.cu
        gaussian_inference/IntersectMTFused.cu
        gaussian_inference/MacroTileIntersect.cu
        gaussian_inference/MacroTileRasterize.cu
        gaussian_inference/Projection.cu
        "${GSPLAT_SOURCE_DIR}/gsplat/cuda/csrc/SegmentedSort.cu"
        gaussian_inference/SHCompression.cu
        gaussian_inference/SphericalHarmonics.cu
)
target_include_directories(
    experimental_gaussian_render_inference_scene_cuda
    PRIVATE
        "${GSPLAT_SOURCE_DIR}/gsplat/cuda/csrc/include"
        "${GSPLAT_SOURCE_DIR}/gsplat/cuda/csrc"
        "${CMAKE_CURRENT_SOURCE_DIR}/gaussian_inference"
)
target_link_libraries(experimental_gaussian_render_inference_scene_cuda PRIVATE gsplat::ext::glm)
