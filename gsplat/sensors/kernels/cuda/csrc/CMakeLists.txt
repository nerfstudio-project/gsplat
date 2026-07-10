# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gsplat_add_python_extension(
    TARGET gsplat_sensors_cuda
    SOURCES
        ext.cpp
        camera_torch.cpp
        external_distortion_torch.cpp
        camera_kernel.cu
        camera_kernel_backward.cu
        ftheta_kernel.cu
        ftheta_kernel_backward.cu
        lidar_torch.cpp
        lidar_kernel.cu
        lidar_kernel_backward.cu
        fisheye_kernel.cu
        fisheye_kernel_backward.cu
)
target_include_directories(
    gsplat_sensors_cuda
    PRIVATE "${GSPLAT_SOURCE_DIR}/gsplat/geometry/kernels/cuda/csrc"
)
