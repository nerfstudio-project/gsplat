# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

gsplat_add_python_extension(TARGET gsplat_geometry_cuda SOURCES ext.cpp quaternion.cu pose.cu)
