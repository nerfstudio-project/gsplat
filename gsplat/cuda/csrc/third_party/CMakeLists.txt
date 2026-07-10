# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GLM ========================================================================
block()
    add_library(gsplat::ext::glm INTERFACE IMPORTED GLOBAL)
    target_include_directories(gsplat::ext::glm SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/glm")
endblock()
