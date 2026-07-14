# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/GSplatProjectInfo.cmake")

set(SKBUILD_PROJECT_NAME gsplat)
set(SKBUILD_PROJECT_VERSION "1.5.3+pt210cu130")
gsplat_read_project_info(_project_name _project_version)
if(NOT _project_name STREQUAL "gsplat")
    message(FATAL_ERROR "Unexpected project name: ${_project_name}")
endif()
if(NOT _project_version STREQUAL "1.5.3")
    message(FATAL_ERROR "Local version was not normalized: ${_project_version}")
endif()

set(SKBUILD_PROJECT_VERSION "1.5.3")
gsplat_read_project_info(_project_name _project_version)
if(NOT _project_version STREQUAL "1.5.3")
    message(FATAL_ERROR "Release version changed: ${_project_version}")
endif()

unset(SKBUILD_PROJECT_NAME)
unset(SKBUILD_PROJECT_VERSION)
gsplat_read_project_info(_fallback_project_name _fallback_project_version)
if(NOT _fallback_project_name STREQUAL "gsplat")
    message(FATAL_ERROR "Could not read the project name from pyproject.toml")
endif()
if(NOT _fallback_project_version STREQUAL "1.5.3")
    message(FATAL_ERROR "Could not resolve dynamic version metadata: ${_fallback_project_version}")
endif()
