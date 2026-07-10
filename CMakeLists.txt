# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

# Build products must never land among the sources: sdists and other
# source-snapshot consumers glob this tree. Nested build directories such as
# build/<preset> remain legal.
if(CMAKE_CURRENT_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(
        FATAL_ERROR
        "In-source builds are not supported; use a separate build directory "
        "(cmake -B build)."
    )
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/GSplatProjectInfo.cmake)
include(cmake/GSplatCudaArch.cmake)
include(cmake/GSplatCompileOptions.cmake)
include(cmake/GSplatPrintConfig.cmake)

# -----------------------------------------------------------------------------
# Project metadata and CUDA architecture defaults
# -----------------------------------------------------------------------------

gsplat_read_project_info(_gsplat_project_name _gsplat_project_version)

project("${_gsplat_project_name}" VERSION "${_gsplat_project_version}" LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(
        FATAL_ERROR
        "The gsplat CMake source build currently supports Linux only. "
        "On Windows, install a matching precompiled wheel instead."
    )
endif()

gsplat_setup_cuda()

include(cmake/GSplatBuildConfig.cmake)

# The interpreter floor is [project].requires-python in pyproject.toml.
gsplat_read_requires_python(_gsplat_python_minimum)
find_package(Python ${_gsplat_python_minimum} COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)

# Report every unmet Python dependency before FindTorch fails on the first.
include(cmake/GSplatCheckPythonDeps.cmake)
gsplat_check_python_dependencies()

find_package(Torch MODULE REQUIRED)

# Torch injects -gencode flags globally during discovery. Replace them with
# CMake's exact architecture selection, including virtual-only targets.
gsplat_finalize_cuda_architectures()

# -----------------------------------------------------------------------------
# Build configuration and shared target properties
# -----------------------------------------------------------------------------

gsplat_resolve_build_config()
gsplat_define_compile_options()

# Vendored dependencies build only as link dependencies of gsplat targets and
# contribute nothing to the default build or the install set.
add_subdirectory(third_party obj/third_party EXCLUDE_FROM_ALL)
add_subdirectory(gsplat obj/gsplat)
add_subdirectory(tests obj/tests)

# ccache statistics depend on the complete target graph.
gsplat_finalize_ccache_stats()

gsplat_print_config()
