# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

set(GSPLAT_SOURCE_DIR "/tmp/gsplat-worktree")
set(GSPLAT_BINARY_DIR "/tmp/gsplat-build")
set(GSPLAT_ENABLE_CCACHE ON)
set(GSPLAT_CCACHE_STATS ON)
set(GSPLAT_CCACHE_EXECUTABLE "/usr/bin/ccache")
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/GSplatCCache.cmake")

if(NOT GSPLAT_CCACHE_NORMALIZE_PATHS)
    message(FATAL_ERROR "GSPLAT_CCACHE_NORMALIZE_PATHS must default to ON.")
endif()

gsplat_configure_ccache()
set(_expected_statslog "${GSPLAT_BINARY_DIR}/ccache-build-stats.log")
if(NOT "${_gsplat_ccache_statslog}" STREQUAL "${_expected_statslog}")
    message(FATAL_ERROR "The private ccache stats log does not survive function scope.")
endif()
list(FIND CMAKE_CXX_COMPILER_LAUNCHER "CCACHE_STATSLOG=${_expected_statslog}" _statslog_index)
if(_statslog_index EQUAL -1)
    message(FATAL_ERROR "The ccache launcher does not record this build's statistics.")
endif()

list(FIND CMAKE_CXX_COMPILER_LAUNCHER "CCACHE_BASEDIR=${GSPLAT_SOURCE_DIR}" _basedir_index)
if(_basedir_index EQUAL -1)
    message(FATAL_ERROR "The normalized launcher does not set CCACHE_BASEDIR.")
endif()

list(FIND CMAKE_CXX_COMPILER_LAUNCHER "CCACHE_NOHASHDIR=true" _nohashdir_index)
if(NOT _nohashdir_index EQUAL -1)
    message(FATAL_ERROR "Path normalization must retain ccache's directory hashing.")
endif()

foreach(_launcher IN ITEMS CMAKE_C_COMPILER_LAUNCHER CMAKE_CUDA_COMPILER_LAUNCHER)
    list(FIND ${_launcher} "CCACHE_BASEDIR=${GSPLAT_SOURCE_DIR}" _basedir_index)
    list(FIND ${_launcher} "CCACHE_NOHASHDIR=true" _nohashdir_index)
    if(_basedir_index EQUAL -1)
        message(FATAL_ERROR "${_launcher} does not use the normalized ccache environment.")
    endif()
    if(NOT _nohashdir_index EQUAL -1)
        message(FATAL_ERROR "${_launcher} disables ccache's directory hashing.")
    endif()
endforeach()

set(GSPLAT_CCACHE_NORMALIZE_PATHS OFF)
set(GSPLAT_CCACHE_STATS OFF)
gsplat_configure_ccache()
foreach(_launcher IN ITEMS CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER)
    if(NOT "${${_launcher}}" STREQUAL "${GSPLAT_CCACHE_EXECUTABLE}")
        message(FATAL_ERROR "Disabling normalization must restore the plain ${_launcher}.")
    endif()
endforeach()

foreach(
    _launcher
    IN
    ITEMS CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER CMAKE_CUDA_COMPILER_LAUNCHER
)
    list(FIND ${_launcher} "CCACHE_BASEDIR=${GSPLAT_SOURCE_DIR}" _basedir_index)
    if(NOT _basedir_index EQUAL -1)
        message(FATAL_ERROR "Disabling normalization leaves CCACHE_BASEDIR in ${_launcher}.")
    endif()
endforeach()
