# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/GSplatCudaArch.cmake")

# Run fatal-error cases in child CMake processes so this script can verify both
# the nonzero result and the user-facing diagnostic.
if(DEFINED GSPLAT_CUDA_ARCH_FAILURE_CASE)
    if(GSPLAT_CUDA_ARCH_FAILURE_CASE STREQUAL "empty")
        _gsplat_normalize_native_cuda_architectures(_resolved "")
    elseif(GSPLAT_CUDA_ARCH_FAILURE_CASE STREQUAL "malformed")
        _gsplat_normalize_native_cuda_architectures(_resolved "89-real;not-an-architecture")
    else()
        message(FATAL_ERROR "Unknown failure case: ${GSPLAT_CUDA_ARCH_FAILURE_CASE}")
    endif()
    message(FATAL_ERROR "Failure case unexpectedly succeeded: ${GSPLAT_CUDA_ARCH_FAILURE_CASE}")
endif()

if(NOT DEFINED GSPLAT_CUDA_COMPILER OR "${GSPLAT_CUDA_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "GSPLAT_CUDA_COMPILER is required.")
endif()

# An explicit concrete policy must bypass native resolution byte-for-byte, even
# when the available native result and compiler path are deliberately unusable.
set(_explicit_arches "80;86-real;89-virtual")
set(_GSPLAT_CUDA_ARCH_OWNER CMAKE)
set(_GSPLAT_EXACT_CUDA_ARCHITECTURES "${_explicit_arches}")
set(CMAKE_CUDA_ARCHITECTURES "${_explicit_arches}")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "not-a-native-result")
gsplat_resolve_native_cuda_architectures("/does/not/exist/nvcc")
if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${_explicit_arches}")
    message(FATAL_ERROR "Explicit CUDA architectures changed: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(NOT "${_GSPLAT_EXACT_CUDA_ARCHITECTURES}" STREQUAL "${_explicit_arches}")
    message(FATAL_ERROR "The exact explicit CUDA policy changed.")
endif()

# Simulate an unstable multi-GPU enumeration order. The result must contain one
# sorted entry per distinct visible compute capability.
_gsplat_normalize_native_cuda_architectures(
    _normalized_arches
    "120-real;89-real;120-real;100-real;89-real"
)
if(NOT "${_normalized_arches}" STREQUAL "89-real;100-real;120-real")
    message(FATAL_ERROR "Native CUDA architectures were not normalized: ${_normalized_arches}")
endif()

# Exercise the complete native promotion with a compiler-supported synthetic
# result, keeping this source-level test independent of GPU availability. The
# dev build-tree test separately verifies CMake's actual GPU-derived result.
_gsplat_query_nvcc_architectures(
    _supported_real_arches
    _supported_virtual_arches
    "${GSPLAT_CUDA_COMPILER}"
)
list(GET _supported_real_arches 0 _synthetic_native_arch)

unset(CMAKE_CUDA_ARCHITECTURES CACHE)
unset(_GSPLAT_RESOLVED_NATIVE_CUDA_ARCHITECTURES CACHE)
set(_GSPLAT_EXACT_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "${_synthetic_native_arch}-real")
gsplat_resolve_native_cuda_architectures("${GSPLAT_CUDA_COMPILER}")
if(CMAKE_CUDA_ARCHITECTURES MATCHES "(^|;)native($|;)")
    message(FATAL_ERROR "Native survived architecture resolution: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${_GSPLAT_EXACT_CUDA_ARCHITECTURES}")
    message(FATAL_ERROR "The cached and exact CUDA architecture policies differ.")
endif()
get_property(_cached_arches CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
if(NOT "${_cached_arches}" STREQUAL "${CMAKE_CUDA_ARCHITECTURES}")
    message(FATAL_ERROR "CMakeCache.txt did not receive the concrete CUDA policy.")
endif()
get_property(
    _internally_cached_arches
    CACHE _GSPLAT_RESOLVED_NATIVE_CUDA_ARCHITECTURES
    PROPERTY VALUE
)
if(NOT "${_internally_cached_arches}" STREQUAL "${CMAKE_CUDA_ARCHITECTURES}")
    message(FATAL_ERROR "The internal native CUDA architecture cache differs.")
endif()
if(NOT TORCH_CUDA_ARCH_LIST MATCHES "^[1-9][0-9]*\.[0-9]$")
    message(FATAL_ERROR "Unexpected Torch CUDA architecture bridge: ${TORCH_CUDA_ARCH_LIST}")
endif()

# A dev preset writes native into the public cache on every configure. Prove a
# repeat initialization uses only the hidden concrete result: neither malformed
# detector data nor an unusable compiler may be observed on this path.
set(_first_resolved_arches "${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "malformed-repeat-detector-data")
gsplat_initialize_cuda_architectures("/does/not/exist/nvcc")
if(NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${_first_resolved_arches}")
    message(FATAL_ERROR "Repeat native initialization changed the concrete policy.")
endif()
if(NOT "${_GSPLAT_EXACT_CUDA_ARCHITECTURES}" STREQUAL "${_first_resolved_arches}")
    message(FATAL_ERROR "Repeat native initialization did not restore the exact policy.")
endif()
get_property(_repeat_cached_arches CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
if(NOT "${_repeat_cached_arches}" STREQUAL "${_first_resolved_arches}")
    message(FATAL_ERROR "Repeat native initialization did not restore the public cache.")
endif()

set(_failure_cases empty malformed)
set(_expected_diagnostics "usable architecture" "malformed detected")
foreach(_case _diagnostic IN ZIP_LISTS _failure_cases _expected_diagnostics)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" "-DGSPLAT_CUDA_ARCH_FAILURE_CASE=${_case}" -P
            "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
    )
    if(_result EQUAL 0)
        message(FATAL_ERROR "The ${_case} native-detection case unexpectedly succeeded.")
    endif()
    string(CONCAT _diagnostic_output "${_output}" "${_error}")
    if(NOT _diagnostic_output MATCHES "${_diagnostic}")
        message(
            FATAL_ERROR
            "The ${_case} failure did not contain '${_diagnostic}':\n${_diagnostic_output}"
        )
    endif()
endforeach()
