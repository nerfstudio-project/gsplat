# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/GSplatCCache.cmake")

# This module is included from gsplat's root immediately after project().
# Keeping these values in directory scope avoids forwarding every option and
# compiler launcher through an otherwise unnecessary function scope.
set(GSPLAT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(GSPLAT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

option(GSPLAT_ENABLE_BUILD_TRACES "Generate configure, build, and test trace files" OFF)

option(
    GSPLAT_STRICT_SUBMODULES
    "Fail (vs warn) when a third_party submodule is not at the pinned commit"
    ON
)

option(
    GSPLAT_CHECK_PYTHON_DEPS
    "Verify the declared Python dependencies against the active environment at configure time"
    ON
)

if(PROJECT_IS_TOP_LEVEL AND GSPLAT_ENABLE_BUILD_TRACES)
    if(CMAKE_VERSION VERSION_LESS 4.3)
        message(FATAL_ERROR "GSPLAT_ENABLE_BUILD_TRACES requires CMake 4.3 or newer.")
    endif()

    set(_gsplat_instrumentation_data_version 1)
    set(_gsplat_instrumentation_options trace)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.4)
        set(_gsplat_instrumentation_data_version 1.1)
        list(APPEND _gsplat_instrumentation_options captureOutput compileTrace)
    endif()

    # Archive each completed phase before the next indexing hook replaces
    # CMake's most recent Google Trace file. Both build hooks are needed to
    # cover direct build-tool and `cmake --build` invocations.
    cmake_instrumentation(
        API_VERSION 1
        DATA_VERSION ${_gsplat_instrumentation_data_version}
        HOOKS postGenerate postBuild postCMakeBuild postCTest
        OPTIONS ${_gsplat_instrumentation_options}
        CALLBACK ${CMAKE_COMMAND}
        -P
        "${CMAKE_CURRENT_LIST_DIR}/GSplatArchiveInstrumentation.cmake"
    )
endif()

# Nested consumers get a runtime-only build; a parent project that wants
# gsplat's tests opts in explicitly.
option(GSPLAT_BUILD_TESTS "Build the C++ tests" ${PROJECT_IS_TOP_LEVEL})

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    include(CTest)

    # Catch misspelled target names at generate time instead of link time.
    # Top-level only: a nested gsplat must not impose this policy on targets a
    # parent creates after including us. Flags, absolute paths, and imported
    # targets (gsplat::ext::*) are exempt from the check by definition.
    set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS ON)

    # Compiler launchers are captured when targets are created, so configure
    # ccache before any add_subdirectory() call can create a compiled target.
    gsplat_configure_ccache()
endif()

set(_GSPLAT_BUILD_FAMILIES
    2DGS
    3DGS
    3DGUT
    ADAM
    RELOC
    LOSSES
)
set(_GSPLAT_DEFAULT_NUM_CHANNELS "1,2,3,4,5,6,8,9,16,17,21,23,24,32")

# Resolve gsplat's build options and configure the generated C++ config header.
function(gsplat_resolve_build_config)
    # With single-config generators an empty build type compiles without any
    # optimization flags — a silently slow build of a CUDA library. Presets
    # and scikit-build-core always set one; default the bare-configure case.
    get_property(_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(PROJECT_IS_TOP_LEVEL AND NOT _multi_config AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
        message(STATUS "No CMAKE_BUILD_TYPE requested; defaulting to Release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    endif()

    set(GSPLAT_KERNEL_FAMILIES
        ""
        CACHE STRING
        "Kernel families to AOT-build (comma/semicolon list, case-insensitive; empty selects all)."
    )

    # Normalize the selection: accept both separators and any case.
    string(REPLACE "," ";" _requested_families "${GSPLAT_KERNEL_FAMILIES}")
    string(TOUPPER "${_requested_families}" _requested_families)
    list(REMOVE_ITEM _requested_families "")

    foreach(_family IN LISTS _requested_families)
        if(NOT "${_family}" IN_LIST _GSPLAT_BUILD_FAMILIES)
            list(JOIN _GSPLAT_BUILD_FAMILIES ", " _known_families)
            message(
                FATAL_ERROR
                "Unknown kernel family '${_family}' in GSPLAT_KERNEL_FAMILIES; "
                "known families: ${_known_families}."
            )
        endif()
    endforeach()

    foreach(_family IN LISTS _GSPLAT_BUILD_FAMILIES)
        if("${_requested_families}" STREQUAL "" OR "${_family}" IN_LIST _requested_families)
            set(_gsplat_final_build_${_family} 1)
        else()
            set(_gsplat_final_build_${_family} 0)
        endif()
    endforeach()

    if(NOT DEFINED GSPLAT_BUILD_CAMERA_WRAPPERS)
        # The camera-wrapper test kernels build only when the C++ tests do.
        set(GSPLAT_BUILD_CAMERA_WRAPPERS "${GSPLAT_BUILD_TESTS}")
    endif()
    set(GSPLAT_BUILD_CAMERA_WRAPPERS
        "${GSPLAT_BUILD_CAMERA_WRAPPERS}"
        CACHE BOOL
        "Build Python-exposed camera wrapper test kernels."
    )
    if("${GSPLAT_BUILD_CAMERA_WRAPPERS}")
        set(_gsplat_final_build_camera_wrappers 1)
    else()
        set(_gsplat_final_build_camera_wrappers 0)
    endif()

    if(DEFINED NUM_CHANNELS AND NOT DEFINED GSPLAT_NUM_CHANNELS)
        set(GSPLAT_NUM_CHANNELS "${NUM_CHANNELS}")
    elseif(NOT DEFINED GSPLAT_NUM_CHANNELS)
        # Instantiate the complete supported channel set unless the caller
        # explicitly selects a subset.
        set(GSPLAT_NUM_CHANNELS "${_GSPLAT_DEFAULT_NUM_CHANNELS}")
    endif()
    set(GSPLAT_NUM_CHANNELS
        "${GSPLAT_NUM_CHANNELS}"
        CACHE STRING
        "Comma- or semicolon-separated channel counts to instantiate ahead of time."
    )

    # Accept both comma- and semicolon-separated input and normalize it into
    # the CMake list consumed by target generation. Ignore empty fields.
    string(REPLACE "," ";" _gsplat_final_num_channels "${GSPLAT_NUM_CHANNELS}")
    list(TRANSFORM _gsplat_final_num_channels STRIP)
    list(FILTER _gsplat_final_num_channels EXCLUDE REGEX "^$")

    if(NOT DEFINED GSPLAT_GENERATED_DIR)
        set(GSPLAT_GENERATED_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
            CACHE PATH
            "Directory for generated gsplat build headers."
        )
    endif()

    # The C++ preprocessor needs a comma-separated template argument list.
    # Keep the resolved CMake value as a list and serialize only at this output
    # boundary; configure_file() creates the output directory when necessary.
    list(JOIN _gsplat_final_num_channels "," _gsplat_final_num_channels_csv)
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GSplatBuildConfig.h.in"
        "${GSPLAT_GENERATED_DIR}/GSplatBuildConfig.h"
        @ONLY
    )

    set(_resolved_vars _gsplat_final_build_camera_wrappers _gsplat_final_num_channels)
    foreach(_family IN LISTS _GSPLAT_BUILD_FAMILIES)
        list(APPEND _resolved_vars "_gsplat_final_build_${_family}")
    endforeach()
    return(PROPAGATE ${_resolved_vars})
endfunction()
