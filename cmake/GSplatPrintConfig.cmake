# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

function(gsplat_print_config)
    if(NOT PROJECT_IS_TOP_LEVEL)
        return()
    endif()

    set(_gsplat_enabled_families "")
    foreach(_fam IN LISTS _GSPLAT_BUILD_FAMILIES)
        if(_gsplat_final_build_${_fam})
            list(APPEND _gsplat_enabled_families "${_fam}")
        endif()
    endforeach()
    if(NOT _gsplat_enabled_families)
        set(_gsplat_enabled_families "(none)")
    endif()

    message(STATUS "")
    message(STATUS "${PROJECT_NAME} ${PROJECT_VERSION} -- resolved build configuration:")
    if(GSPLAT_PRESET)
        message(STATUS "  preset                       : ${GSPLAT_PRESET}")
    else()
        message(STATUS "  preset                       : (none)")
    endif()
    message(STATUS "  build directory              : ${GSPLAT_BINARY_DIR}")
    if(CMAKE_BUILD_TYPE)
        message(STATUS "  CMAKE_BUILD_TYPE             : ${CMAKE_BUILD_TYPE}")
    else()
        message(
            STATUS
            "  CMAKE_BUILD_TYPE             : (unset -> unoptimized; pass -DCMAKE_BUILD_TYPE=Release)"
        )
    endif()
    message(STATUS "  CMAKE_CUDA_ARCHITECTURES     : ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "  TORCH_CUDA_ARCH_LIST         : ${TORCH_CUDA_ARCH_LIST}")
    if("${GSPLAT_KERNEL_FAMILIES}" STREQUAL "")
        message(STATUS "  GSPLAT_KERNEL_FAMILIES       : (all)")
    else()
        message(STATUS "  GSPLAT_KERNEL_FAMILIES       : ${GSPLAT_KERNEL_FAMILIES}")
    endif()
    message(STATUS "  AOT kernel families          : ${_gsplat_enabled_families}")
    message(STATUS "  GSPLAT_NUM_CHANNELS          : ${GSPLAT_NUM_CHANNELS}")
    message(STATUS "  GSPLAT_BUILD_CAMERA_WRAPPERS : ${GSPLAT_BUILD_CAMERA_WRAPPERS}")
    message(STATUS "  GSPLAT_FAST_MATH             : ${GSPLAT_FAST_MATH}")
    message(STATUS "  GSPLAT_BUILD_TESTS           : ${GSPLAT_BUILD_TESTS}")
    message(STATUS "  GSPLAT_DEVELOPMENT_MODE      : ${GSPLAT_DEVELOPMENT_MODE}")
    message(STATUS "  GSPLAT_ENABLE_BUILD_TRACES   : ${GSPLAT_ENABLE_BUILD_TRACES}")
    message(STATUS "  GSPLAT_ENABLE_CCACHE         : ${GSPLAT_ENABLE_CCACHE}")
    if(GSPLAT_ENABLE_CCACHE)
        message(STATUS "    GSPLAT_FORCE_CCACHE        : ${GSPLAT_FORCE_CCACHE}")
        message(STATUS "    GSPLAT_CCACHE_STATS        : ${GSPLAT_CCACHE_STATS}")
        message(STATUS "    GSPLAT_CCACHE_NORMALIZE_PATHS : ${GSPLAT_CCACHE_NORMALIZE_PATHS}")
        if(GSPLAT_CCACHE_EXECUTABLE)
            message(STATUS "    ccache                     : ${GSPLAT_CCACHE_EXECUTABLE}")
            # GSPLAT_CCACHE_DIR, when set, is baked into the compiler launcher so
            # the ambient CCACHE_DIR cannot override it, and it is the resolved
            # directory. Otherwise ask ccache which directory it would use, which
            # honors the build-time CCACHE_DIR or ccache's own default.
            if(GSPLAT_CCACHE_DIR)
                message(STATUS "    GSPLAT_CCACHE_DIR          : ${GSPLAT_CCACHE_DIR}")
                set(_gsplat_resolved_ccache_dir "${GSPLAT_CCACHE_DIR}")
            else()
                message(STATUS "    GSPLAT_CCACHE_DIR          : (not set)")
                execute_process(
                    COMMAND "${GSPLAT_CCACHE_EXECUTABLE}" --get-config cache_dir
                    OUTPUT_VARIABLE _gsplat_resolved_ccache_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_VARIABLE _gsplat_ccache_config_error
                    ERROR_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _gsplat_ccache_config_status
                )
                if(NOT _gsplat_ccache_config_status EQUAL 0)
                    if(_gsplat_ccache_config_error STREQUAL "")
                        set(_gsplat_ccache_config_error
                            "exit status ${_gsplat_ccache_config_status}"
                        )
                    endif()
                    set(_gsplat_resolved_ccache_dir "(unknown - ${_gsplat_ccache_config_error})")
                endif()
            endif()
            message(STATUS "    Resolved ccache dir        : ${_gsplat_resolved_ccache_dir}")
        else()
            message(STATUS "    ccache                     : enabled, not found")
        endif()
    endif()
    message(STATUS "")
endfunction()
