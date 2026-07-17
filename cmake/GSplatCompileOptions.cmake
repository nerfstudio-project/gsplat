# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

option(GSPLAT_FAST_MATH "Use CUDA fast-math intrinsics." ON)

function(gsplat_define_compile_options)
    # Keep RelWithDebInfo's metadata while matching Release optimization.
    # Change the GNU/NVCC default in this directory scope so a parent project
    # that adds gsplat as a subdirectory retains its own toolchain flags.
    foreach(_language IN ITEMS CXX CUDA)
        set(_flag_variable "CMAKE_${_language}_FLAGS_RELWITHDEBINFO")
        string(
            REGEX REPLACE "(^|[ \t])-O2([ \t]|$)"
            "\\1-O3\\2"
            _gsplat_relwithdebinfo_flags
            "${${_flag_variable}}"
        )
        set(${_flag_variable} "${_gsplat_relwithdebinfo_flags}" PARENT_SCOPE)

        # Assert the invariant: after the rewrite, RelWithDebInfo may differ
        # from Release only by host debug-metadata flags (-g...), so stripping
        # the debug info yields the Release binary. Device debug (-G) changes
        # code generation and is deliberately not tolerated.
        separate_arguments(
            _gsplat_relwithdebinfo_list
            UNIX_COMMAND
            "${_gsplat_relwithdebinfo_flags}"
        )
        list(FILTER _gsplat_relwithdebinfo_list EXCLUDE REGEX "^-g")
        separate_arguments(_gsplat_release_list UNIX_COMMAND "${CMAKE_${_language}_FLAGS_RELEASE}")
        list(SORT _gsplat_relwithdebinfo_list)
        list(SORT _gsplat_release_list)
        if(NOT _gsplat_relwithdebinfo_list STREQUAL _gsplat_release_list)
            message(
                FATAL_ERROR
                "CMAKE_${_language}_FLAGS_RELWITHDEBINFO "
                "('${_gsplat_relwithdebinfo_flags}') no longer matches "
                "CMAKE_${_language}_FLAGS_RELEASE "
                "('${CMAKE_${_language}_FLAGS_RELEASE}') outside of debug "
                "metadata; RelWithDebInfo builds would not be "
                "Release-optimized."
            )
        endif()
    endforeach()

    set(_gsplat_cuda_definitions
        __CUDA_NO_HALF_OPERATORS__
        __CUDA_NO_HALF_CONVERSIONS__
        __CUDA_NO_HALF2_OPERATORS__
        __CUDA_NO_BFLOAT16_CONVERSIONS__
    )
    set(_gsplat_nvcc_options
        "$<$<BOOL:${GSPLAT_FAST_MATH}>:--use_fast_math>"
        "$<$<CONFIG:Debug,RelWithDebInfo>:-lineinfo>"
        --expt-relaxed-constexpr
        --forward-unknown-opts
        # Keep nvcc intermediates beside the objects: the default shared /tmp
        # collides across sandboxed parallel builds.
        --objdir-as-tempdir
        # fatbinary compresses only PTX/debug images by default; also compress
        # the per-arch SASS images that dominate production wheel size.
        -Xfatbin=--compress-all
        # __host__ or __device__ annotation ignored on an explicitly defaulted function.
        -diag-suppress=20012
        # Pointless comparison of an unsigned integer with zero.
        -diag-suppress=186
        # Address of a kernel template instantiation taken without an in-TU definition.
        -diag-suppress=20281
    )
    set(_gsplat_cxx_options -Wall -Wno-unknown-pragmas)
    set(_gsplat_normalized_source_path
        "$<$<BOOL:${GSPLAT_CCACHE_NORMALIZE_PATHS}>:-ffile-prefix-map=${GSPLAT_SOURCE_DIR}=.>"
    )

    add_library(gsplat_compile_opts INTERFACE)
    target_compile_features(gsplat_compile_opts INTERFACE cxx_std_20 cuda_std_20)
    target_compile_definitions(
        gsplat_compile_opts
        INTERFACE
            GSPLAT_HAVE_GENERATED_CONFIG=1
            "$<$<COMPILE_LANGUAGE:CUDA>:${_gsplat_cuda_definitions}>"
    )
    target_compile_options(
        gsplat_compile_opts
        INTERFACE
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${_gsplat_nvcc_options}>"
            "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:${_gsplat_normalized_source_path}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${_gsplat_cxx_options}>"
            "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:${_gsplat_normalized_source_path}>"
    )
endfunction()
