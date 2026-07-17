# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

foreach(_required_var IN ITEMS GSPLAT_CUOBJDUMP GSPLAT_ARTIFACT)
    if(NOT DEFINED ${_required_var} OR "${${_required_var}}" STREQUAL "")
        message(FATAL_ERROR "${_required_var} is required.")
    endif()
endforeach()

foreach(_kind IN ITEMS elf ptx)
    execute_process(
        COMMAND "${GSPLAT_CUOBJDUMP}" "--list-${_kind}" "${GSPLAT_ARTIFACT}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
    )
    if(NOT _result EQUAL 0)
        message(
            FATAL_ERROR
            "cuobjdump --list-${_kind} failed for ${GSPLAT_ARTIFACT}:\n${_output}${_error}"
        )
    endif()

    # cuobjdump labels PTX images sm_<cc> too (e.g. `foo.1.sm_80.ptx`), so both
    # kinds use sm_; the real/virtual split is from --list-<kind>, not the label.
    set(_prefix sm)

    string(REGEX MATCHALL "${_prefix}_[0-9]+[a-z]?" _architectures "${_output}${_error}")
    list(TRANSFORM _architectures REPLACE "^${_prefix}_" "")
    list(REMOVE_DUPLICATES _architectures)
    list(SORT _architectures COMPARE NATURAL ORDER ASCENDING)
    set(_${_kind}_architectures "${_architectures}")
endforeach()

foreach(_kind IN ITEMS real virtual)
    string(TOUPPER "${_kind}" _kind_upper)
    set(_expected "${GSPLAT_EXPECTED_${_kind_upper}_ARCHITECTURES}")
    if(_kind STREQUAL "real")
        set(_actual "${_elf_architectures}")
    else()
        set(_actual "${_ptx_architectures}")
    endif()
    if(NOT "${_actual}" STREQUAL "${_expected}")
        message(
            FATAL_ERROR
            "${GSPLAT_ARTIFACT} embeds ${_kind} CUDA architectures "
            "'${_actual}', expected '${_expected}'."
        )
    endif()
endforeach()
