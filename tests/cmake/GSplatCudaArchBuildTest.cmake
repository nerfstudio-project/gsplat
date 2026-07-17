# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

foreach(
    _required_var
    IN
    ITEMS GSPLAT_BINARY_DIR GSPLAT_CUOBJDUMP GSPLAT_EXPECTED_CUDA_ARCHITECTURES
)
    if(NOT DEFINED ${_required_var} OR "${${_required_var}}" STREQUAL "")
        message(FATAL_ERROR "${_required_var} is required.")
    endif()
endforeach()

file(READ "${GSPLAT_BINARY_DIR}/CMakeCache.txt" _cache)
string(REGEX MATCH "(^|\n)CMAKE_CUDA_ARCHITECTURES:[^=]*=([^\r\n]+)" _cache_arch_match "${_cache}")
set(_cached_arches "${CMAKE_MATCH_2}")
if(NOT "${_cached_arches}" STREQUAL "${GSPLAT_EXPECTED_CUDA_ARCHITECTURES}")
    message(
        FATAL_ERROR
        "CMakeCache.txt has CUDA architectures '${_cached_arches}', expected "
        "'${GSPLAT_EXPECTED_CUDA_ARCHITECTURES}'."
    )
endif()
if(_cached_arches MATCHES "(^|;)native($|;)")
    message(FATAL_ERROR "CMakeCache.txt still contains the special native architecture.")
endif()

set(_expected_numeric_arches "")
set(_expected_real_arches "")
set(_expected_virtual_arches "")
set(_expected_arches_are_concrete TRUE)
foreach(_arch IN LISTS GSPLAT_EXPECTED_CUDA_ARCHITECTURES)
    if(_arch MATCHES "^([0-9]+[a-z]?)(-(real|virtual))?$")
        set(_numeric_arch "${CMAKE_MATCH_1}")
        set(_arch_kind "${CMAKE_MATCH_3}")
        list(APPEND _expected_numeric_arches "${_numeric_arch}")
        if(NOT _arch_kind STREQUAL "virtual")
            list(APPEND _expected_real_arches "${_numeric_arch}")
        endif()
        if(NOT _arch_kind STREQUAL "real")
            list(APPEND _expected_virtual_arches "${_numeric_arch}")
        endif()
    else()
        set(_expected_arches_are_concrete FALSE)
    endif()
endforeach()
foreach(_arch_list IN ITEMS _expected_numeric_arches _expected_real_arches _expected_virtual_arches)
    list(REMOVE_DUPLICATES ${_arch_list})
    list(SORT ${_arch_list} COMPARE NATURAL ORDER ASCENDING)
endforeach()

file(READ "${GSPLAT_BINARY_DIR}/compile_commands.json" _compile_database)
if(_compile_database MATCHES "-arch=native")
    message(FATAL_ERROR "compile_commands.json contains -arch=native.")
endif()

string(JSON _command_count LENGTH "${_compile_database}")
set(_cuda_command_count 0)
set(_cuda_bearing_object_count 0)
math(EXPR _last_command "${_command_count} - 1")
foreach(_command_index RANGE 0 ${_last_command})
    string(JSON _source_file GET "${_compile_database}" ${_command_index} file)
    if(NOT _source_file MATCHES "\\.cu$")
        continue()
    endif()

    math(EXPR _cuda_command_count "${_cuda_command_count} + 1")
    string(JSON _command GET "${_compile_database}" ${_command_index} command)
    if(_command MATCHES "-arch=native")
        message(FATAL_ERROR "CUDA command for ${_source_file} contains -arch=native.")
    endif()

    if(_expected_arches_are_concrete)
        string(REGEX MATCHALL "arch=compute_[0-9]+[a-z]?" _command_arches "${_command}")
        list(TRANSFORM _command_arches REPLACE "^arch=compute_" "")
        list(REMOVE_DUPLICATES _command_arches)
        list(SORT _command_arches COMPARE NATURAL ORDER ASCENDING)
        if(NOT "${_command_arches}" STREQUAL "${_expected_numeric_arches}")
            message(
                FATAL_ERROR
                "CUDA command for ${_source_file} uses architectures "
                "'${_command_arches}', expected '${_expected_numeric_arches}'."
            )
        endif()

        string(JSON _object_file GET "${_compile_database}" ${_command_index} output)
        if(NOT EXISTS "${_object_file}")
            message(FATAL_ERROR "CUDA compilation output does not exist: ${_object_file}")
        endif()

        foreach(_kind IN ITEMS elf ptx)
            execute_process(
                COMMAND "${GSPLAT_CUOBJDUMP}" "--list-${_kind}" "${_object_file}"
                RESULT_VARIABLE _result
                OUTPUT_VARIABLE _output
                ERROR_VARIABLE _error
            )
            if(NOT _result EQUAL 0)
                message(
                    FATAL_ERROR
                    "cuobjdump --list-${_kind} failed for ${_object_file}:\n${_output}${_error}"
                )
            endif()

            if(_kind STREQUAL "elf")
                set(_prefix sm)
            else()
                set(_prefix compute)
            endif()
            string(REGEX MATCHALL "${_prefix}_[0-9]+[a-z]?" _object_arches "${_output}${_error}")
            list(TRANSFORM _object_arches REPLACE "^${_prefix}_" "")
            list(REMOVE_DUPLICATES _object_arches)
            list(SORT _object_arches COMPARE NATURAL ORDER ASCENDING)
            set(_object_${_kind}_arches "${_object_arches}")
        endforeach()
        set(_object_real_arches "${_object_elf_arches}")
        set(_object_virtual_arches "${_object_ptx_arches}")
        if(_object_real_arches OR _object_virtual_arches)
            math(EXPR _cuda_bearing_object_count "${_cuda_bearing_object_count} + 1")
            if(
                NOT "${_object_real_arches}" STREQUAL "${_expected_real_arches}"
                OR NOT "${_object_virtual_arches}" STREQUAL "${_expected_virtual_arches}"
            )
                message(
                    FATAL_ERROR
                    "CUDA object ${_object_file} embeds real architectures "
                    "'${_object_real_arches}' and virtual architectures "
                    "'${_object_virtual_arches}'; expected "
                    "'${_expected_real_arches}' and '${_expected_virtual_arches}'."
                )
            endif()
        endif()
    endif()
endforeach()

if(_cuda_command_count EQUAL 0)
    message(FATAL_ERROR "compile_commands.json contains no CUDA compilation commands.")
endif()
if(_expected_arches_are_concrete AND _cuda_bearing_object_count EQUAL 0)
    message(FATAL_ERROR "CUDA compilation outputs contain no device code.")
endif()
message(
    STATUS
    "Audited ${_cuda_command_count} CUDA commands and ${_cuda_bearing_object_count} "
    "CUDA-bearing object files."
)
