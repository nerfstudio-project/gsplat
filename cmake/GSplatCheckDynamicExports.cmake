# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(_required_variable IN ITEMS GSPLAT_NM GSPLAT_MODULE GSPLAT_EXPECTED_SYMBOLS)
    if(NOT DEFINED ${_required_variable} OR "${${_required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${_required_variable} must be set")
    endif()
endforeach()

if(NOT EXISTS "${GSPLAT_NM}")
    message(FATAL_ERROR "nm executable does not exist: ${GSPLAT_NM}")
endif()

if(NOT EXISTS "${GSPLAT_MODULE}")
    message(FATAL_ERROR "module does not exist: ${GSPLAT_MODULE}")
endif()

# Inspect only externally visible definitions in the dynamic symbol table.
# Undefined entries are imports rather than symbols exported by the module.
execute_process(
    COMMAND "${GSPLAT_NM}" -D -g --defined-only -P "${GSPLAT_MODULE}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _nm_output
    ERROR_VARIABLE _nm_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "nm failed for ${GSPLAT_MODULE} (exit ${_nm_result}):\n${_nm_error}")
endif()

string(REPLACE "\r\n" "\n" _nm_output "${_nm_output}")
string(REPLACE "\n" ";" _nm_lines "${_nm_output}")

set(_actual_symbols)
foreach(_line IN LISTS _nm_lines)
    string(STRIP "${_line}" _line)
    if(_line STREQUAL "")
        continue()
    endif()

    string(REGEX MATCH "^[^ \t]+" _symbol "${_line}")
    if(_symbol STREQUAL "")
        message(FATAL_ERROR "cannot parse nm output line: ${_line}")
    endif()
    list(APPEND _actual_symbols "${_symbol}")
endforeach()

set(_expected_symbols ${GSPLAT_EXPECTED_SYMBOLS})
list(SORT _actual_symbols)
list(SORT _expected_symbols)

if(NOT "${_actual_symbols}" STREQUAL "${_expected_symbols}")
    list(JOIN _actual_symbols ", " _actual_text)
    list(JOIN _expected_symbols ", " _expected_text)
    if(_actual_text STREQUAL "")
        set(_actual_text "<none>")
    endif()
    message(
        FATAL_ERROR
        "unexpected dynamic exports in ${GSPLAT_MODULE}\n"
        "expected: ${_expected_text}\n"
        "actual:   ${_actual_text}\n"
        "nm output:\n${_nm_output}"
    )
endif()
