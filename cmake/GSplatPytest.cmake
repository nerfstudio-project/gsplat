# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(gsplat_configure_pytest_ini output_file test_path python_path)
    set(_gsplat_pytest_test_path "${test_path}")
    set(_gsplat_pytest_python_path "${python_path}")

    # Keep Python and compiled-code warning policy aligned. The template keeps
    # narrow exceptions for known third-party warnings in both modes; this
    # switch controls whether every other warning is promoted to an error.
    if(CMAKE_COMPILE_WARNING_AS_ERROR)
        set(_gsplat_pytest_warning_error_rule "    error\n")
    else()
        set(_gsplat_pytest_warning_error_rule "")
    endif()

    configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/GSplatPytest.ini.in" "${output_file}" @ONLY)
endfunction()
