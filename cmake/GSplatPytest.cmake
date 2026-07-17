# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(gsplat_configure_pytest_ini output_file test_path)
    set(_gsplat_pytest_test_path "${test_path}")

    # Import roots (ARGN): render one quoted pythonpath entry per path. The
    # installed-wheel configuration passes both the site-packages root and the
    # examples/ payload root (so example modules' sibling imports resolve);
    # build-tree callers pass a single path.
    set(_gsplat_pytest_python_path "")
    foreach(_gsplat_python_root IN LISTS ARGN)
        if(NOT _gsplat_pytest_python_path STREQUAL "")
            string(APPEND _gsplat_pytest_python_path "\n")
        endif()
        string(APPEND _gsplat_pytest_python_path "    \"${_gsplat_python_root}\"")
    endforeach()

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
