# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

option(GSPLAT_DEVELOPMENT_MODE "Enable the developer workflow" OFF)

# Run environment_check.py on the given pyproject.toml dependency sections.
# Report each satisfied requirement's installed version as STATUS and any
# unmet requirements at the given message level. A requirement repeated by a
# later section is checked once, under the first section declaring it.
function(_gsplat_check_dependency_section sections level)
    # Everything after the level is the hint, in as many fragments as the
    # caller needs.
    list(JOIN ARGN "" _hint)
    set(_checker "${GSPLAT_SOURCE_DIR}/gsplat/build_support/environment_check.py")
    execute_process(
        COMMAND
            "${Python_EXECUTABLE}" -B "${_checker}" "${GSPLAT_SOURCE_DIR}/pyproject.toml"
            ${sections}
        OUTPUT_VARIABLE _found
        ERROR_VARIABLE _unmet
        RESULT_VARIABLE _result
    )
    string(REPLACE ";" ", " _sections_label "${sections}")
    message(STATUS "Python dependencies (${_sections_label}):")
    string(STRIP "${_found}" _found)
    string(REPLACE "\n" ";" _found "${_found}")
    foreach(_line IN LISTS _found)
        message(STATUS "  Found ${_line}")
    endforeach()
    if(_result)
        message(
            ${level}
            "The Python environment at ${Python_EXECUTABLE} misses "
            "dependencies:\n${_unmet}${_hint}"
        )
    endif()
endfunction()

# Verify at configure time that the active Python environment satisfies the
# dependencies declared in pyproject.toml, reporting every unmet requirement
# at once instead of failing on the first missing import. The check only
# reports — installing stays an explicit developer action. Severity follows
# what each dependency gates:
# - build inputs (torch): required to compile and link, so configure fails
# - test extra: required by the enabled test targets, so configure fails
# - remaining runtime dependencies: gate importing the built package, not
#   producing it, so configure only warns
# The check is skipped when GSPLAT_CHECK_PYTHON_DEPS is off, for wheel builds
# (pip resolves and validates the dependencies itself), and for nested
# consumers (they own their environment).
function(gsplat_check_python_dependencies)
    if(NOT PROJECT_IS_TOP_LEVEL OR SKBUILD OR NOT GSPLAT_CHECK_PYTHON_DEPS)
        return()
    endif()

    set(_dev_extra "dev")

    _gsplat_check_dependency_section(
        "dependencies:torch"
        FATAL_ERROR
        "The build compiles and links against them. "
        "Install with: python3 -m pip install -e \".[${_dev_extra}]\""
    )
    # Mandatory sections first: a requirement both sections declare is then
    # reported under the more essential one.
    set(_fatal_sections "")
    if(GSPLAT_BUILD_TESTS)
        list(APPEND _fatal_sections test)
    endif()
    if(GSPLAT_DEVELOPMENT_MODE)
        list(APPEND _fatal_sections "${_dev_extra}")
    endif()
    if(_fatal_sections STREQUAL "test")
        set(_fatal_hint
            "The enabled tests need them; install with: python3 -m pip "
            "install -e \".[test]\" or configure with -DGSPLAT_BUILD_TESTS=OFF"
        )
    else()
        set(_fatal_hint
            "The developer workflow relies on them; install with: "
            "python3 -m pip install -e \".[${_dev_extra}]\""
        )
    endif()
    if(NOT _fatal_sections STREQUAL "")
        _gsplat_check_dependency_section("${_fatal_sections}" FATAL_ERROR ${_fatal_hint})
    endif()
    _gsplat_check_dependency_section(
        "dependencies"
        WARNING
        "The build does not need them, but importing the built package "
        "will fail until they are installed."
    )
endfunction()
