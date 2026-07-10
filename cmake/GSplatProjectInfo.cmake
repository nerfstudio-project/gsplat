# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

function(_gsplat_normalize_cmake_project_version out_var version)
    # project(VERSION) accepts one to four numeric components. Keep Python's
    # local-version suffix in package metadata, but omit it at the CMake boundary.
    string(REGEX REPLACE "\\+.*$" "" _normalized_version "${version}")
    if(NOT _normalized_version MATCHES "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?(\\.[0-9]+)?$")
        message(
            FATAL_ERROR
            "Python project version '${version}' cannot be represented by "
            "CMake project(VERSION)."
        )
    endif()

    set(${out_var} "${_normalized_version}" PARENT_SCOPE)
endfunction()

function(gsplat_read_project_info out_name_var out_version_var)
    if(DEFINED SKBUILD_PROJECT_NAME)
        set(_project_name "${SKBUILD_PROJECT_NAME}")
    endif()

    if(DEFINED SKBUILD_PROJECT_VERSION)
        set(_project_version "${SKBUILD_PROJECT_VERSION}")
    endif()

    if(DEFINED _project_name AND DEFINED _project_version)
        _gsplat_normalize_cmake_project_version(_cmake_project_version "${_project_version}")
        set(${out_name_var} "${_project_name}" PARENT_SCOPE)
        set(${out_version_var} "${_cmake_project_version}" PARENT_SCOPE)
        return()
    endif()

    set(_pyproject_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../pyproject.toml")

    if(NOT EXISTS "${_pyproject_file}")
        message(FATAL_ERROR "pyproject.toml not found at ${_pyproject_file}")
    endif()

    get_filename_component(_gsplat_source_dir "${_pyproject_file}" DIRECTORY)

    file(STRINGS "${_pyproject_file}" _pyproject_lines)
    set(_section "")
    set(_version_is_dynamic FALSE)

    foreach(_line IN LISTS _pyproject_lines)
        string(STRIP "${_line}" _line)

        if(_line MATCHES "^\\[([^]]+)\\]$")
            set(_section "${CMAKE_MATCH_1}")
            continue()
        endif()

        if(_section STREQUAL "project")
            if(NOT DEFINED _project_name AND _line MATCHES "^name[ \t]*=[ \t]*[\"']([^\"']+)[\"']")
                set(_project_name "${CMAKE_MATCH_1}")
            elseif(
                NOT DEFINED _project_version
                AND _line MATCHES "^version[ \t]*=[ \t]*[\"']([^\"']+)[\"']"
            )
                set(_project_version "${CMAKE_MATCH_1}")
            elseif(_line MATCHES "dynamic[ \t]*=[ \t]*\\[[^]]*[\"']version[\"']")
                set(_version_is_dynamic TRUE)
            endif()
        elseif(_section STREQUAL "tool.scikit-build.metadata.version")
            if(_line MATCHES "^input[ \t]*=[ \t]*[\"']([^\"']+)[\"']")
                set(_version_input "${CMAKE_MATCH_1}")
            endif()
        endif()
    endforeach()

    if(NOT DEFINED _project_name)
        message(FATAL_ERROR "pyproject.toml must define [project].name")
    endif()

    # Resolve dynamic Python metadata for direct CMake configuration.
    if(NOT DEFINED _project_version AND _version_is_dynamic)
        if(NOT DEFINED _version_input)
            message(
                FATAL_ERROR
                "pyproject.toml declares dynamic version but does not define "
                "[tool.scikit-build.metadata.version].input"
            )
        endif()

        set(_version_file "${_gsplat_source_dir}/${_version_input}")

        if(NOT EXISTS "${_version_file}")
            message(FATAL_ERROR "pyproject.toml version input does not exist: ${_version_file}")
        endif()

        file(READ "${_version_file}" _version_contents)
        string(
            REGEX MATCH "__version__[ \t]*=[ \t]*[\"']([^\"']+)[\"']"
            _version_match
            "${_version_contents}"
        )
        if(CMAKE_MATCH_1)
            set(_project_version "${CMAKE_MATCH_1}")
        else()
            message(FATAL_ERROR "Could not parse __version__ from ${_version_file}")
        endif()
    endif()

    if(NOT DEFINED _project_version)
        message(
            FATAL_ERROR
            "pyproject.toml must define [project].version or dynamic version metadata"
        )
    endif()

    _gsplat_normalize_cmake_project_version(_cmake_project_version "${_project_version}")
    set(${out_name_var} "${_project_name}" PARENT_SCOPE)
    set(${out_version_var} "${_cmake_project_version}" PARENT_SCOPE)
endfunction()

# Return the minimum interpreter version encoded in [project].requires-python.
function(gsplat_read_requires_python out_version_var)
    set(_pyproject_file "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../pyproject.toml")
    if(NOT EXISTS "${_pyproject_file}")
        message(FATAL_ERROR "pyproject.toml not found at ${_pyproject_file}")
    endif()

    file(STRINGS "${_pyproject_file}" _pyproject_lines)
    set(_section "")
    foreach(_line IN LISTS _pyproject_lines)
        string(STRIP "${_line}" _line)
        if(_line MATCHES "^\\[([^]]+)\\]$")
            set(_section "${CMAKE_MATCH_1}")
        elseif(
            _section STREQUAL "project"
            AND _line MATCHES "^requires-python[ \t]*=[ \t]*[\"']([^\"']+)[\"']"
        )
            set(_requires_python "${CMAKE_MATCH_1}")
        endif()
    endforeach()

    if(NOT DEFINED _requires_python)
        message(FATAL_ERROR "pyproject.toml must define [project].requires-python")
    endif()

    # Only the lower bound maps to find_package's minimum-version argument.
    if(NOT _requires_python MATCHES ">=[ \t]*([0-9]+(\\.[0-9]+)*)")
        message(
            FATAL_ERROR
            "Cannot derive a minimum interpreter version from "
            "requires-python = \"${_requires_python}\"."
        )
    endif()

    set(${out_version_var} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()
