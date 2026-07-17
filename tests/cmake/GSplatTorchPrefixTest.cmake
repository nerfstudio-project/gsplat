# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

if(NOT GSPLAT_BINARY_DIR OR NOT GSPLAT_PYTHON OR NOT GSPLAT_CMAKE_DIR)
    message(FATAL_ERROR "GSPLAT_BINARY_DIR, GSPLAT_PYTHON, and GSPLAT_CMAKE_DIR are required")
endif()

set(_fixture_root "${GSPLAT_BINARY_DIR}/torch-prefix-selection")
set(_project_dir "${_fixture_root}/project")

# Build two minimal torch package layouts. Empty library files are sufficient
# because this regression exercises configure-time selection, not the linker.
function(_gsplat_write_fake_torch name out_site out_root)
    set(_site_dir "${_fixture_root}/${name}/site-packages")
    set(_torch_root "${_site_dir}/torch")
    file(
        MAKE_DIRECTORY
            "${_torch_root}/include/ATen"
            "${_torch_root}/lib"
            "${_torch_root}/share/cmake/Caffe2"
            "${_torch_root}/share/cmake/Torch"
    )
    file(
        WRITE "${_torch_root}/__init__.py"
        "class _Version:\n"
        "    cuda = '12.8'\n"
        "version = _Version()\n"
    )
    file(WRITE "${_torch_root}/include/ATen/Config.h" "#define AT_PARALLEL_OPENMP 0\n")
    foreach(_library IN ITEMS torch torch_python)
        file(WRITE "${_torch_root}/lib/lib${_library}.so" "")
    endforeach()

    file(WRITE "${_torch_root}/share/cmake/Caffe2/Caffe2Config.cmake" "set(Caffe2_FOUND TRUE)\n")
    file(
        WRITE "${_torch_root}/share/cmake/Torch/TorchConfig.cmake"
        [=[
if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    set(TORCH_INSTALL_PREFIX "$ENV{TORCH_INSTALL_PREFIX}")
else()
    get_filename_component(TORCH_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
endif()
find_package(Caffe2 CONFIG REQUIRED)
set(
    TORCH_INCLUDE_DIRS
    "${TORCH_INSTALL_PREFIX}/include"
    "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
)
find_library(TORCH_LIBRARY NAMES torch PATHS "${TORCH_INSTALL_PREFIX}/lib" REQUIRED)
set(TORCH_LIBRARIES "${TORCH_LIBRARY}")
set(Torch_FOUND TRUE)
]=]
    )

    set(${out_site} "${_site_dir}" PARENT_SCOPE)
    set(${out_root} "${_torch_root}" PARENT_SCOPE)
endfunction()

_gsplat_write_fake_torch("first" _first_site _first_root)
_gsplat_write_fake_torch("second" _second_site _second_root)

file(MAKE_DIRECTORY "${_project_dir}")
file(
    WRITE "${_project_dir}/CMakeLists.txt"
    [=[
cmake_minimum_required(VERSION 3.26)
project(gsplat_torch_prefix_test LANGUAGES NONE)

list(PREPEND CMAKE_MODULE_PATH "${GSPLAT_CMAKE_DIR}")
set(Python_EXECUTABLE "${GSPLAT_PYTHON}")
set(CMAKE_CUDA_COMPILER_VERSION "12.8")
find_package(Torch MODULE REQUIRED)

file(
    WRITE "${CMAKE_BINARY_DIR}/selected-prefix.txt"
    "Torch_DIR=${Torch_DIR}\n"
    "TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}\n"
    "GSPLAT_TORCH_INSTALL_PREFIX=${GSPLAT_TORCH_INSTALL_PREFIX}\n"
)
]=]
)

function(_gsplat_check_bound_tree)
    set(_build_dir "${_fixture_root}/build")
    set(_common_args
        -S
        "${_project_dir}"
        -B
        "${_build_dir}"
        "-DGSPLAT_CMAKE_DIR=${GSPLAT_CMAKE_DIR}"
        "-DGSPLAT_PYTHON=${GSPLAT_PYTHON}"
    )

    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_first_site}"
            "TORCH_INSTALL_PREFIX=${_first_root}" "${CMAKE_COMMAND}" ${_common_args}
        RESULT_VARIABLE _first_result
        OUTPUT_VARIABLE _first_stdout
        ERROR_VARIABLE _first_stderr
    )
    if(_first_result)
        message(FATAL_ERROR "Initial configure failed:\n${_first_stdout}${_first_stderr}")
    endif()

    # An environment override must not displace the package imported by the
    # active Python when the build tree is reconfigured normally.
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_first_site}"
            "TORCH_INSTALL_PREFIX=${_second_root}" "${CMAKE_COMMAND}" ${_common_args}
        RESULT_VARIABLE _second_result
        OUTPUT_VARIABLE _second_stdout
        ERROR_VARIABLE _second_stderr
    )
    if(_second_result)
        message(
            FATAL_ERROR
            "Same-environment reconfigure failed:\n${_second_stdout}${_second_stderr}"
        )
    endif()

    # Switching the Python environment in place would leave PyTorch's private
    # cache entries ambiguous. Require a fresh build tree instead of trying to
    # maintain a release-specific list of entries to rewrite.
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_second_site}"
            "TORCH_INSTALL_PREFIX=${_first_root}" "${CMAKE_COMMAND}" ${_common_args}
        RESULT_VARIABLE _changed_result
        OUTPUT_VARIABLE _changed_stdout
        ERROR_VARIABLE _changed_stderr
    )
    string(CONCAT _changed_output "${_changed_stdout}" "${_changed_stderr}")
    if(NOT _changed_result OR NOT _changed_output MATCHES "build tree is bound to")
        message(
            FATAL_ERROR
            "Changing the active Python torch root did not fail clearly:\n"
            "${_changed_output}"
        )
    endif()

    file(READ "${_build_dir}/selected-prefix.txt" _selection)
    string(FIND "${_selection}" "${_second_root}" _changed_prefix_position)
    if(NOT _changed_prefix_position EQUAL -1)
        message(FATAL_ERROR "Reconfigure mixed torch prefixes:\n${_selection}")
    endif()

    foreach(_selection_name IN ITEMS Torch_DIR TORCH_INSTALL_PREFIX GSPLAT_TORCH_INSTALL_PREFIX)
        string(REGEX MATCH "(^|\n)${_selection_name}=([^\n]+)" _selection_line "${_selection}")
        if(NOT _selection_line)
            message(FATAL_ERROR "${_selection_name} was not recorded:\n${_selection}")
        endif()
        set(_selection_value "${CMAKE_MATCH_2}")
        cmake_path(
            IS_PREFIX
            _first_root
            "${_selection_value}"
            NORMALIZE
            _selection_uses_bound_torch
        )
        if(NOT _selection_uses_bound_torch)
            message(
                FATAL_ERROR
                "${_selection_name} did not use the bound Python torch prefix: "
                "${_selection_value}"
            )
        endif()
    endforeach()
endfunction()

_gsplat_check_bound_tree()
