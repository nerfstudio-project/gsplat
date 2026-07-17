# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/GSplatTorchCuda.cmake")

if(DEFINED GSPLAT_TORCH_CUDA_CHILD)
    if(GSPLAT_TORCH_CUDA_USE_FIND_MODULE)
        set(Python_EXECUTABLE "${GSPLAT_PYTHON}")
        set(CMAKE_CUDA_COMPILER_VERSION "${GSPLAT_COMPILER_CUDA}")
        list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake")
        find_package(Torch MODULE REQUIRED)
    else()
        gsplat_check_torch_cuda_compatibility("${GSPLAT_PYTHON}" "${GSPLAT_COMPILER_CUDA}")
    endif()
    return()
endif()

if(NOT GSPLAT_BINARY_DIR OR NOT GSPLAT_PYTHON)
    message(FATAL_ERROR "GSPLAT_BINARY_DIR and GSPLAT_PYTHON are required")
endif()

function(
    _gsplat_run_torch_cuda_case
    name
    torch_cuda
    use_find_module
    should_succeed
    expected_diagnostic
)
    set(_case_dir "${GSPLAT_BINARY_DIR}/torch-cuda-compatibility/${name}")
    file(MAKE_DIRECTORY "${_case_dir}")
    file(
        WRITE "${_case_dir}/torch.py"
        "class _Version:\n"
        "    cuda = ${torch_cuda}\n"
        "version = _Version()\n"
    )

    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" -E env "PYTHONPATH=${_case_dir}" "${CMAKE_COMMAND}"
            -DGSPLAT_TORCH_CUDA_CHILD=ON "-DGSPLAT_TORCH_CUDA_USE_FIND_MODULE=${use_find_module}"
            "-DGSPLAT_PYTHON=${GSPLAT_PYTHON}" -DGSPLAT_COMPILER_CUDA=12.6.85 -DSKBUILD=ON
            -DGSPLAT_CHECK_PYTHON_DEPS=OFF -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
        WORKING_DIRECTORY "${_case_dir}"
    )
    string(CONCAT _output "${_stdout}" "${_stderr}")

    if(should_succeed)
        if(_result)
            message(FATAL_ERROR "${name} unexpectedly failed:\n${_output}")
        endif()
    elseif(NOT _result)
        message(FATAL_ERROR "${name} unexpectedly succeeded")
    endif()

    if(expected_diagnostic AND NOT _output MATCHES "${expected_diagnostic}")
        message(
            FATAL_ERROR
            "${name} did not report the expected diagnostic "
            "'${expected_diagnostic}':\n${_output}"
        )
    endif()
endfunction()

_gsplat_run_torch_cuda_case("matching" [["12.8"]] OFF TRUE "")
_gsplat_run_torch_cuda_case("mismatch" [["13.0"]] ON FALSE "does not match the CUDA 13 ABI")
_gsplat_run_torch_cuda_case("cpu" "None" ON FALSE "PyTorch is not CUDA-enabled")
_gsplat_run_torch_cuda_case("malformed" [["unknown"]] OFF FALSE "invalid CUDA version")
