# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(_gsplat_validate_torch_cuda_versions torch_cuda_version compiler_cuda_version)
    if(torch_cuda_version STREQUAL "")
        message(
            FATAL_ERROR
            "The active PyTorch is not CUDA-enabled. Building gsplat requires "
            "a PyTorch package built for the CUDA compiler's major version."
        )
    endif()

    if(NOT torch_cuda_version MATCHES "^([0-9]+)(\\.[0-9]+)*$")
        message(FATAL_ERROR "PyTorch reported an invalid CUDA version: ${torch_cuda_version}")
    endif()
    set(_gsplat_torch_cuda_major "${CMAKE_MATCH_1}")

    if(NOT compiler_cuda_version MATCHES "^([0-9]+)(\\.[0-9]+)*$")
        message(
            FATAL_ERROR
            "CMake reported an invalid CUDA compiler version: ${compiler_cuda_version}"
        )
    endif()
    set(_gsplat_compiler_cuda_major "${CMAKE_MATCH_1}")

    if(NOT _gsplat_torch_cuda_major STREQUAL _gsplat_compiler_cuda_major)
        message(
            FATAL_ERROR
            "The CUDA ${_gsplat_compiler_cuda_major} compiler selected for gsplat does not "
            "match the CUDA ${_gsplat_torch_cuda_major} ABI used to build PyTorch. "
            "Select a matching CUDA compiler and PyTorch package."
        )
    endif()
endfunction()

# Validate the active interpreter rather than inferring compatibility from the
# GPU driver. CuPy wheels and PyTorch expose CUDA-major-specific ABIs, while a
# driver can support toolkits newer than the one used to build either package.
function(gsplat_check_torch_cuda_compatibility python_executable cuda_compiler_version)
    if(NOT python_executable)
        message(FATAL_ERROR "Python must be discovered before checking PyTorch's CUDA ABI.")
    endif()

    execute_process(
        COMMAND "${python_executable}" -c "import torch; print(torch.version.cuda or '')"
        RESULT_VARIABLE _gsplat_torch_cuda_result
        OUTPUT_VARIABLE _gsplat_torch_cuda_version
        ERROR_VARIABLE _gsplat_torch_cuda_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_gsplat_torch_cuda_result)
        message(
            FATAL_ERROR
            "Failed to query torch.version.cuda from ${python_executable}:\n"
            "${_gsplat_torch_cuda_error}"
        )
    endif()

    _gsplat_validate_torch_cuda_versions("${_gsplat_torch_cuda_version}" "${cuda_compiler_version}")
endfunction()
