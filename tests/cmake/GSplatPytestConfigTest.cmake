# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${GSPLAT_SOURCE_DIR}/cmake/GSplatPytest.cmake")

set(_test_root "${GSPLAT_BINARY_DIR}/pytest-config-test")
file(MAKE_DIRECTORY "${_test_root}")
file(
    WRITE "${_test_root}/test_warning.py"
    "import warnings\n\ndef test_warning():\n    warnings.warn('probe', UserWarning)\n"
)
function(_run_warning_probe warnings_as_errors expected_result)
    if(warnings_as_errors)
        set(_mode "on")
    else()
        set(_mode "off")
    endif()

    set(CMAKE_COMPILE_WARNING_AS_ERROR "${warnings_as_errors}")
    set(_config "${_test_root}/pytest-${_mode}.ini")
    gsplat_configure_pytest_ini("${_config}" "${_test_root}" ".")

    file(READ "${_config}" _config_text)
    set(_expected_rule "filterwarnings =\n    error")
    string(FIND "${_config_text}" "${_expected_rule}" _rule_index)
    if(warnings_as_errors AND _rule_index EQUAL -1)
        message(FATAL_ERROR "warning-as-error pytest config is missing its error rule")
    endif()
    if(NOT warnings_as_errors AND NOT _rule_index EQUAL -1)
        message(FATAL_ERROR "ordinary pytest config unexpectedly promotes warnings")
    endif()

    execute_process(
        COMMAND "${GSPLAT_PYTHON}" -m pytest --quiet -c "${_config}" "${_test_root}/test_warning.py"
        WORKING_DIRECTORY "${_test_root}"
        RESULT_VARIABLE _result
        OUTPUT_VARIABLE _stdout
        ERROR_VARIABLE _stderr
    )
    if(NOT "${_result}" STREQUAL "${expected_result}")
        message(
            FATAL_ERROR
            "pytest warning probe (${_mode}) returned ${_result}, expected "
            "${expected_result}\nstdout:\n${_stdout}\nstderr:\n${_stderr}"
        )
    endif()
endfunction()

cmake_language(CALL _run_warning_probe ON 1)
cmake_language(CALL _run_warning_probe OFF 0)
