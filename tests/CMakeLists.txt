# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the configuration used by direct build-tree pytest runs and CTest.
# Limit conftest discovery to the test tree so parent projects cannot inject
# unrelated pytest plugins into gsplat's suite.
include("${GSPLAT_SOURCE_DIR}/cmake/GSplatPytest.cmake")

gsplat_configure_pytest_ini("${GSPLAT_BINARY_DIR}/pytest.ini" "${GSPLAT_SOURCE_DIR}/tests" ".")

if(GSPLAT_BUILD_TESTS)
    add_subdirectory(cpp)
endif()

# Python test suite registration -- top-level only (a nested subproject must
# not inject into the parent's CTest), but OUTSIDE the GSPLAT_BUILD_TESTS gate
# (which controls only the C++ gtests + the googletest submodule).
if(PROJECT_IS_TOP_LEVEL)
    add_test(
        NAME cmake_project_info
        COMMAND "${CMAKE_COMMAND}" -P "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatProjectInfoTest.cmake"
    )
    set_tests_properties(cmake_project_info PROPERTIES LABELS "cmake;source")

    add_test(
        NAME cmake_torch_cuda
        COMMAND
            "${CMAKE_COMMAND}" "-DGSPLAT_BINARY_DIR=${GSPLAT_BINARY_DIR}"
            "-DGSPLAT_PYTHON=${Python_EXECUTABLE}" -P
            "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatTorchCudaTest.cmake"
    )
    set_tests_properties(cmake_torch_cuda PROPERTIES LABELS "cmake;source")

    add_test(
        NAME cmake_torch_prefix
        COMMAND
            "${CMAKE_COMMAND}" "-DGSPLAT_BINARY_DIR=${GSPLAT_BINARY_DIR}"
            "-DGSPLAT_PYTHON=${Python_EXECUTABLE}" "-DGSPLAT_CMAKE_DIR=${GSPLAT_SOURCE_DIR}/cmake"
            -P "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatTorchPrefixTest.cmake"
    )
    set_tests_properties(cmake_torch_prefix PROPERTIES LABELS "cmake;source")

    add_test(
        NAME cmake_pytest_config
        COMMAND
            "${CMAKE_COMMAND}" "-DGSPLAT_SOURCE_DIR=${GSPLAT_SOURCE_DIR}"
            "-DGSPLAT_BINARY_DIR=${GSPLAT_BINARY_DIR}" "-DGSPLAT_PYTHON=${Python_EXECUTABLE}" -P
            "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatPytestConfigTest.cmake"
    )
    set_tests_properties(cmake_pytest_config PROPERTIES LABELS "cmake;source")

    add_test(
        NAME cmake_ccache
        COMMAND "${CMAKE_COMMAND}" -P "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatCCacheTest.cmake"
    )
    set_tests_properties(cmake_ccache PROPERTIES LABELS "cmake;source")

    function(gsplat_add_pytest name)
        set(_test_name "python_${name}")
        add_test(
            NAME "${_test_name}"
            COMMAND Python::Interpreter -m pytest -c "${GSPLAT_BINARY_DIR}/pytest.ini" ${ARGN}
            WORKING_DIRECTORY "${GSPLAT_BINARY_DIR}"
        )
        set_tests_properties(
            "${_test_name}"
            PROPERTIES
                # Keep source-path imports from creating __pycache__ entries
                # beneath the checkout during an out-of-tree CTest run.
                ENVIRONMENT "PYTHONPYCACHEPREFIX=${GSPLAT_BINARY_DIR}/pycache"
                LABELS "python;${name}"
        )
    endfunction()

    gsplat_add_pytest(core "${GSPLAT_SOURCE_DIR}/tests/core")
    gsplat_add_pytest(examples "${GSPLAT_SOURCE_DIR}/tests/examples")
    gsplat_add_pytest(experimental "${GSPLAT_SOURCE_DIR}/tests/experimental")
    gsplat_add_pytest(geometry "${GSPLAT_SOURCE_DIR}/tests/geometry")
    gsplat_add_pytest(scene "${GSPLAT_SOURCE_DIR}/tests/scene")
    gsplat_add_pytest(sensors "${GSPLAT_SOURCE_DIR}/tests/sensors")
    gsplat_add_pytest(source "${GSPLAT_SOURCE_DIR}/tests/source")
    gsplat_add_pytest(stage "${GSPLAT_SOURCE_DIR}/tests/stage")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Every Python extension module must export exactly its PyInit symbol.
        get_property(_gsplat_extension_targets GLOBAL PROPERTY GSPLAT_PYTHON_EXTENSION_TARGETS)
        foreach(_gsplat_extension IN LISTS _gsplat_extension_targets)
            add_test(
                NAME ${_gsplat_extension}_dynamic_exports
                COMMAND
                    "${CMAKE_COMMAND}" "-DGSPLAT_NM=${CMAKE_NM}"
                    "-DGSPLAT_MODULE=$<TARGET_FILE:${_gsplat_extension}>"
                    "-DGSPLAT_EXPECTED_SYMBOLS=PyInit_${_gsplat_extension}" -P
                    "${GSPLAT_SOURCE_DIR}/cmake/GSplatCheckDynamicExports.cmake"
            )
            set_tests_properties(
                ${_gsplat_extension}_dynamic_exports
                PROPERTIES LABELS "binary;python"
            )
        endforeach()
    endif()
endif()
