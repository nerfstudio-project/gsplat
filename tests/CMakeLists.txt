# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the configuration used by direct build-tree pytest runs and CTest.
# Limit conftest discovery to the test tree so parent projects cannot inject
# unrelated pytest plugins into gsplat's suite.
include("${GSPLAT_SOURCE_DIR}/cmake/GSplatPytest.cmake")

gsplat_configure_pytest_ini("${GSPLAT_BINARY_DIR}/pytest.ini" "${GSPLAT_SOURCE_DIR}/tests" ".")

# Keep the command implementation with the test sources, but install it as an
# importable module so the test entry point can resolve it.
# The runtime component keeps that static entry point valid when a wheel omits
# the optional test payload.
install(
    FILES "${GSPLAT_SOURCE_DIR}/tests/gsplat-test.py"
    DESTINATION gsplat
    RENAME _test.py
    COMPONENT gsplat_runtime
)

if(GSPLAT_BUILD_TESTS)
    add_subdirectory(cpp)

    # Unlike the build-tree configuration, the installed test configuration
    # must remain valid after the wheel is relocated. Both roots are relative to
    # _testdata: ../.. names the directory holding the installed gsplat package,
    # and examples is the packaged examples/ tree, so example modules' sibling
    # imports (e.g. av_trainer's `from utils import ...`) resolve from the wheel.
    gsplat_configure_pytest_ini(
        "${CMAKE_CURRENT_BINARY_DIR}/pytest-wheel.ini"
        "tests"
        "../.."
        "examples"
    )

    # Preserve the source-tree layout beneath one private root. Existing
    # tests can therefore keep their relative paths to assets, examples,
    # and sibling test helpers when run from an installed wheel.
    install(
        DIRECTORY "${GSPLAT_SOURCE_DIR}/tests/"
        DESTINATION gsplat/_testdata/tests
        COMPONENT gsplat_tests
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.json"
        PATTERN "__pycache__" EXCLUDE
        PATTERN "cpp" EXCLUDE
        PATTERN "gsplat-test.py" EXCLUDE
        PATTERN "source" EXCLUDE
    )

    # The tests/ install above ships only *.py/*.json, which drops the ncore CI
    # fixture's binary scene stores (*.zarr.itar). Ship the whole test_data tree
    # so the ncore e2e tests can load their scene from an installed wheel.
    install(
        DIRECTORY "${GSPLAT_SOURCE_DIR}/tests/examples/test_data/"
        DESTINATION gsplat/_testdata/tests/examples/test_data
        COMPONENT gsplat_tests
        PATTERN "__pycache__" EXCLUDE
    )

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/pytest-wheel.ini"
        DESTINATION gsplat/_testdata
        RENAME pytest.ini
        COMPONENT gsplat_tests
    )

    install(
        FILES "${GSPLAT_SOURCE_DIR}/assets/test_garden.npz"
        DESTINATION gsplat/_testdata/assets
        COMPONENT gsplat_tests
    )

    # Ship the entire examples/ tree so an installed wheel can use it exactly as
    # a source checkout would: scripts, data, requirements, and the Python
    # modules that import each other (e.g. av_trainer -> utils, simple_trainer ->
    # datasets.*), which a hand-maintained allowlist silently breaks whenever a
    # new intra-example import is added. Resolved at install time, so files added
    # after configure are still packaged. Tracked-exact gathering is deferred
    # (see .claude/plans/cmake-tracked-testdata-install.md); this interim copy
    # also picks up any untracked files present in a dirty worktree.
    install(
        DIRECTORY "${GSPLAT_SOURCE_DIR}/examples/"
        DESTINATION gsplat/_testdata/examples
        COMPONENT gsplat_tests
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )
endif()

# Register Python source-tree tests independently of the optional C++ test
# target. A nested subproject must not inject tests into its parent's CTest.
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
