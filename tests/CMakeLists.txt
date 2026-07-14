# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the configuration used by direct build-tree pytest runs and CTest.
# Limit conftest discovery to the test tree so parent projects cannot inject
# unrelated pytest plugins into gsplat's suite.
include("${GSPLAT_SOURCE_DIR}/cmake/GSplatPytest.cmake")

if(GSPLAT_TESTS_FORCE_CUDA)
    set(_gsplat_tests_force_cuda 1)
else()
    set(_gsplat_tests_force_cuda 0)
endif()

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

    # Derive the expected device-image census from the same concrete policy
    # that initialized every target. Native has already been resolved here.
    set(_gsplat_expected_real_arches "")
    set(_gsplat_expected_virtual_arches "")
    set(_gsplat_concrete_arch_policy TRUE)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
        if(_arch MATCHES "^([0-9]+[a-z]?)(-(real|virtual))?$")
            set(_numeric_arch "${CMAKE_MATCH_1}")
            set(_arch_kind "${CMAKE_MATCH_3}")
            if(NOT _arch_kind STREQUAL "virtual")
                list(APPEND _gsplat_expected_real_arches "${_numeric_arch}")
            endif()
            if(NOT _arch_kind STREQUAL "real")
                list(APPEND _gsplat_expected_virtual_arches "${_numeric_arch}")
            endif()
        else()
            # Special policies such as all are compiler-expanded and cannot be
            # predicted from the CMake spelling. Their command-line coverage
            # remains active, but skip the exact binary census below.
            set(_gsplat_concrete_arch_policy FALSE)
        endif()
    endforeach()
    foreach(_arch_list IN ITEMS _gsplat_expected_real_arches _gsplat_expected_virtual_arches)
        list(REMOVE_DUPLICATES ${_arch_list})
        list(SORT ${_arch_list} COMPARE NATURAL ORDER ASCENDING)
    endforeach()

    # Collect the current final artifacts with direct CUDA sources across the
    # complete project tree. The compile-database test separately audits every
    # CUDA object, including private object-library outputs.
    function(_gsplat_collect_buildsystem_targets out_var directory)
        get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        foreach(_subdirectory IN LISTS _subdirectories)
            _gsplat_collect_buildsystem_targets(_subdirectory_targets "${_subdirectory}")
            list(APPEND _targets ${_subdirectory_targets})
        endforeach()
        set(${out_var} "${_targets}" PARENT_SCOPE)
    endfunction()

    if(PROJECT_IS_TOP_LEVEL AND _gsplat_concrete_arch_policy)
        # CUDA container images do not always place the binary utilities next
        # to nvcc. Prefer that stable toolkit location. The current runner's
        # Triton package carries a compatible fallback, although that package
        # layout is private and should not replace cuobjdump in the dev image.
        get_filename_component(_gsplat_torch_site_packages "${TORCH_INSTALL_PREFIX}" DIRECTORY)
        find_program(
            _gsplat_cuobjdump
            NAMES cuobjdump
            HINTS
                "${CUDAToolkit_BIN_DIR}"
                "${_gsplat_torch_site_packages}/triton/backends/nvidia/bin"
                NO_CACHE
        )
        if(NOT _gsplat_cuobjdump)
            message(FATAL_ERROR "cuobjdump is required to validate CUDA-bearing test artifacts.")
        endif()

        _gsplat_collect_buildsystem_targets(_gsplat_buildsystem_targets "${GSPLAT_SOURCE_DIR}")
        list(REMOVE_DUPLICATES _gsplat_buildsystem_targets)
        foreach(_target IN LISTS _gsplat_buildsystem_targets)
            get_target_property(_target_imported "${_target}" IMPORTED)
            get_target_property(_target_type "${_target}" TYPE)
            if(
                _target_imported
                OR NOT _target_type MATCHES "^(EXECUTABLE|MODULE_LIBRARY|SHARED_LIBRARY)$"
            )
                continue()
            endif()

            get_target_property(_target_sources "${_target}" SOURCES)
            set(_target_has_cuda FALSE)
            foreach(_source IN LISTS _target_sources)
                if(_source MATCHES "\\.cu$")
                    set(_target_has_cuda TRUE)
                    break()
                endif()
            endforeach()
            if(NOT _target_has_cuda)
                continue()
            endif()

            string(MAKE_C_IDENTIFIER "${_target}" _test_target_id)
            add_test(
                NAME "cuda_arch_artifact_${_test_target_id}"
                COMMAND
                    "${CMAKE_COMMAND}" "-DGSPLAT_CUOBJDUMP=${_gsplat_cuobjdump}"
                    "-DGSPLAT_ARTIFACT=$<TARGET_FILE:${_target}>"
                    "-DGSPLAT_EXPECTED_REAL_ARCHITECTURES=${_gsplat_expected_real_arches}"
                    "-DGSPLAT_EXPECTED_VIRTUAL_ARCHITECTURES=${_gsplat_expected_virtual_arches}" -P
                    "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatCudaArtifactTest.cmake"
            )
            set_tests_properties(
                "cuda_arch_artifact_${_test_target_id}"
                PROPERTIES LABELS "binary;cmake;cuda"
            )
        endforeach()
    endif()

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

    add_test(
        NAME cmake_cuda_arch
        COMMAND
            "${CMAKE_COMMAND}" "-DGSPLAT_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}" -P
            "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatCudaArchTest.cmake"
    )
    set_tests_properties(cmake_cuda_arch PROPERTIES LABELS "cmake;source")

    if(GSPLAT_BUILD_TESTS AND _gsplat_concrete_arch_policy)
        add_test(
            NAME cmake_cuda_arch_build_tree
            COMMAND
                "${CMAKE_COMMAND}" "-DGSPLAT_BINARY_DIR=${GSPLAT_BINARY_DIR}"
                "-DGSPLAT_CUOBJDUMP=${_gsplat_cuobjdump}"
                "-DGSPLAT_EXPECTED_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}" -P
                "${GSPLAT_SOURCE_DIR}/tests/cmake/GSplatCudaArchBuildTest.cmake"
        )
        set_tests_properties(cmake_cuda_arch_build_tree PROPERTIES LABELS "binary;cmake;cuda")
    endif()

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
                ENVIRONMENT
                    "PYTHONPYCACHEPREFIX=${GSPLAT_BINARY_DIR}/pycache;GSPLAT_TESTS_FORCE_CUDA=${_gsplat_tests_force_cuda}"
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
