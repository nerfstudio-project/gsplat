# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

# Development.Embed provides Python::Python (libpython) for the standalone
# test executable; the rest of the build needs only Development.Module.
find_package(Python 3 COMPONENTS Development.Embed REQUIRED)

add_executable(
    gsplat_cpp_tests
    # New C++/CUDA gtest sources need an explicit edit here.
    test_channel_chunks.cpp
    test_common.cpp
    test_mathutils.cpp
    test_projection_ut_helpers.cu
    test_raster_eval3d.cpp
    test_torchutils.cpp
)

# The standalone executable and Python extension share one installed CUDA
# implementation library. Some core sources contain pybind11 camera wrappers,
# so the executable still needs libpython as well as torch_python. The tests
# include GoogleMock matchers, while gtest_main supplies main().
target_link_libraries(
    gsplat_cpp_tests
    PRIVATE gsplat_core gsplat::ext::torch_python Python::Python GTest::gmock GTest::gtest_main
)

set_target_properties(
    gsplat_cpp_tests
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${GSPLAT_BINARY_DIR}/bin" BUILD_RPATH_USE_ORIGIN ON
)

if(UNIX AND NOT APPLE)
    # The executable is installed two levels below the gsplat package. The
    # final entry reaches the virtual environment's lib directory in the
    # conventional <venv>/lib/pythonX.Y/site-packages layout; the Python runner
    # also augments LD_LIBRARY_PATH for layouts that differ from that pattern.
    set_target_properties(
        gsplat_cpp_tests
        PROPERTIES
            INSTALL_RPATH
                "$ORIGIN/../../lib;$ORIGIN/../../../torch/lib;$ORIGIN/../../../nvidia/cuda_runtime/lib;$ORIGIN/../../../../.."
    )
endif()

install(TARGETS gsplat_cpp_tests RUNTIME DESTINATION gsplat/_testdata/bin COMPONENT gsplat_tests)

# A nested subproject must not inject tests into its parent's CTest.
if(PROJECT_IS_TOP_LEVEL)
    gtest_discover_tests(gsplat_cpp_tests DISCOVERY_MODE PRE_TEST)
endif()
