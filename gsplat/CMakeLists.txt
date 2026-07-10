# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------------------------------
# Build-tree Python package
# -----------------------------------------------------------------------------

set(_gsplat_python_package_dir "${GSPLAT_BINARY_DIR}/gsplat")
configure_file(
    "${GSPLAT_SOURCE_DIR}/cmake/GSplatBuildTreeInit.py.in"
    "${_gsplat_python_package_dir}/__init__.py"
    @ONLY
)

include(GSplatPythonExtension)

add_subdirectory(cuda)
add_subdirectory(geometry)
add_subdirectory(scene)
add_subdirectory(sensors)
add_subdirectory(experimental)
