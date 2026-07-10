# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_FOLDER third_party)

function(_gsplat_require_third_party_submodule source_dir name required_by)
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "${required_by} requires a populated ${name} source tree at "
            "'${source_dir}'; for the in-tree default run: "
            "git submodule update --init third_party/${name}"
        )
    endif()
endfunction()

# GLM ========================================================================
block()
    # A parent may already vendor GLM; add_subdirectory has no self-guard.
    if(NOT TARGET glm-header-only)
        _gsplat_require_third_party_submodule(
            "${CMAKE_CURRENT_SOURCE_DIR}/glm"
            glm
            "the gsplat CUDA build"
        )

        # gsplat consumes GLM header-only: skip the static library and keep
        # GLM's install rules out of the wheel components.
        set(GLM_BUILD_LIBRARY OFF)
        set(GLM_BUILD_INSTALL OFF)
        add_subdirectory(glm obj/glm SYSTEM EXCLUDE_FROM_ALL)
    endif()
endblock()
add_library(gsplat::ext::glm ALIAS glm-header-only)

# googletest ================================================================
if(GSPLAT_BUILD_TESTS)
    block()
        # A parent may already vendor googletest; add_subdirectory has no
        # self-guard, and the FORCE-set options below must not clobber a
        # parent's gtest configuration.
        if(NOT TARGET gtest)
            _gsplat_require_third_party_submodule(
                "${CMAKE_CURRENT_SOURCE_DIR}/googletest"
                googletest
                "GSPLAT_BUILD_TESTS=ON"
            )

            # Vendored code is not held to gsplat's target-only link policy;
            # a googletest update must not fail gsplat's generate step.
            set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS OFF)
            set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation." FORCE)
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            add_subdirectory(googletest EXCLUDE_FROM_ALL)
        endif()
    endblock()
endif()
