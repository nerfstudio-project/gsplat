# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(CMAKE_FOLDER third_party)

# Warn or fail (per GSPLAT_STRICT_SUBMODULES) when a submodule is checked out at
# a commit other than the one the gsplat project pins. Only a git work tree can be
# checked; an exported source snapshot has no gitlink to compare against, so the
# comparison is skipped there.
function(_gsplat_check_submodule_revision source_dir name required_by)
    find_package(Git QUIET)
    if(NOT Git_FOUND)
        return()
    endif()
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse --show-superproject-working-tree
        WORKING_DIRECTORY "${source_dir}"
        OUTPUT_VARIABLE superproject
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE rc
        ERROR_QUIET
    )
    if(NOT rc EQUAL 0 OR superproject STREQUAL "")
        return()
    endif()
    file(RELATIVE_PATH relpath "${superproject}" "${source_dir}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule status "${relpath}"
        WORKING_DIRECTORY "${superproject}"
        OUTPUT_VARIABLE status
        RESULT_VARIABLE rc
        ERROR_QUIET
    )
    if(NOT rc EQUAL 0 OR status STREQUAL "")
        return()
    endif()
    # git submodule status leads each entry with a flag: ' ' at the pinned
    # commit, '+' a different commit, 'U' merge conflicts, '-' not initialized.
    string(SUBSTRING "${status}" 0 1 flag)
    if(flag STREQUAL " ")
        return()
    endif()
    set(detail
        "the ${name} submodule at '${source_dir}' is not at the commit "
        "${required_by} pins (git submodule status flag '${flag}'); run: "
        "git submodule update --init ${relpath}"
    )
    if(GSPLAT_STRICT_SUBMODULES)
        message(FATAL_ERROR ${detail})
    else()
        message(WARNING ${detail})
    endif()
endfunction()

function(_gsplat_require_third_party_submodule source_dir name required_by)
    if(NOT EXISTS "${source_dir}/CMakeLists.txt")
        message(
            FATAL_ERROR
            "${required_by} requires a populated ${name} source tree at "
            "'${source_dir}'; for the in-tree default run: "
            "git submodule update --init third_party/${name}"
        )
    endif()
    _gsplat_check_submodule_revision("${source_dir}" "${name}" "${required_by}")
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

            # The wheel installs one standalone executable, not GoogleTest's
            # runtime libraries. Shadow this standard CMake variable only within
            # the block so vendored gtest remains static even when a caller
            # configures the rest of the project with BUILD_SHARED_LIBS=ON.
            set(BUILD_SHARED_LIBS OFF)
            # Vendored code is not held to gsplat's target-only link policy;
            # a googletest update must not fail gsplat's generate step.
            set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS OFF)
            set(INSTALL_GTEST OFF CACHE BOOL "Disable GoogleTest installation." FORCE)
            set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
            add_subdirectory(googletest EXCLUDE_FROM_ALL)
        endif()
    endblock()
endif()
