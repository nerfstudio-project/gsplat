# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compiler caching (ccache/sccache) integration.
#
# - GSPLAT_ENABLE_CCACHE (ON): use ccache as the C/C++/CUDA compiler launcher
#   when found. Incremental rebuilds, branch switches, and bisects can reuse
#   cached objects. ccache >= 4.6 handles nvcc natively. When ccache is absent
#   the build warns and compiles without it (unless GSPLAT_FORCE_CCACHE).
#   GSPLAT_CCACHE_EXECUTABLE is set to the resolved ccache when it is in use.
# - GSPLAT_FORCE_CCACHE (OFF): fail configuration if ccache is requested but not
#   found (for CI that wants the cache guaranteed).
# - GSPLAT_CCACHE_STATS (OFF): after `ninja`/`make`, print THIS build's ccache
#   statistics (hits/misses) and clear them, via a per-build CCACHE_STATSLOG.
# - GSPLAT_CCACHE_NORMALIZE_PATHS (ON): make cache keys portable across
#   worktrees by expressing paths relative to the gsplat source root. The
#   compiler prefix map is applied by GSplatCompileOptions.cmake.
#
# Call gsplat_configure_ccache() once, before any compilable target is defined,
# and gsplat_finalize_ccache_stats() once, after all targets are defined.

include_guard(GLOBAL)
include(CMakeDependentOption)

option(GSPLAT_ENABLE_CCACHE "Use ccache to speed up compilation, if available" ON)
cmake_dependent_option(
    GSPLAT_FORCE_CCACHE
    "Fail configuration if ccache is enabled but not found"
    OFF
    GSPLAT_ENABLE_CCACHE
    OFF
)
cmake_dependent_option(
    GSPLAT_CCACHE_STATS
    "Print this build's ccache statistics after building"
    OFF
    GSPLAT_ENABLE_CCACHE
    OFF
)
cmake_dependent_option(
    GSPLAT_CCACHE_NORMALIZE_PATHS
    "Normalize ccache paths so objects can be reused across worktrees"
    ON
    GSPLAT_ENABLE_CCACHE
    OFF
)
set(GSPLAT_CCACHE_DIR
    ""
    CACHE PATH
    "Cache directory ccache uses for this build; empty keeps ccache's own resolution"
)
function(gsplat_configure_ccache)
    if(NOT GSPLAT_ENABLE_CCACHE)
        return()
    endif()

    find_program(GSPLAT_CCACHE_EXECUTABLE NAMES ccache)
    if(NOT GSPLAT_CCACHE_EXECUTABLE)
        if(GSPLAT_FORCE_CCACHE)
            message(FATAL_ERROR "GSPLAT_FORCE_CCACHE=ON but no ccache executable was found.")
        endif()
        message(
            WARNING
            "GSPLAT_ENABLE_CCACHE=ON but no ccache executable was found; compiling without it."
        )
        return()
    endif()

    set(_common_env "")
    if(GSPLAT_CCACHE_DIR)
        # Pin the cache location at configure time so every compile uses it
        # regardless of the invoking environment (sandboxed drivers cannot
        # export per-compiler environment themselves).
        list(APPEND _common_env "CCACHE_DIR=${GSPLAT_CCACHE_DIR}")
    endif()
    if(GSPLAT_CCACHE_NORMALIZE_PATHS)
        # Each worktree has a different absolute source root. ccache rewrites
        # paths beneath BASEDIR before hashing. The matching compiler prefix
        # map also normalizes debug information and path-valued macros, which
        # lets ccache safely retain its default directory hashing behavior.
        list(APPEND _common_env "CCACHE_BASEDIR=${GSPLAT_SOURCE_DIR}")
    endif()
    if(GSPLAT_CCACHE_STATS)
        # Route ccache through `cmake -E env CCACHE_STATSLOG=...` so
        # `ccache --show-log-stats` can report just this build's activity.
        set(_gsplat_ccache_statslog
            "${GSPLAT_BINARY_DIR}/ccache-build-stats.log"
            CACHE INTERNAL
            "Per-build ccache stats log"
        )
        list(APPEND _common_env "CCACHE_STATSLOG=${_gsplat_ccache_statslog}")
    endif()

    if(_common_env)
        set(_launcher
            "${CMAKE_COMMAND}"
            -E
            env
            ${_common_env}
            --
            "${GSPLAT_CCACHE_EXECUTABLE}"
        )
    else()
        set(_launcher "${GSPLAT_CCACHE_EXECUTABLE}")
    endif()
    # NVCC preprocesses host and device code differently. Depend mode prevents
    # a direct miss from falling back to an incomplete device-only cache key.
    # Both modes ride in as environment variables: per-compilation KEY=VALUE
    # options on the ccache command line would require ccache >= 4.8.
    set(_cuda_launcher
        "${CMAKE_COMMAND}"
        -E
        env
        ${_common_env}
        CCACHE_DIRECT=true
        CCACHE_DEPEND=true
        --
        "${GSPLAT_CCACHE_EXECUTABLE}"
    )

    # Set in the caller's (root directory) scope so targets created afterwards
    # pick up the launcher at creation time.
    set(CMAKE_C_COMPILER_LAUNCHER "${_launcher}" PARENT_SCOPE)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${_launcher}" PARENT_SCOPE)
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${_cuda_launcher}" PARENT_SCOPE)
endfunction()

# Recursively collect BUILDSYSTEM_TARGETS from a directory and all of its
# subdirectories (CMake has no built-in recursive form), skipping vendored deps
# under third_party/. Needed because gsplat's compilable targets live in
# subdirectories (gsplat/cuda, tests) -- a single root-level query would miss
# them and the stats target would silently no-op.
function(_gsplat_collect_targets_recursive _out_var _dir)
    get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
    foreach(_sub IN LISTS _subdirs)
        if(_sub MATCHES "/third_party/")
            continue()
        endif()
        _gsplat_collect_targets_recursive(_sub_targets "${_sub}")
        list(APPEND _targets ${_sub_targets})
    endforeach()
    set(${_out_var} "${_targets}" PARENT_SCOPE)
endfunction()

function(gsplat_finalize_ccache_stats)
    # Sub-project builds defer statistics to the enclosing project.
    if(NOT PROJECT_IS_TOP_LEVEL)
        return()
    endif()

    if(NOT GSPLAT_CCACHE_STATS OR NOT GSPLAT_CCACHE_EXECUTABLE)
        return()
    endif()

    # Targets live in subdirectories, so a root-directory query alone would
    # miss them. Vendored dependencies under third_party/ are excluded.
    _gsplat_collect_targets_recursive(_targets "${CMAKE_CURRENT_SOURCE_DIR}")

    set(_compilable "")
    foreach(_t IN LISTS _targets)
        get_target_property(_type "${_t}" TYPE)
        if(
            _type STREQUAL "EXECUTABLE"
            OR _type STREQUAL "MODULE_LIBRARY"
            OR _type STREQUAL "SHARED_LIBRARY"
            OR _type STREQUAL "STATIC_LIBRARY"
            OR _type STREQUAL "OBJECT_LIBRARY"
        )
            list(APPEND _compilable "${_t}")
        endif()
    endforeach()
    if(NOT _compilable)
        return()
    endif()

    # Clear the stats log before the build (a prerequisite of every compilable
    # target), then print + clear it after (an ALL target that runs last).
    add_custom_target(
        gsplat_ccache_reset_stats
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${_gsplat_ccache_statslog}"
        COMMENT "Resetting per-build ccache stats"
        VERBATIM
    )
    add_custom_target(
        gsplat_ccache_stats
        ALL
        COMMAND "${CMAKE_COMMAND}" -E echo "-- ccache statistics for this build:"
        COMMAND
            "${CMAKE_COMMAND}" -E env "CCACHE_STATSLOG=${_gsplat_ccache_statslog}"
            "${GSPLAT_CCACHE_EXECUTABLE}" --show-log-stats -v
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${_gsplat_ccache_statslog}"
        DEPENDS ${_compilable}
        COMMENT "ccache build statistics"
        VERBATIM
    )
    foreach(_t IN LISTS _compilable)
        add_dependencies("${_t}" gsplat_ccache_reset_stats)
    endforeach()
endfunction()
