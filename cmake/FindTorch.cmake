# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Project-local Torch wrapper.
#
# PyTorch ships a config package (TorchConfig.cmake), not a CMake module. This
# file is intentionally a thin module-mode wrapper around that config package:
# it performs the gsplat-specific setup that must happen immediately before
# PyTorch's package file runs, while keeping the top-level CMakeLists.txt
# readable as plain find_package(...) calls. Torch's package file injects
# -gencode flags that only the caller's directory scope can replace, so the
# caller must finalize the CUDA architectures after this module succeeds.

if(NOT Python_EXECUTABLE)
    find_package(Python 3 COMPONENTS Interpreter REQUIRED)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/GSplatTorchCuda.cmake")
# Deliberately validate on every configure: the Python environment at a cached
# interpreter path can change while a build tree persists.
gsplat_check_torch_cuda_compatibility("${Python_EXECUTABLE}" "${CMAKE_CUDA_COMPILER_VERSION}")

# Resolve TorchConfig.cmake from the active Python environment on every
# configure. Python environments can be replaced at the same interpreter path,
# and CMake otherwise keeps package and library paths from the previous torch.
execute_process(
    COMMAND
        "${Python_EXECUTABLE}" -c
        "import importlib.util as u; from pathlib import Path; print((Path(u.find_spec('torch').origin).resolve().parent / 'share' / 'cmake').resolve())"
    OUTPUT_VARIABLE _gsplat_torch_cmake_prefix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE _gsplat_torch_prefix_error
    RESULT_VARIABLE _gsplat_torch_prefix_result
)
if(
    _gsplat_torch_prefix_result
    OR NOT EXISTS "${_gsplat_torch_cmake_prefix}/Torch/TorchConfig.cmake"
    OR NOT EXISTS "${_gsplat_torch_cmake_prefix}/Caffe2/Caffe2Config.cmake"
)
    message(
        FATAL_ERROR
        "Failed to locate the CMake package files from the torch package used by "
        "${Python_EXECUTABLE}:\n${_gsplat_torch_prefix_error}"
    )
endif()

file(REAL_PATH "${_gsplat_torch_cmake_prefix}/../.." _gsplat_torch_root)

# PyTorch's config package caches private helper and library paths. Rebinding
# only the known variables would risk mixing a new Python package with a stale
# path added by a future PyTorch release. Treat Torch like the compiler: a
# configured build tree remains bound to its original installation.
set(_cached_torch_root "${GSPLAT_TORCH_INSTALL_PREFIX}")
if(_cached_torch_root)
    if(NOT _cached_torch_root STREQUAL _gsplat_torch_root)
        message(
            FATAL_ERROR
            "The active Python interpreter resolves torch at ${_gsplat_torch_root}, "
            "but this build tree is bound to ${_cached_torch_root}. "
            "Configure a fresh build tree when changing Python or Torch environments."
        )
    endif()
endif()

set(GSPLAT_TORCH_INSTALL_PREFIX
    "${_gsplat_torch_root}"
    CACHE INTERNAL
    "Torch package root selected from the active Python interpreter."
    FORCE
)
set(Torch_DIR
    "${_gsplat_torch_cmake_prefix}/Torch"
    CACHE PATH
    "Torch CMake package directory selected from the active Python interpreter."
    FORCE
)
set(Torch_DIR "${_gsplat_torch_cmake_prefix}/Torch")
set(Caffe2_DIR
    "${_gsplat_torch_cmake_prefix}/Caffe2"
    CACHE PATH
    "Caffe2 CMake package directory shipped by the active Python torch package."
    FORCE
)
set(Caffe2_DIR "${_gsplat_torch_cmake_prefix}/Caffe2")

set(_gsplat_torch_saved_cmake_prefix_path "${CMAKE_PREFIX_PATH}")
# The package prefix selects the config files, while the torch root ensures
# find_library calls inside those files search this installation first.
list(PREPEND CMAKE_PREFIX_PATH "${_gsplat_torch_cmake_prefix}" "${_gsplat_torch_root}")

set(_gsplat_torch_config_args)
if(Torch_FIND_VERSION)
    list(APPEND _gsplat_torch_config_args "${Torch_FIND_VERSION}")
    if(Torch_FIND_VERSION_EXACT)
        list(APPEND _gsplat_torch_config_args EXACT)
    endif()
endif()

list(APPEND _gsplat_torch_config_args CONFIG)
if(Torch_FIND_QUIETLY)
    list(APPEND _gsplat_torch_config_args QUIET)
endif()
if(Torch_FIND_REQUIRED)
    list(APPEND _gsplat_torch_config_args REQUIRED)
endif()
if(Torch_FIND_COMPONENTS)
    list(APPEND _gsplat_torch_config_args COMPONENTS ${Torch_FIND_COMPONENTS})
endif()

# Torch uses this hash in its CUDA configuration. Compute it here because its
# package file attempts to execute the Python imported-target name as a command,
# which cannot work with execute_process().
if(TARGET CUDA::nvrtc AND NOT CUDA_NVRTC_SHORTHASH)
    get_target_property(_gsplat_nvrtc_library CUDA::nvrtc IMPORTED_LOCATION)
    if(_gsplat_nvrtc_library AND EXISTS "${_gsplat_nvrtc_library}")
        file(SHA256 "${_gsplat_nvrtc_library}" _gsplat_nvrtc_sha256)
        string(SUBSTRING "${_gsplat_nvrtc_sha256}" 0 8 CUDA_NVRTC_SHORTHASH)
    endif()
endif()

# TorchConfig probes for a static libkineto.a unconditionally and warns when
# the probe fails, but wheel builds link kineto into libtorch_cpu instead of
# shipping the archive. Pre-seed the probe result here:
# - an actual libkineto.a wins
# - otherwise fall back to the library that carries kineto (torch_cpu)
unset(_gsplat_torch_kineto_library)
find_library(
    _gsplat_torch_kineto_library
    kineto
    PATHS "${_gsplat_torch_root}/lib"
    NO_DEFAULT_PATH
    NO_CACHE
)
if(NOT _gsplat_torch_kineto_library)
    find_library(
        _gsplat_torch_kineto_library
        torch_cpu
        PATHS "${_gsplat_torch_root}/lib"
        NO_DEFAULT_PATH
        NO_CACHE
    )
endif()
unset(kineto_LIBRARY CACHE)
unset(kineto_LIBRARY)
if(_gsplat_torch_kineto_library)
    set(kineto_LIBRARY
        "${_gsplat_torch_kineto_library}"
        CACHE FILEPATH
        "Library carrying kineto (linked into torch_cpu in wheel builds)."
        FORCE
    )
endif()

# TorchConfig gives an environment override precedence over its own location.
# The Python package is authoritative for a Python extension build, so pin the
# override only while loading that package and restore the caller's environment
# immediately afterwards.
set(_gsplat_torch_install_prefix_was_defined FALSE)
if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    set(_gsplat_torch_install_prefix_was_defined TRUE)
    set(_gsplat_torch_saved_install_prefix "$ENV{TORCH_INSTALL_PREFIX}")
endif()
set(ENV{TORCH_INSTALL_PREFIX} "${_gsplat_torch_root}")

# Torch's package file ignores CMAKE_CUDA_ARCHITECTURES and warns whenever it
# is defined; it reads TORCH_CUDA_ARCH_LIST instead. Hide the variable in both
# scopes during discovery and restore it afterwards — the caller replaces
# Torch's architecture flags anyway, and the cache entry must survive so a
# user selection stays sticky across reconfigures.
set(_gsplat_torch_saved_cuda_architectures "${CMAKE_CUDA_ARCHITECTURES}")
set(_gsplat_torch_cuda_arch_cache_type "")
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    get_property(_gsplat_torch_cuda_arch_cache_value CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
    get_property(_gsplat_torch_cuda_arch_cache_type CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY TYPE)
    get_property(
        _gsplat_torch_cuda_arch_cache_help
        CACHE CMAKE_CUDA_ARCHITECTURES
        PROPERTY HELPSTRING
    )
endif()
unset(CMAKE_CUDA_ARCHITECTURES)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)

# CONFIG mode is what prevents recursion back into this FindTorch.cmake file.
find_package(Torch ${_gsplat_torch_config_args})

if(_gsplat_torch_install_prefix_was_defined)
    set(ENV{TORCH_INSTALL_PREFIX} "${_gsplat_torch_saved_install_prefix}")
else()
    unset(ENV{TORCH_INSTALL_PREFIX})
endif()

if(NOT _gsplat_torch_cuda_arch_cache_type STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES
        "${_gsplat_torch_cuda_arch_cache_value}"
        CACHE ${_gsplat_torch_cuda_arch_cache_type}
        "${_gsplat_torch_cuda_arch_cache_help}"
    )
endif()
set(CMAKE_CUDA_ARCHITECTURES "${_gsplat_torch_saved_cuda_architectures}")
set(CMAKE_PREFIX_PATH "${_gsplat_torch_saved_cmake_prefix_path}")

if(NOT Torch_FOUND)
    return()
endif()

unset(_gsplat_torch_python_library)
find_library(
    _gsplat_torch_python_library
    NAMES torch_python
    PATHS "${_gsplat_torch_root}/lib"
    NO_DEFAULT_PATH
    NO_CACHE
    REQUIRED
)
set(TORCH_PYTHON_LIBRARY
    "${_gsplat_torch_python_library}"
    CACHE FILEPATH
    "Python library shipped by the active Python torch package."
    FORCE
)

# ATen/Config.h records the parallel backend selected when this exact Torch
# package was built. Reading it avoids importing torch a second time.
set(_gsplat_torch_parallel_config_found FALSE)
set(_gsplat_torch_uses_openmp FALSE)
foreach(_gsplat_torch_include_dir IN LISTS TORCH_INCLUDE_DIRS)
    set(_gsplat_torch_config_header "${_gsplat_torch_include_dir}/ATen/Config.h")
    if(EXISTS "${_gsplat_torch_config_header}")
        file(
            STRINGS "${_gsplat_torch_config_header}"
            _gsplat_torch_openmp_definition
            REGEX "^[ \t]*#[ \t]*define[ \t]+AT_PARALLEL_OPENMP[ \t]+[01]([ \t]|$)"
        )
        if(_gsplat_torch_openmp_definition)
            set(_gsplat_torch_parallel_config_found TRUE)
            if(_gsplat_torch_openmp_definition MATCHES "AT_PARALLEL_OPENMP[ \t]+1([ \t]|$)")
                set(_gsplat_torch_uses_openmp TRUE)
            endif()
            break()
        endif()
    endif()
endforeach()
if(NOT _gsplat_torch_parallel_config_found)
    message(
        FATAL_ERROR
        "Torch's include directories do not contain an ATen/Config.h with "
        "a supported AT_PARALLEL_OPENMP definition: ${TORCH_INCLUDE_DIRS}"
    )
endif()

set(_gsplat_torch_openmp_flag "")
if(_gsplat_torch_uses_openmp AND NOT APPLE)
    if(WIN32)
        set(_gsplat_torch_openmp_flag "/openmp")
    else()
        set(_gsplat_torch_openmp_flag "-fopenmp")
    endif()
endif()

# Torch's package exposes most usage requirements through imported targets,
# but TORCH_LIBRARIES remains the authoritative complete link closure. Wrap it
# once so gsplat targets consume only target-based dependencies.
if(NOT TARGET gsplat::ext::torch)
    add_library(gsplat::ext::torch INTERFACE IMPORTED GLOBAL)
    target_link_libraries(gsplat::ext::torch INTERFACE ${TORCH_LIBRARIES})

    # Torch already links its OpenMP runtime. Propagate only the host compiler
    # mode; FindOpenMP would run unnecessary compiler probes for this known
    # package configuration.
    if(_gsplat_torch_openmp_flag)
        target_compile_definitions(gsplat::ext::torch INTERFACE AT_PARALLEL_OPENMP)
        target_compile_options(
            gsplat::ext::torch
            INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:${_gsplat_torch_openmp_flag}>"
        )
    endif()
endif()

if(NOT TARGET gsplat::ext::torch_python)
    add_library(gsplat::ext::torch_python INTERFACE IMPORTED GLOBAL)
    target_link_libraries(
        gsplat::ext::torch_python
        INTERFACE gsplat::ext::torch "${TORCH_PYTHON_LIBRARY}"
    )
endif()
