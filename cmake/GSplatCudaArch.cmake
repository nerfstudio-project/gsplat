# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

# CMP0104 controls CUDA architecture initialization from CUDAARCHS / compiler
# defaults. Keep NEW semantics so architecture handling is explicit and
# warning-free.
cmake_policy(SET CMP0104 NEW)

function(_gsplat_split_cuda_arch_list out_var arch_list)
    string(STRIP "${arch_list}" _raw_arch_list)
    if(_raw_arch_list STREQUAL "")
        set(${out_var} "" PARENT_SCOPE)
        return()
    endif()

    # CMake lists are semicolon-separated; TORCH_CUDA_ARCH_LIST values also
    # commonly arrive as shell-space-separated or comma-separated CI strings.
    # Internally we use a normal CMake list.
    string(REGEX REPLACE "[,; \t\r\n]+" ";" _tokens "${_raw_arch_list}")
    set(${out_var} "${_tokens}" PARENT_SCOPE)
endfunction()

function(_gsplat_query_nvcc_architectures out_real_arches out_virtual_arches cuda_compiler)
    if(cuda_compiler STREQUAL "")
        message(
            FATAL_ERROR
            "Cannot filter CUDA architectures because the selected CUDA "
            "toolkit did not provide an nvcc executable."
        )
    endif()

    execute_process(
        COMMAND "${cuda_compiler}" --list-gpu-code --list-gpu-arch
        OUTPUT_VARIABLE _output
        ERROR_VARIABLE _error
        RESULT_VARIABLE _result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_result)
        message(
            FATAL_ERROR
            "Failed to query CUDA architectures from ${cuda_compiler}: "
            "${_error}"
        )
    endif()

    string(REGEX MATCHALL "code=sm_[0-9]+[a-z]?" _real_arches "${_output}")
    list(TRANSFORM _real_arches REPLACE "^code=sm_" "")
    list(REMOVE_DUPLICATES _real_arches)

    string(REGEX MATCHALL "arch=compute_[0-9]+[a-z]?" _virtual_arches "${_output}")
    list(TRANSFORM _virtual_arches REPLACE "^arch=compute_" "")
    list(REMOVE_DUPLICATES _virtual_arches)

    if(NOT _real_arches OR NOT _virtual_arches)
        message(FATAL_ERROR "${cuda_compiler} returned no usable CUDA architecture list.")
    endif()

    set(${out_real_arches} "${_real_arches}" PARENT_SCOPE)
    set(${out_virtual_arches} "${_virtual_arches}" PARENT_SCOPE)
endfunction()

function(_gsplat_probe_nvcc_architecture out_supported arch kind cuda_compiler)
    if(kind STREQUAL "real")
        set(_code "sm_${arch}")
    elseif(kind STREQUAL "virtual")
        set(_code "compute_${arch}")
    elseif(kind STREQUAL "both")
        set(_code "[sm_${arch},compute_${arch}]")
    else()
        message(FATAL_ERROR "Internal error: unknown CUDA architecture kind '${kind}'.")
    endif()

    # NVCC's list commands omit accelerated variants such as sm_90a. A dry
    # run validates one exact code-generation request without executing any
    # compilation command or writing an output file. --fatbin also avoids a
    # dependency on the host compiler during this pre-language check.
    # --dryrun never opens the -x cu input (a nonexistent path also succeeds),
    # so this module file doubles as the input to avoid creating a stub file.
    execute_process(
        COMMAND
            "${cuda_compiler}" --dryrun --fatbin -x cu "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            "--generate-code=arch=compute_${arch},code=${_code}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        OUTPUT_QUIET
        ERROR_VARIABLE _probe_error
        RESULT_VARIABLE _probe_result
    )

    if(_probe_result STREQUAL "0")
        set(${out_supported} TRUE PARENT_SCOPE)
    elseif(_probe_error MATCHES "[Uu]nsupported gpu architecture")
        set(${out_supported} FALSE PARENT_SCOPE)
    else()
        message(
            FATAL_ERROR
            "Failed to probe CUDA architecture '${arch}-${kind}' with "
            "${cuda_compiler}: ${_probe_error}"
        )
    endif()
endfunction()

function(_gsplat_filter_cuda_architectures out_filtered arch_list cuda_compiler)
    _gsplat_split_cuda_arch_list(_tokens "${arch_list}")

    set(_filtered "")
    set(_removed "")
    set(_capabilities_queried FALSE)
    foreach(_token IN LISTS _tokens)
        string(TOLOWER "${_token}" _token_lower)
        set(_supported TRUE)

        if(_token_lower MATCHES "^([0-9]+a?)(-(real|virtual))?$")
            set(_arch "${CMAKE_MATCH_1}")
            set(_kind "${CMAKE_MATCH_3}")
            if(_kind STREQUAL "")
                set(_kind both)
            endif()

            # NVCC's list commands omit accelerated variants, so probe those
            # exact requests and use the queried lists for standard variants.
            if(_arch MATCHES "a$")
                _gsplat_probe_nvcc_architecture(_supported "${_arch}" "${_kind}" "${cuda_compiler}")
            else()
                if(NOT _capabilities_queried)
                    _gsplat_query_nvcc_architectures(
                        _supported_real_arches
                        _supported_virtual_arches
                        "${cuda_compiler}"
                    )
                    set(_capabilities_queried TRUE)
                endif()
                if(_kind STREQUAL "real")
                    if(NOT "${_arch}" IN_LIST _supported_real_arches)
                        set(_supported FALSE)
                    endif()
                elseif(_kind STREQUAL "virtual")
                    if(NOT "${_arch}" IN_LIST _supported_virtual_arches)
                        set(_supported FALSE)
                    endif()
                elseif(
                    NOT "${_arch}" IN_LIST _supported_real_arches
                    OR NOT "${_arch}" IN_LIST _supported_virtual_arches
                )
                    set(_supported FALSE)
                endif()
            endif()
        elseif(_token_lower MATCHES "^[0-9]+[a-z](-(real|virtual))?$")
            message(FATAL_ERROR "Unsupported CMAKE_CUDA_ARCHITECTURES token '${_token}'.")
        endif()

        if(_supported)
            list(APPEND _filtered "${_token}")
        else()
            list(APPEND _removed "${_token}")
        endif()
    endforeach()

    if(NOT _filtered)
        message(
            FATAL_ERROR
            "The selected CUDA compiler supports none of the requested "
            "architectures: ${arch_list}"
        )
    endif()

    if(_removed)
        message(
            WARNING
            "${cuda_compiler} does not support these requested CUDA "
            "architectures: ${_removed}. They will be omitted."
        )
    endif()

    set(${out_filtered} "${_filtered}" PARENT_SCOPE)
endfunction()

function(_gsplat_normalize_torch_cuda_arch_list out_var arch_list)
    _gsplat_split_cuda_arch_list(_tokens "${arch_list}")

    set(_normalized_tokens "")
    foreach(_token IN LISTS _tokens)
        if(_token STREQUAL "")
            continue()
        endif()

        string(TOLOWER "${_token}" _token_lower)
        if(_token_lower STREQUAL "all")
            list(APPEND _normalized_tokens "All")
        elseif(_token_lower STREQUAL "common")
            list(APPEND _normalized_tokens "Common")
        elseif(_token_lower STREQUAL "auto")
            list(APPEND _normalized_tokens "Auto")
        elseif(_token_lower MATCHES "\\+ptx$")
            string(REGEX REPLACE "\\+[Pp][Tt][Xx]$" "" _arch_without_ptx "${_token}")
            list(APPEND _normalized_tokens "${_arch_without_ptx}+PTX")
        else()
            list(APPEND _normalized_tokens "${_token}")
        endif()
    endforeach()

    set(${out_var} "${_normalized_tokens}" PARENT_SCOPE)
endfunction()

# Resolve one architecture-policy owner before CUDA is enabled. CMake inputs
# have priority, followed by Torch inputs, then the top-level native default.
# Torch still requires an architecture input while its package is discovered,
# but its generated architecture flags are discarded later. Therefore a CMake-
# owned policy only needs one valid bridge for Torch, not an exact second
# representation of the CMake list.
# Discover the toolkit, resolve the architecture policy, and enable the CUDA
# language, in the required order (arch filtering needs nvcc before
# enable_language(CUDA); the Torch bridge only resolves afterwards). A macro:
# enable_language() must run at directory scope.
macro(gsplat_setup_cuda)
    find_package(CUDAToolkit REQUIRED)
    gsplat_initialize_cuda_architectures("${CUDAToolkit_NVCC_EXECUTABLE}")
    enable_language(CUDA)
    gsplat_refine_torch_cuda_arch_bridge()
endmacro()

function(gsplat_initialize_cuda_architectures cuda_compiler)
    set(_torch_arch_list "")
    if(DEFINED TORCH_CUDA_ARCH_LIST AND NOT "${TORCH_CUDA_ARCH_LIST}" STREQUAL "")
        set(_torch_arch_list "${TORCH_CUDA_ARCH_LIST}")
    elseif(DEFINED ENV{TORCH_CUDA_ARCH_LIST} AND NOT "$ENV{TORCH_CUDA_ARCH_LIST}" STREQUAL "")
        set(_torch_arch_list "$ENV{TORCH_CUDA_ARCH_LIST}")
    endif()

    set(_cmake_arches "")
    if(DEFINED ENV{CUDAARCHS} AND NOT "$ENV{CUDAARCHS}" STREQUAL "")
        set(_cmake_arches "$ENV{CUDAARCHS}")
    endif()
    if(DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "")
        string(TOLOWER "${CMAKE_CUDA_ARCHITECTURES}" _cmake_arches_lower)
        if(
            TARGET torch
            AND (
                _cmake_arches_lower STREQUAL "off"
                OR _cmake_arches_lower STREQUAL "false"
                OR _cmake_arches_lower STREQUAL "no"
            )
        )
            # PyTorch writes OFF after consuming the parent's architecture
            # choice. Recover the cache entry hidden by that normal-variable
            # sentinel so nested gsplat targets retain the parent's exact
            # selection, including virtual-only architectures.
            get_property(_cached_cmake_arches CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY VALUE)
            if(NOT _cached_cmake_arches STREQUAL "")
                set(_cmake_arches "${_cached_cmake_arches}")
            endif()
        else()
            set(_cmake_arches "${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()

    if(NOT _cmake_arches STREQUAL "")
        set(_arch_owner CMAKE)
    elseif(NOT _torch_arch_list STREQUAL "")
        set(_arch_owner TORCH)
    elseif(PROJECT_IS_TOP_LEVEL)
        set(_arch_owner CMAKE)
        set(_cmake_arches native)
    else()
        # A nested project with no explicit policy follows Torch's default.
        set(_arch_owner TORCH)
    endif()

    set(_exact_cmake_arches "")
    if(_arch_owner STREQUAL "CMAKE")
        _gsplat_filter_cuda_architectures(
            _filtered_cmake_arches
            "${_cmake_arches}"
            "${cuda_compiler}"
        )
        set(_exact_cmake_arches "${_filtered_cmake_arches}")
        set(CMAKE_CUDA_ARCHITECTURES "${_exact_cmake_arches}" PARENT_SCOPE)

        # Torch only needs one valid bridge target because its architecture
        # flags are removed after package discovery. Prefer the first filtered
        # numeric target; Common handles special CMake values such as native.
        set(_torch_arch_list Common)
        foreach(_arch IN LISTS _exact_cmake_arches)
            string(TOLOWER "${_arch}" _arch_lower)
            if(_arch_lower MATCHES "^([0-9]+)([0-9])(a?)(-(real|virtual))?$")
                set(_torch_arch_list "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}")
                break()
            endif()
        endforeach()
    else()
        # Prevent CUDA language initialization from caching a compiler-default
        # architecture that would masquerade as CMake-owned policy on the next
        # configure. Torch's resolved list replaces this temporary sentinel.
        set(CMAKE_CUDA_ARCHITECTURES OFF PARENT_SCOPE)
    endif()

    _gsplat_normalize_torch_cuda_arch_list(_torch_arch_list "${_torch_arch_list}")

    set(TORCH_CUDA_ARCH_LIST "${_torch_arch_list}" PARENT_SCOPE)
    set(_GSPLAT_CUDA_ARCH_OWNER "${_arch_owner}" PARENT_SCOPE)
    set(_GSPLAT_EXACT_CUDA_ARCHITECTURES "${_exact_cmake_arches}" PARENT_SCOPE)

    if(PROJECT_IS_TOP_LEVEL AND _arch_owner STREQUAL "TORCH")
        # Persist only user-selected Torch policy. CMake-side inputs already
        # remain available on every configure, while caching the Torch bridge
        # would make a derived value look user-authored later.
        set(TORCH_CUDA_ARCH_LIST
            "${_torch_arch_list}"
            CACHE STRING
            "CUDA architectures passed to PyTorch's CMake package."
            FORCE
        )
    endif()

    # Keep nvcc's host compiler identical to the CXX compiler so host-side ABI
    # and flags match across .cpp and .cu translation units. Respect explicit
    # non-empty policy from the user or the environment; an empty value means
    # unset.
    if("${CMAKE_CUDA_HOST_COMPILER}" STREQUAL "" AND "$ENV{CUDAHOSTCXX}" STREQUAL "")
        set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" PARENT_SCOPE)
    endif()
endfunction()

# PyTorch appends architecture options directly to CMAKE_CUDA_FLAGS. Remove all
# such options from gsplat's directory scope, including inherited blocks from a
# parent that already found Torch. CMake then generates one authoritative
# architecture set, including virtual-only code. The parent scope is unchanged.
function(_gsplat_remove_torch_cuda_arch_flags)
    set(_cuda_arch_flags ${NVCC_FLAGS_EXTRA})
    set(_filtered_cuda_nvcc_flags "")
    foreach(_flag IN LISTS CUDA_NVCC_FLAGS)
        if(
            _flag STREQUAL "-gencode"
            OR _flag MATCHES "^arch=compute_[0-9]+[a-z]?,code=(sm|compute)_[0-9]+[a-z]?$"
            OR _flag MATCHES "^-gencode=arch=compute_[0-9]+[a-z]?,code=(sm|compute)_[0-9]+[a-z]?$"
        )
            list(APPEND _cuda_arch_flags "${_flag}")
        else()
            list(APPEND _filtered_cuda_nvcc_flags "${_flag}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _cuda_arch_flags)

    if(NOT _cuda_arch_flags)
        message(
            FATAL_ERROR
            "PyTorch did not expose CUDA architecture flags for gsplat to "
            "replace with CMake's architecture selection."
        )
    endif()

    set(_cmake_cuda_flags " ${CMAKE_CUDA_FLAGS} ")
    foreach(_flag IN LISTS _cuda_arch_flags)
        string(REPLACE " ${_flag} " " " _cmake_cuda_flags "${_cmake_cuda_flags}")
    endforeach()
    string(STRIP "${_cmake_cuda_flags}" _cmake_cuda_flags)

    set(CMAKE_CUDA_FLAGS "${_cmake_cuda_flags}" PARENT_SCOPE)
    set(CUDA_NVCC_FLAGS "${_filtered_cuda_nvcc_flags}" PARENT_SCOPE)
endfunction()

# Convert the -gencode flags emitted by Torch into CMake's CUDA_ARCHITECTURES
# spelling. This is the only Torch-to-CMake conversion we need, because it
# observes the exact expansion Torch chose for Auto, Common, named arches, and
# CUDA-version-specific defaults.
function(_gsplat_torch_nvcc_flags_to_cmake_architectures out_var)
    set(_cmake_arches "")
    foreach(_flag IN LISTS ARGN)
        if(_flag MATCHES "^arch=compute_([0-9]+a?),code=(sm|compute)_([0-9]+a?)$")
            set(_compute_arch "${CMAKE_MATCH_1}")
            set(_code_type "${CMAKE_MATCH_2}")
            set(_code_arch "${CMAKE_MATCH_3}")

            if(NOT _compute_arch STREQUAL _code_arch)
                message(
                    FATAL_ERROR
                    "Torch generated '${_flag}', which CMake's "
                    "CUDA_ARCHITECTURES property cannot represent exactly."
                )
            endif()
            if(_code_type STREQUAL "sm")
                set(_kind real)
            else()
                set(_kind virtual)
            endif()
            list(APPEND _cmake_arches "${_code_arch}-${_kind}")
        endif()
    endforeach()

    list(REMOVE_DUPLICATES _cmake_arches)
    set(${out_var} "${_cmake_arches}" PARENT_SCOPE)
endfunction()

# Upgrade the Common fallback bridge to the detected numeric architecture.
# CMAKE_CUDA_ARCHITECTURES_NATIVE only materializes when the CUDA language is
# enabled, after gsplat_initialize_cuda_architectures() computed the bridge;
# call this before Torch discovery so Torch reports the real target.
function(gsplat_refine_torch_cuda_arch_bridge)
    if(
        NOT _GSPLAT_CUDA_ARCH_OWNER STREQUAL "CMAKE"
        OR NOT TORCH_CUDA_ARCH_LIST STREQUAL "Common"
        OR NOT DEFINED CMAKE_CUDA_ARCHITECTURES_NATIVE
    )
        return()
    endif()

    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
        string(TOLOWER "${_arch}" _arch_lower)
        if(_arch_lower MATCHES "^([0-9]+)([0-9])(a?)(-(real|virtual))?$")
            set(TORCH_CUDA_ARCH_LIST
                "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}${CMAKE_MATCH_3}"
                PARENT_SCOPE
            )
            return()
        endif()
    endforeach()
endfunction()

# Remove Torch's generated architecture flags after discovery, then restore the
# selected policy in CMake's native representation.
function(gsplat_finalize_cuda_architectures)
    set(_torch_nvcc_flags ${NVCC_FLAGS_EXTRA})
    if(NOT _torch_nvcc_flags)
        set(_torch_nvcc_flags ${CUDA_NVCC_FLAGS})
    endif()

    if(_GSPLAT_CUDA_ARCH_OWNER STREQUAL "CMAKE")
        set(_cmake_arches "${_GSPLAT_EXACT_CUDA_ARCHITECTURES}")
    elseif(_GSPLAT_CUDA_ARCH_OWNER STREQUAL "TORCH")
        _gsplat_torch_nvcc_flags_to_cmake_architectures(_cmake_arches ${_torch_nvcc_flags})
    else()
        message(FATAL_ERROR "Internal error: CUDA architecture policy has no owner.")
    endif()
    if(_cmake_arches STREQUAL "")
        message(
            FATAL_ERROR
            "Torch did not expose CUDA -gencode flags, so gsplat cannot "
            "synchronize CMAKE_CUDA_ARCHITECTURES with TORCH_CUDA_ARCH_LIST."
        )
    endif()

    _gsplat_remove_torch_cuda_arch_flags()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CUDA_ARCHITECTURES "${_cmake_arches}" PARENT_SCOPE)
endfunction()
