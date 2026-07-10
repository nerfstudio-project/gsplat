# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
include_guard(GLOBAL)

# Define a Python extension named TARGET from the SOURCES files and include
# it in both raw build-tree imports and the gsplat_runtime install
# component. DESTINATION selects the install directory (default: the platlib
# root). The module's dynamic exports are restricted to its PyInit symbol
# through a generated linker version script.
function(gsplat_add_python_extension)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "TARGET;DESTINATION" "SOURCES")
    if(NOT arg_TARGET)
        message(FATAL_ERROR "gsplat_add_python_extension: TARGET is required.")
    endif()
    if(NOT arg_DESTINATION)
        set(arg_DESTINATION ".")
    endif()
    set(target "${arg_TARGET}")

    python_add_library(${target} MODULE WITH_SOABI ${arg_SOURCES})
    target_compile_definitions(${target} PRIVATE TORCH_EXTENSION_NAME=${target})
    target_link_libraries(
        ${target}
        PRIVATE gsplat_compile_opts gsplat::ext::torch_python CUDA::cudart
    )
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${GSPLAT_BINARY_DIR}")

    if(NOT WIN32)
        # Strip configurations without debug information and restrict the
        # module's exported dynamic symbols to its PyInit entry point.
        set(_gsplat_module_target "${target}")
        set(_ldscript "${GSPLAT_BINARY_DIR}/ldscripts/${target}.ldscript")
        configure_file("${GSPLAT_SOURCE_DIR}/cmake/gsplat_module.ldscript.in" "${_ldscript}" @ONLY)
        target_link_options(
            ${target}
            PRIVATE
                "$<$<NOT:$<CONFIG:Debug,RelWithDebInfo>>:-s>"
                "-Wl,--version-script,${_ldscript}"
        )
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_ldscript}")
    endif()

    set_property(GLOBAL APPEND PROPERTY GSPLAT_PYTHON_EXTENSION_TARGETS "${target}")

    install(TARGETS ${target} LIBRARY DESTINATION "${arg_DESTINATION}" COMPONENT gsplat_runtime)
endfunction()
