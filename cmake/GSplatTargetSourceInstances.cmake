# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Compile one source with a channel-specific definition and add its objects to
# an existing target. The instance inherits the target's compile requirements
# without linking the target itself, which would create a dependency cycle.
function(gsplat_target_add_source_instance target source)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "CHANNEL" "")

    if(NOT TARGET "${target}")
        message(FATAL_ERROR "gsplat_target_add_source_instance: unknown target '${target}'.")
    endif()
    if(NOT DEFINED arg_CHANNEL OR NOT arg_CHANNEL MATCHES "^[1-9][0-9]*$")
        message(
            FATAL_ERROR
            "gsplat_target_add_source_instance: CHANNEL must be a positive integer."
        )
    endif()
    if(arg_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
            "gsplat_target_add_source_instance: unexpected arguments: ${arg_UNPARSED_ARGUMENTS}"
        )
    endif()

    string(MAKE_C_IDENTIFIER "${source}" _source_id)
    set(_instance_target "${target}_${_source_id}_channel_${arg_CHANNEL}")

    add_library("${_instance_target}" OBJECT "${source}")
    set_target_properties("${_instance_target}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(
        "${_instance_target}"
        PRIVATE
            GSPLAT_INSTANTIATE_TEMPLATE
            "GSPLAT_CHANNEL_INSTANCE=${arg_CHANNEL}"
            "$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>"
    )
    target_include_directories(
        "${_instance_target}"
        PRIVATE "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>"
    )
    target_compile_features(
        "${_instance_target}"
        PRIVATE "$<TARGET_PROPERTY:${target},COMPILE_FEATURES>"
    )
    target_compile_options(
        "${_instance_target}"
        PRIVATE "$<TARGET_PROPERTY:${target},COMPILE_OPTIONS>"
    )
    target_link_libraries(
        "${_instance_target}"
        PRIVATE "$<TARGET_PROPERTY:${target},LINK_LIBRARIES>"
    )

    # Channel instances are implementation details of the target. Keeping the
    # object input private is especially important when the target is shared:
    # propagating it would compile the same CUDA implementation into every
    # consumer as well as into the shared library.
    target_sources("${target}" PRIVATE "$<TARGET_OBJECTS:${_instance_target}>")
endfunction()
