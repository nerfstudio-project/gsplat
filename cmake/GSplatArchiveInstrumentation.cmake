# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26)

# CMake appends the generated instrumentation index path to every callback.
math(EXPR _index_argument "${CMAKE_ARGC} - 1")
set(_index_variable "CMAKE_ARGV${_index_argument}")
set(_index_file "${${_index_variable}}")

file(READ "${_index_file}" _index_json)
string(JSON _hook GET "${_index_json}" hook)

if(_hook STREQUAL "postGenerate")
    set(_output_name configure-trace.json)
elseif(_hook STREQUAL "postBuild" OR _hook STREQUAL "postCMakeBuild")
    set(_output_name build-trace.json)
elseif(_hook STREQUAL "postCTest")
    set(_output_name tests-trace.json)
else()
    return()
endif()

string(JSON _build_dir GET "${_index_json}" buildDir)
string(JSON _data_dir GET "${_index_json}" dataDir)
string(JSON _trace_file GET "${_index_json}" trace)

set(_trace_path "${_data_dir}")
cmake_path(APPEND _trace_path "${_trace_file}")

file(COPY_FILE "${_trace_path}" "${_build_dir}/${_output_name}" ONLY_IF_DIFFERENT)
