#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

script_path="$(readlink -f "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "${script_path}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
script_name="$(basename "${BASH_SOURCE[0]}")"

# Selection mode: full | changed. Set by the mutually-related --full/--changed
# flags; --full is the default.
mode="full"
changed_base=""
check_mode=false
black_args=()
python_extensions=(
    '*.py'
    '*.pyi'
)
cpp_extensions=(
    '*.c'
    '*.cc'
    '*.cpp'
    '*.cppm'
    '*.cxx'
    '*.h'
    '*.hh'
    '*.hpp'
    '*.hxx'
    '*.cu'
    '*.cuh'
)

# Read a formatter's version from its single pin in the shared dev dependencies.
read_dev_pin() {
    local package="$1"

    python3 "${repo_root}/gsplat/build_support/pyproject_metadata.py" \
        "${repo_root}/pyproject.toml" dev --pin "${package}"
}

usage() {
    cat <<EOF
Format code in the gsplat repository.

Usage: ${script_name} [--check] [--full] [--changed <ref>] [--help|-h]

Options:
  --check          Check formatting without modifying files.
  --full           Format all tracked source files (default).
  --changed <ref>  Format only source files changed since <ref> (e.g. for CI).
  --help, -h       Show this help message.
EOF
}

die() {
    echo "ERROR: $*"
    usage
    exit 1
} >&2

# Command line argument processing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            check_mode=true
            black_args+=(
                --check
                --diff
            )
            shift
            ;;
        --full)
            mode="full"
            shift
            ;;
        --changed)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--changed requires a git ref argument"
            mode="changed"
            changed_base="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

find_repo_files() {
    local git_diff_args=()

    case "${mode}" in
        full)
            # Index vs the empty tree enumerates every tracked source file.
            local empty_tree
            empty_tree="$(git -C "${repo_root}" hash-object -t tree /dev/null)"
            git_diff_args=(
                --cached
                --diff-filter=ACMR
                "${empty_tree}"
            )
            ;;
        changed)
            git_diff_args=(
                --diff-filter=ACMR
                "${changed_base}"
            )
            ;;
    esac

    git -C "${repo_root}" diff --name-only "${git_diff_args[@]}" -z -- "$@"
}

find_code_format() {
    local expected_version="$1"
    local expected_major="${expected_version%%.*}"
    local code_format=""
    local code_format_version_output=""
    local installed_major=""

    if command -v "clang-format-${expected_major}" >/dev/null 2>&1; then
        code_format="clang-format-${expected_major}"
    elif command -v clang-format >/dev/null 2>&1; then
        code_format="clang-format"
    else
        echo "ERROR: clang-format is required to format C/C++/CUDA files." >&2
        echo "Install clang-format-${expected_major}, or install clang-format at version ${expected_version}." >&2
        exit 1
    fi

    code_format_version_output="$("${code_format}" --version)"
    if [[ "${code_format_version_output}" =~ version[[:space:]]+([0-9]+)([.[:space:]]|$) ]]; then
        installed_major="${BASH_REMATCH[1]}"
    else
        echo "ERROR: Could not determine ${code_format} major version." >&2
        echo "Found: ${code_format_version_output}" >&2
        exit 1
    fi

    if [[ "${installed_major}" != "${expected_major}" ]]; then
        echo "ERROR: clang-format major version mismatch." >&2
        echo "Expected clang-format major version ${expected_major}." >&2
        echo "Found: ${code_format_version_output}" >&2
        echo "Install clang-format-${expected_major}, or install clang-format at version ${expected_version}." >&2
        exit 1
    fi

    printf '%s\n' "${code_format}"
}


# Python code formatting ====================
mapfile -d '' python_files < <(
    find_repo_files "${python_extensions[@]}"
)

if (( ${#python_files[@]} > 0 )); then
    black_version="$(read_dev_pin black)"
    black_args+=(
        --required-version "${black_version}"
        --color
    )

    python -m black "${black_args[@]}" "${python_files[@]}"
else
    echo "Python formatting was not performed because there are no selected Python files."
fi


# C/C++/CUDA code formatting ===============
mapfile -d '' cpp_files < <(
    find_repo_files "${cpp_extensions[@]}"
)

if (( ${#cpp_files[@]} > 0 )); then
    clang_format_version="$(read_dev_pin clang-format)"
    if [[ -z "${clang_format_version}" ]]; then
        echo "ERROR: pyproject.toml's dev extra must pin clang-format==<version>." >&2
        exit 1
    fi

    code_format="$(find_code_format "${clang_format_version}")"

    if $check_mode; then
        printf '%s\n' "${cpp_files[@]}" | "${code_format}" --dry-run --Werror --files=/dev/stdin
    else
        printf '%s\n' "${cpp_files[@]}" | "${code_format}" -i --files=/dev/stdin
    fi
else
    echo "C/C++/CUDA formatting was not performed because there are no selected C/C++/CUDA files."
fi
