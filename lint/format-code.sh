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

# Selection mode: full | changed. `mode_flag` records the flag that set it so
# a second, different mode flag is a hard error rather than a silent last-wins.
mode="full"
mode_flag=""
changed_base=""
check_mode=false
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

Usage: ${script_name} [--check] [--full | --changed <ref>] [--help|-h]

Options:
  --check          Check formatting without modifying files.
  --full           Format all tracked source files (default).
  --changed <ref>  Format only source files changed since <ref> (e.g. for CI).
  --help, -h       Show this help message.

--full and --changed select what to act on and are mutually exclusive.
EOF
}

die() {
    echo "ERROR: $*"
    usage
    exit 1
} >&2

# Select a mode, rejecting a conflicting second selection flag.
set_mode() {
    local requested="$1" flag="$2"
    if [[ -n "${mode_flag}" && "${mode_flag}" != "${flag}" ]]; then
        die "${flag} cannot be combined with ${mode_flag}; pass at most one of --full / --changed"
    fi
    mode="${requested}"
    mode_flag="${flag}"
}

# Command line argument processing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            check_mode=true
            shift
            ;;
        --full)
            set_mode full --full
            shift
            ;;
        --changed)
            [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "--changed requires a git ref argument"
            set_mode changed --changed
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


# Run the pinned formatters over the given files. ACTION is "format" (rewrite in
# place) or "check" (report only, non-zero if unformatted). Each formatter's
# failure is captured into rc and returned rather than left to set -e --
# callers invoke this in an `if !`/`||` context where set -e is ignored inside.
run_formatters() {
    local action="$1"
    shift
    local py=() cpp=() f rc=0
    for f in "$@"; do
        case "${f}" in
            *.py | *.pyi) py+=("${f}") ;;
            *) cpp+=("${f}") ;;
        esac
    done
    if (( ${#py[@]} > 0 )); then
        local black_pin
        black_pin="$(read_dev_pin black)"
        if [[ -z "${black_pin}" ]]; then
            echo "ERROR: pyproject.toml's dev extra must pin black==<version>." >&2
            return 1
        fi
        local black_flags=(--required-version "${black_pin}" --color)
        if [[ "${action}" == check ]]; then
            black_flags+=(--check --diff)
        fi
        python -m black "${black_flags[@]}" -- "${py[@]}" || rc=1
    fi
    if (( ${#cpp[@]} > 0 )); then
        local cf pin clang_flags=(-i)
        if [[ "${action}" == check ]]; then
            clang_flags=(--dry-run --Werror)
        fi
        pin="$(read_dev_pin clang-format)"
        if [[ -z "${pin}" ]]; then
            echo "ERROR: pyproject.toml's dev extra must pin clang-format==<version>." >&2
            return 1
        fi
        # Propagate a lookup failure explicitly: callers invoke this in an
        # `if !`/`||` context where set -e is ignored inside.
        cf="$(find_code_format "${pin}")" || return 1
        printf '%s\n' "${cpp[@]}" | "${cf}" "${clang_flags[@]}" --files=/dev/stdin || rc=1
    fi
    return "${rc}"
}

mapfile -d '' -t src_files < <(
    find_repo_files "${python_extensions[@]}" "${cpp_extensions[@]}"
)

if (( ${#src_files[@]} == 0 )); then
    echo "No source files selected."
elif $check_mode; then
    run_formatters check "${src_files[@]}"
else
    run_formatters format "${src_files[@]}"
fi
