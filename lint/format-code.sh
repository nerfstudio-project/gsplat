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

full_mode=false
git_mode_args=()
black_args=()

usage() {
    cat <<EOF
Format code in the gsplat repository.

Usage: ${script_name} [--check] [--full] [--help|-h]

Options:
  --check      Check formatting without modifying files.
  --full       Format all tracked Python files instead of only modified tracked files.
  --help, -h   Show this help message.
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
            black_args+=(
                --check
                --diff
            )
            shift
            ;;
        --full)
            full_mode=true
            shift
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

    if $full_mode; then
        local empty_tree
        empty_tree="$(git -C "${repo_root}" hash-object -t tree /dev/null)"
        git_diff_args=(
            --cached
            --diff-filter=ACMR
            "${empty_tree}"
        )
    else
        git_diff_args=(
            --diff-filter=ACMR
            HEAD
        )
    fi

    git -C "${repo_root}" diff --name-only "${git_diff_args[@]}" -z -- "$@"
}


# Python code formatting ====================
mapfile -d '' python_files < <(
    find_repo_files '*.py' '*.pyi'
)

if (( ${#python_files[@]} > 0 )); then
    black_args+=(
        --required-version 22.3.0
        --color
    )

    python -m black "${black_args[@]}" "${python_files[@]}"
else
    echo "Python formatting was not performed because there are no modified Python files."
fi
