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

# Installing the pre-commit hook needs git; without it there is nothing to do.
# Check with a shell builtin before any external command (dirname below) so a
# git-less PATH exits cleanly instead of failing on a missing utility.
if ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git is not installed; skipping pre-commit hook installation." >&2
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
support="${repo_root}/gsplat/build_support/bootstrap_support.py"

# Print usage and option help.
usage()
{
    cat <<'EOF'
Provision a gsplat development environment.

Usage: ./bootstrap.sh [options] [-- <pip-options>...]

Options:
  --python <path>         Base/current Python interpreter. By default, use
                          python from the active environment, then python3.
  --venv <directory>      Create or reuse this virtual environment. By default,
                          install into the active Python environment.
  --cuda <major.minor>    CUDA version used to select binary dependencies. By
                          default, reuse the CUDA of the environment's Torch,
                          then detect nvcc (CUDACXX, CUDA_HOME, CUDA_PATH, PATH).
  --dry-run               Print environment-changing commands without running
                          them. Detection and dependency parsing still run.
  -h, --help              Show this help.

Re-running is idempotent: an omitted option keeps what the environment already
has (the installed Torch records its CUDA; the active or --venv interpreter
records the environment). Passing an option applies that change and leaves the
rest untouched.

Arguments after -- are forwarded to each pip install command. Explicit pip
options take precedence over bootstrap defaults.
EOF
}

# Print an error message to stderr and exit non-zero.
die()
{
    echo "ERROR: $*" >&2
    exit 1
}

# Run a command, or with --dry-run print it (shell-quoted) instead of running.
run_command()
{
    if [[ ${dry_run} == true ]]; then
        # Echo the command, shell-quoted so the line stays copy-pasteable,
        # rather than running it.
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

# Print the absolute path of a command; a bare name is looked up on PATH.
# Return non-zero when it is not an executable.
resolve_executable()
{
    local executable=$1

    if [[ ${executable} == */* ]]; then
        [[ -x ${executable} ]] || return 1
        (
            cd "$(dirname "${executable}")"
            printf '%s/%s\n' "$PWD" "$(basename "${executable}")"
        )
    else
        command -v "${executable}"
    fi
}

# Exit unless the given interpreter meets pyproject.toml's requires-python
# floor. bootstrap_support does the parse with the standard library alone, so
# the check runs before the interpreter has any dependency installed.
require_supported_python()
{
    "$1" -B "${support}" check-python "${repo_root}/pyproject.toml" \
        || die "unsupported Python interpreter: $1"
}

# -----------------------------------------------------------------------------
# Command-line options
# -----------------------------------------------------------------------------

python_option=""
venv_dir=""
cuda_option=""
dry_run=false
pip_options=()

while (( $# > 0 )); do
    case "$1" in
        --python)
            (( $# >= 2 )) || die "--python requires a path"
            python_option=$2
            shift 2
            ;;
        --venv)
            (( $# >= 2 )) || die "--venv requires a directory"
            venv_dir=$2
            shift 2
            ;;
        --cuda)
            (( $# >= 2 )) || die "--cuda requires a major.minor version"
            cuda_option=$2
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            pip_options=("$@")
            break
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Pre-commit hook
# -----------------------------------------------------------------------------

# Install the formatting hook first, using only git and bash, so it lands in any
# checkout whether or not the development-environment provisioning below runs.

# A pre-commit hook is "ours" if it carries the dispatcher's marker (any
# version) or is the symlink to lint/format-code.sh that older bootstrap
# versions installed. We replace our own hook but preserve a foreign one.
hook_is_ours()
{
    local hook="$1"
    [[ -L "${hook}" && "$(readlink "${hook}")" == */lint/format-code.sh ]] && return 0
    grep -q "GSPLAT PRE-COMMIT HOOK" "${hook}" 2>/dev/null
}

hook_src="${repo_root}/hooks/pre-commit"
hook_path=$(git -C "${repo_root}" rev-parse --git-path hooks/pre-commit) \
    || die "pre-commit hook installation requires a Git checkout"
# --git-path is relative to repo_root in the main worktree, absolute in a linked
# worktree; absolutize it so the install lands in the repo regardless of the
# directory bootstrap was invoked from.
[[ ${hook_path} == /* ]] || hook_path="${repo_root}/${hook_path}"
run_command mkdir -p "$(dirname "${hook_path}")"

# Install a copy (not a symlink) so the hook keeps working after the worktree
# that ran bootstrap is removed. A foreign hook is rotated aside with a numbered
# backup so bootstrap never clobbers one it did not install.
if [[ -e ${hook_path} || -L ${hook_path} ]] && ! hook_is_ours "${hook_path}"; then
    run_command mv --backup=numbered "${hook_path}" "${hook_path}.backup"
    [[ ${dry_run} == false ]] && echo "Backed up your existing pre-commit hook to ${hook_path}.backup"
fi
# Remove first so an existing symlink is replaced rather than followed.
run_command rm -f "${hook_path}"
run_command cp -f "${hook_src}" "${hook_path}"
run_command chmod +x "${hook_path}"
[[ ${dry_run} == false ]] && echo "Installed pre-commit hook at ${hook_path}"

# The hook needs only git + bash. Everything below provisions a full development
# environment and needs this checkout's build_support helper; a minimal checkout
# (e.g. the hook tests' throwaway repo) has only the scripts, so stop here.
if [[ ! -f "${support}" ]]; then
    exit 0
fi

# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------

# The interpreter bootstrap installs into.
if [[ -n ${python_option} ]]; then
    # Prefer an explicit --python.
    base_python=$(resolve_executable "${python_option}") \
        || die "Python interpreter is not executable: ${python_option}"
elif command -v python >/dev/null 2>&1; then
    # Else the active environment's python,
    base_python=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    # then python3.
    base_python=$(command -v python3)
else
    die "Python was not found; pass --python <path>"
fi
require_supported_python "${base_python}"

# With --venv, provision a dedicated environment: create it when missing and
# install into its interpreter. A dry run does not create it, so fall back to
# the base interpreter for the read-only inspections that follow.
target_python=${base_python}
if [[ -n ${venv_dir} ]]; then
    venv_dir=$("${base_python}" -c \
        'import os, sys; print(os.path.abspath(sys.argv[1]))' "${venv_dir}")
    target_python="${venv_dir}/bin/python"
    if [[ ! -x ${target_python} ]]; then
        run_command "${base_python}" -m venv "${venv_dir}"
        [[ ${dry_run} == true ]] && target_python=${base_python}
    fi
    [[ ${target_python} == "${base_python}" ]] || require_supported_python "${target_python}"
fi

"${target_python}" -m pip --version >/dev/null 2>&1 \
    || die "pip is unavailable for ${target_python}"
echo "Python: $("${target_python}" -c \
    'import sys; print(f"{sys.executable} (Python {sys.version.split()[0]})")')"

# inspect-torch runs in the target environment; requirements only needs pip.
inspect_torch()
{
    "${target_python}" -B "${support}" inspect-torch "$@"
}

# -----------------------------------------------------------------------------
# Source submodules
# -----------------------------------------------------------------------------

# The build reads third_party/glm and third_party/googletest. Bring them to the
# pinned commits; a no-op once they already are, so re-runs stay idempotent.
if git -C "${repo_root}" rev-parse --git-dir >/dev/null 2>&1; then
    run_command git -C "${repo_root}" submodule update --init --recursive
else
    echo "WARNING: ${repo_root} is not a Git checkout; skipping submodule update" >&2
fi

# -----------------------------------------------------------------------------
# CUDA version
# -----------------------------------------------------------------------------

# The CUDA that binary dependencies target, chosen in precedence order.
cuda_version=""
cuda_source=""
if [[ -n ${cuda_option} ]]; then
    # An explicit --cuda wins.
    [[ ${cuda_option} =~ ^[0-9]+\.[0-9]+$ ]] \
        || die "--cuda must be a major.minor version, such as 12.8"
    cuda_version=${cuda_option}
    cuda_source="requested"
elif [[ ${dry_run} == false ]] && torch_info=$(inspect_torch) \
    && [[ ${torch_info} != missing ]]; then
    # Else stay on the CUDA the installed Torch already targets. (A dry run has
    # no interpreter to inspect, so it falls through to nvcc below.)
    IFS=$'\t' read -r _ _ _ torch_cuda <<< "${torch_info}"
    if [[ -n ${torch_cuda} && ${torch_cuda} != cpu ]]; then
        cuda_version=${torch_cuda}
        cuda_source="installed Torch"
    fi
fi
if [[ -z ${cuda_version} ]]; then
    # Still unselected: probe the toolkit -- CUDACXX (its first word), then
    # CUDA_HOME, CUDA_PATH, then nvcc on PATH.
    if [[ -n ${CUDACXX:-} ]]; then
        nvcc=${CUDACXX%% *}
    elif [[ -n ${CUDA_HOME:-} && -x ${CUDA_HOME}/bin/nvcc ]]; then
        nvcc="${CUDA_HOME}/bin/nvcc"
    elif [[ -n ${CUDA_PATH:-} && -x ${CUDA_PATH}/bin/nvcc ]]; then
        nvcc="${CUDA_PATH}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc=$(command -v nvcc)
    else
        die "CUDA nvcc was not found; pass --cuda <major.minor> or select a toolkit"
    fi
    nvcc=$(resolve_executable "${nvcc}") \
        || die "CUDA compiler is not executable: ${nvcc}"
    cuda_version=$(sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' \
        <<< "$("${nvcc}" --version)" | head -1)
    [[ ${cuda_version} =~ ^[0-9]+\.[0-9]+$ ]] \
        || die "could not determine the CUDA version from ${nvcc}"
    cuda_source="nvcc ${nvcc}"
fi
cuda_major=${cuda_version%%.*}
torch_index_url="https://download.pytorch.org/whl/cu${cuda_version/./}"
echo "CUDA: ${cuda_version} (${cuda_source})"

# -----------------------------------------------------------------------------
# Dependency metadata
# -----------------------------------------------------------------------------

# Resolve the whole dependency set through the Python helper, which returns
# Torch first (it installs from the CUDA-tagged index) followed by everything
# that installs from the default index.
requirements_file=$(mktemp -t bootstrap-requirements.XXXXXX)
trap 'rm -f "${requirements_file}"' EXIT
"${target_python}" -B "${support}" requirements \
    "${repo_root}/pyproject.toml" "${cuda_version}" > "${requirements_file}" \
    || die "could not resolve the bootstrap requirements for CUDA ${cuda_version}"
mapfile -d '' -t requirements < "${requirements_file}"
(( ${#requirements[@]} > 1 )) || die "no bootstrap dependencies were found"
torch_requirement=${requirements[0]}
dependency_requirements=("${requirements[@]:1}")

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------

# Fail when Torch's CUDA major differs from the selected one; warn on a
# minor-version difference (pip picks a compatible minor).
validate_torch_cuda()
{
    local torch_version=$1 torch_cuda=$2
    if [[ ${torch_cuda} == cpu || ${torch_cuda%%.*} != "${cuda_major}" ]]; then
        die "installed Torch ${torch_version} uses CUDA ${torch_cuda}, but CUDA "\
"${cuda_version} was selected; create a compatible environment with --venv"
    fi
    if [[ ${torch_cuda} != "${cuda_version}" ]]; then
        echo "WARNING: Torch uses CUDA ${torch_cuda} while CUDA ${cuda_version} "\
"was selected; compatible CUDA minor versions will be used." >&2
    fi
}

# Install Torch from the CUDA-tagged index; extra flags (e.g.
# --force-reinstall) pass through, and it honors --dry-run.
pip_install_torch()
{
    run_command "${target_python}" -m pip install --index-url "${torch_index_url}" \
        "$@" "${pip_options[@]}" "${torch_requirement}"
}

# Bring Torch to a build that is CUDA-correct and satisfies the requirement,
# disturbing the environment as little as possible.
if [[ ${dry_run} == true ]]; then
    # Dry run: only show the install a real run would perform.
    pip_install_torch
else
    torch_info=$(inspect_torch --requirement "${torch_requirement}") \
        || die "Torch could not be inspected"
    if [[ ${torch_info} == missing ]]; then
        # Absent: install it.
        pip_install_torch
    else
        IFS=$'\t' read -r torch_status _ torch_version torch_cuda \
            <<< "${torch_info}"
        if [[ ${torch_cuda} == cpu || ${torch_cuda%%.*} != "${cuda_major}" ]]; then
            # Installed Torch targets a different CUDA. Reinstall in place, but
            # only when --cuda asked for the switch -- an inferred version was
            # read from this Torch and cannot disagree.
            [[ -n ${cuda_option} ]] || die "installed Torch ${torch_version} uses "\
"CUDA ${torch_cuda}, but CUDA ${cuda_version} was selected"
            echo "Reinstalling Torch for CUDA ${cuda_version} (was CUDA ${torch_cuda})."
            pip_install_torch --force-reinstall
        elif [[ ${torch_status} != compatible ]]; then
            # Right CUDA but the version misses the requirement: refuse rather
            # than upgrade Torch under the user.
            die "installed Torch ${torch_version} does not satisfy "\
"${torch_requirement}; create a new environment with --venv"
        fi
        # Otherwise Torch is already correct: leave it untouched.
    fi

    # Whatever branch ran, hold the resulting Torch to the contract before the
    # dependency install proceeds.
    torch_info=$(inspect_torch --requirement "${torch_requirement}") \
        || die "Torch could not be inspected after installation"
    IFS=$'\t' read -r torch_status torch_distribution_version torch_version \
        torch_cuda <<< "${torch_info}"
    [[ ${torch_status} == compatible ]] \
        || die "installed Torch does not satisfy ${torch_requirement}"
    validate_torch_cuda "${torch_version}" "${torch_cuda}"
fi

dependency_install=(
    "${target_python}"
    -m pip install
    "${pip_options[@]}"
    "${dependency_requirements[@]}"
)
if [[ ${dry_run} == false ]]; then
    # Pin Torch so a transitive dependency cannot replace the CUDA-compatible
    # build selected above; pip reports a resolution error instead.
    dependency_install+=("torch==${torch_distribution_version}")
fi
run_command "${dependency_install[@]}"

# The pin should keep Torch fixed, but a dependency can still drag it in; make
# sure the exact CUDA-correct build survived before reporting success.
if [[ ${dry_run} == false ]]; then
    torch_info=$(inspect_torch --requirement "${torch_requirement}") \
        || die "Torch could not be inspected"
    IFS=$'\t' read -r torch_status final_distribution_version final_version \
        torch_cuda <<< "${torch_info}"
    [[ ${torch_status} == compatible ]] \
        || die "Torch became incompatible during dependency installation"
    [[ ${final_distribution_version} == "${torch_distribution_version}" ]] \
        || die "Torch changed from ${torch_distribution_version} to "\
"${final_distribution_version} during dependency installation"
    validate_torch_cuda "${final_version}" "${torch_cuda}"
    echo "Torch: ${final_version} (CUDA ${torch_cuda})"
fi

echo "Bootstrap complete."
# Point the user at the venv they provisioned, unless they are already in it.
if [[ -n ${venv_dir} ]]; then
    if [[ ${VIRTUAL_ENV:-} == "${venv_dir}" ]]; then
        echo "Environment ${venv_dir} is active."
    else
        printf 'Activate the environment before building:\n  source %q\n' \
            "${venv_dir}/bin/activate"
    fi
fi
