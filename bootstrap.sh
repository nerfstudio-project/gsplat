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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

usage()
{
    cat <<'EOF'
Provision a gsplat development environment.

Usage: ./bootstrap.sh [options] [-- <pip-options>...]

Options:
  --python <path>          Base/current Python interpreter. By default, use
                           python from the active environment, then python3.
  --venv <directory>      Create or reuse this virtual environment. By default,
                           install into the active Python environment.
  --cuda <major.minor>    CUDA version used to select binary dependencies. By
                           default, inspect CUDACXX, CUDA_HOME, CUDA_PATH, then
                           PATH for nvcc and detect its version.
  --dry-run               Print environment-changing commands without running
                           them. Detection and dependency parsing still run.
  -h, --help              Show this help.

Arguments after -- are forwarded to each pip install command. Explicit pip
options take precedence over bootstrap defaults.
EOF
}

die()
{
    echo "ERROR: $*" >&2
    exit 1
}

print_command()
{
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run_command()
{
    if [[ ${dry_run} == true ]]; then
        print_command "$@"
    else
        "$@"
    fi
}

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

require_supported_python()
{
    local interpreter=$1

    "${interpreter}" - <<'PY' \
        || die "gsplat requires Python 3.10 or newer: ${interpreter}"
import sys

raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
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
# Python environment
# -----------------------------------------------------------------------------

if [[ -n ${python_option} ]]; then
    base_python=$(resolve_executable "${python_option}") \
        || die "Python interpreter is not executable: ${python_option}"
elif command -v python >/dev/null 2>&1; then
    base_python=$(command -v python)
elif command -v python3 >/dev/null 2>&1; then
    base_python=$(command -v python3)
else
    die "Python was not found; pass --python <path>"
fi

require_supported_python "${base_python}"

metadata_python=${base_python}
target_python=${base_python}
target_python_exists=true

if [[ -n ${venv_dir} ]]; then
    venv_dir=$("${base_python}" -c \
        'import os, sys; print(os.path.abspath(sys.argv[1]))' "${venv_dir}")
    target_python="${venv_dir}/bin/python"
    if [[ ! -x ${target_python} ]]; then
        target_python_exists=false
        run_command "${base_python}" -m venv "${venv_dir}"
        if [[ ${dry_run} == false ]]; then
            target_python_exists=true
        fi
    fi
    if [[ ${target_python_exists} == true ]]; then
        require_supported_python "${target_python}"
        metadata_python=${target_python}
    fi
fi

if [[ ${target_python_exists} == true ]]; then
    "${target_python}" -m pip --version >/dev/null 2>&1 \
        || die "pip is unavailable for ${target_python}"
else
    "${metadata_python}" -m pip --version >/dev/null 2>&1 \
        || die "pip is unavailable for ${metadata_python}"
fi

python_description=$("${metadata_python}" -c \
    'import sys; print(f"{sys.executable} (Python {sys.version.split()[0]})")')
echo "Python: ${python_description}"

# -----------------------------------------------------------------------------
# CUDA toolkit
# -----------------------------------------------------------------------------

cuda_compiler=""
cuda_compiler_args=()
cudacxx_value=${CUDACXX:-}

if [[ -n ${cuda_option} ]]; then
    [[ ${cuda_option} =~ ^[0-9]+\.[0-9]+$ ]] \
        || die "--cuda must be a major.minor version, such as 12.8"
    cuda_version=${cuda_option}

    # --cuda selects binary dependencies only; the build uses whatever
    # toolkit CMake selects. Detect opportunistically and warn on drift.
    detected_nvcc=""
    if [[ -x ${cudacxx_value//[[:space:]]/} ]]; then
        detected_nvcc=${cudacxx_value}
    elif [[ -n ${CUDA_HOME:-} && -x ${CUDA_HOME}/bin/nvcc ]]; then
        detected_nvcc="${CUDA_HOME}/bin/nvcc"
    elif [[ -n ${CUDA_PATH:-} && -x ${CUDA_PATH}/bin/nvcc ]]; then
        detected_nvcc="${CUDA_PATH}/bin/nvcc"
    elif command -v nvcc >/dev/null 2>&1; then
        detected_nvcc=$(command -v nvcc)
    fi
    if [[ -n ${detected_nvcc} ]] \
        && detected_output=$("${detected_nvcc}" --version 2>/dev/null); then
        detected_version=$(sed -nE \
            's/.*release ([0-9]+\.[0-9]+).*/\1/p' <<< "${detected_output}" | head -1)
        if [[ -n ${detected_version} && ${detected_version} != "${cuda_version}" ]]; then
            echo "WARNING: binary dependencies target CUDA ${cuda_version}, but" \
                "the detected nvcc is ${detected_version} (${detected_nvcc})." \
                "The build uses the detected toolkit unless a compiler is" \
                "selected via CUDACXX or -DCMAKE_CUDA_COMPILER." >&2
        fi
    fi
elif [[ -n ${cudacxx_value//[[:space:]]/} ]]; then
    # CMake permits fixed compiler arguments in CUDACXX. Preserve those only
    # for the version probe; the executable itself is reported separately.
    if [[ -x ${cudacxx_value} ]]; then
        cuda_command=("${cudacxx_value}")
    else
        printf '%s' "${cudacxx_value}" | xargs true 2> /dev/null \
            || die "CUDACXX value has unbalanced quoting: ${cudacxx_value}"
        mapfile -d '' -t cuda_command < <(
            printf '%s' "${cudacxx_value}" | xargs printf '%s\0' 2> /dev/null
        )
    fi
    (( ${#cuda_command[@]} > 0 )) \
        || die "CUDACXX does not contain a CUDA compiler command"
    cuda_compiler=${cuda_command[0]}
    if (( ${#cuda_command[@]} > 1 )); then
        cuda_compiler_args=("${cuda_command[@]:1}")
    fi
elif [[ -n ${CUDA_HOME:-} && -x ${CUDA_HOME}/bin/nvcc ]]; then
    cuda_compiler="${CUDA_HOME}/bin/nvcc"
elif [[ -n ${CUDA_PATH:-} && -x ${CUDA_PATH}/bin/nvcc ]]; then
    cuda_compiler="${CUDA_PATH}/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
    cuda_compiler=$(command -v nvcc)
else
    die "CUDA nvcc was not found; pass --cuda <major.minor> or select a toolkit"
fi

if [[ -n ${cuda_compiler} ]]; then
    cuda_compiler=$(resolve_executable "${cuda_compiler}") \
        || die "CUDA compiler is not executable: ${cuda_compiler}"
    nvcc_output=$("${cuda_compiler}" "${cuda_compiler_args[@]}" --version) \
        || die "failed to query ${cuda_compiler} --version"
    cuda_version=$(sed -nE \
        's/.*release ([0-9]+\.[0-9]+).*/\1/p' <<< "${nvcc_output}" | head -1)
    [[ -n ${cuda_version} ]] \
        || die "could not determine the CUDA version from ${cuda_compiler}"
    echo "CUDA: ${cuda_version} (${cuda_compiler})"
else
    echo "CUDA: ${cuda_version} (selected by --cuda)"
fi

torch_cuda_tag="cu${cuda_version/./}"
torch_index_url="https://download.pytorch.org/whl/${torch_cuda_tag}"

cuda_major=${cuda_version%%.*}
cupy_requirement=${CUPY_PACKAGE:-"cupy-cuda${cuda_major}x"}

# -----------------------------------------------------------------------------
# Dependency metadata
# -----------------------------------------------------------------------------

requirements_file=$(mktemp -t bootstrap-requirements.XXXXXX)
cleanup()
{
    rm -f "${requirements_file}"
}
trap cleanup EXIT

"${metadata_python}" -B - "${repo_root}/pyproject.toml" "${cupy_requirement}" \
    > "${requirements_file}" <<'PY'
"""Emit bootstrap requirements as NUL-delimited strings.

The first item is the Torch requirement. Remaining items are the union of the
build requirements, project dependencies, and recursively expanded shared
development extra. Same-project extra references are expanded instead of
emitted.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        try:
            from pip._vendor import tomli as tomllib
        except ImportError:
            try:
                from pip._vendor import toml as tomllib
            except ImportError as error:
                raise SystemExit(
                    "unable to read pyproject.toml; install a current pip or tomli"
                ) from error

try:
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.utils import canonicalize_name


pyproject_path = pathlib.Path(sys.argv[1])
cupy_requirement = sys.argv[2]
pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

build_requirements = list(pyproject.get("build-system", {}).get("requires", []))
project = pyproject.get("project", {})
project_name = canonicalize_name(project.get("name", ""))
project_requirements = list(project.get("dependencies", []))


# Load the shared extras-expansion helper by file path so every caller uses
# the same logic.
_metadata_spec = importlib.util.spec_from_file_location(
    "gsplat_pyproject_metadata",
    pyproject_path.parent / "gsplat" / "build_support" / "pyproject_metadata.py",
)
if _metadata_spec is None or _metadata_spec.loader is None:
    raise SystemExit("unable to load gsplat/build_support/pyproject_metadata.py")
_metadata = importlib.util.module_from_spec(_metadata_spec)
_metadata_spec.loader.exec_module(_metadata)


requirements = [
    *build_requirements,
    *project_requirements,
    *_metadata.expand_optional_group(project, "dev"),
    # torchpq imports CuPy without declaring it. Bootstrap already selected
    # nvcc, so choose the corresponding CUDA-major wheel deterministically.
    cupy_requirement,
]

deduplicated: list[str] = []
seen: set[str] = set()
for requirement_text in requirements:
    if requirement_text not in seen:
        seen.add(requirement_text)
        deduplicated.append(requirement_text)

torch_requirements = [
    requirement_text
    for requirement_text in deduplicated
    if canonicalize_name(Requirement(requirement_text).name) == "torch"
]
if len(torch_requirements) != 1:
    raise SystemExit(
        "expected exactly one direct Torch requirement, found "
        f"{len(torch_requirements)}"
    )

output = [
    torch_requirements[0],
    *(
        requirement_text
        for requirement_text in deduplicated
        if canonicalize_name(Requirement(requirement_text).name) != "torch"
    ),
]
for requirement_text in output:
    sys.stdout.buffer.write(requirement_text.encode() + b"\0")
PY

mapfile -d '' -t requirements < "${requirements_file}"
(( ${#requirements[@]} > 1 )) || die "no bootstrap dependencies were found"

torch_requirement=${requirements[0]}
dependency_requirements=("${requirements[@]:1}")

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------

inspect_torch()
{
    "${target_python}" - "${torch_requirement}" <<'PY'
import sys
from importlib.metadata import version

try:
    from packaging.requirements import Requirement
except ImportError:
    from pip._vendor.packaging.requirements import Requirement

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    print("missing")
    raise SystemExit

requirement = Requirement(sys.argv[1])
distribution_version = version("torch")
status = (
    "compatible"
    if requirement.specifier.contains(distribution_version, prereleases=True)
    else "incompatible"
)
print(
    status,
    distribution_version,
    torch.__version__,
    torch.version.cuda or "cpu",
    sep="\t",
)
PY
}

validate_torch_cuda()
{
    local torch_version=$1
    local torch_cuda=$2
    local torch_cuda_major=${torch_cuda%%.*}

    if [[ ${torch_cuda} == cpu || ${torch_cuda_major} != "${cuda_major}" ]]; then
        die "installed Torch ${torch_version} uses CUDA ${torch_cuda}, but CUDA "\
"${cuda_version} was selected; create a compatible environment with --venv"
    fi
    if [[ ${torch_cuda} != "${cuda_version}" ]]; then
        echo "WARNING: Torch uses CUDA ${torch_cuda} while CUDA ${cuda_version} "\
"was selected; compatible CUDA minor versions will be used." >&2
    fi
}

torch_is_installed=false
if [[ ${target_python_exists} == true && ${dry_run} == false ]]; then
    torch_info=$(inspect_torch) || die "Torch could not be inspected"
    IFS=$'\t' read -r torch_status installed_torch_distribution_version \
        installed_torch_version installed_torch_cuda <<< "${torch_info}"
    case "${torch_status}" in
        compatible)
            validate_torch_cuda \
                "${installed_torch_version}" "${installed_torch_cuda}"
            torch_is_installed=true
            ;;
        incompatible)
            die "installed Torch ${installed_torch_version} does not satisfy "\
"${torch_requirement}; create a new environment with --venv"
            ;;
        missing)
            ;;
        *)
            die "unexpected Torch inspection result: ${torch_status}"
            ;;
    esac
fi

if [[ ${torch_is_installed} == false ]]; then
    torch_install=(
        "${target_python}"
        -m pip install
        --index-url "${torch_index_url}"
        "${pip_options[@]}"
        "${torch_requirement}"
    )
    run_command "${torch_install[@]}"
fi

if [[ ${dry_run} == false ]]; then
    torch_info=$(inspect_torch) || die "Torch could not be inspected after installation"
    IFS=$'\t' read -r torch_status installed_torch_distribution_version \
        installed_torch_version installed_torch_cuda <<< "${torch_info}"
    [[ ${torch_status} == compatible ]] \
        || die "installed Torch does not satisfy ${torch_requirement}"
    validate_torch_cuda "${installed_torch_version}" "${installed_torch_cuda}"
fi

dependency_install=(
    "${target_python}"
    -m pip install
    "${pip_options[@]}"
    "${dependency_requirements[@]}"
)
if [[ ${dry_run} == false ]]; then
    # Prevent a transitive dependency from replacing the CUDA-compatible Torch
    # selected above. Pip reports a resolution error instead.
    dependency_install+=("torch==${installed_torch_distribution_version}")
fi
run_command "${dependency_install[@]}"

if [[ ${dry_run} == false ]]; then
    torch_info=$(inspect_torch) || die "Torch could not be inspected"
    IFS=$'\t' read -r torch_status final_torch_distribution_version \
        final_torch_version installed_torch_cuda <<< "${torch_info}"
    [[ ${torch_status} == compatible ]] \
        || die "Torch became incompatible during dependency installation"
    [[ ${final_torch_distribution_version} == \
       "${installed_torch_distribution_version}" ]] \
        || die "Torch changed from ${installed_torch_distribution_version} to "\
"${final_torch_distribution_version} during dependency installation"
    validate_torch_cuda "${final_torch_version}" "${installed_torch_cuda}"
    echo "Torch: ${final_torch_version} (CUDA ${installed_torch_cuda})"
fi

# -----------------------------------------------------------------------------
# Repository hook and handoff
# -----------------------------------------------------------------------------

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
if [[ ${dry_run} == true ]]; then
    print_command cp -f "${hook_src}" "${hook_path}"
else
    if [[ -e ${hook_path} || -L ${hook_path} ]] && ! hook_is_ours "${hook_path}"; then
        mv --backup=numbered "${hook_path}" "${hook_path}.backup"
        echo "Backed up your existing pre-commit hook to ${hook_path}.backup"
    fi
    # Remove first so an existing symlink is replaced rather than followed.
    rm -f "${hook_path}"
    cp -f "${hook_src}" "${hook_path}"
    chmod +x "${hook_path}"
    echo "Installed pre-commit hook at ${hook_path}"
fi

echo "Bootstrap complete."
if [[ -n ${venv_dir} ]]; then
    printf 'Activate the environment before building:\n  source %q\n' \
        "${venv_dir}/bin/activate"
fi
