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
script_rel="${script_path#"${repo_root}/"}"

# Selection mode: local (default) | full | changed | staged. `mode_flag` records
# the explicit flag that set a non-default mode, so a second, different mode
# flag is a hard error rather than a silent last-wins.
mode="local"
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

Usage: ${script_name} [--check] [--full | --changed <ref> | --staged] [--precommit] [--help|-h]

Options:
  --check          Check formatting without modifying files.
  --full           Format all tracked source files.
  --changed <ref>  Format only source files changed since <ref> (e.g. for CI).
  --staged         Format the staged content and re-stage it (with --check,
                   check it instead of modifying anything).
  --precommit      Run what the installed pre-commit hook needs.
  --help, -h       Show this help message.

With no mode flag, format the source files with local changes (staged or
unstaged). --full, --changed, and --staged select what to act on instead and
are mutually exclusive.
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
        die "${flag} cannot be combined with ${mode_flag}; pass at most one of --full / --changed / --staged"
    fi
    mode="${requested}"
    mode_flag="${flag}"
}

# Command line argument processing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --precommit)
            # Stable entry point for the installed pre-commit hook. The name must
            # never change and usage() must keep it as an options-list entry --
            # the hook greps --help for that entry to learn it can be run; only
            # the expansion below may change.
            set -- --staged "${@:2}"
            continue
            ;;
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
        --staged)
            set_mode staged --staged
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

# The base tree that changes are diffed against — HEAD normally, the empty tree
# for the very first commit (no HEAD yet).
staged_base() {
    if git -C "${repo_root}" rev-parse --verify -q HEAD >/dev/null; then
        printf 'HEAD\n'
    else
        git -C "${repo_root}" hash-object -t tree /dev/null
    fi
}

find_repo_files() {
    local git_diff_args=()

    case "${mode}" in
        local)
            # Files that differ from HEAD in the working tree, so both staged
            # and unstaged edits are covered.
            git_diff_args=(
                --diff-filter=ACMR
                "$(staged_base)"
            )
            ;;
        staged)
            # Exactly the files this commit records, index vs HEAD.
            git_diff_args=(
                --cached
                --diff-filter=ACMR
                "$(staged_base)"
            )
            ;;
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

# A symlink's staged blob is its target string, not source code: formatting or
# judging it would corrupt the link, so staged modes leave symlinks alone.
staged_symlink() {
    [[ "$(git -C "${repo_root}" ls-files --stage -- "$1" | awk '{print $1}')" == "120000" ]]
}

# In staged mode, reproduce each file's staged blob into a scratch tree that
# mirrors the repo's relative layout. The formatters then run there with cwd
# set to the scratch root, so: they judge exactly the bytes being committed
# (a partially-staged file's unstaged hunks are never seen), no working-tree
# file is ever touched, and their output shows clean repo-relative paths.
materialize_staged_copies() {
    local path dest
    for path in "$@"; do
        dest="${staged_scratch_dir}/${path}"
        mkdir -p "$(dirname "${dest}")"
        # Drop any pre-seeded config symlink at this path before writing, so a
        # staged config file lands as its staged content and the real repo
        # file is never written through the symlink.
        rm -f "${dest}"
        git -C "${repo_root}" show ":${path}" > "${dest}"
    done
}

staged_scratch_dir=""
if [[ "${mode}" == "staged" ]] && $check_mode; then
    staged_scratch_dir="$(mktemp -d)"
    trap 'rm -rf "${staged_scratch_dir}"' EXIT
    # Mirror the formatter configs into the scratch root so black's and
    # clang-format's upward config search finds the same files they would in
    # the real tree. Without .clang-format, clang-format silently falls back
    # to LLVM defaults and would misjudge every C++ file.
    for cfg in .clang-format pyproject.toml; do
        if [[ -e "${repo_root}/${cfg}" ]]; then
            ln -s "${repo_root}/${cfg}" "${staged_scratch_dir}/${cfg}"
        fi
    done
fi

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

# Rollback state for perform_staged_format, kept in globals so the EXIT trap can
# still reach them after the function has returned:
#   perform_scratch - scratch tree to remove on exit
#   perform_index   - the pre-format index (as a tree) to restore on an abort
#   perform_done    - set to 1 at every controlled return; the trap restores only
#                     while it is still 0, i.e. set -e killed the function partway
#   perform_touched - working-tree files rewritten so far, to restore on an abort
perform_scratch=""
perform_index=""
perform_done=0
perform_touched=()

# Safety net run from the EXIT trap: on an uncontrolled abort restore the index
# and every rewritten working-tree file so an unexpected failure never leaves a
# half-formatted tree or loses an edit; always remove the scratch tree.
perform_cleanup() {
    # Keep going past individual failures: under set -e one unrestorable file
    # would otherwise abort the rest of the restore, the report, and the
    # scratch cleanup.
    set +e
    if (( perform_done == 0 )) && [[ -n "${perform_index}" ]]; then
        git -C "${repo_root}" read-tree "${perform_index}"
        local f failed=()
        for f in "${perform_touched[@]}"; do
            cp -- "${perform_scratch}/work/${f}" "${repo_root}/${f}" || failed+=("${f}")
        done
        if (( ${#failed[@]} > 0 )); then
            # Keep the scratch tree: it holds the only backup of these files.
            echo "ERROR: formatting was interrupted; the index was restored, but these" >&2
            echo "files could not be (backups in ${perform_scratch}/work):" >&2
            printf '    %s\n' "${failed[@]}" >&2
            return
        fi
        echo "ERROR: formatting was interrupted; restored the index and working tree." >&2
    fi
    [[ -n "${perform_scratch}" ]] && rm -rf "${perform_scratch}"
}

# Format the staged content and re-stage it.
perform_staged_format() {
    cd "${repo_root}"

    local staged=()
    mapfile -d '' -t staged < <(
        find_repo_files "${python_extensions[@]}" "${cpp_extensions[@]}"
    )
    if (( ${#staged[@]} == 0 )); then
        echo "No staged source files to format."
        return 0
    fi

    perform_done=0
    perform_touched=()
    trap 'perform_cleanup' EXIT
    perform_scratch="$(mktemp -d)"
    perform_index="$(git write-tree)"

    # The staged copies get their own subtree: a repo file living under a
    # directory literally named orig/ or work/ must not collide with another
    # file's bookkeeping copies below.
    mkdir -p "${perform_scratch}/staged"
    # Mirror the formatter configs so black and clang-format resolve the same
    # config in the scratch tree as in the real one.
    local cfg
    for cfg in .clang-format pyproject.toml; do
        [[ -e "${repo_root}/${cfg}" ]] && ln -s "${repo_root}/${cfg}" "${perform_scratch}/staged/${cfg}"
    done

    # First pass, read-only: format each staged blob in the scratch tree (its
    # original kept beside it), so nothing is mutated until every file is known
    # good.
    local pending=() f dest orig
    for f in "${staged[@]}"; do
        staged_symlink "${f}" && continue
        dest="${perform_scratch}/staged/${f}"
        orig="${perform_scratch}/orig/${f}"
        mkdir -p "$(dirname "${dest}")" "$(dirname "${orig}")"
        rm -f "${dest}"
        git show ":${f}" > "${dest}"
        cp -- "${dest}" "${orig}"
        # A failure here leaves the working tree untouched, so nothing is lost.
        if ! ( cd "${perform_scratch}/staged" && run_formatters format "${f}" ); then
            perform_done=1
            echo "ERROR: cannot format the staged content of ${f}." >&2
            echo "Nothing was changed; format it and re-stage by hand." >&2
            return 1
        fi
        # Already formatted: leave the index and working tree untouched, so an
        # unchanged file is not rewritten and does not spuriously trigger a build.
        git show ":${f}" | cmp -s - "${dest}" && continue
        pending+=("${f}")
    done

    # Second pass: re-stage each formatted blob, then bring the working tree
    # along.
    local restaged=() mode blob patch fully
    for f in "${pending[@]}"; do
        dest="${perform_scratch}/staged/${f}"
        orig="${perform_scratch}/orig/${f}"
        # Classify against the pre-format index entry, before it is replaced.
        fully=0
        git diff --quiet -- "${f}" && fully=1
        mode="$(git ls-files --stage -- "${f}" | awk '{print $1}')"
        # --no-filters: dest holds exact blob bytes (from git show); ambient
        # .gitattributes of whatever tree encloses $TMPDIR must not re-convert.
        blob="$(git hash-object --no-filters -w "${dest}")"
        if [[ ! -e "${f}" && ! -L "${f}" ]]; then
            # Staged but deleted from the working tree: format the index entry
            # only; there is no working-tree copy to sync.
            git update-index --cacheinfo "${mode},${blob},${f}"
            restaged+=("${f}")
            continue
        fi
        # Back up the working-tree file before rewriting it, so the safety net can
        # restore it if a later step aborts.
        mkdir -p "$(dirname "${perform_scratch}/work/${f}")"
        cp -- "${f}" "${perform_scratch}/work/${f}"
        perform_touched+=("${f}")
        git update-index --cacheinfo "${mode},${blob},${f}"
        if (( fully )); then
            # Fully staged: the working-tree copy is the old staged content, so
            # mirror the formatted content over it exactly.
            cp -- "${dest}" "${f}"
        else
            # Partially staged: the working tree also carries unstaged edits that
            # are not this commit's content, so never rewrite it wholesale.
            # Instead apply just the formatting delta to it; the unstaged edits
            # stay physically in place.
            patch="$(mktemp -p "${perform_scratch}")"
            diff -u --label "a/${f}" --label "b/${f}" -- "${orig}" "${dest}" > "${patch}" || true
            if ! git apply --whitespace=nowarn -- "${patch}" 2>/dev/null; then
                # An unstaged edit overlaps the formatting, so the delta does not
                # apply. The commit still gets the formatted content; the
                # working-tree copy keeps the old formatting until a later
                # format run.
                echo "NOTE: ${f}: unstaged edits overlap or adjoin the formatting; its" >&2
                echo "      working-tree copy was left as is (the committed content is" >&2
                echo "      formatted)." >&2
            fi
        fi
        restaged+=("${f}")
    done

    # Formatting may have normalized every staged change back to HEAD, so the
    # commit would be empty; refuse rather than record it, leaving the formatted
    # content staged.
    if git diff --cached --quiet "$(staged_base)"; then
        perform_done=1
        echo "ERROR: after formatting, no staged change remains -- nothing to commit." >&2
        echo "Your staged edits differed from HEAD only in formatting; the staged" >&2
        echo "content has been normalized. Nothing was committed." >&2
        echo "To record an empty commit anyway: git commit --allow-empty --no-verify" >&2
        return 1
    fi

    perform_done=1
    if (( ${#restaged[@]} > 0 )); then
        echo "Formatted and re-staged: ${restaged[*]}"
    else
        echo "Staged content already formatted."
    fi
    return 0
}

# In --staged mode without --check, format the staged content and re-stage it,
# then stop; the shared sections below only check or format the other modes.
if [[ "${mode}" == "staged" ]] && ! $check_mode; then
    perform_staged_format
    exit
fi

staged_failed=false

mapfile -d '' -t src_files < <(
    find_repo_files "${python_extensions[@]}" "${cpp_extensions[@]}"
)

if (( ${#src_files[@]} == 0 )); then
    echo "No source files selected."
elif [[ "${mode}" == "staged" ]]; then
    # --staged --check: check each file's staged content in a scratch tree.
    checkable=()
    for f in "${src_files[@]}"; do
        staged_symlink "${f}" && continue
        checkable+=("${f}")
    done
    if (( ${#checkable[@]} > 0 )); then
        materialize_staged_copies "${checkable[@]}"
        ( cd "${staged_scratch_dir}" && run_formatters check "${checkable[@]}" ) || staged_failed=true
    fi
elif $check_mode; then
    run_formatters check "${src_files[@]}"
else
    run_formatters format "${src_files[@]}"
fi


# Report staged-mode failures loudly. Nothing was modified, so the fix is to
# format the working tree and restage.
if [[ "${mode}" == "staged" ]] && $staged_failed; then
    {
        echo
        echo "ERROR: staged content is not formatted (the diff above is the fix)."
        echo "Format the files, restage them, and commit again:"
        echo "    ${script_rel}        # format the working tree in place"
        echo "    git add -- <files>   # restage the reformatted files"
        echo
        echo "To commit without this check (e.g. history surgery): git commit --no-verify"
    } >&2
    exit 1
fi
