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

# Installing a git hook needs git; without it there is nothing to do.
if ! command -v git >/dev/null 2>&1; then
    echo "WARNING: git is not installed; skipping pre-commit hook installation." >&2
    exit 0
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"
hook_src="${repo_root}/hooks/pre-commit"

# A pre-commit hook is "ours" if it carries the dispatcher's marker (any
# version) or is the symlink to lint/format-code.sh that older bootstrap
# versions installed. We replace our own hook but preserve a foreign one.
hook_is_ours() {
    local hook="$1"
    [[ -L "${hook}" && "$(readlink "${hook}")" == */lint/format-code.sh ]] && return 0
    grep -q "GSPLAT PRE-COMMIT HOOK" "${hook}" 2>/dev/null
}

# Install the pre-commit hook by copying it into the shared hooks directory. A
# copy (not a symlink) keeps working after the worktree that ran bootstrap is
# removed.
hook_path="$(git -C "${repo_root}" rev-parse --git-path hooks/pre-commit)"
# --git-path is relative to repo_root in the main worktree, absolute in a linked
# worktree; absolutize it so the install lands in the repo regardless of the
# directory bootstrap was invoked from.
[[ "${hook_path}" = /* ]] || hook_path="${repo_root}/${hook_path}"
mkdir -p "$(dirname "${hook_path}")"

# Preserve a pre-commit hook we did not install before overwriting it. Numbered
# backups rotate any earlier backup aside instead of clobbering it, so bootstrap
# always succeeds without losing a foreign hook.
if [[ -e "${hook_path}" || -L "${hook_path}" ]] && ! hook_is_ours "${hook_path}"; then
    backup="${hook_path}.backup"
    mv --backup=numbered "${hook_path}" "${backup}"
    echo "Backed up your existing pre-commit hook to ${backup}"
fi

# Remove first so an existing symlink is replaced rather than followed (which
# would overwrite its target).
rm -f "${hook_path}"
cp -f "${hook_src}" "${hook_path}"
chmod +x "${hook_path}"

echo "Installed pre-commit hook at ${hook_path}"
