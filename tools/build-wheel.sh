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
repo_root="$(cd "$(dirname "${script_path}")/.." && pwd)"
script_name="$(basename "${BASH_SOURCE[0]}")"

usage() {
    cat <<EOF
Build a gsplat wheel with pip and scikit-build-core.

Usage: ${script_name} [preset] [pip-arg...]

The optional configure preset selects the build configuration
(default: full-release). It stays the one source of the build
configuration; this script only forwards it to scikit-build-core and
reuses the preset's build tree under build/<preset>. Any remaining
arguments are passed to pip unchanged, after the preset-derived
settings (e.g. --wheel-dir to choose the output directory, or extra
--config-settings entries). The output directory and its cleanup are the
caller's responsibility; without --wheel-dir, pip writes to the current
directory.
EOF
}

preset=full-release
extra_args=()
case "${1:-}" in
    "") ;;
    --help|-h)
        usage
        exit 0
        ;;
    -*)
        echo "ERROR: the first argument must be a preset name: $1" >&2
        usage >&2
        exit 1
        ;;
    *)
        preset="$1"
        extra_args=("${@:2}")
        ;;
esac

# Fail on a typo before pip spends time collecting the build.
if ! cmake --list-presets 2>/dev/null | grep -qE "^  \"${preset}\""; then
    echo "ERROR: unknown configure preset '${preset}'." >&2
    cmake --list-presets >&2 || true
    exit 1
fi

cd "${repo_root}"

# cmake.build-type is emptied so scikit-build-core's Release default cannot
# override the preset's CMAKE_BUILD_TYPE.
exec python -m pip wheel \
    --verbose \
    --no-build-isolation \
    --no-deps \
    --config-settings=build-dir="build/${preset}" \
    --config-settings=cmake.build-type= \
    --config-settings=cmake.args=--preset="${preset}" \
    "${extra_args[@]}" \
    .
