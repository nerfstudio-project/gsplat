#!/bin/bash
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

# Install script for gsplat shared libraries
# Usage: ./install.sh [package_name]
# If no package name is provided, lists the supported packages
#
# Install order matters for editable dev: stage depends on scene, so
# `./install.sh scene` before `./install.sh stage` to pick up local edits.
# sensors depends on geometry: `./install.sh geometry` before `./install.sh sensors`.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Available packages:"
    echo "  - geometry"
    echo "  - scene"
    echo "  - sensors"
    echo "  - stage"
    echo ""
    echo "Usage: $0 <package_name>"
    echo "Example: $0 geometry"
    exit 0
fi

# Skip `pip install -e` when the package is already installed in the active
# interpreter and no source file is newer than the egg-info marker (which pip
# rewrites on every editable install). pip install -e always rebuilds the
# editable wheel and reinstalls — even when nothing changed — so this guard is
# the only way to make repeated runs cheap. Removing the egg-info dir or pip
# uninstalling the package both force a fresh install.
install_one()
{
    local pkg="$1"
    local dist="gsplat_${pkg}"
    local dir="$SCRIPT_DIR/$pkg"
    local marker="$dir/${dist}.egg-info/PKG-INFO"

    # find_spec checks the package is importable without actually loading it
    # (a real `import` of any of these triggers torch and costs ~2s each).
    if [[ -f "$marker" ]] \
       && python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('${dist}') else 1)" >/dev/null 2>&1 \
       && [[ -z "$(find "$dir" -type f \
                \( -name '*.py' -o -name '*.toml' -o -name '*.cu' \
                   -o -name '*.cuh' -o -name '*.cpp' -o -name '*.cc' \
                   -o -name '*.h'  -o -name '*.hpp' \) \
                -newer "$marker" -print -quit)" ]]
    then
        echo "Skipping ${dist}: up to date"
        return 0
    fi

    echo "Installing ${dist}..."
    (cd "$dir" && pip install -e .)
}

PACKAGE=$1

check_no_build_isolation_deps() {
    python -c "import setuptools, wheel" >/dev/null 2>&1 || {
        echo "setuptools and wheel must be installed before using --no-build-isolation" >&2
        exit 1
    }
}

case $PACKAGE in
    geometry|scene|stage)
        install_one "$PACKAGE"
        ;;
    sensors)
        python -c "import importlib.metadata as m; m.version('gsplat-geometry')" >/dev/null 2>&1 || {
            echo "gsplat-geometry must be installed first: bash libs/install.sh geometry" >&2
            exit 1
        }
        echo "Installing gsplat-sensors..."
        check_no_build_isolation_deps
        (cd "$SCRIPT_DIR/sensors" && pip install --no-build-isolation -e .)
        ;;
    *)
        echo "Unknown package: $PACKAGE" >&2
        echo "Available packages: geometry, scene, sensors, stage" >&2
        exit 1
        ;;
esac

echo "Installation complete!"
