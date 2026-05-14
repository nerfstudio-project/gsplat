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

PACKAGE=$1

check_no_build_isolation_deps() {
    python -c "import setuptools, wheel" >/dev/null 2>&1 || {
        echo "setuptools and wheel must be installed before using --no-build-isolation" >&2
        exit 1
    }
}

case $PACKAGE in
    geometry)
        echo "Installing gsplat-geometry..."
        (cd "$SCRIPT_DIR/geometry" && pip install -e .)
        ;;
    scene)
        echo "Installing gsplat-scene..."
        (cd "$SCRIPT_DIR/scene" && pip install -e .)
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
    stage)
        echo "Installing gsplat-stage..."
        (cd "$SCRIPT_DIR/stage" && pip install -e .)
        ;;
    *)
        echo "Unknown package: $PACKAGE" >&2
        echo "Available packages: geometry, scene, sensors, stage" >&2
        exit 1
        ;;
esac

echo "Installation complete!"
