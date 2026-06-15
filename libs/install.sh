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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $# -eq 0 ]; then
    echo "Available packages:"
    echo "  - geometry"
    echo "  - scene"
    echo "  - stage"
    echo ""
    echo "Usage: $0 <package_name>"
    echo "Example: $0 geometry"
    exit 0
fi

PACKAGE=$1

case $PACKAGE in
    geometry)
        echo "Installing gsplat-geometry..."
        (cd "$SCRIPT_DIR/geometry" && pip install -e .)
        ;;
    scene)
        echo "Installing gsplat-scene..."
        (cd "$SCRIPT_DIR/scene" && pip install -e .)
        ;;
    stage)
        echo "Installing gsplat-stage..."
        (cd "$SCRIPT_DIR/stage" && pip install -e .)
        ;;
    *)
        echo "Unknown package: $PACKAGE"
        echo "Available packages: geometry, scene, stage"
        exit 1
        ;;
esac

echo "Installation complete!"
