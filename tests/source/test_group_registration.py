# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every Python test file must belong to a registered CTest group.

CTest runs one pytest invocation per registered directory, so a test file
outside every registered directory is silently never executed. Recursive
pytest discovery used to catch such files; this check replaces that safety
net.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

_REGISTRATION = re.compile(
    r'gsplat_add_pytest\(\s*\w+\s+"\$\{GSPLAT_SOURCE_DIR\}/(tests/[^"]+)"'
)


def _registered_directories() -> list[Path]:
    cmakelists = (TESTS_ROOT / "CMakeLists.txt").read_text()
    directories = [REPO_ROOT / match for match in _REGISTRATION.findall(cmakelists)]
    assert directories, "no gsplat_add_pytest registrations found"
    return directories


def test_every_test_file_belongs_to_a_registered_group() -> None:
    registered = _registered_directories()
    unregistered = [
        path.relative_to(REPO_ROOT)
        for path in TESTS_ROOT.rglob("test_*.py")
        if not any(path.is_relative_to(directory) for directory in registered)
    ]
    assert not unregistered, (
        "test files outside every registered CTest group (register the "
        f"directory in tests/CMakeLists.txt or move the file): {unregistered}"
    )
