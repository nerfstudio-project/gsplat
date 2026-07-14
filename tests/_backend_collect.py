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

"""Shared collection guard for the native-extension test suites.

Each of tests/{geometry,sensors,experimental} sets
``collect_ignore_glob = cuda_collect_ignore_glob(...)`` so CUDA-only tests are
de-collected when the configured CUDA policy allows automatic detection and
PyTorch reports no CUDA device. Forced coverage instead collects the tests so
missing CUDA or native extensions fail visibly.
"""

from __future__ import annotations

from typing import Callable, Optional

from tests._cuda import cuda_is_available

# Glob (relative to the conftest's dir) matching every test module in the tree.
_ALL_TESTS = ["**/test_*.py"]


def cuda_collect_ignore_glob(probe: Optional[Callable[[], object]] = None) -> list[str]:
    """Return the collect_ignore_glob for a CUDA-backed test directory.

    De-collects all tests only when the shared CUDA policy reports unavailable.
    When ``probe`` is given for a package that binds its native extension during
    collection, its exception deliberately propagates: a missing or broken
    extension must fail instead of silently reducing test coverage.
    """
    if not cuda_is_available():
        return list(_ALL_TESTS)
    if probe is not None:
        probe()
    return []
