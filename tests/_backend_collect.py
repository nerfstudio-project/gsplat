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
de-collected — rather than ERRORing at import-time collection — on hosts that
cannot run them. Centralized here so the policy lives in one place.
"""

from __future__ import annotations

from typing import Callable, Optional

# Glob (relative to the conftest's dir) matching every test module in the tree.
_ALL_TESTS = ["**/test_*.py"]


def cuda_collect_ignore_glob(probe: Optional[Callable[[], object]] = None) -> list[str]:
    """Return the collect_ignore_glob for a CUDA-backed test directory.

    De-collects all tests when CUDA is unavailable. When ``probe`` is given (for
    packages whose test modules bind the native extension at import time, e.g.
    sensors), it is also called to detect a GPU host that nonetheless lacks a
    loadable/buildable extension; if the probe raises, the tests are de-collected
    so collection skips cleanly instead of erroring on the import.
    """
    import torch

    if not torch.cuda.is_available():
        return list(_ALL_TESTS)
    if probe is not None:
        try:
            probe()
        except Exception:
            return list(_ALL_TESTS)
    return []
