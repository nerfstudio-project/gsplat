# SPDX-FileCopyrightText: Copyright 2025-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
Pytest configuration and shared fixtures for gsplat tests.

This file is automatically discovered by pytest and applies to all test files
in this directory and subdirectories.
"""

import pytest
import torch
import gc


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Autouse fixture that runs before every test to ensure:
    1. Deterministic random seed
    2. CUDA cache is cleared
    3. Garbage collection is performed

    This fixture automatically applies to all tests in this directory
    without needing to be explicitly requested.
    """

    seed = 42

    # Set seed based on test name for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    # Run garbage collection
    gc.collect()

    # Yield to run the test
    yield

    # Optional: cleanup after test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
