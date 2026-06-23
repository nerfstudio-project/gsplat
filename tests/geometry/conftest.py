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

"""Shared pytest configuration for the gsplat.geometry test suite."""

from __future__ import annotations

from tests._backend_collect import cuda_collect_ignore_glob

# Geometry ops require CUDA. Importing gsplat.geometry is lazy (no build at
# collection), but as a defensive guard for CPU-only CI we de-collect every
# geometry test when no GPU is available so an unguarded CUDA test can't error.
collect_ignore_glob = cuda_collect_ignore_glob()
