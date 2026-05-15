# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend implementation: extension loading, autograd, and CUDA dispatch.

This package is not a curated user-facing API. Prefer
:mod:`gsplat_sensors.functional` for sensor operators. Submodules such as
``kernels.cameras.ops`` remain importable for in-package use and tests.
"""

__all__: list[str] = []
