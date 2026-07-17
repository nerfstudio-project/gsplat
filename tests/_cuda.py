# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared CUDA-availability policy for the Python test suite."""

from __future__ import annotations

import os

import torch

_TESTS_FORCE_CUDA_ENV = "GSPLAT_TESTS_FORCE_CUDA"
_detect_cuda = torch.cuda.is_available


def cuda_is_available() -> bool:
    """Return whether CUDA-backed tests should run.

    A forced configuration deliberately avoids querying PyTorch. This makes a
    missing or broken GPU fail in the CUDA test itself instead of silently
    reducing coverage through collection guards or skip conditions.

    Returns:
        ``True`` when CUDA is forced; otherwise, PyTorch's detected
        availability.

    Raises:
        RuntimeError: If the ``GSPLAT_TESTS_FORCE_CUDA`` environment variable
            contains a value other than ``0`` or ``1``.
    """

    force_cuda = os.environ.get(_TESTS_FORCE_CUDA_ENV, "1")
    if force_cuda == "1":
        return True
    if force_cuda == "0":
        return bool(_detect_cuda())
    raise RuntimeError(
        f"{_TESTS_FORCE_CUDA_ENV} must be configured as 0 or 1, got {force_cuda!r}"
    )
