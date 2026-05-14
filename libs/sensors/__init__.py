# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor projection APIs for gsplat.

This package exposes two sub-packages: ``functional`` (Layer 1 stateless ops that
wrap the kernel layer with argument validation and device-transfer guards) and
``models`` (Layer 2 stateful ``nn.Module`` wrappers with learnable parameters).
"""

from . import functional, models

__all__ = ["functional", "models"]
