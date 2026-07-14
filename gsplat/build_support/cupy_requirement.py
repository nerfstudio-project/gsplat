# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single owner of the dynamic CuPy distribution selection.

The bare ``cupy`` package on PyPI is a source distribution that compiles
against the local CUDA toolkit and routinely fails when stub libs or optional
headers (e.g. cusparseLt) aren't present. Prefer a prebuilt wheel keyed on the
detected CUDA major version. This selection cannot live in the static
``pyproject.toml`` metadata, so the docker dependency provisioning and
``setup.py`` load this module to resolve the wheel name.

Also usable as a command, printing the resolved requirement:

    python3 cupy_requirement.py
"""

from __future__ import annotations

import os
import re
import subprocess
import warnings


def detect_cupy_requirement() -> str:
    """Pick a CuPy distribution that matches the local CUDA toolkit.

    Set ``CUPY_PACKAGE`` to override (e.g. ``CUPY_PACKAGE=cupy-cuda12x`` or
    ``CUPY_PACKAGE=cupy``).
    """
    override = os.getenv("CUPY_PACKAGE")
    if override:
        return override

    cuda_roots = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            cuda_roots.append(root)
    cuda_roots.append("/usr/local/cuda")

    # 1. Try ``nvcc --version`` from each candidate root, then PATH.
    nvcc_candidates = [os.path.join(r, "bin", "nvcc") for r in cuda_roots]
    nvcc_candidates.append("nvcc")

    for nvcc in nvcc_candidates:
        try:
            out = subprocess.check_output(
                [nvcc, "--version"], stderr=subprocess.STDOUT, text=True
            )
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            continue
        m = re.search(r"release (\d+)\.", out)
        if m:
            return f"cupy-cuda{m.group(1)}x"
        # nvcc ran but its --version output didn't match. That's distinct from
        # the "nvcc not found" case (silently skipped above) — it usually
        # signals a broken shim (ccache wrapper, non-NVIDIA stub) rather than
        # a legitimate format change, since the CUDA toolkit's --version has
        # been stable for years. Surface the anomaly so a developer who later
        # ends up with a source-built cupy isn't left guessing why.
        warnings.warn(
            f"nvcc at {nvcc} returned an unparseable --version output; "
            "skipping this candidate.",
            stacklevel=2,
        )

    # 2. Fall back to cuda.h's CUDA_VERSION macro for environments where
    # nvcc is missing (runtime-only CUDA install) or wrapped in a way
    # that breaks ``--version`` (e.g. a misbehaving ccache shim).
    # CUDA_VERSION encodes major as the integer division by 1000
    # (e.g. 13020 → 13). cuda.h ships with every CUDA toolkit.
    for root in cuda_roots:
        try:
            with open(os.path.join(root, "include", "cuda.h")) as f:
                content = f.read()
        except (FileNotFoundError, OSError):
            continue
        m = re.search(r"^#define\s+CUDA_VERSION\s+(\d+)", content, re.MULTILINE)
        if m:
            return f"cupy-cuda{int(m.group(1)) // 1000}x"

    return "cupy"


if __name__ == "__main__":
    print(detect_cupy_requirement())
