# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the configured CUDA-availability policy."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests import _backend_collect, _cuda


def test_force_cuda_defaults_to_enabled_without_probe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GSPLAT_TESTS_FORCE_CUDA", raising=False)

    def unexpected_probe() -> bool:
        raise AssertionError("default forced CUDA must not probe PyTorch")

    monkeypatch.setattr(_cuda, "_detect_cuda", unexpected_probe)

    assert _cuda.cuda_is_available() is True


def test_forced_cuda_does_not_probe(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GSPLAT_TESTS_FORCE_CUDA", "1")

    def unexpected_probe() -> bool:
        raise AssertionError("forced CUDA must not probe PyTorch")

    monkeypatch.setattr(_cuda, "_detect_cuda", unexpected_probe)

    assert _cuda.cuda_is_available() is True


@pytest.mark.parametrize("detected", [False, True])
def test_automatic_cuda_uses_torch_probe(
    monkeypatch: pytest.MonkeyPatch, detected: bool
):
    monkeypatch.setenv("GSPLAT_TESTS_FORCE_CUDA", "0")
    calls = 0

    def detect() -> bool:
        nonlocal calls
        calls += 1
        return detected

    detector: Callable[[], bool] = detect
    monkeypatch.setattr(_cuda, "_detect_cuda", detector)

    assert _cuda.cuda_is_available() is detected
    assert calls == 1


def test_collection_probe_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_backend_collect, "cuda_is_available", lambda: True)

    def broken_extension():
        raise RuntimeError("native extension failed to load")

    with pytest.raises(RuntimeError, match="native extension failed to load"):
        _backend_collect.cuda_collect_ignore_glob(probe=broken_extension)
