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

import importlib
from pathlib import Path
import sys
import types

import pytest


TRACE_PATH = Path(__file__).resolve().parents[1] / "gsplat" / "trace.py"


class _FakeDomain:
    # Record NVTX-like calls so tests can assert the exact emitted hierarchy.
    def __init__(self):
        self.calls = []

    def push_range(self, message=None, **kwargs):
        self.calls.append(("push", message, kwargs))

    def pop_range(self):
        self.calls.append(("pop", None, {}))


@pytest.fixture
def load_trace(monkeypatch):
    def loader(with_nvtx: bool):
        if not with_nvtx:
            # Force the import-time "nvtx unavailable" branch.
            monkeypatch.setitem(sys.modules, "nvtx", None)
            domain = None
        else:
            domain = _FakeDomain()

            nvtx_module = types.ModuleType("nvtx")
            nvtx_module.__path__ = []
            nvtx_module.enabled = lambda: True
            nvtx_module.get_domain = lambda name: domain
            monkeypatch.setitem(sys.modules, "nvtx", nvtx_module)

        # Provide a lightweight package shell so importing `gsplat.trace` does
        # not execute the real `gsplat/__init__.py`, which pulls in CUDA-heavy
        # imports unrelated to these unit tests.
        gsplat_pkg = types.ModuleType("gsplat")
        gsplat_pkg.__path__ = [str(TRACE_PATH.parent)]
        monkeypatch.setitem(sys.modules, "gsplat", gsplat_pkg)

        sys.modules.pop("gsplat.trace", None)
        module = importlib.import_module("gsplat.trace")
        module = importlib.reload(module)
        return module, domain

    return loader


@pytest.fixture
def enabled_trace(load_trace):
    return load_trace(True)


def _run_range(trace, name: str):
    with trace.trace_range(name):
        pass


def _run_decorated(trace, name: str):
    @trace.trace_function(name)
    def fn():
        return 7

    return fn()


# Disabled tracing should bind all helpers to behavior-preserving no-ops,
# regardless of whether nvtx is missing entirely or the gsplat domain is
# disabled.
def test_trace_noops_when_nvtx_is_unavailable(load_trace):
    trace, _ = load_trace(False)

    def fn():
        return "ok"

    assert trace.trace_function("leaf")(fn) is fn
    trace.trace_push("outer")
    trace.trace_pop()
    with trace.trace_range("inner"):
        assert fn() == "ok"


# Each public tracing mechanism should emit the same single-level range shape
# for a simple one-segment name.
@pytest.mark.parametrize(
    "mechanism",
    [
        lambda trace: (trace.trace_push("outer"), trace.trace_pop()),
        lambda trace: _run_range(trace, "outer"),
        lambda trace: _run_decorated(trace, "outer"),
    ],
    ids=["push-pop", "range", "decorator"],
)
def test_trace_records_single_level_name(enabled_trace, mechanism):
    trace, domain = enabled_trace
    mechanism(trace)
    assert domain.calls == [("push", "outer", {}), ("pop", None, {})]


def test_trace_forwards_nvtx_kwargs(enabled_trace):
    trace, domain = enabled_trace
    trace.trace_push("outer", color="blue", category="unit", payload=7)
    with trace.trace_range("inner", color="green", category="test"):
        pass

    @trace.trace_function("leaf", payload=11)
    def fn():
        return 7

    assert fn() == 7
    assert domain.calls == [
        ("push", "outer", {"color": "blue", "category": "unit", "payload": 7}),
        ("push", "inner", {"color": "green", "category": "test"}),
        ("pop", None, {}),
        ("push", "leaf", {"payload": 11}),
        ("pop", None, {}),
    ]
