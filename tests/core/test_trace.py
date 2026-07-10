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
import sys
import types

import pytest

from .._package_paths import gsplat_package_file

TRACE_PATH = gsplat_package_file("trace.py")


class _FakeDomain:
    # Record NVTX-like calls so tests can assert the exact emitted hierarchy.
    def __init__(self):
        self.calls = []

    def get_event_attributes(self, **kwargs):
        return kwargs

    def push_range(self, attrs):
        self.calls.append(
            (
                "push",
                attrs["message"],
                {k: v for k, v in attrs.items() if k != "message"},
            )
        )

    def pop_range(self):
        self.calls.append(("pop", None, {}))


class _IncompleteDomain:
    # Missing per-domain push_range / pop_range / get_event_attributes —
    # simulates the nvtx 0.2.13 build that triggered the fallback branch.
    def __init__(self):
        # Intentionally empty: probing for "push_range" / "pop_range" /
        # "get_event_attributes" must return False so `_DOMAIN` lands as `None`.
        pass


class _FakeModuleNvtxRecorder:
    # Records module-level nvtx.push_range / nvtx.pop_range calls so the
    # fallback-branch tests can assert the exact emitted call sequence.
    # `push_range` mirrors the real nvtx 0.2.10+ signature exactly so the
    # import-time `inspect.signature` probe in trace.py sees the same
    # parameter set the production code expects.
    def __init__(self):
        self.calls = []

    def push_range(self, message, color=None, domain=None, category=None, payload=None):
        kwargs = {
            k: v
            for k, v in (
                ("color", color),
                ("domain", domain),
                ("category", category),
                ("payload", payload),
            )
            if v is not None
        }
        self.calls.append(("push", message, kwargs))

    def pop_range(self):
        self.calls.append(("pop", None, {}))


@pytest.fixture
def load_trace(monkeypatch):
    """Reload `gsplat.trace` under a controlled `nvtx` shim.

    Modes:
      - `"absent"`: `nvtx` is not importable (no-op branch).
      - `"domain"`: full per-domain API present (Domain branch); records via
        `_FakeDomain.calls`.
      - `"fallback"`: `nvtx.get_domain(...)` returns an object missing the
        per-domain methods; module-level `nvtx.push_range` / `nvtx.pop_range`
        are stubbed to record calls.
    """

    def loader(mode: str):
        recorder = None
        if mode == "absent":
            monkeypatch.setitem(sys.modules, "nvtx", None)
        elif mode == "domain":
            recorder = _FakeDomain()
            nvtx_module = types.ModuleType("nvtx")
            nvtx_module.__path__ = []
            nvtx_module.enabled = lambda: True
            nvtx_module.get_domain = lambda name: recorder
            # Provide module-level push_range / pop_range too so the import-
            # time signature probe in trace.py has something to inspect. The
            # Domain branch never reaches these.
            nvtx_module.push_range = (
                lambda message, color=None, domain=None, category=None, payload=None: None
            )
            nvtx_module.pop_range = lambda: None
            monkeypatch.setitem(sys.modules, "nvtx", nvtx_module)
        elif mode == "fallback":
            # nvtx.get_domain returns an object missing the per-domain API,
            # so `_DOMAIN` ends up `None` in trace.py and the module-level
            # branch fires. The recorder captures the module-level calls.
            recorder = _FakeModuleNvtxRecorder()
            nvtx_module = types.ModuleType("nvtx")
            nvtx_module.__path__ = []
            nvtx_module.enabled = lambda: True
            nvtx_module.get_domain = lambda name: _IncompleteDomain()
            nvtx_module.push_range = recorder.push_range
            nvtx_module.pop_range = recorder.pop_range
            monkeypatch.setitem(sys.modules, "nvtx", nvtx_module)
        else:
            raise ValueError(f"unknown mode: {mode!r}")

        # Provide a lightweight package shell so importing `gsplat.trace` does
        # not execute the real `gsplat/__init__.py`, which pulls in CUDA-heavy
        # imports unrelated to these unit tests.
        gsplat_pkg = types.ModuleType("gsplat")
        gsplat_pkg.__path__ = [str(TRACE_PATH.parent)]
        monkeypatch.setitem(sys.modules, "gsplat", gsplat_pkg)

        sys.modules.pop("gsplat.trace", None)
        module = importlib.import_module("gsplat.trace")
        module = importlib.reload(module)
        return module, recorder

    return loader


@pytest.fixture
def enabled_trace(load_trace):
    return load_trace("domain")


@pytest.fixture
def fallback_trace(load_trace):
    return load_trace("fallback")


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
    trace, _ = load_trace("absent")

    def fn():
        return "ok"

    assert trace.trace_function("leaf")(fn)() == "ok"
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


# --- Fallback (module-level nvtx) branch --------------------------------------
#
# Exercised when nvtx is loaded but the per-domain API is incomplete.
# `_DOMAIN` resolves to None in trace.py and the helpers route through
# module-level nvtx.push_range / pop_range.


@pytest.mark.parametrize(
    "mechanism",
    [
        lambda trace: (trace.trace_push("outer"), trace.trace_pop()),
        lambda trace: _run_range(trace, "outer"),
        lambda trace: _run_decorated(trace, "outer"),
    ],
    ids=["push-pop", "range", "decorator"],
)
def test_fallback_records_single_level_name(fallback_trace, mechanism):
    trace, recorder = fallback_trace
    mechanism(trace)
    assert recorder.calls == [("push", "outer", {}), ("pop", None, {})]


def test_fallback_forwards_supported_kwargs(fallback_trace):
    # The shim's `nvtx.push_range` accepts color / domain / category /
    # payload; all should pass through and land in the recorded calls.
    trace, recorder = fallback_trace
    trace.trace_push("outer", color="blue", category="unit", payload=7)
    with trace.trace_range("inner", color="green", category="test"):
        pass

    @trace.trace_function("leaf", payload=11)
    def fn():
        return 7

    assert fn() == 7
    assert recorder.calls == [
        ("push", "outer", {"color": "blue", "category": "unit", "payload": 7}),
        ("push", "inner", {"color": "green", "category": "test"}),
        ("pop", None, {}),
        ("push", "leaf", {"payload": 11}),
        ("pop", None, {}),
    ]


def test_fallback_drops_unsupported_kwargs(fallback_trace):
    # An unknown kwarg (not in the shim's `push_range` signature) is silently
    # filtered out instead of raising TypeError.
    trace, recorder = fallback_trace
    trace.trace_push("outer", unsupported_kwarg=42)
    trace.trace_pop()
    assert recorder.calls == [("push", "outer", {}), ("pop", None, {})]
