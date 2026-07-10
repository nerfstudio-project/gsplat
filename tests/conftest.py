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

import functools
import gc
import os
import socket
import threading
import time
from pathlib import Path
from typing import List, Optional

import gsplat as _gsplat
import pytest
import torch
import torch.distributed

# The generated pytest configuration puts either the build-tree package or the
# installed wheel ahead of the test sources. Import it before collection so
# example modules that later prepend their own directory cannot make Python
# switch to the unbuilt source package.

# pytest's `pythonpath` setting changes only this process's sys.path. Preserve
# the selected package for tests that launch a fresh Python interpreter by
# forwarding its parent directory, without exposing the source or test root.
_python_path_entries = [str(Path(_gsplat.__file__).resolve().parent.parent)]
_inherited_python_path = os.environ.get("PYTHONPATH")
if _inherited_python_path:
    _python_path_entries.append(_inherited_python_path)
os.environ["PYTHONPATH"] = os.pathsep.join(_python_path_entries)

# Default fraction of *post-CUDA-context* free VRAM the test process is
# allowed to use. Caps the PyTorch caching allocator so an over-allocation
# OOMs cleanly instead of spilling into the OS-managed shared/system memory
# pool — that pool is ~30+ GB on Windows/WSL2 hosts but orders of magnitude
# slower than dedicated VRAM, and the spill is what makes the machine
# appear to hang.
#
# Sized off `mem_get_info` (free bytes after torch's CUDA context init)
# rather than `total_memory`, so the budget auto-adapts to driver / cuBLAS
# / cuDNN context overhead (~1 GB on modern stacks) and to any other
# process already using the GPU at session start.
#
# Default 1.0 = use everything the driver reports as free, but no more.
# Lower it via --cuda-mem-fraction on shared hosts that need headroom.
_DEFAULT_CUDA_FREE_FRACTION = 1.0
_CAMERA_WRAPPERS_SKIP_REASON = "rebuild with -DGSPLAT_BUILD_CAMERA_WRAPPERS=ON"


@functools.lru_cache(maxsize=1)
def _camera_wrappers_enabled():
    """Return whether the current extension build includes camera wrappers."""
    try:
        import gsplat.csrc as _C
    except ImportError:
        return False

    return bool(_C.build_config().get("camera_wrappers", False))


def pytest_addoption(parser):
    group = parser.getgroup(
        "gsplat-mem", "gsplat: GPU memory cap and per-test tracking"
    )
    group.addoption(
        "--cuda-mem-fraction",
        type=float,
        default=_DEFAULT_CUDA_FREE_FRACTION,
        metavar="FRACTION",
        help=(
            "Cap torch's CUDA caching allocator at this fraction of the free "
            "VRAM observed after CUDA context init. Default 1.0 (use all "
            "free memory; over-allocations OOM cleanly instead of spilling)."
        ),
    )
    group.addoption(
        "--mem-track",
        action="store_true",
        default=False,
        help=(
            "Track per-test CUDA memory peaks (torch caching allocator + "
            "driver-visible) and print a top-N summary at session end. Also "
            "logs the chosen cap at session start. Off by default; default "
            "pytest output is unchanged when omitted."
        ),
    )
    group.addoption(
        "--mem-track-interval",
        type=float,
        default=0.05,
        metavar="SECONDS",
        help=(
            "Background sampler poll interval (default 0.05 s). Lower = "
            "catches shorter spikes at higher overhead. Only meaningful "
            "with --mem-track."
        ),
    )
    group.addoption(
        "--mem-track-top",
        type=int,
        default=25,
        metavar="N",
        help=(
            "How many tests to print in the session-end summary (default 25). "
            "Only meaningful with --mem-track."
        ),
    )
    group.addoption(
        "--mem-track-csv",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Write full per-test peaks as CSV to this path for offline "
            "analysis. Only meaningful with --mem-track."
        ),
    )
    group.addoption(
        "--xfail-oom",
        action="store_true",
        default=False,
        help=(
            "Convert tests that fail with torch.cuda.OutOfMemoryError into "
            "XFAIL instead of FAIL. Useful on memory-constrained hosts where "
            "some test parametrizations exceed the allocator cap. Detection "
            "is strict: only torch.cuda.OutOfMemoryError counts; allocation "
            "failures from sub-libraries that surface as plain RuntimeError "
            "still fail."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that need optional camera wrappers when the build omits them."""
    del config
    camera_wrapper_items = [
        item for item in items if item.get_closest_marker("camera_wrappers") is not None
    ]
    if camera_wrapper_items and not _camera_wrappers_enabled():
        skip = pytest.mark.skip(reason=_CAMERA_WRAPPERS_SKIP_REASON)
        for item in camera_wrapper_items:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _cap_cuda_memory_fraction(request):
    """Cap torch CUDA allocations at a fraction of post-context free VRAM.

    Always installed (the cap is the safety net that turns a would-be
    spill into shared/system memory into a clean OOM). The informational
    announcement is only printed under --mem-track so default pytest
    output is unchanged.
    """
    if not torch.cuda.is_available():
        yield
        return

    free_fraction = max(
        0.05, min(1.0, float(request.config.getoption("--cuda-mem-fraction")))
    )

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Force CUDA context init so `mem_get_info` accounts for driver + cuBLAS
    # + cuDNN overhead before we measure free memory.
    torch.cuda.synchronize(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    cap_bytes = int(free_bytes * free_fraction)
    fraction_of_total = cap_bytes / total_bytes

    torch.cuda.set_per_process_memory_fraction(fraction_of_total, device)

    if request.config.getoption("--mem-track"):
        print(
            f"\n[gsplat tests] {props.name} (cuda:{device}): "
            f"total {total_bytes / 1024**3:.2f} GiB, "
            f"free after context init {free_bytes / 1024**3:.2f} GiB. "
            f"Capping torch allocator at {free_fraction:.0%} of free "
            f"= {cap_bytes / 1024**3:.2f} GiB "
            f"({fraction_of_total:.1%} of total). "
            f"Override with --cuda-mem-fraction=<0.05..1.0>."
        )

    yield


# --------------------------------------------------------------------------
# Per-test GPU memory tracking — opt-in via --mem-track.
#
# When enabled, records two metrics per test:
#   * `torch_peak`  — peak bytes managed by torch's caching allocator
#                     (`torch.cuda.max_memory_allocated`). Tightly attributed
#                     to the test's torch operations.
#   * `device_peak` — peak total VRAM in use as seen by the CUDA driver,
#                     sampled from a background thread at ~20 Hz via
#                     `cudaMemGetInfo`. Includes context overhead and any
#                     other CUDA allocator on the same device.
#
# At session end `pytest_terminal_summary` prints the top-N heaviest tests.
# When --mem-track is omitted the fixture is a no-op (no sampler thread, no
# overhead, no extra console output). Sampler poll interval, summary length,
# and CSV path are tuned via the --mem-track-{interval,top,csv} CLI flags.
# --------------------------------------------------------------------------

_TEST_MEM_PEAKS: List[dict] = []


class _CudaMemSampler:
    """Background thread that polls cudaMemGetInfo and records peak usage."""

    def __init__(self, device: int, interval: float = 0.05):
        self._device = device
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_used_bytes = 0

    def start(self):
        free, total = torch.cuda.mem_get_info(self._device)
        self.peak_used_bytes = total - free
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        # One final read in case the test allocated and freed between samples.
        free, total = torch.cuda.mem_get_info(self._device)
        used = total - free
        if used > self.peak_used_bytes:
            self.peak_used_bytes = used

    def _run(self):
        while not self._stop_event.is_set():
            try:
                free, total = torch.cuda.mem_get_info(self._device)
                used = total - free
                if used > self.peak_used_bytes:
                    self.peak_used_bytes = used
            except Exception:
                pass  # device may transiently be in an unusable state
            time.sleep(self._interval)


@pytest.fixture(autouse=True)
def _track_cuda_memory_peak(request):
    if not torch.cuda.is_available() or not request.config.getoption("--mem-track"):
        yield
        return

    interval = float(request.config.getoption("--mem-track-interval"))
    device = torch.cuda.current_device()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    sampler = _CudaMemSampler(device, interval=interval)
    sampler.start()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        sampler.stop()
        _TEST_MEM_PEAKS.append(
            {
                "nodeid": request.node.nodeid,
                "torch_peak": torch.cuda.max_memory_allocated(device),
                "device_peak": sampler.peak_used_bytes,
            }
        )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Reroute torch CUDA OOM failures to XFAIL when --xfail-oom is set."""
    outcome = yield
    report = outcome.get_result()
    if (
        report.when == "call"
        and report.failed
        and call.excinfo is not None
        and item.config.getoption("--xfail-oom")
        and isinstance(call.excinfo.value, torch.cuda.OutOfMemoryError)
    ):
        report.outcome = "skipped"
        report.wasxfail = "torch.cuda.OutOfMemoryError (--xfail-oom)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--mem-track") or not _TEST_MEM_PEAKS:
        return
    tr = terminalreporter
    sorted_peaks = sorted(_TEST_MEM_PEAKS, key=lambda r: -r["device_peak"])
    n_total = len(sorted_peaks)
    n_show = min(int(config.getoption("--mem-track-top")), n_total)

    overall_torch = max(r["torch_peak"] for r in sorted_peaks)
    overall_device = max(r["device_peak"] for r in sorted_peaks)

    tr.write_sep("=", f"CUDA memory peaks (top {n_show} of {n_total} tests)")
    tr.write_line(
        f"session max:  device={overall_device / 1024**2:>9.1f} MiB   "
        f"torch={overall_torch / 1024**2:>9.1f} MiB"
    )
    tr.write_line(f"  {'device_peak':>12s}  {'torch_peak':>12s}   test")
    for r in sorted_peaks[:n_show]:
        tr.write_line(
            f"  {r['device_peak'] / 1024**2:>9.1f} MiB  "
            f"{r['torch_peak'] / 1024**2:>9.1f} MiB   {r['nodeid']}"
        )

    out_path = config.getoption("--mem-track-csv")
    if out_path:
        import csv

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["nodeid", "device_peak_bytes", "torch_peak_bytes"])
            for r in sorted_peaks:
                writer.writerow([r["nodeid"], r["device_peak"], r["torch_peak"]])
        tr.write_line(f"  (full per-test CSV written to {out_path})")


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
        # Sync first so any pending kernels from the previous test release
        # their tensors before we ask the allocator to free unreferenced blocks.
        torch.cuda.synchronize()

    # Run garbage collection (drops Python-side refs before empty_cache).
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Yield to run the test
    yield

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free_port() -> int:
    """Return an OS-assigned free TCP port.

    Used for the distributed rendezvous so concurrent test runs sharing the
    GPU each get their own port instead of colliding on a fixed one.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def dist_init():
    """Initialize a single-process distributed group for testing distributed code paths.

    With world_size=1 the all-gather / all-to-all ops become identity operations,
    but the code path inside ``rasterization(distributed=True)`` is still exercised.
    """
    if not torch.cuda.is_available():
        yield
        return

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
        # Warm up the communicator required by batch_isend_irecv.
        _ = [None]
        torch.distributed.all_gather_object(_, 0)

    yield

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
