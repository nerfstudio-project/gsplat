# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Profiling helpers and a replay harness for gsplat rasterization.

Tutorial
========

This module supports a two-step profiling workflow:

1. Capture a real rasterization call from an existing application.
2. Replay that exact input under Nsight Systems or Nsight Compute.

Capture inputs
--------------

`gsplat.rendering.rasterization()` is decorated with
`@capture_inputs(envvar="GSPLAT_INPUT_CAPTURE_RASTERIZATION")`.
Set that environment variable before running your application:

```bash
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs:1
python your_app.py
```

The value is a comma-delimited list of specs, each of the form
`<output_path>:<range_spec>`:

- `<path>:<stop>` captures calls `0..stop-1`
- `<path>:<start>:<stop>` captures calls `start..stop-1`
- `<path>:<start>:<stop>:<step>` captures strided calls

Multiple specs let you capture different call indices to different output
paths in a single run — useful when one process makes several kinds of
rasterization calls (e.g. a camera pass followed by a lidar pass) and you
want each written to its own file. The call-index ranges across specs must
not overlap.

Examples:

```bash
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs.pt:1
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs:5:8
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=raster:10:20:2
export GSPLAT_INPUT_CAPTURE_DIR=/tmp/gsplat-captures
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=raster:3
# Capture call 0 as camera and call 1 as lidar:
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/camera:1,/tmp/lidar:1:2
```

Captured files are written with a numeric suffix such as
`/tmp/gsplat_inputs_0.pt`. Under DDP (when the `RANK` environment variable is
set by a launcher), an `_r<rank>` infix is added so ranks do not race on the
same path: e.g. `/tmp/gsplat_inputs_r0_0.pt`. Outside DDP, a `_p<pid>` infix
is used to keep concurrent processes from colliding. Relative paths are
resolved under `GSPLAT_INPUT_CAPTURE_DIR` when that variable is set. Once
every requested capture has been written, the process exits intentionally
with `SystemExit`.

You can use the same decorator on other functions to build a more focused
benchmarking path, for example by capturing `isect_tiles()` inputs separately
instead of replaying the full `rasterization()` pipeline. If you do that, you
also need to adjust the replay loop in `main()` so it calls the captured
function and applies the right forward/backward logic for that workload.

Replay without a profiler
-------------------------

You can validate the captured input directly:

```bash
python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt
```

The harness loads the saved arguments, moves tensors to CUDA, restores
`requires_grad=True` on differentiable Gaussian inputs, then runs warmup and
profiled forward/backward iterations of `rasterization()`.

Replay under Nsight Systems
---------------------------

To collect rich GPU-side trace data without spending effort on secondary CPU
metrics, trace CUDA/NVTX plus the CUDA math libraries you care about, enable
CUDA memory tracking, and export a report. Use `cudaProfilerStart/Stop` to keep
the capture focused on profiled iterations:

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx,osrt,cublas,cudnn,cusolver,cusparse \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --stats=true \
  --export=sqlite \
  --output /tmp/gsplat_nsys \
  python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt --warmup 5 --iterations 20
```

The replay harness emits NVTX ranges for `iteration`, `forward`, and
`backward`, which makes it easier to inspect the trace in Nsight Systems.

Replay under Nsight Compute
---------------------------

To collect as many kernel metrics as practical while staying focused on the GPU
work, use the `full` set, keep NVTX filtering so the replay stays focused, and
profile a single iteration without warmup:

```bash
ncu \
  --set full \
  --nvtx \
  --nvtx-include forward \
  --target-processes all \
  --replay-mode kernel \
  --force-overwrite \
  --export /tmp/gsplat_ncu \
  python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt --warmup 0 --iterations 1
```

This is still heavy-weight because `--set full` can require many replay passes.
For faster iteration, replace `--set full` with a narrower metric set and add
kernel filters as needed.

"""

import inspect
import os
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

import torch
from torch.utils._pytree import tree_map

from gsplat.trace import trace_range


def _detach_for_capture(x: Any) -> Any:
    """Break autograd-graph connectivity before ``torch.save``.

    Non-leaf tensors would otherwise drag their autograd graph into the
    pickle, bloating the file and coupling it to live model code.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


profiler = {}

# Parameter names that should have requires_grad=True for backward pass
_GRAD_PARAMS = {"means", "quats", "scales", "opacities", "colors", "extra_signals"}

# --- Input capture registry ---
# Tracks all active capture_inputs decorators so the program only exits when
# every one of them has finished its range.
_pending_captures: set[int] = set()
_next_capture_id: int = 0


class timeit(object):
    """Profiler that is controled by the TIMEIT environment variable.

    If TIMEIT is set to 1, the profiler will measure the time taken by the decorated function.

    Usage:

    ```python
    @timeit()
    def my_function():
        pass

    # Or

    with timeit(name="stage1"):
        my_function()

    print(profiler)
    ```
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: Optional[float] = None
        self.enabled = os.environ.get("TIMEIT", "0") == "1"

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - self.start_time
            if self.name not in profiler:
                profiler[self.name] = total_time
            else:
                profiler[self.name] += total_time

    def __call__(self, f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            with self:
                self.name = f.__name__
                return f(*args, **kwargs)

        return decorated


def capture_inputs(*, envvar: str) -> Callable[[_F], _F]:
    """Decorator that captures a function's inputs to disk when envvar is set.

    When the environment variable is not set the original function is returned
    unchanged with zero overhead.

    Format: ``<envvar>=<spec>[,<spec>...]`` where each ``<spec>`` is
    ``<output_path>:<range_spec>``.

    <range_spec> follows Python range() / tensor-slice conventions:
      - ``stop``             — capture calls 0, 1, ..., stop-1
      - ``start:stop``       — capture calls start, start+1, ..., stop-1
      - ``start:stop:step``  — capture calls start, start+step, ..., < stop

    Within each spec, integers are scanned from the right; everything to the
    left of them is the output path.  If the path is relative and
    ``GSPLAT_INPUT_CAPTURE_DIR`` is set, that directory is prepended.  If no
    file extension is given, ``.pt`` is used.  Each captured call is saved as
    ``<stem>_<zero-padded index><ext>``, with enough zero-padding so that all
    indices across all specs share the same number of digits.

    Passing multiple comma-separated specs lets a single run route different
    call indices to different output paths (e.g. camera vs lidar calls).
    Ranges across specs must be disjoint.

    All gsplat C++ custom classes support pickle via ``def_pickle`` in
    ``ext.cpp``, so ``torch.save`` handles them natively.

    .. note::

       On PyTorch 2.6+, ``torch.load`` defaults to ``weights_only=True`` and
       refuses to instantiate custom ``torch::class_`` instances. Reload
       captured files with ``torch.load(path, weights_only=False)`` (the
       bundled replay harness in ``gsplat.profile.main`` already does this).

    Example::

        @capture_inputs(envvar="MY_APP_CAPTURE_FOO")
        def foo(...):
            ...
    """

    def decorator(fn: _F) -> _F:
        global _next_capture_id

        env = os.environ.get(envvar)
        if not env:
            return fn

        capture_dir = os.environ.get("GSPLAT_INPUT_CAPTURE_DIR")

        # Parse the comma-delimited list of specs. Each spec is parsed
        # independently into its own (stem, ext, capture_range).
        specs: list[tuple[str, str, range]] = []
        for spec_str in env.split(","):
            spec_str = spec_str.strip()
            if not spec_str:
                continue
            parts = spec_str.split(":")
            range_ints: list[int] = []
            path_end = len(parts)
            for i in range(len(parts) - 1, -1, -1):
                try:
                    range_ints.insert(0, int(parts[i]))
                    path_end = i
                except ValueError:
                    break
            if not range_ints:
                raise ValueError(
                    f"{envvar}: expected <path>:<stop>, <path>:<start>:<stop>, or "
                    f"<path>:<start>:<stop>:<step> in each spec, got {spec_str!r}"
                )
            if any(v < 0 for v in range_ints):
                raise ValueError(
                    f"{envvar}: negative values are not supported, got {spec_str!r}"
                )
            if len(range_ints) >= 2 and range_ints[1] < range_ints[0]:
                raise ValueError(
                    f"{envvar}: stop ({range_ints[1]}) must be >= start ({range_ints[0]}), got {spec_str!r}"
                )
            output_path = ":".join(parts[:path_end])
            capture_range = range(*range_ints)
            if not capture_range:
                raise ValueError(
                    f"{envvar}: empty range (nothing to capture), got {spec_str!r}"
                )

            # Resolve output path: apply capture dir if path is relative, add
            # default extension if none given.
            if capture_dir and not os.path.isabs(output_path):
                output_path = os.path.join(capture_dir, output_path)
            stem, ext = os.path.splitext(output_path)
            if not ext:
                ext = ".pt"

            specs.append((stem, ext, capture_range))

        if not specs:
            raise ValueError(f"{envvar}: no specs provided, got {env!r}")

        # Ensure call-index ranges don't overlap across specs — otherwise we'd
        # need an arbitrary tie-break when a call matches several specs.
        seen_calls: dict[int, int] = {}
        for spec_idx, (_stem, _ext, capture_range) in enumerate(specs):
            for c in capture_range:
                if c in seen_calls:
                    raise ValueError(
                        f"{envvar}: call index {c} is claimed by multiple specs "
                        f"(specs {seen_calls[c]} and {spec_idx}), got {env!r}"
                    )
                seen_calls[c] = spec_idx

        # Zero-pad index so all filenames sort lexicographically across specs.
        n_digits = len(str(max(seen_calls)))
        total_captures = sum(len(r) for _s, _e, r in specs)

        # Track all active capture decorators so multiple instrumented functions
        # can coexist in one process without exiting early.
        capture_id = _next_capture_id
        _next_capture_id += 1
        _pending_captures.add(capture_id)

        call_count = 0
        captures_done = 0
        sig = inspect.signature(fn)

        @wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, captures_done

            matching_spec: Optional[tuple[str, str, range]] = None
            for spec in specs:
                if call_count in spec[2]:
                    matching_spec = spec
                    break

            if matching_spec is not None:
                stem, ext, _capture_range = matching_spec
                # Bind by signature so saved inputs are stable even when callers
                # mix positional and keyword arguments.
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                # Qualify by RANK (DDP) or PID (single-process concurrency) so
                # concurrent writers never race on the same output file.
                rank = os.environ.get("RANK")
                worker_tag = f"r{rank}" if rank is not None else f"p{os.getpid()}"
                save_path = f"{stem}_{worker_tag}_{call_count:0{n_digits}d}{ext}"
                # Detach tensors so non-leaf inputs don't drag their autograd
                # graph into the capture. main() re-applies requires_grad on
                # load for the parameters that need it.
                captured = tree_map(_detach_for_capture, bound.arguments)
                torch.save(captured, save_path)
                captures_done += 1
                print(
                    f"[gsplat.profile] Captured {fn.__name__} inputs "
                    f"({captures_done}/{total_captures}, call {call_count}) "
                    f"to {save_path}"
                )
                print(
                    "[gsplat.profile] Reload with "
                    "torch.load(path, weights_only=False) on PyTorch 2.6+."
                )
                for k, v in bound.arguments.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                    elif v is not None:
                        print(f"  {k}: {type(v).__name__} = {v}")
                if captures_done >= total_captures:
                    _pending_captures.discard(capture_id)
                    if not _pending_captures:
                        raise SystemExit("[gsplat.profile] All captures done, exiting.")

            call_count += 1
            return fn(*args, **kwargs)

        return _wrapper  # type: ignore[return-value]

    return decorator


def main() -> None:
    """Standalone profiling harness entry point.

    Usage:
        nsys profile --capture-range=cudaProfilerApi \\
            python -m gsplat.profile --input gsplat_inputs.pt --warmup 5 --iterations 20
    """
    import argparse

    from gsplat.rendering import rasterization

    parser = argparse.ArgumentParser(description="GSplat profiling harness")
    parser.add_argument(
        "--input", required=True, help="Path to captured inputs (.pt file)"
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of profiled iterations"
    )
    args = parser.parse_args()

    print(f"[gsplat.profile] Loading inputs from {args.input}")
    inputs: dict[str, Any] = torch.load(
        args.input, map_location="cuda", weights_only=False
    )

    # Recreate the training-style inputs expected by rasterization replay.
    for k in _GRAD_PARAMS:
        if (
            k in inputs
            and isinstance(inputs[k], torch.Tensor)
            and inputs[k].is_floating_point()
        ):
            inputs[k] = inputs[k].requires_grad_(True)

    n_gaussians = inputs["means"].shape[-2] if "means" in inputs else "?"
    width = inputs.get("width", "?")
    height = inputs.get("height", "?")
    print(f"[gsplat.profile] {n_gaussians} gaussians, {width}x{height} image")
    print(
        f"[gsplat.profile] Warmup: {args.warmup}, Profiled iterations: {args.iterations}"
    )

    def run_iteration(iteration: int) -> None:
        # Keep the replay structure explicit so NVTX ranges map cleanly to the
        # forward and backward phases in Nsight tools.
        with trace_range("iteration", payload=iteration):
            with trace_range("forward"):
                render_colors, _render_alphas, _meta = rasterization(**inputs)
                loss = render_colors.sum()

            with trace_range("backward"):
                loss.backward()
            # Clear grads in place so every iteration starts from the same state.
            for k in _GRAD_PARAMS:
                if (
                    k in inputs
                    and isinstance(inputs[k], torch.Tensor)
                    and inputs[k].grad is not None
                ):
                    inputs[k].grad = None

    print(f"[gsplat.profile] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        run_iteration(i)
    print("[gsplat.profile] Warmup done.")

    torch.autograd.set_multithreading_enabled(False)
    # Gate profiler collection to the measured iterations only.
    print(
        f"[gsplat.profile] Starting profiler, running {args.iterations} iterations..."
    )
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iterations):
        run_iteration(i)
    torch.cuda.cudart().cudaProfilerStop()
    print("[gsplat.profile] Profiling done.")


if __name__ == "__main__":
    main()
