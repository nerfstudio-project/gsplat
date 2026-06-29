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

"""Benchmark fused batched SE(3) pose-track interpolation.

This is a standalone CUDA benchmark, not a pytest test. It measures the public
``gsplat.geometry.functional.se3_interpolate_tracks`` helper against a
benchmark-only naive Python loop.

Run directly::

    python examples/benchmarks/geometry/se3_interpolate_tracks_bench.py
    python examples/benchmarks/geometry/se3_interpolate_tracks_bench.py --iters 500
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

import gsplat.geometry.functional as geom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SE(3) packed track interpolation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", default="cuda:0", help="CUDA device to use.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    return parser.parse_args()


def _make_counts(
    n_tracks: int, keyframes: int, *, skewed: bool, device: torch.device
) -> torch.Tensor:
    counts = torch.full((n_tracks,), keyframes, device=device, dtype=torch.int32)
    if skewed and n_tracks > 1:
        counts.fill_(2)
        counts[0] = keyframes
    return counts


def make_case(
    n_tracks: int,
    keyframes: int,
    *,
    skewed: bool,
    per_track_query: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    counts = _make_counts(n_tracks, keyframes, skewed=skewed, device=device)
    offsets = torch.cumsum(counts, dim=0) - counts
    total_keyframes = int(counts.sum().item())

    torch.manual_seed(1234 + n_tracks * 17 + keyframes)
    translations = torch.randn(total_keyframes, 3, device=device, dtype=dtype)
    rotations = torch.nn.functional.normalize(
        torch.randn(total_keyframes, 4, device=device, dtype=dtype), dim=-1
    )

    pose_times = torch.empty(total_keyframes, device=device, dtype=dtype)
    counts_list = [int(x) for x in counts.cpu().tolist()]
    offsets_list = [int(x) for x in offsets.cpu().tolist()]
    for start, count in zip(offsets_list, counts_list):
        pose_times[start : start + count] = torch.linspace(
            0.0, 1.0, count, device=device, dtype=dtype
        )

    if per_track_query:
        query_time = torch.linspace(0.1, 0.9, n_tracks, device=device, dtype=dtype)
    else:
        query_time = torch.tensor(0.37, device=device, dtype=dtype)

    return translations, rotations, pose_times, offsets, counts, query_time


def naive_se3_interpolate_tracks(
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    offsets_list: list[int],
    counts_list: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Benchmark-only Python loop reference for fused interpolation speedups."""

    (
        pose_translations,
        pose_rotations,
        pose_times,
        _pose_offsets,
        _pose_counts,
        query_time,
    ) = inputs
    if not offsets_list:
        return pose_translations.new_empty((0, 3)), pose_rotations.new_empty((0, 4))

    times = pose_times.reshape(-1)
    query_times = (
        query_time.expand(len(offsets_list))
        if query_time.ndim == 0
        else query_time.reshape(-1)
    )

    out_translations: list[torch.Tensor] = []
    out_rotations: list[torch.Tensor] = []
    for track_idx, (start, count) in enumerate(zip(offsets_list, counts_list)):
        track_times = times[start : start + count]
        track_translations = pose_translations[start : start + count]
        track_rotations = pose_rotations[start : start + count]
        if count == 1:
            out_translations.append(track_translations[0])
            out_rotations.append(track_rotations[0])
            continue

        clamped_query = torch.clamp(
            query_times[track_idx], track_times[0], track_times[-1]
        )
        right = int(torch.searchsorted(track_times, clamped_query, right=False).item())
        if right <= 0:
            left = right = 0
        elif right >= count:
            left = right = count - 1
        elif bool((track_times[right] == clamped_query).item()):
            left = right
        else:
            left = right - 1

        if left == right:
            out_translations.append(track_translations[left])
            out_rotations.append(track_rotations[left])
            continue

        left_time = track_times[left]
        right_time = track_times[right]
        alpha = (clamped_query - left_time) / (right_time - left_time)
        out_translations.append(
            track_translations[left]
            + alpha * (track_translations[right] - track_translations[left])
        )
        out_rotations.append(
            geom.quat_slerp(
                track_rotations[left : left + 1],
                track_rotations[right : right + 1],
                alpha.reshape(1, 1),
            )[0]
        )

    return torch.stack(out_translations, dim=0), torch.stack(out_rotations, dim=0)


def latency_benchmark(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples_ms)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    cases = [
        ("tiny uniform", 16, 2, False, False),
        ("small uniform", 128, 8, False, True),
        ("medium uniform", 512, 8, False, True),
        ("skewed", 128, 512, True, True),
    ]

    header = [
        "case",
        "n_tracks",
        "keyframes",
        "skewed",
        "total_keyframes",
        "query",
        "fused_latency_ms",
        "naive_latency_ms",
        "speedup",
    ]
    writer = csv.writer(sys.stdout)
    writer.writerow(header)

    for name, n_tracks, keyframes, skewed, per_track_query in cases:
        inputs = make_case(
            n_tracks,
            keyframes,
            skewed=skewed,
            per_track_query=per_track_query,
            device=device,
            dtype=dtype,
        )
        total_keyframes = int(inputs[4].sum().item())
        fused_fn = lambda: geom.se3_interpolate_tracks(*inputs)
        fused_fn()

        fused_latency_ms = latency_benchmark(
            fused_fn, warmup=args.warmup, iters=args.iters
        )
        offsets_list = [int(x) for x in inputs[3].cpu().tolist()]
        counts_list = [int(x) for x in inputs[4].cpu().tolist()]
        naive_fn = lambda: naive_se3_interpolate_tracks(
            inputs, offsets_list, counts_list
        )
        fused_out = fused_fn()
        naive_out = naive_fn()
        torch.cuda.synchronize()
        if not torch.allclose(
            fused_out[0], naive_out[0], atol=1e-5
        ) or not torch.allclose(fused_out[1], naive_out[1], atol=1e-5):
            raise RuntimeError(f"naive reference mismatch for benchmark case {name!r}")
        naive_latency_ms = latency_benchmark(
            naive_fn, warmup=args.warmup, iters=args.iters
        )
        speedup = naive_latency_ms / fused_latency_ms

        query_kind = "per-track" if per_track_query else "scalar"
        row = [
            name,
            n_tracks,
            keyframes,
            skewed,
            total_keyframes,
            query_kind,
            fmt(fused_latency_ms),
            fmt(naive_latency_ms),
            fmt(speedup),
        ]
        writer.writerow(row)


if __name__ == "__main__":
    main()
