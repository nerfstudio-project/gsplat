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

"""Benchmark script for the Inference Gaussian rasterization path.

Measures GPU latency (via CUDA events) for the stateful Inference renderer:

1. **Stateful renderer** -- ``GaussianInferenceRenderer``
   Scene is packed once, a persistent renderer is created once, and each
   frame is rendered via ``renderer.render()``.  The renderer caches GPU
   buffers across frames.

Optionally, pass ``--reference-tile-size`` to benchmark the reference
``gsplat.rasterization()`` path and print speedup/image-quality comparisons.

This is a standalone script (NOT a pytest test). Run it directly::

    python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py
    python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py --num-gaussians 1000000 --num-frames 200
    python examples/benchmarks/gaussian_render_inference_scene/gaussian_render_inference_scene_bench.py --inference-tile-size 8 --reference-tile-size 16
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the Inference rasterization path.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-gaussians",
        type=int,
        default=500_000,
        help="Number of Gaussians in the synthetic scene.",
    )
    p.add_argument("--width", type=int, default=1920, help="Image width in pixels.")
    p.add_argument("--height", type=int, default=1080, help="Image height in pixels.")
    p.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of timed frames per benchmark.",
    )
    p.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Number of warmup frames (not timed) before each benchmark.",
    )
    p.add_argument(
        "--sh-degree",
        type=int,
        default=3,
        help="SH degree (0-3). Use -1 for pre-activated RGB.",
    )
    p.add_argument(
        "--inference-tile-size",
        type=int,
        default=8,
        choices=[8, 16],
        help="Tile size for Inference rasterization kernels.",
    )
    p.add_argument(
        "--reference-tile-size",
        type=int,
        default=None,
        choices=[4, 16],
        help=(
            "Tile size for the gsplat reference rasterization path. When omitted, "
            "the reference benchmark, speedup, and image-quality comparisons are skipped."
        ),
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility alias for --inference-tile-size. "
            "Use --reference-tile-size separately to enable the reference path."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device to use.",
    )
    p.add_argument(
        "--ply-path",
        type=str,
        default=None,
        help=(
            "Path to a trained 3DGS PLY file. When set, the bench loads the "
            "scene via gsplat.exporter.load_ply_to_splats instead of "
            "generating synthetic Gaussians. --num-gaussians is ignored and "
            "--sh-degree is inferred from the PLY contents."
        ),
    )
    p.add_argument(
        "--save-image",
        type=str,
        default=None,
        metavar="PATH",
        help="Render one frame with the inference path and save it as a PNG.",
    )
    p.add_argument(
        "--quality-frames",
        type=int,
        default=1,
        help=(
            "Number of frames used for Inference-vs-reference PSNR/LPIPS comparison. "
            "Only used when --reference-tile-size is provided. Set to 0 to skip "
            "image-quality metrics."
        ),
    )
    p.add_argument(
        "--lpips-net",
        type=str,
        default="alex",
        choices=["alex", "vgg"],
        help="LPIPS backbone used for Inference-vs-reference image-quality comparison.",
    )
    args = p.parse_args()
    if args.tile_size is not None:
        if args.tile_size not in (8, 16):
            p.error(
                "--tile-size can only be 8 or 16. Use --reference-tile-size "
                "separately to enable the reference path."
            )
        args.inference_tile_size = args.tile_size
    if args.quality_frames < 0:
        p.error("--quality-frames must be non-negative.")
    return args


# ---------------------------------------------------------------------------
# Synthetic scene and camera helpers
# ---------------------------------------------------------------------------


def make_gaussians(
    N: int,
    device: torch.device,
    sh_degree: Optional[int],
) -> tuple:
    """Create random Gaussians placed in front of the camera."""
    torch.manual_seed(42)
    means = torch.randn(N, 3, device=device)
    # Spread Gaussians in a 10-unit cube in front of the camera (z in [2, 12])
    means[:, 0] *= 5.0
    means[:, 1] *= 5.0
    means[:, 2] = means[:, 2].abs() * 5.0 + 2.0

    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    scales = torch.exp(torch.rand(N, 3, device=device) * 0.1 - 2.0)

    opacities = torch.sigmoid(torch.rand(N, device=device) * 2.0 - 1.0)

    if sh_degree is not None and sh_degree >= 0:
        K_sh = (sh_degree + 1) ** 2
        colors = torch.randn(N, K_sh, 3, device=device) * 0.1
    else:
        colors = torch.sigmoid(torch.randn(N, 3, device=device))

    return means, quats, scales, opacities, colors


def make_gaussians_from_ply(
    path: str,
    device: torch.device,
) -> tuple:
    """Load Gaussians from a trained 3DGS PLY file.

    Returns the same (means, quats, scales, opacities, colors) tuple as
    :func:`make_gaussians`. ``colors`` is built from the PLY's sh0 (DC) and
    shN (higher-order) coefficients with shape ``(N, K, 3)`` where
    ``K = (sh_degree+1)**2``. The inferred SH degree is returned alongside.
    """
    from gsplat.exporter import load_ply_to_splats

    splats = load_ply_to_splats(path)
    # Concatenate SH on CPU to avoid tripling GPU memory (sh0 + shN + colors).
    colors_cpu = torch.cat([splats["sh0"], splats["shN"]], dim=1)  # (N, K, 3)
    inferred_sh_degree = int(round(colors_cpu.shape[1] ** 0.5)) - 1
    means = splats["means"].to(device)
    quats = F.normalize(splats["quats"].to(device), dim=-1)
    # PLY stores log-scales and logit-opacities; activate them.
    scales = torch.exp(splats["scales"]).to(device)
    opacities = torch.sigmoid(splats["opacities"]).to(device)
    del splats
    colors = colors_cpu.to(device)
    del colors_cpu
    return means, quats, scales, opacities, colors, inferred_sh_degree


def make_intrinsics(width: int, height: int, device: torch.device) -> torch.Tensor:
    """Simple pinhole intrinsics with ~50-degree horizontal FOV."""
    focal = width / (2.0 * math.tan(math.radians(25)))
    K = torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    return K


def make_orbit_viewmats(
    num_frames: int,
    device: torch.device,
    radius: float = 7.0,
    height: float = 0.0,
    look_at: tuple = (0.0, 0.0, 5.0),
) -> List[torch.Tensor]:
    """Generate a circular orbit camera trajectory.

    Returns a list of [4, 4] world-to-camera matrices.
    """
    viewmats = []
    for i in range(num_frames):
        angle = 2.0 * math.pi * i / num_frames
        cx = radius * math.cos(angle) + look_at[0]
        cy = height + look_at[1]
        cz = radius * math.sin(angle) + look_at[2]
        cam_pos = torch.tensor([cx, cy, cz], dtype=torch.float32)

        # Look-at direction
        target = torch.tensor(look_at, dtype=torch.float32)
        forward = target - cam_pos
        forward = forward / forward.norm()

        # World up
        up = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)

        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        up = torch.linalg.cross(right, forward)

        # Build rotation (rows = right, -up, forward for OpenGL-style)
        R = torch.stack([right, -up, forward], dim=0)
        t = -R @ cam_pos

        viewmat = torch.eye(4, dtype=torch.float32, device=device)
        viewmat[:3, :3] = R.to(device)
        viewmat[:3, 3] = t.to(device)
        viewmats.append(viewmat)
    return viewmats


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


class CudaTimer:
    """GPU timer using CUDA events for accurate kernel timing."""

    def __init__(self, device: torch.device):
        self.device = device
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()

    def stop(self) -> float:
        """Stop timer and return elapsed time in milliseconds."""
        self._end.record()
        torch.cuda.synchronize(self.device)
        return self._start.elapsed_time(self._end)


def summarize_times(times_ms: List[float], label: str, num_frames: int) -> None:
    """Print a formatted summary of per-frame GPU times."""
    if not times_ms:
        print(f"  [{label}]  No timing data collected.\n")
        return
    t = torch.tensor(times_ms)
    mean = t.mean().item()
    median = t.median().item()
    mn = t.min().item()
    mx = t.max().item()
    std = t.std().item() if len(t) > 1 else 0.0
    p5 = t.quantile(0.05).item() if len(t) >= 20 else mn
    p95 = t.quantile(0.95).item() if len(t) >= 20 else mx
    fps = 1000.0 / median if median > 0 else 0.0

    print(f"  [{label}]")
    print(f"    Frames : {num_frames}")
    print(f"    Mean   : {mean:8.3f} ms")
    print(f"    Median : {median:8.3f} ms")
    print(f"    Min    : {mn:8.3f} ms")
    print(f"    Max    : {mx:8.3f} ms")
    print(f"    StdDev : {std:8.3f} ms")
    print(f"    P5     : {p5:8.3f} ms")
    print(f"    P95    : {p95:8.3f} ms")
    print(f"    FPS    : {fps:8.1f}  (from median)")
    print()


def print_gpu_memory(device: torch.device) -> None:
    """Print peak GPU memory statistics."""
    allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    print(f"  Peak allocated : {allocated:8.1f} MiB")
    print(f"  Peak reserved  : {reserved:8.1f} MiB")
    print()


# ---------------------------------------------------------------------------
# Image-quality helpers
# ---------------------------------------------------------------------------


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PSNR between two images after clamping to [0, 1]."""
    a = a.float().clamp(0.0, 1.0)
    b = b.float().clamp(0.0, 1.0)
    mse = ((a - b) ** 2).mean().item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def to_lpips_nchw(image: torch.Tensor) -> torch.Tensor:
    """Convert [1, H, W, 3] or [H, W, 3] image tensors to LPIPS NCHW input."""
    image = image.float().clamp(0.0, 1.0)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.permute(0, 3, 1, 2).contiguous()


def make_lpips_metric(net_type: str, device: torch.device):
    """Create the torchmetrics LPIPS metric if the optional dependency is available."""
    try:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except Exception as exc:
        print(f"    LPIPS skipped: torchmetrics LPIPS unavailable ({exc}).")
        return None

    try:
        return LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=True,
        ).to(device)
    except Exception as exc:
        print(f"    LPIPS skipped: failed to initialize {net_type!r} network ({exc}).")
        return None


def compare_render_quality(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    K,
    width,
    height,
    sh_degree,
    inference_tile_size,
    reference_tile_size,
    quality_frames,
    lpips_net,
    device,
) -> Optional[dict]:
    """Compare Inference functional output against the reference rasterizer."""
    if quality_frames == 0:
        print("  SKIPPED -- --quality-frames=0.\n")
        return None

    from experimental import GaussianInferenceScene, rasterize_gaussian_inference_scene
    from gsplat import rasterization

    sh_deg_arg = sh_degree if (sh_degree is not None and sh_degree >= 0) else None
    Ks = K.unsqueeze(0)
    lpips_metric = make_lpips_metric(lpips_net, device)

    scene = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=sh_deg_arg,
        sh_compression="none",
        id="quality",
    )

    psnrs = []
    lpips_values = []

    with torch.inference_mode():
        for i in range(quality_frames):
            vm = viewmats[i % len(viewmats)]
            inference_ret = rasterize_gaussian_inference_scene(
                scene,
                viewmat=vm,
                K=K,
                width=width,
                height=height,
                tile_size=inference_tile_size,
            )
            reference_colors, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=vm.unsqueeze(0),
                Ks=Ks,
                width=width,
                height=height,
                render_mode="RGB",
                packed=False,
                sh_degree=sh_deg_arg,
                tile_size=reference_tile_size,
            )

            inference_frame = inference_ret.frame
            reference_frame = reference_colors
            psnrs.append(compute_psnr(reference_frame, inference_frame))

            if lpips_metric is not None:
                try:
                    lpips_values.append(
                        lpips_metric(
                            to_lpips_nchw(inference_frame),
                            to_lpips_nchw(reference_frame),
                        ).item()
                    )
                except Exception as exc:
                    print(f"    LPIPS skipped: metric evaluation failed ({exc}).")
                    lpips_metric = None
                    lpips_values = []

    torch.cuda.synchronize(device)

    psnr_tensor = torch.tensor(psnrs)
    mean_psnr = psnr_tensor.mean().item()
    min_psnr = psnr_tensor.min().item()
    max_psnr = psnr_tensor.max().item()

    print(f"    Frames compared : {quality_frames}")
    print(f"    PSNR mean       : {mean_psnr:8.3f} dB")
    print(f"    PSNR min/max    : {min_psnr:8.3f} / {max_psnr:8.3f} dB")

    result = {
        "psnr_mean": mean_psnr,
        "psnr_min": min_psnr,
        "psnr_max": max_psnr,
        "lpips_mean": None,
    }

    if lpips_values:
        lpips_tensor = torch.tensor(lpips_values)
        mean_lpips = lpips_tensor.mean().item()
        min_lpips = lpips_tensor.min().item()
        max_lpips = lpips_tensor.max().item()
        print(f"    LPIPS ({lpips_net})  : {mean_lpips:8.5f}")
        print(f"    LPIPS min/max  : {min_lpips:8.5f} / {max_lpips:8.5f}")
        result["lpips_mean"] = mean_lpips
        result["lpips_min"] = min_lpips
        result["lpips_max"] = max_lpips

    print()
    return result


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def bench_stateful(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    K,
    width,
    height,
    sh_degree,
    tile_size,
    warmup,
    num_frames,
    device,
) -> tuple:
    """Benchmark GaussianInferenceRenderer (scene packed once, renderer created once, render per frame).

    Returns:
        (times, memory_info) where *memory_info* is a dict with keys:
        ``alloc_before``, ``alloc_after``, ``per_frame_allocs``,
        ``per_frame_delta``. All allocation values are in bytes.
    """
    from experimental import GaussianInferenceScene, GaussianInferenceRenderer

    sh_deg_arg = sh_degree if (sh_degree is not None and sh_degree >= 0) else None

    # Build scene (timed)
    pack_timer = CudaTimer(device)
    pack_timer.start()
    scene = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=sh_deg_arg,
        sh_compression="none",
        id="bench_stateful",
    )
    pack_time = pack_timer.stop()
    print(
        f"    Scene packing  : {pack_time:8.3f} ms  "
        f"({scene.num_gaussians} Gaussians)"
    )

    # Create renderer (timed separately)
    renderer_timer = CudaTimer(device)
    renderer_timer.start()
    renderer = GaussianInferenceRenderer(scene, tile_size=tile_size)
    renderer_time = renderer_timer.stop()
    print(f"    Renderer setup : {renderer_time:8.3f} ms")

    timer = CudaTimer(device)

    # Warmup
    with torch.inference_mode():
        for i in range(warmup):
            vm = viewmats[i % len(viewmats)]
            renderer.render(
                viewmat=vm,
                K=K,
                width=width,
                height=height,
            )
    torch.cuda.synchronize(device)

    alloc_before = torch.cuda.memory_allocated(device)

    # Timed
    times = []
    per_frame_allocs = []
    with torch.inference_mode():
        for i in range(num_frames):
            vm = viewmats[(warmup + i) % len(viewmats)]
            timer.start()
            renderer.render(
                viewmat=vm,
                K=K,
                width=width,
                height=height,
            )
            elapsed = timer.stop()
            times.append(elapsed)
            per_frame_allocs.append(torch.cuda.memory_allocated(device))

    alloc_after = torch.cuda.memory_allocated(device)
    per_frame_delta = (
        per_frame_allocs[-1] - per_frame_allocs[0] if per_frame_allocs else 0
    )

    memory_info = {
        "alloc_before": alloc_before,
        "alloc_after": alloc_after,
        "per_frame_allocs": per_frame_allocs,
        "per_frame_delta": per_frame_delta,
    }
    return times, memory_info


def bench_reference(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    K,
    width,
    height,
    sh_degree,
    tile_size,
    warmup,
    num_frames,
    device,
) -> Optional[List[float]]:
    """Benchmark the training rasterization() path under torch.no_grad()."""
    from gsplat import rasterization

    timer = CudaTimer(device)
    sh_deg_arg = sh_degree if (sh_degree is not None and sh_degree >= 0) else None
    Ks = K.unsqueeze(0)

    # Warmup
    for i in range(warmup):
        vm = viewmats[i % len(viewmats)].unsqueeze(0)
        with torch.no_grad():
            rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=vm,
                Ks=Ks,
                width=width,
                height=height,
                render_mode="RGB",
                packed=False,
                sh_degree=sh_deg_arg,
                tile_size=tile_size,
            )
    torch.cuda.synchronize(device)

    # Timed
    times = []
    for i in range(num_frames):
        vm = viewmats[(warmup + i) % len(viewmats)].unsqueeze(0)
        timer.start()
        with torch.no_grad():
            rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=vm,
                Ks=Ks,
                width=width,
                height=height,
                render_mode="RGB",
                packed=False,
                sh_degree=sh_deg_arg,
                tile_size=tile_size,
            )
        elapsed = timer.stop()
        times.append(elapsed)

    return times


# ---------------------------------------------------------------------------
# Host-sync detection
# ---------------------------------------------------------------------------


def measure_host_sync_overhead(device: torch.device) -> float:
    """Measure the cost of a bare torch.cuda.synchronize() call (ms)."""
    import time

    # Warm up the sync path
    for _ in range(50):
        torch.cuda.synchronize(device)

    N = 200
    start = time.perf_counter()
    for _ in range(N):
        torch.cuda.synchronize(device)
    end = time.perf_counter()
    return (end - start) / N * 1000.0  # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This benchmark requires a GPU.")
        sys.exit(1)

    device = torch.device(args.device)

    # Import gsplat and probe capabilities
    import gsplat

    has_3dgs_flag = gsplat.has_3dgs()

    sh_degree = args.sh_degree if args.sh_degree >= 0 else None
    run_reference = args.reference_tile_size is not None

    # Header
    print("=" * 70)
    print("  gsplat Inference benchmark")
    print("=" * 70)
    print()
    print(f"  Device           : {torch.cuda.get_device_name(device)}")
    print(f"  gsplat version   : {gsplat.__version__}")
    print(f"  Num Gaussians    : {args.num_gaussians:,}")
    print(f"  Resolution       : {args.width} x {args.height}")
    print(
        f"  SH degree        : {args.sh_degree} "
        f"({'RGB' if sh_degree is None else f'SH K={(sh_degree+1)**2}'})"
    )
    print(f"  Inference tile size    : {args.inference_tile_size}")
    if run_reference:
        print(f"  Reference tile   : {args.reference_tile_size}")
    else:
        print("  Reference path   : disabled")
    print(f"  Warmup frames    : {args.warmup_frames}")
    print(f"  Timed frames     : {args.num_frames}")
    if run_reference:
        print(f"  Quality frames   : {args.quality_frames}")
        print(f"  LPIPS net        : {args.lpips_net}")
    print()
    print(f"  has_3dgs()                    : {has_3dgs_flag}")
    print()

    if args.ply_path is not None:
        print(f"Loading PLY scene {args.ply_path!r} ... ", end="", flush=True)
        (
            means,
            quats,
            scales,
            opacities,
            colors,
            ply_sh_degree,
        ) = make_gaussians_from_ply(args.ply_path, device)
        sh_degree = ply_sh_degree if ply_sh_degree >= 0 else None
        print(f"done ({means.size(0):,} Gaussians, SH degree {ply_sh_degree}).")
        scene_center = means.median(dim=0).values.cpu()
        radius = 2.0
    else:
        print("Generating synthetic scene ... ", end="", flush=True)
        means, quats, scales, opacities, colors = make_gaussians(
            args.num_gaussians,
            device,
            sh_degree,
        )
        print("done.")
        scene_center = torch.tensor([0.0, 0.0, 5.0])
        radius = 7.0

    # Generate camera intrinsics and orbit trajectory
    K = make_intrinsics(args.width, args.height, device)
    quality_frames_needed = args.quality_frames if run_reference else 0
    total_frames_needed = max(
        1,
        args.warmup_frames + args.num_frames,
        quality_frames_needed,
    )
    viewmats = make_orbit_viewmats(
        total_frames_needed,
        device,
        radius=radius,
        look_at=tuple(scene_center.tolist()),
    )
    print(f"Generated {total_frames_needed} camera poses (circular orbit).")
    print()

    # ------------------------------------------------------------------
    # Optional: save a single rendered frame for visual sanity check
    # ------------------------------------------------------------------
    if args.save_image is not None:
        from PIL import Image
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        sh_deg_arg = sh_degree if (sh_degree is not None and sh_degree >= 0) else None

        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=sh_deg_arg,
            sh_compression="none",
            id="preview",
        )
        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene,
                viewmat=viewmats[0],
                K=K,
                width=args.width,
                height=args.height,
                tile_size=args.inference_tile_size,
            )
        img = (ret.frame[0].clamp(0.0, 1.0) * 255).to(torch.uint8).cpu()
        Image.fromarray(img.numpy()).save(args.save_image)
        print(f"Saved Inference preview to {args.save_image!r}")

        if run_reference and has_3dgs_flag:
            from gsplat import rasterization

            vm = viewmats[0].unsqueeze(0)
            Ks = K.unsqueeze(0)
            with torch.no_grad():
                render_colors, _, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=vm,
                    Ks=Ks,
                    width=args.width,
                    height=args.height,
                    render_mode="RGB",
                    packed=False,
                    sh_degree=sh_deg_arg,
                    tile_size=args.reference_tile_size,
                )
            img = (render_colors[0].clamp(0.0, 1.0) * 255).to(torch.uint8).cpu()
            from pathlib import Path

            p = Path(args.save_image)
            ref_path = str(p.with_stem(p.stem + "_ref"))
            Image.fromarray(img.numpy()).save(ref_path)
            print(f"Saved reference preview to {ref_path!r}")
        elif run_reference:
            print("WARNING: --save-image reference preview requires 3DGS; skipping.")

        print()

    # Common args tuples for benchmark calls. Tile sizes are path-specific
    # because Inference supports {8, 16} while the 3DGS reference path supports {4, 16}.
    inference_bench_args = (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        K,
        args.width,
        args.height,
        sh_degree,
        args.inference_tile_size,
        args.warmup_frames,
        args.num_frames,
        device,
    )

    # ------------------------------------------------------------------
    # 1. Stateful renderer benchmark (GaussianInferenceRenderer)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  1. Stateful renderer  (GaussianInferenceRenderer)")
    print("-" * 70)
    times_stateful = None
    stateful_memory_info = None
    torch.cuda.reset_peak_memory_stats(device)
    try:
        times_stateful, stateful_memory_info = bench_stateful(*inference_bench_args)
        summarize_times(times_stateful, "Stateful", args.num_frames)
        # Print workspace-reuse memory diagnostics
        alloc_before = stateful_memory_info["alloc_before"]
        alloc_after = stateful_memory_info["alloc_after"]
        delta = stateful_memory_info["per_frame_delta"]
        grew = alloc_after > alloc_before
        print(f"    Alloc before timed loop : {alloc_before / (1024**2):8.1f} MiB")
        print(f"    Alloc after  timed loop : {alloc_after / (1024**2):8.1f} MiB")
        print(f"    Allocation grew         : {'YES' if grew else 'no'}")
        print(f"    Per-frame delta (first->last) : {delta / 1024:+.1f} KiB")
        print()
    except Exception as exc:
        print(f"  FAILED -- {exc}\n")
    print_gpu_memory(device)

    # ------------------------------------------------------------------
    # 2. Reference path benchmark
    # ------------------------------------------------------------------
    times_ref = None
    if run_reference:
        reference_bench_args = (
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            K,
            args.width,
            args.height,
            sh_degree,
            args.reference_tile_size,
            args.warmup_frames,
            args.num_frames,
            device,
        )
        print("-" * 70)
        print("  2. Reference path  (rasterization + no_grad)")
        print("-" * 70)
    if run_reference and has_3dgs_flag:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            times_ref = bench_reference(*reference_bench_args)
            summarize_times(times_ref, "Reference", args.num_frames)
        except Exception as exc:
            print(f"  FAILED -- {exc}\n")
            times_ref = None
        print_gpu_memory(device)
    elif run_reference:
        print("  SKIPPED -- 3DGS not built.\n")

    # ------------------------------------------------------------------
    # 3. Image-quality comparison
    # ------------------------------------------------------------------
    quality_result = None
    if run_reference:
        print("-" * 70)
        print("  3. Image quality  (Inference vs reference)")
        print("-" * 70)
    if run_reference and has_3dgs_flag:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            quality_result = compare_render_quality(
                means,
                quats,
                scales,
                opacities,
                colors,
                viewmats,
                K,
                args.width,
                args.height,
                sh_degree,
                args.inference_tile_size,
                args.reference_tile_size,
                args.quality_frames,
                args.lpips_net,
                device,
            )
        except Exception as exc:
            print(f"  FAILED -- {exc}\n")
            quality_result = None
        print_gpu_memory(device)
    elif run_reference:
        print("  SKIPPED -- 3DGS not built.\n")

    # ------------------------------------------------------------------
    # 4. Host synchronization overhead
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  4. Host synchronization overhead")
    print("-" * 70)
    sync_ms = measure_host_sync_overhead(device)
    print(f"    Mean sync cost : {sync_ms:8.4f} ms  (200 iterations)")
    print()

    # ------------------------------------------------------------------
    # GPU memory summary (aggregate)
    # ------------------------------------------------------------------
    print("-" * 70)
    print("  GPU Memory (peak across all benchmarks)")
    print("-" * 70)
    print_gpu_memory(device)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("  Summary comparison")
    print("=" * 70)
    print()
    header = (
        f"  {'Path':<30s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}"
        f"  {'Max':>8s}  {'StdDev':>8s}  {'P5':>8s}  {'P95':>8s}  {'FPS':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header.strip())))

    na = "N/A"

    def _row(label: str, times: Optional[List[float]]) -> None:
        if times is None:
            print(
                f"  {label:<30s}  {na:>8s}  {na:>8s}  {na:>8s}"
                f"  {na:>8s}  {na:>8s}  {na:>8s}  {na:>8s}  {na:>8s}"
            )
            return
        t = torch.tensor(times)
        mean = t.mean().item()
        median = t.median().item()
        mn = t.min().item()
        mx = t.max().item()
        std = t.std().item() if len(t) > 1 else 0.0
        p5 = t.quantile(0.05).item() if len(t) >= 20 else mn
        p95 = t.quantile(0.95).item() if len(t) >= 20 else mx
        fps = 1000.0 / median if median > 0 else 0.0
        print(
            f"  {label:<30s}  {mean:8.3f}  {median:8.3f}  {mn:8.3f}"
            f"  {mx:8.3f}  {std:8.3f}  {p5:8.3f}  {p95:8.3f}  {fps:8.1f}"
        )

    _row("Stateful (InferenceRenderer)", times_stateful)
    if run_reference:
        _row("Reference (rasterization)", times_ref)

    # Speedup calculations
    if run_reference and times_ref is not None:
        ref_median = torch.tensor(times_ref).median().item()
        print()
        if times_stateful is not None:
            stat_median = torch.tensor(times_stateful).median().item()
            speedup = ref_median / stat_median if stat_median > 0 else float("inf")
            print(f"  Stateful    vs Reference speedup : {speedup:.2f}x")

    if quality_result is not None:
        print()
        print(f"  Inference vs Reference PSNR  : {quality_result['psnr_mean']:.3f} dB")
        if quality_result["lpips_mean"] is not None:
            print(
                f"  Inference vs Reference LPIPS : "
                f"{quality_result['lpips_mean']:.5f} ({args.lpips_net})"
            )

    print()
    print("=" * 70)
    print("  Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
