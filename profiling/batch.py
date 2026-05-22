# SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Profile Memory Usage.

Usage:
```bash
pytest <THIS_PY_FILE>
```
"""

import time

import torch
from typing_extensions import Callable, Literal

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization

RESOLUTIONS = {
    "360p": (640, 360),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

device = torch.device("cuda")


def timeit(repeats: int, f: Callable, *args, **kwargs) -> float:
    for _ in range(5):  # warmup
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        results = f(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / repeats, results


def main(
    model: Literal["3DGS", "3DGUT", "NHT"] = "3DGS",
    n_gaussians: int = 1000,
    n_cameras: int = 1,
    n_batches: int = 1,
    nht_feature_dim: int = 48,
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    repeats: int = 100,
    memory_history: bool = False,
    world_rank: int = 0,
    world_size: int = 1,
):
    # NHT (Neural Harmonic Textures) builds on the 3DGUT pipeline and stores
    # per-primitive feature vectors (divided across 4 tetrahedral vertices)
    # instead of RGB colors. It also requires non-packed rasterization.
    if model == "NHT":
        if nht_feature_dim % 4 != 0:
            raise ValueError(
                f"NHT feature_dim must be divisible by 4 "
                f"(4 tetrahedron vertices); got {nht_feature_dim}."
            )

    (
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    ) = load_test_data(device=device, scene_grid=1)

    if model == "NHT":
        colors = colors[:, :1].repeat(1, nht_feature_dim)

    tensors = [means, quats, scales, opacities, colors]
    for i, tensor in enumerate(tensors):
        tensor = tensor[:n_gaussians]
        tensor = torch.broadcast_to(tensor, (n_batches, *tensor.shape)).contiguous()
        tensor.requires_grad = True
        tensors[i] = tensor
    means, quats, scales, opacities, colors = tensors

    viewmats = torch.broadcast_to(
        viewmats[None, None, 0, ...], (n_batches, n_cameras, *viewmats.shape[1:])
    ).clone()
    Ks = torch.broadcast_to(
        Ks[None, None, 0, ...], (n_batches, n_cameras, *Ks.shape[1:])
    ).clone()

    render_width, render_height = RESOLUTIONS[reso]  # desired resolution
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    if memory_history:
        torch.cuda.memory._record_memory_history()

    extra_kwargs = {}
    if model == "NHT":
        from gsplat.nht import NHTParams

        # NHT routes through the 3DGUT kernels (with_ut + with_eval3d) and
        # enables a separate NHT-specific raster path via nht_params.
        extra_kwargs.update(
            with_ut=True,
            with_eval3d=True,
            nht_params=NHTParams(),
            sh_degree=None,
        )
    else:
        extra_kwargs.update(
            with_ut=model == "3DGUT",
            with_eval3d=model == "3DGUT",
        )

    ellipse_time_fwd, outputs = timeit(
        repeats,
        rasterization,
        means,  # [B, N, 3]
        quats,  # [B, N, 4]
        scales,  # [B, N, 3]
        opacities,  # [B, N]
        colors,  # [B, N, 3] for 3DGS/3DGUT, [B, N, feature_dim] for NHT
        viewmats,  # [B, C, 4, 4]
        Ks,  # [B, C, 3, 3]
        render_width,
        render_height,
        packed=False,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=3.0,
        distributed=False,
        **extra_kwargs,
    )
    mem_toc_fwd = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic

    render_colors = outputs[0]
    loss = render_colors.sum()

    def backward():
        loss.backward(retain_graph=True)
        for v in [means, quats, scales, opacities, colors]:
            v.grad = None

    ellipse_time_bwd, _ = timeit(repeats, backward)
    mem_toc_all = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic
    print(
        f"Rasterization Mem Allocation: [FWD]{mem_toc_fwd:.2f} GB, [All]{mem_toc_all:.2f} GB "
        f"Time: [FWD]{ellipse_time_fwd:.3f}s, [BWD]{ellipse_time_bwd:.3f}s "
        f"N Gaussians: {means.shape[1]}"
    )

    if memory_history:
        torch.cuda.memory._dump_snapshot(
            f"snapshot_{reso}_{n_batches}_{n_cameras}.pickle"
        )

    return {
        "mem_fwd": mem_toc_fwd,
        "mem_all": mem_toc_all,
        "time_fwd": ellipse_time_fwd,
        "time_bwd": ellipse_time_bwd,
    }


def worker(local_rank: int, world_rank: int, world_size: int, args):
    try:
        from tabulate import tabulate
    except ImportError:
        # Fallback. tabulate is not part of the standard dependencies
        def tabulate(rows, headers, tablefmt="rst"): 
            widths = [
                max(len(str(h)), *(len(str(r[i])) for r in rows))
                for i, h in enumerate(headers)
            ]
            sep = "  ".join("-" * w for w in widths)
            line = lambda row: "  ".join(  # noqa: E731
                str(c).ljust(w) for c, w in zip(row, widths)
            )
            return "\n".join([line(headers), sep, *(line(r) for r in rows)])

    # Tested on a NVIDIA TITAN RTX with (24 GB).

    collection = []
    for model in args.model:
        if model == "NHT":
            feature_dims = args.nht_feature_dim
        else:
            feature_dims = [3]
        for feature_dim in feature_dims:
            for n_gaussians in args.n_gaussians:
                for n_cameras in args.n_cameras:
                    for n_batches in args.n_batches:
                        print("========================================")
                        suffix = (
                            f", Feature Dim: {feature_dim}" if model == "NHT" else ""
                        )
                        print(
                            f"Model: {model}, N Gaussians: {n_gaussians}, "
                            f"N Cameras: {n_cameras}, N Batches: {n_batches}{suffix}"
                        )
                        print("========================================")
                        stats = main(
                            model=model,
                            n_gaussians=n_gaussians,
                            n_cameras=n_cameras,
                            n_batches=n_batches,
                            nht_feature_dim=feature_dim,
                            reso="360p",
                            repeats=args.repeats,
                            world_rank=world_rank,
                            world_size=world_size,
                        )
                        label = (
                            f"NHT (feat={feature_dim})" if model == "NHT" else model
                        )
                        collection.append(
                            [
                                label,
                                n_gaussians,
                                n_cameras,
                                n_batches,
                                f"{stats['mem_all']:0.3f}",
                                f"{1.0 / stats['time_fwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                                f"{1.0 / stats['time_bwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                                f"{stats['time_fwd']:0.5f}",
                                f"{stats['time_bwd']:0.5f}",
                            ]
                        )
                        torch.cuda.empty_cache()

    if world_rank == 0:
        headers = [
            # model
            "Model",
            # configs
            "N Gaussians",
            "N Cameras",
            "N Batches",
            # stats
            "Mem (GB)",
            "FPS[fwd]",
            "FPS[bwd]",
            "Time[fwd]",
            "Time[bwd]",
        ]

        # pop config columns that has only one option
        if len(args.n_batches) == 1:
            headers.pop(3)
            for row in collection:
                row.pop(3)
        if len(args.n_cameras) == 1:
            headers.pop(2)
            for row in collection:
                row.pop(2)
        if len(args.n_gaussians) == 1:
            headers.pop(1)
            for row in collection:
                row.pop(1)

        print(tabulate(collection, headers, tablefmt="rst"))


if __name__ == "__main__":
    """
    # Distributed rendering is not supported yet for batchified API.
    CUDA_VISIBLE_DEVICES=9 python -m profiling.batch
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["3DGS", "3DGUT"],
        help=(
            "Model used for rasterization. One or more of: "
            "3DGS, 3DGUT, NHT."
        ),
    )
    parser.add_argument(
        "--nht_feature_dim",
        type=int,
        nargs="+",
        default=[48],
        help=(
            "Per-primitive feature dimension(s) for the NHT model. Must be "
            "divisible by 4 (4 tetrahedral vertices). Supported values: "
            "4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 64, 80, 96, 128, 256. "
            "Ignored for 3DGS / 3DGUT."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of repeats for profiling",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="Number of batches for profiling",
    )
    parser.add_argument(
        "--n_cameras",
        type=int,
        nargs="+",
        default=[1],
        help="Number of cameras for profiling",
    )
    parser.add_argument(
        "--n_gaussians",
        type=int,
        nargs="+",
        default=[10000],
        help="Number of gaussians for profiling",
    )

    args = parser.parse_args()
    for n_gaussian in args.n_gaussians:
        if n_gaussian > 100000:
            raise ValueError(
                f"Number of gaussians ({n_gaussian}) exceeds maximum allowed (100000)"
            )

    parser.add_argument(
        "--memory_history",
        action="store_true",
        help="Record memory history and dump a snapshot. Use https://pytorch.org/memory_viz to visualize.",
    )
    args = parser.parse_args()
    if args.memory_history:
        args.repeats = 1  # only run once for memory history

    cli(worker, args, verbose=True)
