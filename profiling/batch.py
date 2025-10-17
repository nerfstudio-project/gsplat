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
    model: Literal["3DGS", "3DGUT"] = "3DGS",
    n_gaussians: int = 1000,
    n_cameras: int = 1,
    n_batches: int = 1,
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    repeats: int = 100,
    memory_history: bool = False,
    world_rank: int = 0,
    world_size: int = 1,
):
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

    ellipse_time_fwd, outputs = timeit(
        repeats,
        rasterization,
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, K, 3]
        viewmats,  # [C, 4, 4]
        Ks,  # [C, 3, 3]
        render_width,
        render_height,
        packed=False,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=3.0,
        distributed=False,
        with_ut=model == "3DGUT",
        with_eval3d=model == "3DGUT",
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
    from tabulate import tabulate

    # Tested on a NVIDIA TITAN RTX with (24 GB).

    collection = []
    for model in args.model:
        for n_gaussians in args.n_gaussians:
            for n_cameras in args.n_cameras:
                for n_batches in args.n_batches:
                    print("========================================")
                    print(
                        f"N Gaussians: {n_gaussians}, N Cameras: {n_cameras}, N Batches: {n_batches}"
                    )
                    print("========================================")
                    stats = main(
                        model=model,
                        n_gaussians=n_gaussians,
                        n_cameras=n_cameras,
                        n_batches=n_batches,
                        reso="360p",
                        repeats=args.repeats,
                        world_rank=world_rank,
                        world_size=world_size,
                    )
                    collection.append(
                        [
                            model,
                            # configs
                            n_gaussians,
                            n_cameras,
                            n_batches,
                            # stats
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
        help="Model used for rasterization",
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
