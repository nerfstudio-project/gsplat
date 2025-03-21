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
from gsplat.camera import to_params
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
    with_ut: bool = False,
    with_eval3d: bool = False,
    batch_size: int = 1,
    channels: int = 3,
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    scene_grid: int = 15,
    packed: bool = True,
    sparse_grad: bool = False,
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
    ) = load_test_data(device=device, scene_grid=scene_grid)

    # to batch
    viewmats = viewmats[:1].repeat(batch_size, 1, 1)
    Ks = Ks[:1].repeat(batch_size, 1, 1)

    cm_params, rs_params = to_params(
        viewmats, Ks, width, height, camera_model="pinhole"
    )

    # more channels
    colors = colors[:, :1].repeat(1, channels)

    # distribute the gaussians
    means = means[world_rank::world_size].contiguous()
    quats = quats[world_rank::world_size].contiguous()
    scales = scales[world_rank::world_size].contiguous()
    opacities = opacities[world_rank::world_size].contiguous()
    colors = colors[world_rank::world_size].contiguous()

    means.requires_grad = True
    quats.requires_grad = True
    scales.requires_grad = True
    opacities.requires_grad = True
    colors.requires_grad = True

    render_width, render_height = RESOLUTIONS[reso]  # desired resolution
    Ks[..., 0, :] *= render_width / width
    Ks[..., 1, :] *= render_height / height

    torch.cuda.reset_peak_memory_stats()
    mem_tic = torch.cuda.max_memory_allocated() / 1024**3

    if memory_history:
        torch.cuda.memory._record_memory_history()

    rasterization_fn = rasterization
    ellipse_time_fwd, outputs = timeit(
        repeats,
        rasterization_fn,
        means,  # [N, 3]
        quats,  # [N, 4]
        scales,  # [N, 3]
        opacities,  # [N]
        colors,  # [N, K, 3]
        viewmats,  # [C, 4, 4]
        Ks,  # [C, 3, 3]
        render_width,
        render_height,
        packed=packed,
        near_plane=0.01,
        far_plane=100.0,
        radius_clip=3.0,
        sparse_grad=sparse_grad,
        distributed=world_size > 1,
        cm_params=cm_params,
        rs_params=rs_params,
        with_ut=with_ut,
        with_eval3d=with_eval3d,
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
        f"N Gaussians: {means.shape[0]}"
    )

    if memory_history:
        torch.cuda.memory._dump_snapshot(
            f"snapshot_{reso}_{scene_grid}_{batch_size}_{channels}.pickle"
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
    batch_size = 1
    channels = 3
    scene_grid = 5
    packed = False
    sparse_grad = False

    print("========================================")
    print("gsplat with_ut[False] with_eval3d[False]")
    stats = main(
        with_ut=False,
        with_eval3d=False,
        batch_size=batch_size,
        channels=channels,
        reso="1080p",
        scene_grid=scene_grid,
        packed=packed,
        sparse_grad=sparse_grad,
        repeats=args.repeats,
        # only care about memory for the packed version implementation
        memory_history=args.memory_history,
        world_rank=world_rank,
        world_size=world_size,
    )
    collection.append(
        [
            "gsplat",
            False,
            False,
            # configs
            # scene_grid,
            # stats
            # f"{stats['mem_fwd']:0.2f}",
            f"{stats['mem_all']:0.2f}",
            f"{1.0 / stats['time_fwd']:0.1f} x {(batch_size)}",
            f"{1.0 / stats['time_bwd']:0.1f} x {(batch_size)}",
        ]
    )
    torch.cuda.empty_cache()

    print("gsplat with_ut[False] with_eval3d[True]")
    stats = main(
        with_ut=False,
        with_eval3d=True,
        batch_size=batch_size,
        channels=channels,
        reso="1080p",
        scene_grid=scene_grid,
        packed=packed,
        sparse_grad=sparse_grad,
        repeats=args.repeats,
        world_rank=world_rank,
        world_size=world_size,
    )
    collection.append(
        [
            "gsplat",
            False,
            True,
            # configs
            # scene_grid,
            # stats
            # f"{stats['mem_fwd']:0.2f}",
            f"{stats['mem_all']:0.2f}",
            f"{1.0 / stats['time_fwd']:0.1f} x {(batch_size)}",
            f"{1.0 / stats['time_bwd']:0.1f} x {(batch_size)}",
        ]
    )
    torch.cuda.empty_cache()

    if world_rank == 0:
        headers = [
            "With UT",
            "With Eval3D",
            # configs
            # "Scene Size",
            # stats
            # "Mem[fwd] (GB)",
            "Mem (GB)",
            "FPS[fwd]",
            "FPS[bwd]",
        ]

        print(tabulate(collection, headers, tablefmt="rst"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats for profiling",
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
