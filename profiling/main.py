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
    n_batches: int = 1,
    n_cameras: int = 1,
    channels: int = 3,
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    scene_grid: int = 15,
    packed: bool = True,
    sparse_grad: bool = False,
    backend: Literal["gsplat", "inria"] = "gsplat",
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

    means = torch.broadcast_to(means, (n_batches, *means.shape))
    quats = torch.broadcast_to(quats, (n_batches, *quats.shape))
    scales = torch.broadcast_to(scales, (n_batches, *scales.shape))
    opacities = torch.broadcast_to(opacities, (n_batches, *opacities.shape))
    colors = torch.broadcast_to(
        colors[None, ..., :1], (n_batches, *colors.shape[:-1], channels)
    )
    viewmats = torch.broadcast_to(
        viewmats[None, None, 0, ...], (n_batches, n_cameras, *viewmats.shape[1:])
    ).clone()
    Ks = torch.broadcast_to(
        Ks[None, None, 0, ...], (n_batches, n_cameras, *Ks.shape[1:])
    ).clone()

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

    if backend == "gsplat":
        rasterization_fn = rasterization
    elif backend == "inria":
        from gsplat import rasterization_inria_wrapper

        rasterization_fn = rasterization_inria_wrapper
    else:
        assert False, f"Backend {backend} is not valid."

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
            f"snapshot_{backend}_{reso}_{scene_grid}_{n_batches}_{n_cameras}_{channels}.pickle"
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
    for n_batches in args.n_batches:
        for n_cameras in args.n_cameras:
            for channels in args.channels:
                print("========================================")
                print(
                    f"N Batches: {n_batches}, N Cameras: {n_cameras}, Channels: {channels}"
                )
                print("========================================")
                if "gsplat" in args.backends:
                    # print("gsplat packed[True] sparse_grad[True]")
                    # for scene_grid in args.scene_grid:
                    #     stats = main(
                    #         n_batches=n_batches,
                    #         n_cameras=n_cameras,
                    #         channels=channels,
                    #         reso="1080p",
                    #         scene_grid=scene_grid,
                    #         packed=True,
                    #         sparse_grad=True,
                    #         repeats=args.repeats,
                    #         # only care about memory for the packed version implementation
                    #         memory_history=args.memory_history,
                    #         world_rank=world_rank,
                    #         world_size=world_size,
                    #     )
                    #     collection.append(
                    #         [
                    #             "gsplat v1.0.0",
                    #             True,
                    #             True,
                    #             # configs
                    #             n_batches,
                    #             n_cameras,
                    #             channels,
                    #             scene_grid,
                    #             # stats
                    #             # f"{stats['mem_fwd']:0.2f}",
                    #             f"{stats['mem_all']:0.2f}",
                    #             f"{1.0 / stats['time_fwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                    #             f"{1.0 / stats['time_bwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                    #         ]
                    #     )
                    #     torch.cuda.empty_cache()

                    # print("gsplat packed[True] sparse_grad[False]")
                    # for scene_grid in args.scene_grid:
                    #     stats = main(
                    #         n_batches=n_batches,
                    #         n_cameras=n_cameras,
                    #         channels=channels,
                    #         reso="1080p",
                    #         scene_grid=scene_grid,
                    #         packed=True,
                    #         sparse_grad=False,
                    #         repeats=args.repeats,
                    #         world_rank=world_rank,
                    #         world_size=world_size,
                    #     )
                    #     collection.append(
                    #         [
                    #             "gsplat v1.0.0",
                    #             True,
                    #             False,
                    #             # configs
                    #             n_batches,
                    #             n_cameras,
                    #             channels,
                    #             scene_grid,
                    #             # stats
                    #             # f"{stats['mem_fwd']:0.2f}",
                    #             f"{stats['mem_all']:0.2f}",
                    #             f"{1.0 / stats['time_fwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                    #             f"{1.0 / stats['time_bwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                    #         ]
                    #     )
                    #     torch.cuda.empty_cache()

                    print("gsplat packed[False] sparse_grad[False]")
                    for scene_grid in args.scene_grid:
                        stats = main(
                            n_batches=n_batches,
                            n_cameras=n_cameras,
                            channels=channels,
                            reso="1080p",
                            scene_grid=scene_grid,
                            packed=False,
                            sparse_grad=False,
                            repeats=args.repeats,
                            world_rank=world_rank,
                            world_size=world_size,
                        )
                        collection.append(
                            [
                                "gsplat v1.0.0",
                                False,
                                False,
                                # configs
                                n_batches,
                                n_cameras,
                                channels,
                                scene_grid,
                                # stats
                                # f"{stats['mem_fwd']:0.2f}",
                                f"{stats['mem_all']:0.2f}",
                                f"{1.0 / stats['time_fwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                                f"{1.0 / stats['time_bwd']:0.1f} x {(n_batches)} x {(n_cameras)}",
                            ]
                        )
                        torch.cuda.empty_cache()

                # if "inria" in args.backends:
                #     print("inria")
                #     for scene_grid in args.scene_grid:
                #         stats = main(
                #             batch_size=batch_size,
                #             channels=channels,
                #             reso="1080p",
                #             scene_grid=scene_grid,
                #             backend="inria",
                #             repeats=args.repeats,
                #         )
                #         collection.append(
                #             [
                #                 "diff-gaussian-rasterization",
                #                 "n/a",
                #                 "n/a",
                #                 # configs
                #                 batch_size,
                #                 channels,
                #                 scene_grid,
                #                 # stats
                #                 # f"{stats['mem_fwd']:0.2f}",
                #                 f"{stats['mem_all']:0.2f}",
                #                 f"{1.0 / stats['time_fwd']:0.1f} x {(batch_size)}",
                #                 f"{1.0 / stats['time_bwd']:0.1f} x {(batch_size)}",
                #             ]
                #         )
                #         torch.cuda.empty_cache()

    if world_rank == 0:
        headers = [
            "Backend",
            "Packed",
            "Sparse Grad",
            # configs
            "N Batches",
            "N Cameras",
            "Channels",
            "Scene Size",
            # stats
            # "Mem[fwd] (GB)",
            "Mem (GB)",
            "FPS[fwd]",
            "FPS[bwd]",
        ]

        # pop config columns that has only one option
        if len(args.scene_grid) == 1:
            headers.pop(6)
            for row in collection:
                row.pop(6)
        if len(args.channels) == 1:
            headers.pop(5)
            for row in collection:
                row.pop(5)
        if len(args.n_cameras) == 1:
            headers.pop(4)
            for row in collection:
                row.pop(4)
        if len(args.n_batches) == 1:
            headers.pop(3)
            for row in collection:
                row.pop(3)

        print(tabulate(collection, headers, tablefmt="rst"))


if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=9 python -m profiling.main --n_batches 1 2 4 8 --n_cameras 1 2 4 8 --scene_grid 1
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        type=str,
        default=["gsplat"],
        help="gsplat, inria",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats for profiling",
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        nargs="+",
        default=[1],
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
        "--scene_grid",
        nargs="+",
        type=int,
        default=[1, 11, 21],
        help="Scene grid size for profiling",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        type=int,
        default=[3],
        help="Number of color channels for profiling",
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
