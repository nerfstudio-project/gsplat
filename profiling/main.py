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
    batch_size: int = 1,
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    scene_grid: int = 15,
    packed: bool = True,
    sparse_grad: bool = False,
    backend: Literal["gsplat2", "gsplat", "inria"] = "gsplat2",
    repeats: int = 100,
):
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = (
        load_test_data(device=device, scene_grid=scene_grid)
    )
    sh_degree = None
    viewmats, Ks = viewmats[:1], Ks[:1]  # a single image

    # to batch
    viewmats = viewmats.repeat(batch_size, 1, 1)
    Ks = Ks.repeat(batch_size, 1, 1)

    # # to SH colors
    # sh_degree = 3
    # shs = torch.zeros((means.shape[0], (sh_degree + 1) ** 2, 3), device=device)
    # shs[:, 0, :] = colors / 0.28209479177387814
    # colors = shs

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

    if backend == "gsplat2":
        rasterization_fn = rasterization
    elif backend == "gsplat":
        from gsplat._helper import rasterization_legacy_wrapper

        rasterization_fn = rasterization_legacy_wrapper
    elif backend == "inria":
        from gsplat._helper import rasterization_inria_wrapper

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
        sh_degree=sh_degree,
        sparse_grad=sparse_grad,
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
        f"Time: [FWD]{ellipse_time_fwd:.3f}s, [BWD]{ellipse_time_bwd:.3f}s"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends",
        nargs="+",
        type=str,
        default=["gsplat"],
        help="gsplat, gsplat-legacy, inria",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats for profiling",
    )
    parser.add_argument(
        "--batch_size",
        nargs="+",
        type=int,
        default=[1],
        help="Batch size for profiling",
    )
    parser.add_argument(
        "--scene_grid",
        nargs="+",
        type=int,
        default=[1, 11, 21],
        help="Scene grid size for profiling",
    )
    args = parser.parse_args()

    # Tested on a NVIDIA TITAN RTX with (24 GB).

    for batch_size in args.batch_size:
        print("========================================")
        print(f"Batch Size: {batch_size}")
        print("========================================")
        if "gsplat" in args.backends:
            print("gsplat packed[True] sparse_grad[True]")
            # [FWD]2.21 GB, [All]2.54 GB Time: [FWD]0.028s, [BWD]0.060s
            # [FWD]0.17 GB, [All]0.17 GB Time: [FWD]0.004s, [BWD]0.008s
            # [FWD]0.83 GB, [All]0.83 GB Time: [FWD]0.012s, [BWD]0.020s
            for scene_grid in args.scene_grid:
                main(
                    batch_size=batch_size,
                    reso="1080p",
                    scene_grid=scene_grid,
                    packed=True,
                    sparse_grad=True,
                    repeats=args.repeats,
                )
                torch.cuda.empty_cache()

            print("gsplat packed[True] sparse_grad[False]")
            # [FWD]0.17 GB, [All]0.17 GB Time: [FWD]0.004s, [BWD]0.006s
            # [FWD]0.83 GB, [All]1.13 GB Time: [FWD]0.012s, [BWD]0.021s
            # [FWD]2.21 GB, [All]3.81 GB Time: [FWD]0.028s, [BWD]0.063s
            for scene_grid in args.scene_grid:
                main(
                    batch_size=batch_size,
                    reso="1080p",
                    scene_grid=scene_grid,
                    packed=True,
                    sparse_grad=False,
                    repeats=args.repeats,
                )
                torch.cuda.empty_cache()

            print("gsplat packed[False] sparse_grad[False]")
            # [FWD]0.17 GB, [All]0.17 GB Time: [FWD]0.003s, [BWD]0.006s
            # [FWD]1.50 GB, [All]1.66 GB Time: [FWD]0.011s, [BWD]0.017s
            # [FWD]4.69 GB, [All]5.79 GB Time: [FWD]0.027s, [BWD]0.048s
            for scene_grid in args.scene_grid:
                main(
                    batch_size=batch_size,
                    reso="1080p",
                    scene_grid=scene_grid,
                    packed=False,
                    sparse_grad=False,
                    repeats=args.repeats,
                )
                torch.cuda.empty_cache()

        if "gsplat-legacy" in args.backends:
            # [FWD]0.20 GB, [All]0.20 GB Time: [FWD]0.005s, [BWD]0.009s
            # [FWD]2.04 GB, [All]2.76 GB Time: [FWD]0.016s, [BWD]0.019s
            # [FWD]6.53 GB, [All]9.86 GB Time: [FWD]0.042s, [BWD]0.047s
            print("gsplat-legacy")
            for scene_grid in args.scene_grid:
                main(
                    batch_size=batch_size,
                    reso="1080p",
                    scene_grid=scene_grid,
                    backend="gsplat",
                    repeats=args.repeats,
                )
                torch.cuda.empty_cache()

        if "inria" in args.backends:
            print("inria")
            # [FWD]0.34 GB, [All]0.34 GB Time: [FWD]0.004s, [BWD]0.018s
            # [FWD]3.18 GB, [All]3.18 GB Time: [FWD]0.010s, [BWD]0.032s
            # [FWD]10.25 GB, [All]10.83 GB Time: [FWD]0.026s, [BWD]0.053s
            for scene_grid in args.scene_grid:
                main(
                    batch_size=batch_size,
                    reso="1080p",
                    scene_grid=1,
                    backend="inria",
                    repeats=args.repeats,
                )
                torch.cuda.empty_cache()
