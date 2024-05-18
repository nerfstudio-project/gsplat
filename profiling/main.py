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
    reso: Literal["360p", "720p", "1080p", "4k"] = "4k",
    scene_grid: int = 15,
    packed: bool = True,
    backend: Literal["gsplat2", "gsplat", "inria"] = "gsplat2",
    repeats: int = 100,
):
    means, quats, scales, opacities, colors, viewmats, Ks, width, height = (
        load_test_data(device=device, scene_grid=scene_grid)
    )
    sh_degree = None
    viewmats, Ks = viewmats[:1], Ks[:1]  # a single image

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
    )
    mem_toc_fwd = torch.cuda.max_memory_allocated() / 1024**3 - mem_tic

    render_colors = outputs[0]
    loss = render_colors.sum()
    ellipse_time_bwd, _ = timeit(repeats, loss.backward, retain_graph=True)
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
        default=["gsplat2"],
        help="gsplat2, gsplat, inria",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats for profiling",
    )
    args = parser.parse_args()

    # Tested on a NVIDIA TITAN RTX with (24 GB).
    if "gsplat2" in args.backends:
        print("gsplat2 packed[True]")
        main(
            reso="1080p", scene_grid=1, packed=True, repeats=args.repeats
        )  # [FWD]0.17 GB, [All]0.17 GB Time: [FWD]0.004s, [BWD]0.009s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=11, packed=True, repeats=args.repeats
        )  # [FWD]0.77 GB, [All]1.62 GB Time: [FWD]0.011s, [BWD]0.023s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=21, packed=True, repeats=args.repeats
        )  # [FWD]1.99 GB, [All]5.58 GB Time: [FWD]0.027s, [BWD]0.069s
        torch.cuda.empty_cache()

        print("gsplat2 packed[False]")
        main(
            reso="1080p", scene_grid=1, packed=False, repeats=args.repeats
        )  # [FWD]0.17 GB, [All]0.17 GB Time: [FWD]0.003s, [BWD]0.006s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=11, packed=False, repeats=args.repeats
        )  # [FWD]1.40 GB, [All]2.12 GB Time: [FWD]0.011s, [BWD]0.021s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=21, packed=False, repeats=args.repeats
        )  # [FWD]4.33 GB, [All]7.44 GB Time: [FWD]0.026s, [BWD]0.061s
        torch.cuda.empty_cache()

    if "gsplat" in args.backends:
        print("gsplat")
        main(
            reso="1080p", scene_grid=1, backend="gsplat", repeats=args.repeats
        )  # [FWD]0.20 GB, [All]0.20 GB Time: [FWD]0.005s, [BWD]0.009s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=11, backend="gsplat", repeats=args.repeats
        )  # [FWD]2.04 GB, [All]3.26 GB Time: [FWD]0.016s, [BWD]0.023s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=21, backend="gsplat", repeats=args.repeats
        )  # [FWD]6.53 GB, [All]11.70 GB Time: [FWD]0.042s, [BWD]0.061s
        torch.cuda.empty_cache()

    if "inria" in args.backends:
        print("inria")
        main(
            reso="1080p", scene_grid=1, backend="inria", repeats=args.repeats
        )  # [FWD]0.34 GB, [All]0.34 GB Time: [FWD]0.004s, [BWD]0.020s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=11, backend="inria", repeats=args.repeats
        )  # [FWD]3.18 GB, [All]3.87 GB Time: [FWD]0.011s, [BWD]0.036s
        torch.cuda.empty_cache()
        main(
            reso="1080p", scene_grid=21, backend="inria", repeats=args.repeats
        )  # [FWD]10.25 GB, [All]13.41 GB Time: [FWD]0.027s, [BWD]0.068s
        torch.cuda.empty_cache()
