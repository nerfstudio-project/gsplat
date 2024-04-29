import time

import numpy as np
import torch
import torch.nn.functional as F

device = torch.device("cuda:0")


def timeit(func, cnt: int = 1000, *args, **kwargs):
    for _ in range(5):  # warmup
        func(*args, **kwargs)
    torch.cuda.synchronize()
    tic = time.perf_counter()
    for _ in range(cnt):
        func(*args, **kwargs)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    return (toc - tic) / cnt


def load_test_data(B: int = 1, C: int = 3):
    torch.manual_seed(42)

    data_path = "/workspace/gsplat2/assets/data.npz"
    data = np.load(data_path)  # 3 cameras but we keep the first one
    height, width = data["height"].item(), data["width"].item()
    viewmats = torch.from_numpy(data["viewmats"][:1]).to(device)
    Ks = torch.from_numpy(data["Ks"][:1]).to(device)
    means = torch.from_numpy(data["means3d"]).to(device)
    scales = torch.rand((len(means), 3), device=device) * 0.01
    quats = F.normalize(torch.randn(len(means), 4), dim=-1).to(device)
    opacities = torch.rand((len(means),), device=device)
    colors = torch.rand((len(Ks), len(means), C), device=device)

    if B > 1:
        viewmats = viewmats.repeat(B, 1, 1)
        Ks = Ks.repeat(B, 1, 1)
        colors = colors.repeat(B, 1, 1)
    return viewmats, Ks, means, scales, quats, opacities, colors, width, height


def profile_fwd(batch_size: int = 1, channels: int = 3):
    from gsplat.experimental.cuda import _rendering_gsplat, rendering

    (
        viewmats,
        Ks,
        means,
        scales,
        quats,
        opacities,
        colors,
        width,
        height,
    ) = load_test_data(batch_size, channels)

    elapsed_time = timeit(
        rendering,
        100,
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    )
    print(f"fwd[B={batch_size}][C={channels}][CUDA2]: {elapsed_time * 1000:.3f}ms")

    elapsed_time = timeit(
        _rendering_gsplat,
        100,
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    )
    print(f"fwd[B={batch_size}][C={channels}][Gsplat]: {elapsed_time * 1000:.3f}ms")


def profile_bwd(batch_size: int = 1, channels: int = 3, viewmats_grad: bool = False):
    from gsplat.experimental.cuda import _rendering_gsplat, rendering

    (
        viewmats,
        Ks,
        means,
        scales,
        quats,
        opacities,
        colors,
        width,
        height,
    ) = load_test_data(batch_size, channels)

    viewmats.requires_grad = viewmats_grad
    means.requires_grad = True
    scales.requires_grad = True
    quats.requires_grad = True

    render_colors, render_alphas = rendering(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height
    )
    loss = render_colors.sum() + render_alphas.sum()
    elapsed_time = timeit(
        torch.autograd.grad,
        100,
        loss,
        (means, scales, quats, viewmats) if viewmats_grad else (means, scales, quats),
        retain_graph=True,
    )
    print(f"bwd[B={batch_size}][C={channels}][CUDA2]: {elapsed_time * 1000:.3f}ms")

    render_colors, render_alphas = _rendering_gsplat(
        means, quats, scales, opacities, colors, viewmats, Ks, width, height
    )
    loss = render_colors.sum() + render_alphas.sum()
    elapsed_time = timeit(
        torch.autograd.grad,
        100,
        loss,
        (means, scales, quats, viewmats) if viewmats_grad else (means, scales, quats),
        retain_graph=True,
    )
    print(f"bwd[B={batch_size}][C={channels}][Gsplat]: {elapsed_time * 1000:.3f}ms")


if __name__ == "__main__":
    # profile_fwd(batch_size=1, channels=1) # 1.146ms/1.492ms
    # profile_fwd(batch_size=8, channels=1) # 3.593ms/9.921ms
    # profile_fwd(batch_size=64, channels=1) # 28.948ms/76.295ms

    # profile_fwd(batch_size=1, channels=3)  # 1.172ms/1.540ms
    # profile_fwd(batch_size=8, channels=3)  # 3.801ms/9.995ms
    # profile_fwd(batch_size=64, channels=3)  # 30.247ms/76.098ms

    # profile_fwd(batch_size=1, channels=32)  # 1.930ms/5.078ms
    # profile_fwd(batch_size=8, channels=32)  # 7.773ms/32.824ms
    # profile_fwd(batch_size=64, channels=32)  # 64.461ms/264.146ms

    # profile_bwd(batch_size=1, channels=1, viewmats_grad=False) # 1.562ms/1.594ms
    # profile_bwd(batch_size=8, channels=1, viewmats_grad=False) # 7.409ms/10.600ms
    # profile_bwd(batch_size=64, channels=1, viewmats_grad=False) # 60.402ms/84.395ms

    # profile_bwd(batch_size=1, channels=3, viewmats_grad=False)  # 2.010ms/1.891ms
    # profile_bwd(batch_size=8, channels=3, viewmats_grad=False)  # 10.132ms/11.498ms
    # profile_bwd(batch_size=64, channels=3, viewmats_grad=False)  # 80.269ms/86.210ms

    # profile_bwd(batch_size=1, channels=32, viewmats_grad=False)  # 7.850ms/10.242ms
    # profile_bwd(batch_size=8, channels=32, viewmats_grad=False)  # 43.392ms/80.935ms
    # profile_bwd(batch_size=64, channels=32, viewmats_grad=False)  # 353.733ms/653.627ms

    print("Profile Done.")
