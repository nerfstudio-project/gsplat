"""Benchmark AdaptiveStrategy vs DefaultStrategy on a synthetic scene.

Generates a test scene, trains with both strategies, compares PSNR.
"""

import time
import torch
import numpy as np
from torch import Tensor
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

# Import our adaptive strategy from local
import sys
sys.path.insert(0, ".")
from gsplat.strategy.adaptive import AdaptiveStrategy


def generate_test_scene(n_points=5000, device="cuda"):
    """Generate a random colored point cloud as initial Gaussians."""
    torch.manual_seed(42)
    means = torch.randn(n_points, 3, device=device) * 2.0
    colors = torch.rand(n_points, 3, device=device)
    scales = torch.full((n_points, 3), -3.0, device=device)  # log scale
    quats = torch.zeros(n_points, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n_points,), 2.0, device=device)  # logit

    params = {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "colors": torch.nn.Parameter(colors),
    }
    return params


def generate_target_views(n_views=8, H=256, W=256, device="cuda"):
    """Generate random camera views and render target images from a dense point cloud."""
    # Create a denser scene for targets
    torch.manual_seed(0)
    n_target_pts = 50000
    means = torch.randn(n_target_pts, 3, device=device) * 2.0
    colors = torch.rand(n_target_pts, 3, device=device)
    scales = torch.full((n_target_pts, 3), -4.0, device=device)
    quats = torch.zeros(n_target_pts, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.full((n_target_pts,), 3.0, device=device)

    # Camera intrinsics
    fx, fy = 200.0, 200.0
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=device, dtype=torch.float32)

    views = []
    targets = []
    for i in range(n_views):
        angle = 2 * np.pi * i / n_views
        R = torch.eye(3, device=device)
        t = torch.tensor([3*np.cos(angle), 0.5, 3*np.sin(angle)], device=device, dtype=torch.float32)
        # Look at origin
        forward = -t / t.norm()
        right = torch.cross(torch.tensor([0., 1., 0.], device=device), forward)
        right = right / right.norm()
        up = torch.cross(forward, right)
        R = torch.stack([right, up, forward], dim=0)

        viewmat = torch.eye(4, device=device)
        viewmat[:3, :3] = R
        viewmat[:3, 3] = -(R @ t)

        with torch.no_grad():
            rendered, _, _ = rasterization(
                means=means, quats=quats, scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities), colors=colors,
                viewmats=viewmat[None], Ks=K[None],
                width=W, height=H, packed=False,
            )
        views.append((viewmat, K))
        targets.append(rendered[0].detach())

    return views, targets


def train_with_strategy(strategy, params, views, targets, n_steps=3000, H=256, W=256):
    """Train Gaussians with a given strategy."""
    device = params["means"].device

    # Create optimizers
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=0.005),
        "scales": torch.optim.Adam([params["scales"]], lr=0.005),
        "quats": torch.optim.Adam([params["quats"]], lr=0.001),
        "opacities": torch.optim.Adam([params["opacities"]], lr=0.05),
        "colors": torch.optim.Adam([params["colors"]], lr=0.01),
    }

    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=2.0)

    losses = []
    t0 = time.time()

    for step in range(n_steps):
        # Pick a random view
        idx = step % len(views)
        viewmat, K = views[idx]
        target = targets[idx]

        # Render
        rendered, _, info = rasterization(
            means=params["means"],
            quats=torch.nn.functional.normalize(params["quats"], dim=-1),
            scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]),
            colors=params["colors"],
            viewmats=viewmat[None],
            Ks=K[None],
            width=W, height=H, packed=False,
        )

        loss = torch.nn.functional.l1_loss(rendered[0], target)

        # Strategy pre-backward
        strategy.step_pre_backward(params, optimizers, state, step, info)

        loss.backward()

        # Strategy post-backward
        if isinstance(strategy, AdaptiveStrategy):
            strategy.step_post_backward(
                params, optimizers, state, step, info, loss=loss.item()
            )
        else:
            strategy.step_post_backward(params, optimizers, state, step, info)

        # Optimizer step
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad()

        losses.append(loss.item())
        if step % 500 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, n_gs={len(params['means'])}")

    elapsed = time.time() - t0

    # Compute final PSNR
    total_psnr = 0
    with torch.no_grad():
        for idx in range(len(views)):
            viewmat, K = views[idx]
            target = targets[idx]
            rendered, _, _ = rasterization(
                means=params["means"],
                quats=torch.nn.functional.normalize(params["quats"], dim=-1),
                scales=torch.exp(params["scales"]),
                opacities=torch.sigmoid(params["opacities"]),
                colors=params["colors"],
                viewmats=viewmat[None], Ks=K[None],
                width=W, height=H, packed=False,
            )
            mse = ((rendered[0] - target) ** 2).mean()
            psnr = -10 * torch.log10(mse)
            total_psnr += psnr.item()

    avg_psnr = total_psnr / len(views)
    return avg_psnr, len(params["means"]), elapsed, losses


def main():
    device = "cuda"
    H, W = 256, 256
    n_steps = 3000

    print("Generating test scene...")
    views, targets = generate_target_views(n_views=8, H=H, W=W, device=device)
    print(f"  {len(views)} views, {H}x{W}")

    # Benchmark DefaultStrategy
    print("\n=== DefaultStrategy (hand-tuned) ===")
    params_default = generate_test_scene(n_points=5000, device=device)
    default_strategy = DefaultStrategy(
        refine_start_iter=200,
        refine_every=100,
        refine_stop_iter=2500,
        reset_every=1000,
        grow_grad2d=0.0002,
        prune_opa=0.005,
        verbose=True,
    )
    psnr_default, n_gs_default, time_default, losses_default = train_with_strategy(
        default_strategy, params_default, views, targets, n_steps=n_steps
    )
    print(f"\n  Final: PSNR={psnr_default:.2f} dB, {n_gs_default} Gaussians, {time_default:.1f}s")

    # Benchmark AdaptiveStrategy
    print("\n=== AdaptiveStrategy (auto-tuned) ===")
    params_adaptive = generate_test_scene(n_points=5000, device=device)
    adaptive_strategy = AdaptiveStrategy(
        refine_start_iter=200,
        refine_every=100,
        refine_stop_iter=2500,
        reset_every=1000,
        grow_rate=0.5,
        prune_percentile=5.0,
        cap_max=200_000,
        verbose=True,
    )
    psnr_adaptive, n_gs_adaptive, time_adaptive, losses_adaptive = train_with_strategy(
        adaptive_strategy, params_adaptive, views, targets, n_steps=n_steps
    )
    print(f"\n  Final: PSNR={psnr_adaptive:.2f} dB, {n_gs_adaptive} Gaussians, {time_adaptive:.1f}s")

    # Summary
    print(f"\n{'='*50}")
    print(f"{'Strategy':<20} {'PSNR':>8} {'#GS':>10} {'Time':>8}")
    print(f"{'-'*50}")
    print(f"{'Default':<20} {psnr_default:>7.2f}  {n_gs_default:>9}  {time_default:>6.1f}s")
    print(f"{'Adaptive':<20} {psnr_adaptive:>7.2f}  {n_gs_adaptive:>9}  {time_adaptive:>6.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
