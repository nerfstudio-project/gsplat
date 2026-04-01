"""Multi-camera autonomous driving trainer with gsplat.

Trains 3D Gaussians on a surround-view driving scene (6 pinhole cameras).
Gaussians are initialized from LiDAR point clouds and optimized using:
- L1 loss with sky masking (reduce_mean with non-sky mask)
- SSIM loss for structural similarity
- Depth supervision: LiDAR points projected into camera views provide
  sparse GT depth via disparity-space L1 (depth_l1_loss)
- gaussian_scale_reg and gaussian_density_reg for regularization
- Sky-aware penalty using rasterization alpha output

Each training step renders a single (frame, camera) pair, cycling
through all views.

Usage:
    python av_trainer.py --scene assets/test_pandaset.npz --max-steps 15000
    python av_trainer.py --scene assets/test_pandaset.npz --max-steps 500

Data format (npz):
    images:             [N_frames, N_cams, H, W, 3] uint8
    cam_intrinsics:     [N_cams, 4] float32 (fx, fy, cx, cy)
    cam_to_worlds:      [N_frames, N_cams, 4, 4] float32
    lidar_points:       [M, 4] float32 (x, y, z, intensity) in world coords
    lidar_frame_indices:[M] int32
    is_test:            [N_frames] bool
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import imageio
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gsplat
from gsplat.strategy import MCMCStrategy


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scene(path: str, device: str = "cuda") -> SimpleNamespace:
    """Load a scene from npz file and prepare per-camera tensors."""
    data = dict(np.load(path, allow_pickle=True))

    scene = SimpleNamespace()
    scene.images = torch.from_numpy(data["images"]).float().to(device) / 255.0
    cam_intrinsics = torch.from_numpy(data["cam_intrinsics"]).float().to(device)
    cam_to_worlds = torch.from_numpy(data["cam_to_worlds"]).float().to(device)
    scene.lidar_points = torch.from_numpy(data["lidar_points"]).float().to(device)
    scene.lidar_frame_indices = (
        torch.from_numpy(data["lidar_frame_indices"]).long().to(device)
    )
    scene.is_test = data["is_test"]
    scene.n_frames, scene.n_cams, scene.H, scene.W, _ = scene.images.shape
    scene.camera_names = list(
        data.get("camera_names", [f"cam{i}" for i in range(scene.n_cams)])
    )

    # Per-camera K matrices [N_cams, 1, 3, 3]
    scene.Ks = torch.zeros(scene.n_cams, 1, 3, 3, device=device)
    scene.Ks[:, 0, 0, 0] = cam_intrinsics[:, 0]  # horizontal focal length
    scene.Ks[:, 0, 1, 1] = cam_intrinsics[:, 1]  # vertical focal length
    scene.Ks[:, 0, 0, 2] = cam_intrinsics[:, 2]  # x principal point
    scene.Ks[:, 0, 1, 2] = cam_intrinsics[:, 3]  # y principal point
    scene.Ks[:, 0, 2, 2] = 1.0

    # Per-frame, per-camera viewmats
    scene.viewmats = torch.linalg.inv(cam_to_worlds)  # [N_frames, N_cams, 4, 4]

    # Build per-frame LiDAR point index for fast lookup
    scene.lidar_by_frame = {}
    for fi in range(scene.n_frames):
        mask = scene.lidar_frame_indices == fi
        if mask.any():
            scene.lidar_by_frame[fi] = scene.lidar_points[mask, :3]

    return scene


SH_C0 = 0.28209479177387814


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0, 1] to 0th-order SH coefficient."""
    return (rgb - 0.5) / SH_C0


def init_gaussians_from_lidar(
    scene: SimpleNamespace, device: str = "cuda", sh_degree: int = 0
) -> torch.nn.ParameterDict:
    """Initialize one Gaussian per training-frame LiDAR point.

    LiDAR provides the only source of 3D geometry: each point becomes
    a Gaussian centered at that world-space position.  Colors are
    initialized from LiDAR intensity (grayscale, expanded to 3 channels).
    No additional Gaussians are created during training — the set is
    fixed at initialization.
    """
    train_mask = ~scene.is_test
    train_frame_ids = torch.where(torch.from_numpy(train_mask))[0]

    pts_mask = torch.zeros(len(scene.lidar_points), dtype=torch.bool, device=device)
    for fid in train_frame_ids:
        pts_mask |= scene.lidar_frame_indices == fid.item()

    points = scene.lidar_points[pts_mask, :3]
    intensities = scene.lidar_points[pts_mask, 3:4]
    N = points.shape[0]

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(points.clone()),
            "scales": torch.nn.Parameter(torch.full((N, 3), -5.0, device=device)),
            "quats": torch.nn.Parameter(
                F.normalize(torch.randn(N, 4, device=device), dim=-1)
            ),
            "opacities": torch.nn.Parameter(
                torch.logit(torch.full((N,), 0.1, device=device))
            ),
        }
    )
    if sh_degree > 0:
        K = (sh_degree + 1) ** 2
        sh0 = rgb_to_sh(intensities.expand(-1, 3).clamp(0, 1)).unsqueeze(1)
        shN = torch.zeros(N, K - 1, 3, device=device)
        params["sh0"] = torch.nn.Parameter(sh0)
        params["shN"] = torch.nn.Parameter(shN)
    else:
        params["colors"] = torch.nn.Parameter(
            intensities.expand(-1, 3).clamp(0, 1).clone()
        )
    return params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def estimate_sky_mask(
    image: torch.Tensor,
    height_fraction: float = 0.4,
    brightness_threshold: float = 0.85,
) -> torch.Tensor:
    """Estimate a binary sky mask from an RGB image using brightness.

    Marks pixels in the upper portion of the image that exceed a
    brightness threshold as sky.  Returns a bool mask [H, W].
    """
    H = image.shape[0]
    upper = int(H * height_fraction)
    gray = image.mean(dim=-1)  # [H, W]
    mask = torch.zeros_like(gray, dtype=torch.bool)
    mask[:upper] = gray[:upper] > brightness_threshold
    return mask


def project_lidar_to_camera(
    points_world: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space LiDAR points into a camera, returning sparse GT depth.

    Uses pinhole projection, which must match the camera_model used in
    gsplat.rasterization().

    Args:
        points_world: [M, 3] LiDAR points in world coordinates.
        viewmat: [4, 4] world-to-camera matrix.
        K: [3, 3] camera intrinsics (pinhole: fx, fy, cx, cy).
        H, W: image dimensions.

    Returns:
        pixel_y, pixel_x: [P] int pixel coordinates of valid projections.
        gt_depth: [P] depth values in camera space.
    """
    # Transform to camera space
    R = viewmat[:3, :3]  # [3, 3]
    t = viewmat[:3, 3]  # [3]
    pts_cam = (R @ points_world.T).T + t  # [M, 3]

    # Keep points in front of camera
    valid = pts_cam[:, 2] > 0.1
    pts_cam = pts_cam[valid]

    # Project to pixels
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * pts_cam[:, 0] / pts_cam[:, 2] + cx
    v = fy * pts_cam[:, 1] / pts_cam[:, 2] + cy

    # Keep points within image bounds
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depth = u[in_bounds], v[in_bounds], pts_cam[in_bounds, 2]

    return v.long(), u.long(), depth


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return float("inf")
    return -10.0 * np.log10(mse)


# ---------------------------------------------------------------------------
# Forward pass and loss
# ---------------------------------------------------------------------------


def render_gaussians(
    params: torch.nn.ParameterDict,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    W: int,
    H: int,
    render_mode: str = "RGB",
    sh_degree_to_use: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians into a single camera view.

    Applies post-activation transforms (exp for scales, sigmoid for opacities)
    before passing to gsplat.rasterization with pinhole camera model.

    Args:
        params: ParameterDict with means, quats, scales, opacities, colors.
        viewmat: [1, 4, 4] world-to-camera matrix.
        K: [1, 3, 3] camera intrinsics.
        W, H: image dimensions.
        render_mode: "RGB" for color only, "RGB+ED" for color + expected depth.
            ED (Expected Depth) is the alpha-weighted sum of depths, which is
            differentiable w.r.t. opacities — needed for depth_l1_loss
            supervision.  "D" (median depth) would not be differentiable.

    Returns:
        renders: [1, H, W, C] rendered output (C=3 for RGB, C=4 for RGB+ED).
        alphas: [1, H, W, 1] accumulated opacity per pixel.
    """
    if "sh0" in params:
        colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    else:
        colors = params["colors"]

    renders, alphas, _ = gsplat.rasterization(
        params["means"],
        F.normalize(params["quats"], dim=-1),
        torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]),
        colors,
        viewmat,
        K,
        W,
        H,
        camera_model="pinhole",
        sh_degree=sh_degree_to_use,
        packed=True,
        render_mode=render_mode,
    )
    return renders, alphas


def compute_loss(
    params: torch.nn.ParameterDict,
    rgb_render: torch.Tensor,
    depth_render: torch.Tensor,
    alphas: torch.Tensor,
    rgb_gt: torch.Tensor,
    scene: SimpleNamespace,
    idxframe: int,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute composite training loss for a single rendered view.

    Loss = 0.8 * L1 (sky-masked) + 0.2 * SSIM
         + 0.01 * depth (LiDAR disparity L1)
         + 0.005 * scale_reg + 0.005 * density_reg
         + 0.001 * sky_penalty
    """
    # Sky mask from image brightness (upper image, bright pixels)
    sky_mask = estimate_sky_mask(rgb_gt[0])  # [H, W] bool
    non_sky = (~sky_mask).to(torch.int32).unsqueeze(0)  # [1, H, W] int

    # Per-pixel L1 loss, masked to exclude sky pixels
    per_pixel_l1 = gsplat.losses.l1_loss(rgb_render, rgb_gt).mean(dim=-1)
    loss_l1 = gsplat.losses.reduce_mean(per_pixel_l1, mask=non_sky)

    # Structural similarity (clamp to [0,1] — rasterizer may produce tiny negatives)
    loss_ssim = gsplat.losses.ssim_loss(
        rgb_render.clamp(0, 1).permute(0, 3, 1, 2),
        rgb_gt.permute(0, 3, 1, 2),
    )

    # Depth supervision: LiDAR points are projected into the camera via
    # pinhole math (project_lidar_to_camera), producing sparse GT depth
    # at known pixels.  The rendered expected depth at those pixels is
    # compared to the LiDAR GT using disparity-space L1 (depth_l1_loss),
    # which weights nearby objects more heavily.  This does NOT use
    # gsplat's LiDAR renderer — it only uses LiDAR as a depth reference.
    loss_depth = torch.tensor(0.0, device=device)
    if idxframe in scene.lidar_by_frame:
        lidar_pts = scene.lidar_by_frame[idxframe]  # [M, 3] world
        py, px, gt_depth = project_lidar_to_camera(
            lidar_pts, viewmat[0], K[0], H, W
        )
        if len(py) > 0:
            pred_depth = depth_render[py, px]
            loss_depth = gsplat.losses.depth_l1_loss(
                pred_depth.unsqueeze(0), gt_depth.unsqueeze(0)
            )

    # Sky penalty: penalize high opacity in sky regions
    coverage = alphas[0, :, :, 0]  # [H, W], accumulated opacity
    sky_penalty = (coverage * sky_mask.float()).mean()

    # Gaussian regularization (post-activation values)
    activated_scales = torch.exp(params["scales"])
    activated_opacities = torch.sigmoid(params["opacities"])
    loss_scale = gsplat.losses.gaussian_scale_reg(activated_scales).mean()
    loss_density = gsplat.losses.gaussian_density_reg(activated_opacities).mean()

    return (
        0.8 * loss_l1
        + 0.2 * loss_ssim
        + 0.01 * loss_depth
        + 0.005 * loss_scale
        + 0.005 * loss_density
        + 0.001 * sky_penalty
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    params: torch.nn.ParameterDict,
    scene: SimpleNamespace,
    test_frame_ids: list[int],
    W: int,
    H: int,
    renders_dir: str | None = None,
    sh_degree: int | None = None,
) -> float:
    """Render test views and compute mean PSNR.

    Args:
        params: ParameterDict with Gaussian parameters.
        scene: SimpleNamespace from load_scene().
        test_frame_ids: list of frame indices to evaluate.
        W, H: image dimensions.
        renders_dir: if not None, save side-by-side GT/render PNGs here.

    Returns:
        Mean PSNR in dB across all (frame, camera) test views.
    """
    psnrs = []
    with torch.no_grad():
        for tfid in test_frame_ids:
            for tci in range(scene.n_cams):
                gt_t = scene.images[tfid, tci].unsqueeze(0)
                vm_t = scene.viewmats[tfid, tci].unsqueeze(0)
                K_t = scene.Ks[tci]
                rd, _ = render_gaussians(params, vm_t, K_t, W, H,
                                         sh_degree_to_use=sh_degree)
                psnrs.append(compute_psnr(rd[0].clamp(0, 1), gt_t[0]))

                if renders_dir is not None:
                    gt_np = (gt_t[0].cpu().numpy() * 255).astype(np.uint8)
                    rd_np = (rd[0].clamp(0, 1).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                    canvas = np.concatenate([gt_np, rd_np], axis=1)
                    cam_name = scene.camera_names[tci]
                    imageio.imwrite(
                        os.path.join(renders_dir, f"eval_f{tfid}_{cam_name}.png"),
                        canvas,
                    )

    return float(np.mean(psnrs)) if psnrs else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def create_optimizers(
    params: torch.nn.ParameterDict, lr: float
) -> dict[str, torch.optim.Adam]:
    """Set up per-parameter Adam optimizers.

    MCMC strategy requires one optimizer per parameter key so it can
    add/remove Gaussians and update optimizer state accordingly.
    """
    lr_map = {
        "means": lr * 0.032,
        "scales": lr,
        "quats": lr * 0.2,
        "opacities": lr * 10,
    }
    if "sh0" in params:
        lr_map["sh0"] = lr * 0.5
        lr_map["shN"] = lr * 0.5 / 20
    else:
        lr_map["colors"] = lr * 0.5
    return {
        name: torch.optim.Adam(
            [{"params": params[name], "lr": lr_map[name], "name": name}],
            eps=1e-15,
        )
        for name in lr_map
    }


def log_training_step(
    step: int, max_steps: int, total_loss: torch.Tensor, start_time: float
) -> None:
    """Print training progress: step, loss, throughput, GPU memory."""
    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    print(
        f"Step {step}/{max_steps} | loss={total_loss.item():.4f} | "
        f"{max(1, step) / elapsed:.0f} it/s | {mem_peak:.0f} MB"
    )


def run_evaluation(
    params: torch.nn.ParameterDict,
    scene: SimpleNamespace,
    test_frame_ids: list[int],
    W: int,
    H: int,
    step: int,
    total_loss: torch.Tensor,
    start_time: float,
    N: int,
    renders_dir: str,
    stats_dir: str,
    is_last_step: bool,
    sh_degree: int | None = None,
) -> dict:
    """Run evaluation and save checkpoint JSON. Returns checkpoint dict."""
    elapsed = time.time() - start_time
    mem_peak = torch.cuda.max_memory_allocated() / 1e6
    checkpoint: dict = {
        "step": step + 1,
        "loss": total_loss.item(),
        "elapsed_s": elapsed,
        "gpu_peak_mb": mem_peak,
        "num_gaussians": N,
    }
    if test_frame_ids:
        save_dir = renders_dir if is_last_step else None
        mean_psnr = evaluate(
            params, scene, test_frame_ids, W, H,
            renders_dir=save_dir, sh_degree=sh_degree,
        )
        checkpoint["mean_psnr"] = mean_psnr
        print(
            f"  Eval PSNR: {mean_psnr:.2f} dB "
            f"({len(test_frame_ids) * scene.n_cams} views)"
        )
    with open(os.path.join(stats_dir, f"step{step + 1:05d}.json"), "w") as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint


def train(
    scene_path: str,
    max_steps: int,
    lr: float,
    log_every: int,
    eval_every: int,
    result_dir: str,
    use_mcmc: bool = False,
    cap_max: int = 500_000,
    sh_degree: int = 0,
    sh_degree_interval: int = 1000,
    save_model: bool = True,
) -> tuple[list[float], list[dict]]:
    """Train Gaussians on a multi-camera driving scene.

    High-level flow:
        load scene → init Gaussians from LiDAR → create optimizer →
        for each step: render view → compute loss → backward → update →
        periodically evaluate on held-out test views.

    Args:
        scene_path: path to the npz scene file.
        max_steps: number of training iterations.
        lr: base learning rate.
        log_every: print training stats every N steps.
        eval_every: run evaluation on test views every N steps.
        result_dir: directory for outputs (renders, stats, summary).

    Returns:
        (losses_history, checkpoints): per-step loss list and eval dicts.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    os.makedirs(result_dir, exist_ok=True)
    renders_dir = os.path.join(result_dir, "renders")
    stats_dir = os.path.join(result_dir, "stats")
    os.makedirs(renders_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)

    # --- Load data and initialize model ---
    scene = load_scene(scene_path, device=device)
    params = init_gaussians_from_lidar(scene, device=device, sh_degree=sh_degree)
    optimizers = create_optimizers(params, lr)

    # --- MCMC strategy (optional) ---
    strategy = None
    strategy_state = None
    if use_mcmc:
        strategy = MCMCStrategy(
            cap_max=cap_max, noise_lr=5e5,
            refine_start_iter=500, refine_stop_iter=int(max_steps * 0.8),
            refine_every=100, verbose=True,
        )
        strategy.check_sanity(params, optimizers)
        strategy_state = strategy.initialize_state()
        print(f"MCMC strategy enabled: cap_max={cap_max}")

    # --- Learning rate schedule: exponential decay to 1% for means ---
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        ),
    ]

    N = params["means"].shape[0]
    H, W = scene.H, scene.W
    train_mask = ~scene.is_test
    train_views = [
        (fi, ci)
        for fi in range(scene.n_frames)
        if train_mask[fi]
        for ci in range(scene.n_cams)
    ]
    test_frame_ids = [i for i in range(scene.n_frames) if scene.is_test[i]]
    n_views = len(train_views)

    print(
        f"Device: {device} | {scene.n_frames} frames, {scene.n_cams} cameras, "
        f"{scene.H}x{scene.W} | {len(scene.lidar_points)} LiDAR pts | {N} Gaussians"
    )

    losses_history: list[float] = []
    checkpoints: list[dict] = []
    start_time = time.time()

    # --- Training loop ---
    for step in range(max_steps):
        # Select training view (cycle through all frame-camera pairs)
        idxframe, idxcam = train_views[step % n_views]
        rgb_gt = scene.images[idxframe, idxcam].unsqueeze(0)  # [1, H, W, 3]
        viewmat = scene.viewmats[idxframe, idxcam].unsqueeze(0)  # [1, 4, 4]
        K = scene.Ks[idxcam]  # [1, 3, 3]

        # Progressive SH schedule
        sh_degree_to_use = min(step // sh_degree_interval, sh_degree) if sh_degree > 0 else None

        # Forward pass: rasterize Gaussians into this view
        renders, alphas = render_gaussians(
            params, viewmat, K, W, H, "RGB+ED", sh_degree_to_use=sh_degree_to_use
        )
        rgb_render = renders[..., :3]  # [1, H, W, 3]
        depth_render = renders[0, :, :, 3]  # [H, W]

        # Compute composite loss (L1 + SSIM + depth + regularization + sky)
        total_loss = compute_loss(
            params, rgb_render, depth_render, alphas, rgb_gt,
            scene, idxframe, viewmat, K, H, W, device,
        )

        # Backward pass and parameter update
        for opt in optimizers.values():
            opt.zero_grad()
        total_loss.backward()
        for opt in optimizers.values():
            opt.step()
        for sched in schedulers:
            sched.step()
        losses_history.append(total_loss.item())

        # MCMC: relocate dead Gaussians, add new ones, inject noise
        if strategy is not None:
            strategy.step_post_backward(
                params, optimizers, strategy_state, step,
                info={}, lr=schedulers[0].get_last_lr()[0],
            )

        N = len(params["means"])  # may change with MCMC

        # Periodic logging and evaluation
        if step % log_every == 0:
            log_training_step(step, max_steps, total_loss, start_time)
        if step % eval_every == 0 or step == max_steps - 1:
            ckpt = run_evaluation(
                params, scene, test_frame_ids, W, H,
                step, total_loss, start_time, N,
                renders_dir, stats_dir, is_last_step=(step == max_steps - 1),
                sh_degree=sh_degree_to_use,
            )
            checkpoints.append(ckpt)

    # --- Summary ---
    elapsed = time.time() - start_time
    summary = {
        "max_steps": max_steps,
        "total_time_s": elapsed,
        "initial_loss": losses_history[0] if losses_history else None,
        "final_loss": losses_history[-1] if losses_history else None,
        "num_gaussians": N,
        "checkpoints": checkpoints,
    }
    with open(os.path.join(result_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if save_model:
        model_path = os.path.join(result_dir, "model.pt")
        torch.save({k: v.detach().cpu() for k, v in params.items()}, model_path)
        print(f"Model saved to {model_path} ({N} Gaussians)")

    print(f"\nDone: {max_steps} steps in {elapsed:.1f}s")
    print(f"Results saved to {result_dir}/")

    return losses_history, checkpoints


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Gaussians on multi-camera autonomous driving data"
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../assets/test_pandaset.npz"),
        help="path to the npz scene file (images, intrinsics, poses, LiDAR)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=15000,
        help="number of training iterations (default: 15000)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.005,
        help="base learning rate; per-param LRs are scaled from this (default: 0.005)",
    )
    parser.add_argument(
        "--log-every", type=int, default=3000,
        help="print training stats every N steps (default: 3000)",
    )
    parser.add_argument(
        "--eval-every", type=int, default=3000,
        help="evaluate PSNR on held-out test views every N steps (default: 3000)",
    )
    parser.add_argument(
        "--result-dir", type=str, default="results/av_pandaset",
        help="output directory for renders, stats, and summary (default: results/av_pandaset)",
    )
    parser.add_argument(
        "--sh-degree", type=int, default=0,
        help="spherical harmonics degree (0 = flat RGB, 3 = full SH; default: 0)",
    )
    parser.add_argument(
        "--sh-degree-interval", type=int, default=1000,
        help="progressively enable one more SH band every N steps (default: 1000)",
    )
    parser.add_argument(
        "--no-save-model", action="store_true",
        help="disable saving trained Gaussian parameters to model.pt",
    )
    parser.add_argument(
        "--mcmc", action="store_true",
        help="enable MCMC densification strategy: adaptively add/remove Gaussians",
    )
    parser.add_argument(
        "--cap-max", type=int, default=500_000,
        help="maximum number of Gaussians when using --mcmc (default: 500000)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device required. gsplat rasterization relies on CUDA kernels "
            "and does not support CPU-only execution."
        )
    train(
        scene_path=args.scene,
        max_steps=args.max_steps,
        lr=args.lr,
        log_every=args.log_every,
        eval_every=args.eval_every,
        result_dir=args.result_dir,
        use_mcmc=args.mcmc,
        cap_max=args.cap_max,
        sh_degree=args.sh_degree,
        sh_degree_interval=args.sh_degree_interval,
        save_model=not args.no_save_model,
    )


if __name__ == "__main__":
    main()
