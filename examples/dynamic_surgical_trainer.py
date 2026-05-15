"""Dynamic surgical scene trainer (G-SHARP v0.2 port, experimental).

End-to-end recipe wiring together every module landed in TDD steps 1–8:

- :class:`examples.datasets.endonerf.EndoNeRFParser` /
  :class:`EndoNeRFDataset` for on-disk data.
- :func:`gsplat.init_utils.multi_frame_depth_unprojection` for SfM-free
  point-cloud initialisation.
- :func:`gsplat.init_utils.knn_scale_init` for per-Gaussian log-scale init.
- :class:`gsplat.contrib.dynamic.HexPlaneField` +
  :class:`DeformNetwork` for per-step deformation of
  ``(means, quats, opacities)``.
- :class:`gsplat.contrib.dynamic.DynamicStrategy` for densification +
  :class:`DeformationTable` bookkeeping.
- Mask-aware depth + photometric losses from :mod:`gsplat.losses` and the
  TV / smoothness regularizers from :mod:`gsplat.regularizers` +
  :mod:`gsplat.contrib.dynamic.regulation`.
- :class:`gsplat.training.TwoStageScheduler` for the coarse → fine schedule.

The MVP exposed here is the integration test surface: build all components
from a config + parser, then run training-step iterations. No eval /
checkpointing / viz in v1 — those land in follow-ups.

CUDA is required at runtime (``gsplat.rasterization`` is CUDA-only).
Component construction (build_* helpers) and config validation work on CPU
and are covered by ``tests/test_dynamic_surgical_trainer.py``; the actual
training step is covered by a CUDA-only integration test in the same file.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch import Tensor

from gsplat import rasterization
from gsplat.contrib.dynamic import (
    DeformNetwork,
    DynamicStrategy,
    HexPlaneField,
    plane_smoothness,
    time_l1,
    time_smoothness,
)
from gsplat.init_utils import knn_scale_init, multi_frame_depth_unprojection
from gsplat.losses import (
    binocular_disparity_l1,
    masked_l1,
    masked_ssim,
    pearson_depth_loss,
)
from gsplat.regularizers import compute_tv_loss_targeted
from gsplat.training import TwoStageScheduler

# Make the trainer runnable as `python examples/dynamic_surgical_trainer.py`
# from the repo root (examples/ isn't installed).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from examples.datasets.endonerf import EndoNeRFDataset, EndoNeRFParser  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """CLI / programmatic config for the dynamic surgical trainer."""

    data_dir: Path = Path("./data/EndoNeRF/pulling")
    output_dir: Path = Path("./output/dynamic_surgical")

    # Schedule
    coarse_steps: int = 500
    fine_steps: int = 3000

    # SfM-free initialisation
    init_max_points: int = 50_000
    init_scale_multiplier: float = 1.0  # G-SHARP convention: scales = log(knn_dist * init_scale)
    init_opacity: float = 0.1

    # HexPlane field
    hex_bounds: float = 1.6
    hex_output_dim: int = 32
    hex_resolution: Tuple[int, int, int, int] = (64, 64, 64, 25)
    hex_multires: Tuple[int, ...] = (1, 2)

    # Deformation MLP
    deform_hidden_dim: int = 64
    deform_num_layers: int = 3

    # Loss weights
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_depth: float = 1.0
    lambda_tv: float = 1e-3
    lambda_plane_smooth: float = 1e-4
    lambda_time_smooth: float = 1e-4
    lambda_time_l1: float = 1e-4

    # Optimisation
    lr_means: float = 1.6e-4
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    lr_hexplane: float = 5e-3
    lr_deform_mlp: float = 1.6e-4

    # Loss config
    depth_mode: str = "binocular"  # 'binocular' or 'monocular'

    # Misc
    seed: int = 42
    device: str = "cuda"

    # Render-GIF (post-training)
    render_gif_after_train: bool = False
    gif_filename: str = "preview.gif"
    gif_fps: int = 10
    gif_frame_indices: Optional[Tuple[int, ...]] = None  # None = parser.video_idxs

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        if self.depth_mode not in ("binocular", "monocular"):
            raise ValueError(
                f"Config: depth_mode must be 'binocular' or 'monocular', "
                f"got {self.depth_mode!r}."
            )
        if self.coarse_steps < 0 or self.fine_steps < 0:
            raise ValueError("Config: step counts must be non-negative.")


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _load_train_tensors(
    parser: EndoNeRFParser, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Materialise the train-split frames into ``(N, ...)`` tensors on *device*."""
    from PIL import Image

    idxs = parser.train_idxs
    images_np: list[np.ndarray] = []
    depths_np: list[np.ndarray] = []
    masks_np: list[np.ndarray] = []
    for idx in idxs:
        images_np.append(
            np.array(Image.open(parser.image_paths[idx])).astype(np.float32) / 255.0
        )
        depths_np.append(np.array(Image.open(parser.depth_paths[idx])).astype(np.float32))
        mask_raw = np.array(Image.open(parser.mask_paths[idx]))
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]
        # G-SHARP convention (gsplat_train.py:2102):
        # ``1 - mask/255`` → on-disk 255 (tool) becomes 0, on-disk 0 (tissue)
        # becomes 1. The returned mask is a *tissue / include* mask
        # (``1 = keep this pixel``, ``0 = it's a tool, drop``).
        masks_np.append(1.0 - mask_raw.astype(np.float32) / 255.0)

    images = torch.from_numpy(np.stack(images_np)).to(device)
    depths = torch.from_numpy(np.stack(depths_np)).to(device)
    masks = torch.from_numpy(np.stack(masks_np)).to(device)
    poses = torch.from_numpy(parser.camtoworlds[idxs]).to(device)
    intrinsics = torch.from_numpy(parser.K).unsqueeze(0).expand(len(idxs), -1, -1).to(device)
    return images, depths, masks, poses, intrinsics


def build_splats_from_parser(
    parser: EndoNeRFParser, cfg: Config, device: torch.device
) -> Tuple[nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Initialise splat parameters + optimizers from a parsed EndoNeRF directory.

    Pipeline: multi-frame depth unprojection across the train split →
    optional random subsample to ``cfg.init_max_points`` → kNN log-scale
    init (with G-SHARP's ``log(knn_dist * init_scale_multiplier)`` rescaling).
    Quaternions are random-normal then unit-normalised; opacities are
    initialised in logit space to match ``cfg.init_opacity`` after sigmoid;
    colors are taken from the unprojected pixel RGBs.

    Returns:
        ``(params, optimizers)`` where *params* is a ``ParameterDict`` with
        keys ``means``, ``quats``, ``scales``, ``opacities``, ``colors``,
        ``hexplane_params``, ``deform_mlp_params``. The last two are
        bookkeeping placeholders (zero-d Parameters) that satisfy
        :meth:`DynamicStrategy.check_sanity`; the real HexPlane and
        DeformNet trainables are wired separately via
        :func:`build_deform_modules`.
    """
    images, depths, masks, poses, intrinsics = _load_train_tensors(parser, device)

    # ``masks`` is already in tissue=1 / tool=0 convention (see
    # :func:`_load_train_tensors`), which is exactly the include-mask that
    # :func:`multi_frame_depth_unprojection` expects (keep where ``!= 0``).
    xyz, rgb = multi_frame_depth_unprojection(
        images=images,
        depths=depths,
        masks=masks,
        poses=poses,
        intrinsics=intrinsics,
        max_points=cfg.init_max_points,
    )
    n = xyz.shape[0]
    if n < 4:  # need at least k+1 = 4 points for default knn_scale_init.
        raise RuntimeError(
            f"build_splats_from_parser: only {n} valid points after "
            f"unprojection — too few to initialise. Check that the dataset "
            f"depth + mask combination yields valid pixels."
        )

    log_scales_1d = knn_scale_init(xyz, k=3) + math.log(cfg.init_scale_multiplier)
    log_scales = log_scales_1d.unsqueeze(-1).repeat(1, 3).contiguous()
    quats = torch.randn(n, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    # Init opacities in logit space so sigmoid(...) == init_opacity.
    logit_opa = math.log(cfg.init_opacity / max(1.0 - cfg.init_opacity, 1e-12))
    opacities = torch.full((n, 1), logit_opa, device=device)

    # Strategy params are *per-Gaussian only* — gsplat densification ops
    # iterate every entry and split/duplicate/prune per-Gaussian. HexPlane
    # plane grids and DeformNet MLP weights belong to separate optimizers
    # managed by :func:`build_deform_modules`.
    params = nn.ParameterDict(
        {
            "means": nn.Parameter(xyz.contiguous()),
            "quats": nn.Parameter(quats),
            "scales": nn.Parameter(log_scales),
            "opacities": nn.Parameter(opacities),
            "colors": nn.Parameter(rgb.contiguous()),
        }
    )
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=cfg.lr_means),
        "quats": torch.optim.Adam([params["quats"]], lr=cfg.lr_quats),
        "scales": torch.optim.Adam([params["scales"]], lr=cfg.lr_scales),
        "opacities": torch.optim.Adam([params["opacities"]], lr=cfg.lr_opacities),
        "colors": torch.optim.Adam([params["colors"]], lr=cfg.lr_colors),
    }
    return params, optimizers


def build_deform_modules(
    cfg: Config, device: torch.device
) -> Tuple[HexPlaneField, DeformNetwork, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Construct the HexPlane field, the DeformNet, and their optimizers."""
    plane_cfg = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": cfg.hex_output_dim,
        "resolution": list(cfg.hex_resolution),
    }
    hexplane = HexPlaneField(
        bounds=cfg.hex_bounds,
        planes_config=plane_cfg,
        multires=cfg.hex_multires,
    ).to(device)
    deform_net = DeformNetwork(
        feature_dim=hexplane.feat_dim,
        hidden_dim=cfg.deform_hidden_dim,
        num_layers=cfg.deform_num_layers,
    ).to(device)

    hex_optimizer = torch.optim.Adam(hexplane.parameters(), lr=cfg.lr_hexplane)
    deform_optimizer = torch.optim.Adam(deform_net.parameters(), lr=cfg.lr_deform_mlp)
    return hexplane, deform_net, hex_optimizer, deform_optimizer


# ---------------------------------------------------------------------------
# Train step (CUDA-only — gsplat.rasterization is CUDA-only)
# ---------------------------------------------------------------------------


def train_step(
    cfg: Config,
    item: Dict[str, Tensor],
    params: nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    hexplane: HexPlaneField,
    deform_net: DeformNetwork,
    hex_optimizer: torch.optim.Optimizer,
    deform_optimizer: torch.optim.Optimizer,
    strategy: DynamicStrategy,
    state: Dict[str, Any],
    step: int,
) -> Dict[str, float]:
    """One end-to-end training step. Returns scalar loss components for logging."""
    device = torch.device(cfg.device)

    image_gt = item["image"].to(device).unsqueeze(0)
    depth_gt = item["depth"].to(device).unsqueeze(0)
    mask = item["mask"].to(device).unsqueeze(0)
    camtoworld = item["camtoworld"].to(device).unsqueeze(0)
    K = item["K"].to(device).unsqueeze(0)
    t = item["time"].to(device).view(1)

    H, W = image_gt.shape[1:3]

    means = params["means"]
    quats = params["quats"]
    log_scales = params["scales"]
    opacities_logit = params["opacities"]
    colors = params["colors"]

    # Build per-Gaussian (xyzt) input for HexPlane.
    n = means.shape[0]
    t_per_g = t.expand(n).unsqueeze(-1)
    xyzt = torch.cat([means, t_per_g], dim=-1)

    plane_features = hexplane(xyzt)
    means_d, quats_d, opacities_d_logit = deform_net(
        means, quats, opacities_logit, t_per_g, plane_features
    )

    viewmats = torch.linalg.inv(camtoworld)
    render_colors, _, info = rasterization(
        means=means_d,
        quats=quats_d,
        scales=torch.exp(log_scales),
        opacities=torch.sigmoid(opacities_d_logit).squeeze(-1),
        colors=colors,
        viewmats=viewmats,
        Ks=K,
        width=W,
        height=H,
        sh_degree=None,
        render_mode="RGB+ED",
        packed=True,  # match the strategy's step_post_backward(packed=True) below
    )
    image_rendered = render_colors[..., :3]
    depth_rendered = render_colors[..., 3:4].squeeze(-1)

    rgb_p = image_rendered.permute(0, 3, 1, 2)
    gt_p = image_gt.permute(0, 3, 1, 2)
    # ``mask`` is already tissue=1 / tool=0 (the include-mask convention the
    # mask-aware loss helpers expect). No inversion needed.
    mask_p = mask.unsqueeze(1)

    l1 = masked_l1(rgb_p, gt_p, mask_p)
    ssim = masked_ssim(rgb_p, gt_p, mask_p)
    if cfg.depth_mode == "binocular":
        d_loss = binocular_disparity_l1(depth_rendered, depth_gt, mask)
    else:
        d_loss = pearson_depth_loss(depth_rendered, depth_gt, mask)

    tv = compute_tv_loss_targeted(rgb_p)  # unmasked TV (mask wiring is dataset-specific)

    spatial_planes: list[Tensor] = []
    temporal_planes: list[Tensor] = []
    for scale_grids in hexplane.grids:
        for i in (0, 1, 3):  # xy, xz, yz
            spatial_planes.append(scale_grids[i])
        for i in (2, 4, 5):  # xt, yt, zt
            temporal_planes.append(scale_grids[i])
    plane_smooth = plane_smoothness(spatial_planes)
    time_smooth_v = time_smoothness(temporal_planes)
    time_l1_v = time_l1(temporal_planes)

    loss = (
        cfg.lambda_l1 * l1
        + cfg.lambda_ssim * ssim
        + cfg.lambda_depth * d_loss
        + cfg.lambda_tv * tv
        + cfg.lambda_plane_smooth * plane_smooth
        + cfg.lambda_time_smooth * time_smooth_v
        + cfg.lambda_time_l1 * time_l1_v
    )

    info_with_dims = dict(info)
    info_with_dims["width"] = W
    info_with_dims["height"] = H
    info_with_dims["n_cameras"] = 1

    strategy.step_pre_backward(
        params=params, optimizers=optimizers, state=state, step=step, info=info_with_dims
    )

    for opt in optimizers.values():
        opt.zero_grad(set_to_none=True)
    hex_optimizer.zero_grad(set_to_none=True)
    deform_optimizer.zero_grad(set_to_none=True)

    loss.backward()

    for opt in optimizers.values():
        opt.step()
    hex_optimizer.step()
    deform_optimizer.step()

    strategy.step_post_backward(
        params=params,
        optimizers=optimizers,
        state=state,
        step=step,
        info=info_with_dims,
        packed=True,
    )

    return {
        "loss": float(loss.detach().item()),
        "l1": float(l1.detach().item()),
        "ssim": float(ssim.detach().item()),
        "depth": float(d_loss.detach().item()),
        "tv": float(tv.detach().item()),
        "plane_smooth": float(plane_smooth.detach().item()),
        "time_smooth": float(time_smooth_v.detach().item()),
        "time_l1": float(time_l1_v.detach().item()),
    }


# ---------------------------------------------------------------------------
# Render helpers (post-training)
# ---------------------------------------------------------------------------


@torch.no_grad()
def render_frame(
    parser: EndoNeRFParser,
    frame_idx: int,
    params: nn.ParameterDict,
    hexplane: HexPlaneField,
    deform_net: DeformNetwork,
    device: torch.device,
) -> np.ndarray:
    """Render a single frame at its native time. Returns ``(H, W, 3)`` uint8."""
    camtoworld = torch.from_numpy(parser.camtoworlds[frame_idx]).to(device).unsqueeze(0)
    K = torch.from_numpy(parser.K).to(device).unsqueeze(0)
    t = torch.tensor(float(parser.times[frame_idx]), device=device).view(1)

    means = params["means"]
    quats = params["quats"]
    log_scales = params["scales"]
    opacities_logit = params["opacities"]
    colors = params["colors"]

    n = means.shape[0]
    t_per_g = t.expand(n).unsqueeze(-1)
    xyzt = torch.cat([means, t_per_g], dim=-1)
    plane_features = hexplane(xyzt)
    means_d, quats_d, opacities_d_logit = deform_net(
        means, quats, opacities_logit, t_per_g, plane_features
    )

    viewmats = torch.linalg.inv(camtoworld)
    render_colors, _, _ = rasterization(
        means=means_d,
        quats=quats_d,
        scales=torch.exp(log_scales),
        opacities=torch.sigmoid(opacities_d_logit).squeeze(-1),
        colors=colors,
        viewmats=viewmats,
        Ks=K,
        width=parser.width,
        height=parser.height,
        sh_degree=None,
        render_mode="RGB+ED",
        packed=True,
    )
    image = render_colors[..., :3].squeeze(0).clamp(0.0, 1.0)
    return (image * 255).to(torch.uint8).cpu().numpy()


def render_gif(
    parser: EndoNeRFParser,
    params: nn.ParameterDict,
    hexplane: HexPlaneField,
    deform_net: DeformNetwork,
    output_path: Path,
    device: torch.device,
    frame_indices: Optional[list[int]] = None,
    fps: int = 10,
) -> Path:
    """Render frames across the dataset and write an animated GIF.

    Args:
        parser: A parsed :class:`EndoNeRFParser`.
        params / hexplane / deform_net: Trained trainer components.
        output_path: Destination GIF path. Parent is created if missing.
        device: CUDA device.
        frame_indices: Iterable of parser-relative frame indices to render.
            Defaults to ``parser.video_idxs`` (every frame).
        fps: Output GIF frames-per-second.

    Returns:
        The resolved output path.
    """
    import imageio.v2 as imageio

    if frame_indices is None:
        frame_indices = list(parser.video_idxs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    for idx in tqdm.tqdm(frame_indices, desc=f"rendering -> {output_path.name}"):
        frames.append(render_frame(parser, idx, params, hexplane, deform_net, device))

    imageio.mimsave(str(output_path), frames, duration=max(1, int(1000 / fps)), loop=0)
    print(f"Saved GIF: {output_path} ({len(frames)} frames @ {fps} fps)")
    return output_path


# ---------------------------------------------------------------------------
# Train loop / entry point
# ---------------------------------------------------------------------------


def train(cfg: Config) -> None:
    """Run the full coarse → fine training loop."""
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    parser = EndoNeRFParser(data_dir=cfg.data_dir)
    dataset = EndoNeRFDataset(parser, split="train")

    params, optimizers = build_splats_from_parser(parser, cfg, device)
    hexplane, deform_net, hex_opt, deform_opt = build_deform_modules(cfg, device)

    strategy = DynamicStrategy()
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(
        scene_scale=1.0, num_gaussians=int(params["means"].shape[0]), device=device
    )

    scheduler = TwoStageScheduler(
        coarse_steps=cfg.coarse_steps,
        fine_steps=cfg.fine_steps,
    )
    total_steps = cfg.coarse_steps + cfg.fine_steps

    pbar = tqdm.tqdm(range(total_steps), desc="dynamic_surgical_trainer")
    for step in pbar:
        sched = scheduler.step(global_step=step, num_frames=len(dataset))
        item = dataset[sched.frame_index]
        losses = train_step(
            cfg=cfg,
            item=item,
            params=params,
            optimizers=optimizers,
            hexplane=hexplane,
            deform_net=deform_net,
            hex_optimizer=hex_opt,
            deform_optimizer=deform_opt,
            strategy=strategy,
            state=state,
            step=step,
        )
        if step % 50 == 0:
            pbar.set_postfix(
                stage=sched.stage,
                loss=f"{losses['loss']:.4f}",
                n=int(params["means"].shape[0]),
            )

    if cfg.render_gif_after_train:
        gif_path = cfg.output_dir / cfg.gif_filename
        render_gif(
            parser=parser,
            params=params,
            hexplane=hexplane,
            deform_net=deform_net,
            output_path=gif_path,
            device=device,
            frame_indices=list(cfg.gif_frame_indices)
            if cfg.gif_frame_indices is not None
            else None,
            fps=cfg.gif_fps,
        )


def main() -> None:
    import tyro

    cfg = tyro.cli(Config)
    train(cfg)


if __name__ == "__main__":
    main()
