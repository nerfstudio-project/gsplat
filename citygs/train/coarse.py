"""Coarse global prior with the MCMC strategy (fixed Gaussian budget).

Launch::

    # Single GPU (the PCIe-friendly fallback: buy budget with resolution)
    CUDA_VISIBLE_DEVICES=0 python -m citygs.train.coarse --manifest ...

    # Grendel-style sharding across all visible GPUs (gsplat distributed
    # rasterization, per-iteration all-to-all — measure this on PCIe boxes)
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m citygs.train.coarse ...

MCMC instead of default densification because a fixed total budget (a)
uses all the VRAM you give it and no more, and (b) keeps per-rank counts
balanced in the distributed case, so no single GPU becomes the straggler.

After training, rank checkpoints are consolidated into one ``coarse.pt``
recorded in the manifest.
"""

import glob
import os
import sys

import torch

from gsplat.strategy import MCMCStrategy

from ..ckpt import concat_splats, load_ckpt, save_ckpt
from ..config import CoarseConfig, load_config
from ..scene.dataset import ManifestDataset
from ..scene.manifest import SceneManifest
from ..utils import init_splats_from_points
from .trainer import Trainer


def _train_worker(local_rank: int, world_rank: int, world_size: int, cfg: CoarseConfig):
    manifest = SceneManifest.load(cfg.manifest)
    device = f"cuda:{local_rank}"

    trainset = ManifestDataset(
        manifest,
        manifest.train_indices(),
        load_depths=cfg.train.depth_loss,
        downsample=cfg.downsample,
        cache_size=cfg.train.image_cache_size,
    )
    valset = ManifestDataset(
        manifest,
        manifest.val_indices(),
        downsample=cfg.downsample,
        cache_size=8,
    )

    xyz, rgb, _ = manifest.load_points()
    points = torch.from_numpy(xyz).float()
    rgbs = torch.from_numpy(rgb).float() / 255.0
    if len(points) > cfg.max_init_points:
        sel = torch.randperm(len(points))[: cfg.max_init_points]
        points, rgbs = points[sel], rgbs[sel]
    splats_init = init_splats_from_points(
        points,
        rgbs,
        sh_degree=cfg.train.sh_degree,
        init_opacity=cfg.train.init_opacity,
        init_scale=cfg.train.init_scale,
        world_rank=world_rank,
        world_size=world_size,
    )

    # MCMC prunes via opacity/scale regularization; force them on.
    train_cfg = cfg.train
    if train_cfg.opacity_reg == 0.0:
        train_cfg.opacity_reg = cfg.opacity_reg
    if train_cfg.scale_reg == 0.0:
        train_cfg.scale_reg = cfg.scale_reg

    strategy = MCMCStrategy(
        cap_max=max(1, cfg.cap_max // world_size),  # budget is global
        noise_lr=cfg.mcmc_noise_lr,
        refine_start_iter=cfg.mcmc_refine_start_iter,
        refine_stop_iter=cfg.mcmc_refine_stop_iter,
        refine_every=cfg.mcmc_refine_every,
        min_opacity=cfg.mcmc_min_opacity,
    )

    trainer = Trainer(
        cfg=train_cfg,
        splats_init=splats_init,
        strategy=strategy,
        trainset=trainset,
        valset=valset,
        result_dir=cfg.result_dir,
        scene_scale=manifest.scene_scale * 1.1,
        device=device,
        world_rank=world_rank,
        world_size=world_size,
    )
    trainer.train(ckpt_meta={"stage": "coarse"})

    if world_size > 1:
        torch.distributed.barrier()


def consolidate(cfg: CoarseConfig) -> str:
    """Merge per-rank checkpoints into coarse.pt and record it in the
    manifest."""
    ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
    single = os.path.join(ckpt_dir, "ckpt.pt")
    if os.path.isfile(single):
        parts = [single]
    else:
        parts = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_rank*.pt")))
    assert parts, f"no checkpoints found in {ckpt_dir}"

    loaded = [load_ckpt(p) for p in parts]
    splats = concat_splats([s for s, _ in loaded])
    step = loaded[0][1]["step"]
    out = os.path.join(cfg.result_dir, "coarse.pt")
    save_ckpt(out, splats, step, {"stage": "coarse", "world_size": 1, "rank": 0})

    manifest = SceneManifest.load(cfg.manifest)
    manifest.set_artifact("coarse_ckpt", out)
    manifest.save(cfg.manifest)
    print(f"[coarse] {len(splats['means']):,} gaussians -> {out}")
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--config-json":
        cfg = load_config(CoarseConfig, argv[1])
    else:
        import tyro

        cfg = tyro.cli(CoarseConfig, args=argv)

    from gsplat.distributed import cli

    cli(_train_worker, cfg, verbose=True)
    consolidate(cfg)


if __name__ == "__main__":
    main()
