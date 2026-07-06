"""Finetune one block on one GPU. Zero inter-block communication.

Initialization: the coarse Gaussians inside the block's cell expanded by
``init_margin`` (context from neighbors helps boundary consistency).
After training, the model is cropped back to the exact cell, so the merge
stage is a plain concatenation — all blocks started from the same coarse
prior, which is what keeps the seams invisible.

Run (usually via the scheduler):
    python -m citygs.train.finetune_block --manifest ... --block-id 3
"""

import os
import sys

import numpy as np
import torch

from gsplat.strategy import DefaultStrategy

from ..ckpt import crop_splats, load_ckpt, save_ckpt
from ..config import BlockConfig, load_config
from ..partition.contraction import contract
from ..partition.partition import block_mask
from ..scene.dataset import ManifestDataset
from ..scene.manifest import SceneManifest


def block_ckpt_path(result_dir: str, block_id: int) -> str:
    return os.path.join(result_dir, f"block_{block_id:04d}.pt")


def compute_block_steps(cfg: BlockConfig, n_cameras: int, median_cameras: float) -> int:
    """Adaptive per-block step budget: heavy blocks get more iterations."""
    if not cfg.adaptive_steps or median_cameras <= 0:
        return cfg.train.max_steps
    factor = (n_cameras / median_cameras) ** 0.5
    factor = min(max(factor, cfg.min_steps_factor), cfg.max_steps_factor)
    return int(round(cfg.train.max_steps * factor))


def run_block(cfg: BlockConfig) -> str:
    manifest = SceneManifest.load(cfg.manifest)
    assert manifest.contraction is not None, "run the partition stage first"
    block = manifest.block(cfg.block_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = cfg.coarse_ckpt or manifest.artifact("coarse_ckpt")
    splats, _ = load_ckpt(ckpt_path)

    center = manifest.contraction.center
    half_extent = manifest.contraction.half_extent
    contracted = contract(splats["means"].double().numpy(), center, half_extent)
    init_mask = torch.from_numpy(block_mask(contracted, block, margin=cfg.init_margin))
    splats_init = crop_splats(splats, init_mask)
    n_init = len(splats_init["means"])
    assert n_init > 0, f"block {cfg.block_id} has no coarse Gaussians"

    train_indices = [i for i in block.cameras]
    valset_indices = manifest.val_indices()
    # Val cameras that look at this block: reuse the same membership rule.
    cam_contracted = contract(
        manifest.camera_positions()[valset_indices], center, half_extent
    )
    val_in = valset_indices[block_mask(cam_contracted, block, margin=cfg.init_margin)]

    trainset = ManifestDataset(
        manifest,
        train_indices,
        load_depths=cfg.train.depth_loss,
        cache_size=cfg.train.image_cache_size,
    )
    valset = ManifestDataset(manifest, val_in, cache_size=8) if len(val_in) else None

    cam_counts = [len(b.cameras) for b in manifest.blocks]
    steps = block.steps or compute_block_steps(
        cfg, len(train_indices), float(np.median(cam_counts))
    )

    strategy = DefaultStrategy(
        absgrad=cfg.absgrad,
        grow_grad2d=cfg.grow_grad2d,
        refine_start_iter=cfg.refine_start_iter,
        refine_stop_iter=min(cfg.refine_stop_iter, steps),
        refine_every=cfg.refine_every,
        reset_every=cfg.reset_every,
        verbose=False,
    )

    block_dir = os.path.join(cfg.result_dir, f"block_{cfg.block_id:04d}")
    from .trainer import Trainer

    print(
        f"[block {cfg.block_id}] init={n_init:,} gaussians, "
        f"{len(train_indices)} cameras, {steps} steps"
    )
    trainer = Trainer(
        cfg=cfg.train,
        splats_init=splats_init,
        strategy=strategy,
        trainset=trainset,
        valset=valset,
        result_dir=block_dir,
        scene_scale=manifest.scene_scale * 1.1,
        device=device,
    )
    trainer.train(
        max_steps=steps,
        ckpt_meta={
            "stage": "block",
            "block_id": cfg.block_id,
            "bmin": block.bmin,
            "bmax": block.bmax,
        },
    )

    # Final crop to the exact cell: Gaussians that drifted into neighbor
    # cells are dropped (the neighbor's own finetune owns that space).
    final = {k: v.detach() for k, v in trainer.splats.items()}
    contracted = contract(final["means"].double().cpu().numpy(), center, half_extent)
    keep = torch.from_numpy(block_mask(contracted, block))
    final = crop_splats({k: v.cpu() for k, v in final.items()}, keep)
    out = block_ckpt_path(cfg.result_dir, cfg.block_id)
    save_ckpt(
        out,
        final,
        steps,
        {
            "stage": "block",
            "block_id": cfg.block_id,
            "bmin": block.bmin,
            "bmax": block.bmax,
            "num_init": n_init,
            "num_final": len(final["means"]),
        },
    )
    print(
        f"[block {cfg.block_id}] done: {n_init:,} -> {len(final['means']):,} "
        f"gaussians -> {out}"
    )
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--config-json":
        cfg = load_config(BlockConfig, argv[1])
        extra = argv[2:]
        # allow "--config-json cfg.json --block-id N" from the scheduler
        if len(extra) >= 2 and extra[0] == "--block-id":
            cfg.block_id = int(extra[1])
    else:
        import tyro

        cfg = tyro.cli(BlockConfig, args=argv)
    run_block(cfg)


if __name__ == "__main__":
    main()
