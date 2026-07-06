"""Merge stage: concatenate the per-block models.

Each block checkpoint is already cropped to its own cell, and the cells
tile contracted space exactly, so the merge is a plain concat. The crop
is re-applied here anyway (idempotent) as a guard against stale or
hand-edited block checkpoints.

Run: ``python -m citygs.merge --manifest ... --blocks-dir ...``
"""

import os
import sys

import torch

from .ckpt import concat_splats, crop_splats, load_ckpt, save_ckpt
from .config import MergeConfig, load_config
from .partition.contraction import contract
from .partition.partition import block_mask
from .scene.manifest import SceneManifest
from .train.finetune_block import block_ckpt_path


def run_merge(cfg: MergeConfig) -> str:
    manifest = SceneManifest.load(cfg.manifest)
    assert manifest.blocks, "no blocks in manifest — run the partition stage first"
    assert manifest.contraction is not None
    center = manifest.contraction.center
    half_extent = manifest.contraction.half_extent

    parts = []
    total_before = 0
    for block in manifest.blocks:
        path = block_ckpt_path(cfg.blocks_dir, block.block_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"missing checkpoint for block {block.block_id}: {path} — "
                "finish the finetune stage first"
            )
        splats, meta = load_ckpt(path)
        total_before += len(splats["means"])
        contracted = contract(splats["means"].double().numpy(), center, half_extent)
        keep = torch.from_numpy(block_mask(contracted, block))
        cropped = crop_splats(splats, keep)
        if len(cropped["means"]) != len(splats["means"]):
            print(
                f"[merge] block {block.block_id}: cropped "
                f"{len(splats['means']) - len(cropped['means'])} stray gaussians"
            )
        parts.append(cropped)

    merged = concat_splats(parts)
    os.makedirs(cfg.result_dir, exist_ok=True)
    out = os.path.join(cfg.result_dir, "merged.pt")
    save_ckpt(
        out,
        merged,
        0,
        {"stage": "merged", "num_blocks": len(parts)},
    )
    print(f"[merge] {len(parts)} blocks, {len(merged['means']):,} gaussians -> {out}")

    if cfg.save_ply:
        from gsplat import export_splats

        ply = os.path.join(cfg.result_dir, "merged.ply")
        export_splats(
            means=merged["means"],
            scales=merged["scales"],
            quats=merged["quats"],
            opacities=merged["opacities"],
            sh0=merged["sh0"],
            shN=merged["shN"],
            format="ply",
            save_to=ply,
        )
        manifest.set_artifact("merged_ply", ply)
        print(f"[merge] ply -> {ply} (drop into SuperSplat to inspect)")

    manifest.set_artifact("merged_ckpt", out)
    manifest.save(cfg.manifest)
    return out


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--config-json":
        cfg = load_config(MergeConfig, argv[1])
    else:
        import tyro

        cfg = tyro.cli(MergeConfig, args=argv)
    run_merge(cfg)


if __name__ == "__main__":
    main()
