"""Partition stage: contraction bounds -> grid blocks -> Gaussian and
camera assignment -> scheduling costs. Everything is derived from the
coarse checkpoint and written back to the manifest, so the finetune stage
only loads.

Run: ``python -m citygs.partition.run --manifest ... [--config-json ...]``
"""

import os
import sys

import numpy as np
import torch

from ..ckpt import load_ckpt
from ..config import PartitionConfig, load_config
from ..scene.manifest import Contraction, SceneManifest
from .assign import assign_cameras_to_blocks, block_cost
from .contraction import contract, foreground_bounds
from .partition import assign_gaussians_to_blocks, make_grid_blocks


def _device(pref: str) -> str:
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_partition(cfg: PartitionConfig) -> None:
    manifest = SceneManifest.load(cfg.manifest)
    ckpt_path = cfg.coarse_ckpt or manifest.artifact("coarse_ckpt")
    splats, _ = load_ckpt(ckpt_path)
    device = _device(cfg.device)

    # 1. Foreground box -> contraction.
    positions = manifest.camera_positions()
    if cfg.fg_bounds:
        assert len(cfg.fg_bounds) == 6, "--fg-bounds needs cx cy cz ex ey ez"
        center = np.asarray(cfg.fg_bounds[:3], dtype=np.float64)
        half_extent = np.asarray(cfg.fg_bounds[3:], dtype=np.float64)
    else:
        center, half_extent = foreground_bounds(
            positions, mad_threshold=cfg.mad_threshold, margin=cfg.fg_margin
        )
    manifest.contraction = Contraction(
        center=center.tolist(), half_extent=half_extent.tolist()
    )
    print(
        f"[partition] foreground center={center.round(3)} half_extent={half_extent.round(3)}"
    )

    # 2. Grid blocks + Gaussian-to-block assignment (contracted space).
    means = splats["means"].double().numpy()
    contracted = contract(means, center, half_extent)
    blocks = make_grid_blocks(cfg.grid)
    gaussian_block_ids = assign_gaussians_to_blocks(contracted, blocks)

    os.makedirs(cfg.result_dir, exist_ok=True)
    gid_path = os.path.join(cfg.result_dir, "gaussian_block_ids.npy")
    np.save(gid_path, gaussian_block_ids)

    # 3. Camera-to-block assignment (train split only).
    train_idx = manifest.train_indices()
    cams = [manifest.cameras[i] for i in train_idx]
    camtoworlds = torch.tensor(
        np.stack([c.camtoworld_np() for c in cams]), dtype=torch.float32, device=device
    )
    Ks = torch.tensor(
        np.stack([c.K_np() for c in cams]), dtype=torch.float32, device=device
    )
    image_sizes = torch.tensor(
        [[c.width, c.height] for c in cams], dtype=torch.float32, device=device
    )
    cam_contracted = contract(positions[train_idx], center, half_extent)

    contrib = assign_cameras_to_blocks(
        blocks,
        gaussian_block_ids,
        splats["means"].to(device),
        splats["scales"].exp().to(device),
        torch.sigmoid(splats["opacities"]).to(device),
        camtoworlds,
        Ks,
        image_sizes,
        camera_indices=train_idx,
        cam_contracted=cam_contracted,
        coverage_threshold=cfg.coverage_threshold,
        min_cameras=cfg.min_cameras,
        max_proxy_gaussians=cfg.max_proxy_gaussians,
        camera_chunk=cfg.camera_chunk,
        inside_margin=cfg.block_margin / 2,
    )
    np.save(os.path.join(cfg.result_dir, "block_camera_contrib.npy"), contrib)

    # 4. Scheduling costs.
    for block in blocks:
        block.cost = block_cost(block)

    manifest.blocks = blocks
    manifest.set_artifact("gaussian_block_ids", gid_path)
    manifest.save(cfg.manifest)

    print(f"[partition] {len(blocks)} blocks over {len(train_idx)} train cameras:")
    for b in blocks:
        print(
            f"  block {b.block_id:3d} grid={b.grid_index} "
            f"gaussians={b.num_gaussians:>9,} cameras={len(b.cameras):>5} "
            f"cost={b.cost:,.0f}"
        )

    # 5. Top-down overview figure (optional, needs matplotlib).
    try:
        from ..viz.overview import save_partition_overview

        png = os.path.join(cfg.result_dir, "partition.png")
        save_partition_overview(
            manifest, contracted, gaussian_block_ids, cam_contracted, png
        )
        manifest.set_artifact("partition_png", png)
        manifest.save(cfg.manifest)
        print(f"[partition] overview -> {png}")
    except ImportError:
        print("[partition] matplotlib not installed; skipping overview figure")


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--config-json":
        cfg = load_config(PartitionConfig, argv[1])
    else:
        import tyro

        cfg = tyro.cli(PartitionConfig, args=argv)
    run_partition(cfg)


if __name__ == "__main__":
    main()
