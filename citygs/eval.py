"""Evaluate a (merged) checkpoint on the val split: PSNR / SSIM (+LPIPS).

Run: ``python -m citygs.eval --manifest ... --ckpt results/merged/merged.pt``
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from gsplat.losses import ssim_loss
from gsplat.rendering import rasterization

from .ckpt import load_ckpt
from .config import EvalConfig, load_config
from .scene.dataset import ManifestDataset
from .scene.manifest import SceneManifest


@torch.no_grad()
def run_eval(cfg: EvalConfig) -> dict:
    manifest = SceneManifest.load(cfg.manifest)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats, meta = load_ckpt(cfg.ckpt, device=device)
    sh_degree = int(np.sqrt(splats["sh0"].shape[1] + splats["shN"].shape[1])) - 1

    valset = ManifestDataset(manifest, manifest.val_indices(), cache_size=8)
    loader = torch.utils.data.DataLoader(
        valset, batch_size=1, shuffle=False, num_workers=2
    )

    lpips_fn = None
    if cfg.lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)

    os.makedirs(cfg.result_dir, exist_ok=True)
    if cfg.save_images:
        os.makedirs(os.path.join(cfg.result_dir, "renders"), exist_ok=True)

    psnrs, ssims, lpipss = [], [], []
    for i, data in enumerate(loader):
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        colors, _, _ = rasterization(
            means=splats["means"],
            quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            rasterize_mode="antialiased",
        )
        colors = colors[..., :3].clamp(0.0, 1.0)
        mse = F.mse_loss(colors, pixels)
        psnrs.append((10.0 * torch.log10(1.0 / mse)).item())
        ssims.append(
            1.0
            - ssim_loss(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)).item()
        )
        if lpips_fn is not None:
            lpipss.append(
                lpips_fn(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)).item()
            )
        if cfg.save_images:
            import imageio.v2 as imageio

            canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
            imageio.imwrite(
                os.path.join(cfg.result_dir, "renders", f"val_{i:04d}.png"),
                (canvas * 255).astype(np.uint8),
            )

    stats = {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "num_gaussians": len(splats["means"]),
        "num_val_images": len(psnrs),
    }
    if lpipss:
        stats["lpips"] = float(np.mean(lpipss))
    with open(os.path.join(cfg.result_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[eval] {stats}")
    return stats


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) >= 2 and argv[0] == "--config-json":
        cfg = load_config(EvalConfig, argv[1])
    else:
        import tyro

        cfg = tyro.cli(EvalConfig, args=argv)
    run_eval(cfg)


if __name__ == "__main__":
    main()
