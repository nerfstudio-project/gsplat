"""Pure-torch drop-in for `fused_ssim` (the rahul-goel/fused-ssim CUDA extension).

`examples/simple_trainer.py` imports `from fused_ssim import fused_ssim` and uses it
ONLY for the training SSIM **loss** term (`1 - fused_ssim(pred, gt, padding="valid")`).
The upstream package is a CUDA kernel that doesn't build cleanly for ROCm/gfx1151, so
on this box we put this module on PYTHONPATH instead. It computes the standard
11x11 Gaussian-window SSIM (sigma=1.5, C1=0.01^2, C2=0.03^2, data range 1.0) — the
same definition fused-ssim implements — in differentiable torch ops.

This affects only the loss (training dynamics), not the reported benchmark metrics:
simple_trainer's eval PSNR/SSIM/LPIPS are computed with torchmetrics, independent of
this shim. Slower than the fused kernel, but correct and arch-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])  # [w, w]


def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "same",
    train: bool = True,
) -> torch.Tensor:
    """SSIM between two NCHW images in [0,1]; returns the mean SSIM (scalar).

    padding="valid" -> no padding (matches simple_trainer's call); "same" -> keep size.
    """
    assert img1.shape == img2.shape, (img1.shape, img2.shape)
    C = img1.shape[-3]
    win = _gaussian_window(11, 1.5).to(device=img1.device, dtype=img1.dtype)
    window = win.expand(C, 1, 11, 11).contiguous()
    pad = 0 if padding == "valid" else 11 // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=C)
    mu2 = F.conv2d(img2, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=C) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()
