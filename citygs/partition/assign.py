"""Camera-to-block assignment via a batched contribution proxy.

The original CityGaussian decides "does camera i train block b" by
rendering each block for each camera and thresholding SSIM against the
full render — accurate but very slow. Here we use a much cheaper proxy
that needs no rasterization: the block's coarse Gaussians are projected
into the camera, and each contributes ``opacity * projected_area``. A
camera is assigned to a block when the accumulated contribution covers
enough of its image.

Guarantees layered on top of the threshold:
  * a camera physically inside a block's cell always trains it;
  * every camera trains at least one block (its argmax);
  * every block gets at least ``min_cameras`` cameras (top-k fallback).
"""

from typing import List, Optional

import numpy as np
import torch

from ..scene.manifest import BlockMeta
from .partition import block_mask


@torch.no_grad()
def block_camera_contribution(
    means: torch.Tensor,  # [N, 3] world, block subset
    scales: torch.Tensor,  # [N, 3] world units (already exp'd)
    opacities: torch.Tensor,  # [N] (already sigmoid'd)
    camtoworlds: torch.Tensor,  # [C, 4, 4]
    Ks: torch.Tensor,  # [C, 3, 3]
    image_sizes: torch.Tensor,  # [C, 2] (width, height)
    max_gaussians: int = 100_000,
    camera_chunk: int = 128,
    near: float = 1e-4,
) -> torch.Tensor:
    """Fraction of each camera's image covered by these Gaussians. [C]."""
    device = means.device
    N = means.shape[0]
    C = camtoworlds.shape[0]
    if N == 0 or C == 0:
        return torch.zeros(C, device=device)

    ratio = 1.0
    if N > max_gaussians:
        sel = torch.randperm(N, device=device)[:max_gaussians]
        means, scales, opacities = means[sel], scales[sel], opacities[sel]
        ratio = N / max_gaussians

    # Isotropic proxy radius: geometric mean of the three axes.
    radius = scales.clamp_min(1e-12).log().mean(dim=-1).exp()  # [n]
    weight0 = opacities * torch.pi * radius**2  # [n] world-space area x alpha

    w2c = torch.linalg.inv(camtoworlds)  # [C, 4, 4]
    out = torch.zeros(C, device=device)
    for s in range(0, C, camera_chunk):
        e = min(s + camera_chunk, C)
        R = w2c[s:e, :3, :3]  # [c, 3, 3]
        t = w2c[s:e, :3, 3]  # [c, 3]
        p = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]  # [c, n, 3]
        z = p[..., 2]
        fx = Ks[s:e, 0, 0][:, None]
        fy = Ks[s:e, 1, 1][:, None]
        cx = Ks[s:e, 0, 2][:, None]
        cy = Ks[s:e, 1, 2][:, None]
        w = image_sizes[s:e, 0][:, None].to(z.dtype)
        h = image_sizes[s:e, 1][:, None].to(z.dtype)

        valid = z > near
        zc = z.clamp_min(near)
        u = fx * p[..., 0] / zc + cx
        v = fy * p[..., 1] / zc + cy
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

        # Projected pixel area ~ f^2 * r^2 / z^2; cap a single Gaussian at
        # the full image so one giant background splat cannot dominate.
        area = (fx * fy) * weight0[None, :] / zc.square()  # [c, n]
        area = torch.minimum(area, w * h)
        out[s:e] = (area * valid).sum(dim=1) / (w * h).squeeze(1) * ratio
    return out


@torch.no_grad()
def assign_cameras_to_blocks(
    blocks: List[BlockMeta],
    gaussian_block_ids: np.ndarray,  # [N] from assign_gaussians_to_blocks
    means: torch.Tensor,  # [N, 3] world
    scales: torch.Tensor,  # [N, 3] exp'd
    opacities: torch.Tensor,  # [N] sigmoid'd
    camtoworlds: torch.Tensor,  # [C, 4, 4] (train cameras only)
    Ks: torch.Tensor,  # [C, 3, 3]
    image_sizes: torch.Tensor,  # [C, 2]
    camera_indices: np.ndarray,  # [C] global manifest indices
    cam_contracted: np.ndarray,  # [C, 3] contracted camera positions
    coverage_threshold: float = 0.05,
    min_cameras: int = 25,
    max_proxy_gaussians: int = 100_000,
    camera_chunk: int = 128,
    inside_margin: float = 0.05,
) -> np.ndarray:
    """Fill ``block.cameras`` for every block; returns the [B, C]
    contribution matrix (useful for diagnostics)."""
    device = means.device
    C = camtoworlds.shape[0]
    B = len(blocks)
    contrib = torch.zeros(B, C, device=device)
    ids = torch.from_numpy(gaussian_block_ids.astype(np.int64)).to(device)

    for b, block in enumerate(blocks):
        sel = ids == block.block_id
        contrib[b] = block_camera_contribution(
            means[sel],
            scales[sel],
            opacities[sel],
            camtoworlds,
            Ks,
            image_sizes,
            max_gaussians=max_proxy_gaussians,
            camera_chunk=camera_chunk,
        )

    contrib_np = contrib.cpu().numpy()
    assigned = contrib_np >= coverage_threshold  # [B, C]

    # A camera inside the (slightly expanded) cell always trains it.
    for b, block in enumerate(blocks):
        inside = block_mask(cam_contracted, block, margin=inside_margin)
        assigned[b] |= inside

    # Every camera trains at least its best block.
    best_block = contrib_np.argmax(axis=0)
    for c in range(C):
        if not assigned[:, c].any():
            assigned[best_block[c], c] = True

    # Every block gets at least min_cameras cameras (when available).
    k = min(min_cameras, C)
    for b in range(B):
        have = int(assigned[b].sum())
        if have < k:
            top = np.argsort(-contrib_np[b])[:k]
            assigned[b, top] = True

    for b, block in enumerate(blocks):
        block.cameras = [int(camera_indices[c]) for c in np.nonzero(assigned[b])[0]]

    return contrib_np


def block_cost(block: BlockMeta) -> float:
    """Wall-clock proxy used by the finetune scheduler (longest-first).

    Per-step cost scales with the Gaussian count; camera count enters
    through the adaptive step budget (sqrt scaling, see BlockConfig).
    """
    return float(block.num_gaussians) * max(len(block.cameras), 1) ** 0.5
