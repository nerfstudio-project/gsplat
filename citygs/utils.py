"""Small shared helpers."""

import random
from typing import Dict

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def knn_dist(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Mean distance to the k-1 nearest neighbors (excluding self). [N]."""
    try:
        from sklearn.neighbors import NearestNeighbors

        x = points.cpu().numpy()
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(x)
        dist, _ = nn.kneighbors(x)
        return torch.from_numpy(dist[:, 1:].mean(axis=1)).to(points)
    except ImportError:
        # Chunked torch fallback; fine for moderate point counts.
        N = points.shape[0]
        out = torch.empty(N, dtype=points.dtype, device=points.device)
        chunk = 4096
        for s in range(0, N, chunk):
            d = torch.cdist(points[s : s + chunk], points)
            knn = d.topk(k, largest=False).values[:, 1:]
            out[s : s + chunk] = knn.mean(dim=1)
        return out


def init_splats_from_points(
    points: torch.Tensor,  # [N, 3]
    rgbs: torch.Tensor,  # [N, 3] in [0, 1]
    sh_degree: int = 3,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    world_rank: int = 0,
    world_size: int = 1,
) -> Dict[str, torch.Tensor]:
    """Standard 3DGS initialization from a colored point cloud.

    With world_size > 1 the points are strided across ranks, matching
    gsplat's distributed rasterization (each rank owns a shard).
    """
    # Scale init: average distance to the 3 nearest neighbors, computed on
    # the full cloud before sharding so it is rank-count independent.
    dist = knn_dist(points, 4).clamp_min(1e-8)
    scales = torch.log(dist * init_scale)[:, None].repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    return {
        "means": points.contiguous(),
        "scales": scales.contiguous(),
        "quats": quats,
        "opacities": opacities,
        "sh0": colors[:, :1, :].contiguous(),
        "shN": colors[:, 1:, :].contiguous(),
    }
