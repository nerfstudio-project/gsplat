"""Per-axis scene contraction for unbounded city scenes.

The foreground box (center + half-extents, estimated robustly from camera
positions) maps to [-1, 1]^3. Outside it we apply the mip-NeRF-360 style
contraction *independently per axis* (x -> 2 - 1/x beyond 1), so all of
space lands in (-2, 2)^3. Per-axis (rather than the usual norm-coupled
contraction) matters for axis-aligned grid blocks: a point far outside
the box in z only (e.g. ground content under a constant-altitude drone
rig) keeps its x/y placement instead of being crushed toward the center.
Blocks are axis-aligned cells of the contracted cube, which keeps the
partition a true partition (crop + concat is exact) while boundary cells
absorb unbounded background.

All block bookkeeping (Gaussian-to-block assignment, finetune cropping)
happens in contracted coordinates; nothing ever needs to be uncontracted
except for visualization.
"""

from typing import Tuple, Union

import numpy as np
import torch

Array = Union[np.ndarray, torch.Tensor]


def contract(x: Array, center, half_extent) -> Array:
    """Map world points into (-2, 2)^3. x: [..., 3]."""
    is_np = isinstance(x, np.ndarray)
    if is_np:
        x = torch.from_numpy(np.asarray(x, dtype=np.float64))
    center = torch.as_tensor(center, dtype=x.dtype, device=x.device)
    half_extent = torch.as_tensor(half_extent, dtype=x.dtype, device=x.device)
    q = (x - center) / half_extent
    a = q.abs()
    out = torch.where(a <= 1.0, q, torch.sign(q) * (2.0 - 1.0 / a.clamp_min(1e-12)))
    return out.numpy() if is_np else out


def uncontract(y: Array, center, half_extent) -> Array:
    """Inverse of :func:`contract`. Inputs with |y|_inf >= 2 are clamped
    just inside the boundary (they map to infinity otherwise)."""
    is_np = isinstance(y, np.ndarray)
    if is_np:
        y = torch.from_numpy(np.asarray(y, dtype=np.float64))
    center = torch.as_tensor(center, dtype=y.dtype, device=y.device)
    half_extent = torch.as_tensor(half_extent, dtype=y.dtype, device=y.device)
    a = y.abs().clamp(max=2.0 - 1e-6)
    # Inverse of |y| = 2 - 1/|q| beyond 1: |q| = 1 / (2 - |y|).
    q = torch.where(a <= 1.0, y, torch.sign(y) / (2.0 - a))
    out = q * half_extent + center
    return out.numpy() if is_np else out


def foreground_bounds(
    positions: np.ndarray,
    mad_threshold: float = 6.0,
    margin: float = 1.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robust foreground box from camera positions.

    Outliers (stray registrations, fly-in shots) are rejected per axis with
    a median-absolute-deviation test, then the box is the inlier extent
    scaled by ``margin``. Returns (center [3], half_extent [3]).
    """
    positions = np.asarray(positions, dtype=np.float64)
    med = np.median(positions, axis=0)
    abs_dev = np.abs(positions - med)
    mad = np.median(abs_dev, axis=0)
    # Guard degenerate axes (e.g. constant-altitude aerial rigs): fall back
    # to the mean absolute deviation, then to a fraction of the global size.
    floor = np.maximum(abs_dev.mean(axis=0), 1e-6)
    mad = np.where(mad < 1e-8, floor, mad)
    inlier = (abs_dev <= mad_threshold * mad * 1.4826).all(axis=1)
    if not inlier.any():
        inlier = np.ones(len(positions), dtype=bool)
    pts = positions[inlier]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    center = (lo + hi) / 2
    half_extent = (hi - lo) / 2 * margin
    # A box axis must never collapse; keep at least 5% of the largest axis.
    half_extent = np.maximum(half_extent, 0.05 * half_extent.max())
    return center, half_extent
