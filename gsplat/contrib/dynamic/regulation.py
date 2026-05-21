"""HexPlane regularizers (experimental).

Three regularizers ported from G-SHARP v0.2's
``training/scene/regulation.py``. They operate on lists of feature-plane
tensors of shape ``(B, C, H, W)`` — typically the per-scale
:class:`gsplat.contrib.dynamic.HexPlaneField` planes.

- :func:`plane_smoothness` and :func:`time_smoothness` share the same math
  (sum of squared second-difference along the H axis); the distinction is
  *which* planes the caller passes. For HexPlane outputs, the spatial
  planes ``[xy, xz, yz]`` (combo indices ``[0, 1, 3]``) are smoothed
  spatially, and the spatio-temporal planes ``[xt, yt, zt]`` (combo
  indices ``[2, 4, 5]``) are smoothed in time.
- :func:`time_l1` regularizes spatio-temporal planes toward 1.0 (their
  initialization), encouraging static regions to stay put — matches
  ``L1TimePlanes`` in the G-SHARP source.

Caller selects which planes go where; these functions just iterate the
input list and sum the result.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

__all__ = ["plane_smoothness", "time_smoothness", "time_l1"]


def _second_difference_squared(planes: Sequence[Tensor]) -> Tensor:
    """Mean squared second-difference along the H axis, summed across planes.

    Ports ``compute_plane_smoothness`` from
    ``training/scene/regulation.py:45`` in the G-SHARP source.
    """
    total: Optional[Tensor] = None
    for p in planes:
        if p.ndim != 4:
            raise ValueError(
                f"Expected 4D plane tensors (B, C, H, W); got {p.ndim}D "
                f"shape {tuple(p.shape)}."
            )
        if p.shape[-2] < 3:
            continue
        first = p[..., 1:, :] - p[..., :-1, :]
        second = first[..., 1:, :] - first[..., :-1, :]
        contribution = second.pow(2).mean()
        total = contribution if total is None else total + contribution
    if total is None:
        first_p = next(iter(planes), None)
        device = first_p.device if first_p is not None else None
        dtype = first_p.dtype if first_p is not None else torch.float32
        return torch.zeros((), dtype=dtype, device=device)
    return total


def plane_smoothness(planes: Sequence[Tensor]) -> Tensor:
    """Spatial 2D smoothness — sum of mean squared second-difference along H.

    Pass the *spatial* HexPlane planes (combo indices ``[0, 1, 3]`` of a
    4D HexPlane: ``xy``, ``xz``, ``yz``) for the spatial-smoothness
    regularizer described in the G-SHARP proposal.

    Args:
        planes: Iterable of 4D ``(B, C, H, W)`` plane tensors.

    Returns:
        Scalar tensor (sum across planes; mean within each plane).
    """
    return _second_difference_squared(planes)


def time_smoothness(planes: Sequence[Tensor]) -> Tensor:
    """Temporal smoothness — same math as :func:`plane_smoothness`.

    Pass the *spatio-temporal* HexPlane planes (combo indices
    ``[2, 4, 5]`` of a 4D HexPlane: ``xt``, ``yt``, ``zt``). For these
    planes the H axis is the temporal axis (per the
    :func:`HexPlaneField._init_grid_param` reversed-order layout), so
    the second-difference squared is the per-plane temporal smoothness.

    Args:
        planes: Iterable of 4D ``(B, C, H, W)`` spatio-temporal plane tensors.

    Returns:
        Scalar tensor.
    """
    return _second_difference_squared(planes)


def time_l1(planes: Sequence[Tensor]) -> Tensor:
    """L1 deviation from 1.0 on spatio-temporal planes.

    HexPlane spatio-temporal planes are initialised to 1.0 so deformation
    starts identity-like. ``time_l1`` penalises their deviation from 1.0,
    encouraging static tissue to keep an identity (no-time-deformation)
    response. Matches G-SHARP's ``L1TimePlanes`` regularizer.

    Args:
        planes: Iterable of plane tensors (any shape with ``mean()``).

    Returns:
        Scalar L1 deviation summed across planes (mean within each plane).
    """
    total: Optional[Tensor] = None
    for p in planes:
        contribution = (1.0 - p).abs().mean()
        total = contribution if total is None else total + contribution
    if total is None:
        first_p = next(iter(planes), None)
        device = first_p.device if first_p is not None else None
        dtype = first_p.dtype if first_p is not None else torch.float32
        return torch.zeros((), dtype=dtype, device=device)
    return total
