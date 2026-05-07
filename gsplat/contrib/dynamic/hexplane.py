"""HexPlane spatio-temporal feature field (experimental).

Multi-resolution 6-plane decomposition of a 4D ``(x, y, z, t)`` feature
field, ported from G-SHARP v0.2 (`training/scene/hexplane.py`), itself
derived from the K-Planes / 4DGaussians formulation.

Six 2D feature planes — one for each pair of input axes ``(xy, xz, xt,
yz, yt, zt)`` — are sampled with bilinear interpolation, multiplied
element-wise across planes (giving a per-scale feature vector), and
concatenated across multi-resolution scales to produce the final feature.
"""

from __future__ import annotations

import itertools
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["HexPlaneField"]


# ---------------------------------------------------------------------------
# Internal helpers (private; faithful port of the G-SHARP implementations)
# ---------------------------------------------------------------------------


def _normalize_aabb(pts: Tensor, aabb: Tensor) -> Tensor:
    """Linear remap of *pts* into ``[-1, 1]`` using the axis-aligned bounding box.

    Matches the ``normalize_aabb`` helper in the G-SHARP source. With
    ``aabb = [[+b, +b, +b], [-b, -b, -b]]`` the remap is
    ``-pts / b`` (sign-flipped, kept for parity with upstream).
    """
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def _grid_sample_wrapper(
    grid: Tensor, coords: Tensor, align_corners: bool = True
) -> Tensor:
    grid_dim = coords.shape[-1]
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)
    if grid_dim not in (2, 3):
        raise NotImplementedError(
            f"_grid_sample_wrapper supports 2D / 3D coords only; got {grid_dim}D."
        )
    coords = coords.view(
        [coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:])
    )
    b, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(
        grid,
        coords,
        align_corners=align_corners,
        mode="bilinear",
        padding_mode="border",
    )
    return interp.view(b, feature_dim, n).transpose(-1, -2).squeeze()


def _init_grid_param(
    grid_nd: int,
    in_dim: int,
    out_dim: int,
    reso: Sequence[int],
    a: float = 0.1,
    b: float = 0.5,
) -> nn.ParameterList:
    if in_dim != len(reso):
        raise ValueError(
            f"_init_grid_param: in_dim={in_dim} != len(reso)={len(reso)}."
        )
    if grid_nd > in_dim:
        raise ValueError(
            f"_init_grid_param: grid_nd={grid_nd} > in_dim={in_dim}."
        )
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    for coo_comb in coo_combs:
        new_grid_coef = nn.Parameter(
            torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]])
        )
        if has_time_planes and 3 in coo_comb:
            # Spatio-temporal planes initialise to ones so deformation
            # starts identity-like; matches G-SHARP convention.
            nn.init.ones_(new_grid_coef)
        else:
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)
    return grid_coefs


def _interpolate_ms_features(
    pts: Tensor,
    ms_grids: nn.ModuleList,
    grid_dimensions: int,
    concat_features: bool,
) -> Tensor:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    multi_scale_interp: list[Tensor] = [] if concat_features else None
    summed: Tensor | float = 0.0

    for grids in ms_grids:
        interp_space: Tensor | float = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grids[ci].shape[1]
            interp_plane = _grid_sample_wrapper(grids[ci], pts[..., coo_comb]).view(
                -1, feature_dim
            )
            interp_space = interp_space * interp_plane

        if concat_features:
            assert multi_scale_interp is not None
            multi_scale_interp.append(interp_space)
        else:
            summed = summed + interp_space

    if concat_features:
        assert multi_scale_interp is not None
        return torch.cat(multi_scale_interp, dim=-1)
    return summed if isinstance(summed, Tensor) else torch.zeros(0)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


_DEFAULT_PLANE_CONFIG: dict = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 32,
    "resolution": [64, 64, 64, 25],
}
_DEFAULT_MULTIRES: tuple[int, ...] = (1, 2)


class HexPlaneField(nn.Module):
    """Multi-resolution 6-plane decomposition of a 4D feature field.

    For each scale, six 2D feature planes covering every pair of the four
    input axes ``(x, y, z, t)`` are sampled bilinearly and multiplied
    element-wise to produce a per-scale feature vector. With
    ``concat_features=True`` (the default and only mode currently supported)
    feature vectors from all scales are concatenated; the resulting feature
    dimensionality is ``output_coordinate_dim * len(multires)``.

    Spatial coordinates are normalized into ``[-1, 1]`` using the
    axis-aligned bounding box ``[-bounds, bounds]`` along x/y/z; the time
    coordinate is passed through unchanged (callers should pre-normalize it
    into a sensible range — values outside the grid are clamped via
    ``padding_mode="border"`` in :func:`torch.nn.functional.grid_sample`).

    Args:
        bounds: Half-extent of the spatial AABB along each axis (default
            ``1.6``, matching G-SHARP).
        planes_config: Optional dict with keys ``grid_dimensions``,
            ``input_coordinate_dim``, ``output_coordinate_dim``,
            ``resolution``. The default is suitable for surgical-scene
            4D fields (32 features per plane, ``[64, 64, 64, 25]``
            resolution).
        multires: Iterable of integer multipliers applied to the *spatial*
            grid resolution for each scale; the temporal resolution is
            kept fixed across scales. Default ``(1, 2)``.
    """

    def __init__(
        self,
        bounds: float = 1.6,
        planes_config: Optional[dict] = None,
        multires: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        config = dict(planes_config) if planes_config is not None else dict(_DEFAULT_PLANE_CONFIG)
        multires_seq = list(multires) if multires is not None else list(_DEFAULT_MULTIRES)

        self.bounds = float(bounds)
        aabb = torch.tensor(
            [[bounds, bounds, bounds], [-bounds, -bounds, -bounds]],
            dtype=torch.float32,
        )
        self.aabb = nn.Parameter(aabb, requires_grad=False)

        self.grid_config = config
        self.multires = multires_seq
        self.concat_features = True

        self.grids = nn.ModuleList()
        self.feat_dim = 0
        for res in self.multires:
            scale_config = dict(config)
            base_reso = list(scale_config["resolution"])
            # Multi-res only on spatial axes; time grid stays fixed across scales.
            scale_config["resolution"] = [r * res for r in base_reso[:3]] + base_reso[3:]
            gp = _init_grid_param(
                grid_nd=scale_config["grid_dimensions"],
                in_dim=scale_config["input_coordinate_dim"],
                out_dim=scale_config["output_coordinate_dim"],
                reso=scale_config["resolution"],
            )
            if self.concat_features:
                self.feat_dim += gp[-1].shape[1]
            else:
                self.feat_dim = gp[-1].shape[1]
            self.grids.append(gp)

    def forward(self, xyzt: Tensor) -> Tensor:
        """Sample the multi-resolution HexPlane at the given 4D points.

        Args:
            xyzt: ``(N, 4)`` (or any shape ending in 4) with ``[..., :3]``
                being world-space spatial coordinates inside the AABB and
                ``[..., 3]`` being the temporal coordinate.

        Returns:
            ``(N, feat_dim)`` feature tensor where ``feat_dim ==
            output_coordinate_dim * len(multires)`` under
            ``concat_features=True``.
        """
        if xyzt.shape[-1] != 4:
            raise ValueError(
                f"HexPlaneField.forward: xyzt last dim must be 4, "
                f"got shape {tuple(xyzt.shape)}."
            )

        xyz = xyzt[..., :3]
        t = xyzt[..., 3:]
        xyz_norm = _normalize_aabb(xyz, self.aabb)
        pts = torch.cat([xyz_norm, t], dim=-1).reshape(-1, 4)

        return _interpolate_ms_features(
            pts,
            ms_grids=self.grids,
            grid_dimensions=self.grid_config["grid_dimensions"],
            concat_features=self.concat_features,
        )
