"""SfM-free point-cloud initialization ported from G-SHARP v0.2.

Public API:

- :func:`multi_frame_depth_unprojection` — unproject masked depth maps from
  multiple frames into a shared world-space point cloud + per-point RGB.
  Ported from ``accumulate_multiframe_pointcloud`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``
  (lines 2036–2159 in the source).
- :func:`knn_scale_init` — per-Gaussian initial log-scale based on the mean
  distance to the ``k`` nearest neighbours in the point cloud. Pure-torch
  (uses ``torch.cdist`` + ``topk``) so the core library has no sklearn
  dependency. Numerically matches the G-SHARP path
  ``sqrt((knn(points, k+1)[:, 1:] ** 2).mean(dim=-1)).clamp_min(eps).log()``
  inside ``create_splats_with_optimizers`` (lines 1832–1838 in the source).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def multi_frame_depth_unprojection(
    images: Tensor,
    depths: Tensor,
    masks: Tensor,
    poses: Tensor,
    intrinsics: Tensor,
    max_points: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """Multi-frame depth unprojection into a shared world-space point cloud.

    For each frame, identifies valid pixels (``mask != 0`` AND ``depth > 0``),
    unprojects them to camera space using the per-frame pinhole intrinsics,
    and transforms to world space via the camera-to-world pose. All per-frame
    point sets are concatenated and (optionally) random-subsampled to
    ``max_points``.

    Args:
        images: ``(N, H, W, 3)`` RGB. ``uint8`` is normalized to ``[0, 1]``;
            float inputs are taken as-is.
        depths: ``(N, H, W)`` metric depth (zero = no measurement).
        masks: ``(N, H, W)`` mask (``!= 0`` = keep pixel).
        poses: ``(N, 4, 4)`` camera-to-world transforms.
        intrinsics: ``(N, 3, 3)`` per-frame pinhole intrinsics.
        max_points: Optional cap; a random subset is returned if exceeded.

    Returns:
        Tuple ``(xyz, rgb)``:

        - ``xyz`` ``(P, 3)`` — world-space coordinates.
        - ``rgb`` ``(P, 3)`` — float ``[0, 1]`` colors at the source pixels.

        Both are ``(0, 3)`` if no pixel passes the mask + valid-depth gate.

    Raises:
        ValueError: if leading dimensions disagree across the inputs.
    """
    n = images.shape[0]
    for name, t in (("depths", depths), ("masks", masks), ("poses", poses), ("intrinsics", intrinsics)):
        if t.shape[0] != n:
            raise ValueError(
                f"multi_frame_depth_unprojection: leading dim mismatch — "
                f"images has {n} frames but {name} has {t.shape[0]}."
            )

    h, w = images.shape[1:3]
    device = images.device

    images_f = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    depths_f = depths.float()
    masks_b = masks != 0

    v_coords, u_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )

    xyz_chunks: list[Tensor] = []
    rgb_chunks: list[Tensor] = []

    for i in range(n):
        valid = masks_b[i] & (depths_f[i] > 0)
        if not bool(valid.any()):
            continue

        u_i = u_coords[valid]
        v_i = v_coords[valid]
        d_i = depths_f[i][valid]
        rgb_i = images_f[i][valid]

        k = intrinsics[i]
        fx, fy = k[0, 0], k[1, 1]
        cx, cy = k[0, 2], k[1, 2]

        x_cam = (u_i - cx) * d_i / fx
        y_cam = (v_i - cy) * d_i / fy
        z_cam = d_i

        cam_h = torch.stack(
            [x_cam, y_cam, z_cam, torch.ones_like(z_cam)], dim=-1
        )  # (P_i, 4)
        world_h = cam_h @ poses[i].T.float()  # (P_i, 4)
        xyz_chunks.append(world_h[:, :3])
        rgb_chunks.append(rgb_i)

    if not xyz_chunks:
        empty = torch.zeros(0, 3, device=device)
        return empty, empty.clone()

    xyz = torch.cat(xyz_chunks, dim=0)
    rgb = torch.cat(rgb_chunks, dim=0)

    if max_points is not None and xyz.shape[0] > max_points:
        idx = torch.randperm(xyz.shape[0], device=device)[:max_points]
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def knn_scale_init(
    xyz: Tensor, k: int = 3, eps: float = 1e-7, chunk_size: int = 1024
) -> Tensor:
    """Per-point initial log-scale based on the mean distance to the *k* nearest neighbours.

    Pure-torch implementation (``cdist`` + ``topk``) — no sklearn dependency.
    For each point, computes the Euclidean distances to its *k* nearest other
    points (excluding self), takes the root-mean-square of those distances,
    clamps to *eps* (so duplicate points don't yield ``-inf`` after the log),
    then returns the natural log.

    The output is shape ``(N,)`` and is intended to be broadcast to a
    ``(N, 3)`` log-scale tensor by the caller (typical use:
    ``scales = result.unsqueeze(-1).repeat(1, 3)``), matching the convention
    in G-SHARP's ``create_splats_with_optimizers``.

    Args:
        xyz: Point cloud ``(N, 3)``.
        k: Number of nearest neighbours per point, excluding self
            (default ``3``).
        eps: Minimum distance clamp before the log; default ``1e-7``.
        chunk_size: Row-block size for the pairwise distance scan. Memory
            is ``O(chunk_size * N)`` per block; default 1024 fits comfortably
            on a 12 GB consumer GPU at ``N = 50_000``.

    Returns:
        ``(N,)`` per-point log-distance.

    Raises:
        ValueError: if ``N <= k`` (need at least ``k + 1`` points so each
            point has *k* neighbours after excluding self).
    """
    n = xyz.shape[0]
    if n <= k:
        raise ValueError(
            f"knn_scale_init: need at least k+1={k + 1} points, got {n}."
        )

    # Chunked pairwise KNN: a single (N, N) cdist is ~N^2 * 4 bytes, which is
    # ~10 GB at N=50_000 (fits on an A100, OOMs on a 12-16 GB consumer card).
    # Loop over query rows in blocks; per block we only allocate (chunk, N)
    # distances, then top-k.  Memory drops from O(N^2) to O(chunk * N).
    chunk = max(1, min(chunk_size, n))
    out = torch.empty(n, device=xyz.device, dtype=xyz.dtype)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block_dists = torch.cdist(xyz[start:end], xyz)  # (chunk, N)
        knn_dists, _ = block_dists.topk(k + 1, largest=False)  # (chunk, k+1)
        neighbor_dists = knn_dists[:, 1:]  # drop self → (chunk, k)
        rms = neighbor_dists.pow(2).mean(dim=-1).sqrt()
        out[start:end] = rms
    return out.clamp_min(eps).log()
