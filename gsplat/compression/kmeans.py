# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""K-means clustering in plain PyTorch, with optional per-point weights.

Used by :func:`gsplat.compression.png_compression._compress_kmeans` as an alternative to
TorchPQ. It runs on CPU and GPU and needs no extra dependencies.
"""

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["weighted_kmeans"]


def _assign(x: Tensor, centroids: Tensor, chunk_size: int) -> Tuple[Tensor, Tensor]:
    """Nearest centroid (squared L2) of every point, in chunks of ``chunk_size`` points."""
    n = x.shape[0]
    labels = torch.empty(n, dtype=torch.int64, device=x.device)
    min_sq_dist = torch.empty(n, dtype=x.dtype, device=x.device)
    centroids_sq = (centroids * centroids).sum(dim=1)
    for start in range(0, n, chunk_size):
        chunk = x[start : start + chunk_size]
        # ||c||^2 - 2 x.c ; the per-point ||x||^2 does not change the argmin
        scores = torch.addmm(
            centroids_sq[None, :], chunk, centroids.t(), beta=1.0, alpha=-2.0
        )
        best, idx = scores.min(dim=1)
        labels[start : start + chunk_size] = idx
        min_sq_dist[start : start + chunk_size] = best + (chunk * chunk).sum(dim=1)
    return labels, min_sq_dist


def weighted_kmeans(
    x: Tensor,
    n_clusters: int,
    weights: Optional[Tensor] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    chunk_size: int = 4096,
) -> Tuple[Tensor, Tensor]:
    """Lloyd's algorithm with euclidean assignment and a weighted mean update.

    Centroids start at ``n_clusters`` data points drawn without replacement. Each iteration
    assigns every point to its nearest centroid and moves each centroid to the weighted mean
    of its points. It stops once the summed squared centroid change is ``<= tol``, or after
    ``max_iter`` iterations. Means are accumulated in float64, so the result does not depend
    on ``chunk_size``.

    Args:
        x (Tensor): points to cluster, [N, D].
        n_clusters (int): number of clusters. Must be ``<= N``.
        weights (Tensor, optional): non-negative weight per point, [N]. Default: all ones.
        max_iter (int, optional): maximum number of iterations. Default: 100.
        tol (float, optional): stop once the summed squared centroid change is at most this.
            Default: 1e-4.
        seed (int, optional): seed of the initial draw. Default: 0.
        chunk_size (int, optional): points per assignment chunk. Only affects memory use.
            Default: 4096.

    Returns:
        Tuple[Tensor, Tensor]: centroids [n_clusters, D] and labels [N]. The labels are the
        assignment of the last iteration, that is, the one the returned centroids were
        computed from.
    """
    if x.dim() != 2:
        raise ValueError(f"expected x of shape [N, D], got {tuple(x.shape)}")
    n, d = x.shape
    if not 0 < n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}")
    if weights is not None:
        if weights.shape != (n,):
            raise ValueError(
                f"expected weights of shape [{n}], got {tuple(weights.shape)}"
            )
        if bool((weights < 0).any()):
            raise ValueError("weights must be non-negative")

    # The draw uses numpy's global RNG (seeded and restored here) so that a given seed keeps
    # picking the same points, whatever the caller's torch or numpy state is.
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        init = np.random.choice(n, size=n_clusters, replace=False)
    finally:
        np.random.set_state(numpy_state)
    centroids = x[torch.from_numpy(init).to(x.device)].clone()

    w = (
        torch.ones(n, dtype=torch.float64, device=x.device)
        if weights is None
        else weights.to(device=x.device, dtype=torch.float64)
    )
    weighted_x = x.to(torch.float64) * w[:, None]
    labels = torch.zeros(n, dtype=torch.int64, device=x.device)
    for _ in range(max_iter):
        labels, _ = _assign(x, centroids, chunk_size)
        weight_sum = torch.zeros(n_clusters, dtype=torch.float64, device=x.device)
        weight_sum.index_add_(0, labels, w)
        centroid_sum = torch.zeros(n_clusters, d, dtype=torch.float64, device=x.device)
        centroid_sum.index_add_(0, labels, weighted_x)
        # A cluster that no point (or only zero weight) is assigned to becomes the zero
        # vector. That is what TorchPQ does, so both backends behave the same; it is not a
        # recommendation - re-seeding such a cluster from a far-away point would cluster
        # better.
        new_centroids = torch.where(
            weight_sum[:, None] > 0,
            centroid_sum
            / weight_sum.clamp_min(torch.finfo(torch.float64).tiny)[:, None],
            torch.zeros_like(centroid_sum),
        ).to(x.dtype)
        change = float(((centroids - new_centroids) ** 2).sum())
        centroids = new_centroids
        if change <= tol:
            break
    return centroids, labels
