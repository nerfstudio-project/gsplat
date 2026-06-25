# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the sparse-rasterization op tests.

The sparse ops (num-contributing, contributing-ids, top-contributing, ...) are
all validated the same way: build a synthetic 2D-gaussian scene, pick a set of
pixels to render, and compare the sparse op against the dense op gathered at
those pixels. These helpers centralize the scene/pixel construction and the
gather so the per-op test files only express the op-specific comparison.
"""

import math

import torch

device = torch.device("cuda:0")


def grid(width, height, tile_size):
    """(tile_height, tile_width) for an image of the given size."""
    return math.ceil(height / tile_size), math.ceil(width / tile_size)


def make_scene(C, N, width, height, seed):
    """A synthetic scene of N isotropic 2D gaussians per camera (C cameras).

    Returns ``(means2d, conics, opacities, radii, depths)``. conics are
    isotropic with extent set by a per-gaussian radius; radii are the (looser)
    integer bounding boxes used by tile intersection.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    u = lambda *s: torch.rand(*s, device=device, generator=gen)

    means2d = torch.empty(C, N, 2, device=device)
    means2d[..., 0] = u(C, N) * width  # x (col)
    means2d[..., 1] = u(C, N) * height  # y (row)
    r = u(C, N) * 6.0 + 2.0  # gaussian std in pixels
    inv = 1.0 / (r * r)
    conics = torch.stack([inv, torch.zeros_like(inv), inv], dim=-1).contiguous()
    rb = (r * 3.0).ceil().to(torch.int32)
    radii = torch.stack([rb, rb], dim=-1).contiguous()
    depths = (u(C, N) * 9.9 + 0.1).contiguous()
    opacities = (u(C, N) * 0.9 + 0.05).contiguous()
    return means2d, conics, opacities, radii, depths


def all_pixels(C, width, height):
    """Every pixel of every camera, packed ``(pixels [P, 2] (row, col), image_ids [P])``."""
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    one = torch.stack([rr.flatten(), cc.flatten()], dim=-1)  # [H*W, 2]
    pixels = one.repeat(C, 1).to(torch.int32).contiguous()
    image_ids = torch.arange(C, device=device, dtype=torch.int32).repeat_interleave(
        height * width
    )
    return pixels, image_ids


def subset_pixels(C, width, height, frac, seed):
    """A random per-camera subset of pixels (no duplicates within a camera),
    packed ``(pixels [P, 2] (row, col), image_ids [P])``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    pix_list, img_list = [], []
    for c in range(C):
        k = max(1, int(width * height * frac))
        perm = torch.randperm(width * height, device=device, generator=gen)[:k]
        pix_list.append(torch.stack([perm // width, perm % width], dim=-1))
        img_list.append(torch.full((k,), c, device=device, dtype=torch.int32))
    pixels = torch.cat(pix_list).to(torch.int32).contiguous()
    image_ids = torch.cat(img_list).contiguous()
    return pixels, image_ids


def gather(image_tensor, pixels, image_ids):
    """Gather a dense ``[C, H, W, ...]`` tensor at the packed pixels -> ``[P, ...]``."""
    img = image_ids.long()
    rows = pixels[:, 0].long()
    cols = pixels[:, 1].long()
    return image_tensor[img, rows, cols]
