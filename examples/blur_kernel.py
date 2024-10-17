#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
from torch import Tensor
from examples.mlp import create_mlp, _create_mlp_torch, _create_mlp_tcnn
from gsplat.utils import log_transform


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def encode(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_encoder(multires, i=0):
    embed_kwargs = {
        "include_input": True,
        "input_dims": i,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    embedder = Embedder(**embed_kwargs)
    return embedder


class BlurOptModule(nn.Module):
    """Blur optimization module."""

    def __init__(self, n: int, embed_dim: int = 1):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, embed_dim)
        self.embeds.weight.data = torch.linspace(-1, 1, n)[:, None]

        self.depth_encoder = get_encoder(7, 1)
        self.means_encoder = get_encoder(3, 3)
        self.blur_mask_mlp = _create_mlp_torch(
            in_dim=embed_dim + self.depth_encoder.out_dim,
            num_layers=5,
            layer_width=64,
            out_dim=1,
        )
        self.blur_deltas_mlp = _create_mlp_tcnn(
            in_dim=embed_dim + self.means_encoder.out_dim + 7,
            num_layers=5,
            layer_width=64,
            out_dim=7,
        )

    def predict_mask(self, image_ids: Tensor, depths: Tensor):
        depths_log = log_transform(depths)
        depths_emb = self.depth_encoder.encode(depths_log.reshape(-1, 1))
        images_emb = self.embeds(image_ids).repeat(depths_emb.shape[0], 1)
        mlp_out = self.blur_mask_mlp(torch.cat([images_emb, depths_emb], dim=-1))
        blur_mask = torch.sigmoid(mlp_out)
        blur_mask = blur_mask.reshape(depths.shape)
        return blur_mask

    def predict_deltas(
        self, image_ids: Tensor, means: Tensor, scales: Tensor, quats: Tensor
    ):
        means_log = log_transform(means)
        means_emb = self.means_encoder.encode(means_log)
        images_emb = self.embeds(image_ids).repeat(means.shape[0], 1)
        mlp_out = self.blur_deltas_mlp(
            torch.cat([images_emb, means_emb, scales, quats], dim=-1)
        ).float()
        scales_delta = mlp_out[:, :3]
        rotations_delta = mlp_out[:, 3:]
        scales_delta = torch.clamp(scales_delta, min=0.0, max=0.1)
        rotations_delta = torch.clamp(rotations_delta, min=-0.05, max=0.05)
        return scales_delta, rotations_delta

    def mask_reg_loss(self, blur_mask: Tensor):
        """Mask regularization loss."""
        meanloss = (torch.mean(blur_mask) - 0.5) ** 2
        stdloss = (torch.std(blur_mask) - 0.5) ** 2
        return meanloss + 0.1 * stdloss
