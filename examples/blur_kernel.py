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
import numpy as np
import torch.nn.functional as F
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

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,
        "input_dims": i,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class GTnet(nn.Module):
    def __init__(
        self,
        n,
        res_pos=3,
        res_view=10,
        num_hidden=3,
        width=64,
        pos_delta=False,
        num_moments=4,
    ):
        super().__init__()
        self.focals = torch.nn.Embedding(n, 1)
        self.focals.weight.data = torch.linspace(-1, 1, n)[:, None]
        self.embed_depth, self.embed_depth_cnl = get_embedder(14, 1)
        self.depth_mlp = _create_mlp_torch(
            in_dim=self.embed_depth_cnl + 1,
            num_layers=5,
            layer_width=64,
            out_dim=1,
        )

        self.pos_delta = pos_delta
        self.num_moments = num_moments
        self.embed_pos, self.embed_pos_cnl = get_embedder(res_pos, 3)
        self.embed_view, self.embed_view_cnl = get_embedder(res_view, 3)
        in_cnl = (
            self.embed_pos_cnl + self.embed_view_cnl + 7
        )  # 7 for scales and rotations
        hiddens = [
            nn.Linear(width, width) if i % 2 == 0 else nn.ReLU()
            for i in range((num_hidden - 1) * 2)
        ]
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, width),
            nn.ReLU(),
            *hiddens,
        ).to("cuda")
        if not pos_delta:  # Defocus
            self.s = nn.Linear(width, 3).to("cuda")
            self.r = nn.Linear(width, 4).to("cuda")
        else:  # Motion
            self.s = nn.Linear(width, 3 * (num_moments + 1)).to("cuda")
            self.r = nn.Linear(width, 4 * (num_moments + 1)).to("cuda")
            self.p = nn.Linear(width, 3 * num_moments).to("cuda")

        self.linears.apply(init_linear_weights)
        self.s.apply(init_linear_weights)
        self.r.apply(init_linear_weights)
        if pos_delta:
            self.p.apply(init_linear_weights)

    def forward(self, depths, image_ids):
        height, width = depths.shape[1:3]

        depths_emb = self.embed_depth(depths)
        focals_emb = self.focals(image_ids[0])[None, None, None, :].repeat(
            1, height, width, 1
        )
        x = torch.cat([depths_emb, focals_emb], dim=-1)
        x = x.reshape(-1, x.shape[-1])
        mlp_out = self.depth_mlp(x)
        blur_mask = torch.sigmoid(mlp_out)
        blur_mask = blur_mask.reshape(1, height, width, 1)
        return blur_mask

    def forward_deltas(self, pos, scales, rotations, viewdirs):
        pos_delta = None
        pos = self.embed_pos(pos)
        viewdirs = self.embed_view(viewdirs)

        x = torch.cat([pos, viewdirs, scales, rotations], dim=-1)
        x1 = self.linears(x)

        scales_delta = self.s(x1)
        rotations_delta = self.r(x1)

        if self.pos_delta:
            pos_delta = self.p(x1)

        return scales_delta, rotations_delta, pos_delta
