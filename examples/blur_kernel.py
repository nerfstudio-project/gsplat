import torch
import torch.nn as nn
from torch import Tensor
from examples.mlp import create_mlp
from gsplat.utils import log_transform


class BlurOptModule(nn.Module):
    """Blur optimization module."""

    def __init__(self, n: int, embed_dim: int = 4):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, embed_dim)
        self.depth_encoder = get_encoder(7, 1)
        self.means_encoder = get_encoder(3, 3)
        self.blur_mask_mlp = create_mlp(
            in_dim=embed_dim + self.depth_encoder.out_dim,
            num_layers=5,
            layer_width=64,
            out_dim=1,
        )
        self.blur_deltas_mlp = create_mlp(
            in_dim=embed_dim + self.means_encoder.out_dim + 7,
            num_layers=5,
            layer_width=64,
            out_dim=7,
        )

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(
        self,
        image_ids: Tensor,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
    ):
        means_log = log_transform(means)
        means_emb = self.means_encoder.encode(means_log)
        images_emb = self.embeds(image_ids).repeat(means.shape[0], 1)
        mlp_out = self.blur_deltas_mlp(
            torch.cat([images_emb, means_emb, scales, quats], dim=-1)
        ).float()
        scales_delta = torch.clamp(mlp_out[:, :3], min=0.0, max=0.1)
        quats_delta = torch.clamp(mlp_out[:, 3:], min=0.0, max=0.1)
        scales = torch.exp(scales + scales_delta)
        quats = quats + quats_delta
        return scales, quats

    def predict_mask(self, image_ids: Tensor, depths: Tensor):
        depths_emb = self.depth_encoder.encode(log_transform(depths))
        images_emb = self.embeds(image_ids).repeat(*depths_emb.shape[:-1], 1)
        mlp_in = torch.cat([images_emb, depths_emb], dim=-1)
        mlp_out = self.blur_mask_mlp(mlp_in.reshape(-1, mlp_in.shape[-1]))
        blur_mask = torch.sigmoid(mlp_out)
        blur_mask = blur_mask.reshape(depths.shape)
        return blur_mask

    def mask_variation_loss(self, blur_mask: Tensor, eps: float = 1e-6):
        """Mask variation loss."""
        x = blur_mask.mean()
        meanloss = (1 / (1 - x + eps) - 1) + (0.1 / (x + eps))
        return meanloss


def get_encoder(num_freqs: int, input_dims: int):
    kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": num_freqs - 1,
        "num_freqs": num_freqs,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }
    encoder = Encoder(**kwargs)
    return encoder


class Encoder:
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
