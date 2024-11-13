import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from examples.mlp import create_mlp, get_encoder
from gsplat.utils import log_transform


class BlurOptModule(nn.Module):
    """Blur optimization module."""

    def __init__(self, n: int, embed_dim: int = 4):
        super().__init__()
        self.embeds = torch.nn.Embedding(n, embed_dim)
        self.means_encoder = get_encoder(num_freqs=3, input_dims=3)
        self.depths_encoder = get_encoder(num_freqs=3, input_dims=1)
        self.grid_encoder = get_encoder(num_freqs=1, input_dims=2)
        self.blur_mask_mlp = create_mlp(
            in_dim=embed_dim + self.depths_encoder.out_dim + self.grid_encoder.out_dim,
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
        self.bounded_l1_loss = bounded_l1_loss(10.0, 0.5)

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def forward(
        self,
        image_ids: Tensor,
        means: Tensor,
        scales: Tensor,
        quats: Tensor,
    ):
        quats = F.normalize(quats, dim=-1)
        means_emb = self.means_encoder.encode(log_transform(means))
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
        height, width = depths.shape[1:3]
        grid_y, grid_x = torch.meshgrid(
            (torch.arange(height, device=depths.device) + 0.5) / height,
            (torch.arange(width, device=depths.device) + 0.5) / width,
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        grid_emb = self.grid_encoder.encode(grid_xy)
        depths_emb = self.depths_encoder.encode(log_transform(depths))
        images_emb = self.embeds(image_ids).repeat(*depths_emb.shape[:-1], 1)
        mlp_in = torch.cat([images_emb, grid_emb, depths_emb], dim=-1)
        mlp_out = self.blur_mask_mlp(mlp_in.reshape(-1, mlp_in.shape[-1])).reshape(
            depths.shape
        )
        blur_mask = torch.sigmoid(mlp_out)
        return blur_mask

    def mask_loss(self, blur_mask: Tensor):
        """Loss function for regularizing the blur mask by controlling its mean.

        Uses bounded l1 loss which diverges to +infinity at 0 and 1 to prevents the mask
        from collapsing all 0s or 1s.
        """
        x = blur_mask.mean()
        return self.bounded_l1_loss(x)


def bounded_l1_loss(lambda_a: float, lambda_b: float, eps: float = 1e-2):
    """L1 loss function with discontinuities at 0 and 1.

    Args:
        lambda_a (float): Coefficient of L1 loss.
        lambda_b (float): Coefficient of bounded loss.
        eps (float, optional): Epsilon to prevent divide by zero. Defaults to 1e-2.
    """

    def loss_fn(x: Tensor):
        return lambda_a * x + lambda_b * (1 / (1 - x + eps) + 1 / (x + eps))

    # Compute constant that sets min to zero
    xs = torch.linspace(0, 1, 1000)
    ys = loss_fn(xs)
    c = ys.min()
    return lambda x: loss_fn(x) - c
