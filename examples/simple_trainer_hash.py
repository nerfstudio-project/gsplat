import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from diff_rast.project_gaussians import ProjectGaussians
from diff_rast.rasterize import RasterizeGaussians


class GSHashEncoding(torch.nn.Module):
    """Hash Encoding designed for GS."""

    def __init__(
        self,
        output_dim: int,
        # hashmap_size_list: int = [64, 128, 256, 512, 1024],
        # hashmap_size_list: int = [16],
        hashmap_size_list: int = [16, 32],
        hashmap_dim: int = 4,
        reso: int = 2048,
    ):
        super().__init__()
        self.hashmap_size_list = hashmap_size_list
        self.hashmap_dim = hashmap_dim
        self.reso = reso

        codes = []
        sizes = []
        for hashmap_size in hashmap_size_list:
            if reso <= hashmap_size:
                code = torch.randn((reso, hashmap_dim))
                size = reso
            else:
                code = torch.randn((hashmap_size, hashmap_dim))
                size = hashmap_size
                map = torch.randint(0, hashmap_size, size=(reso, hashmap_dim))
                self.register_buffer("map_" + str(hashmap_size), map)

            torch.nn.init.uniform_(code, -1.0, 1.0)
            codes.append(code)
            sizes.append(size)
        codes = torch.cat(codes, dim=0)

        self.codes = torch.nn.Parameter(codes, requires_grad=True)
        self.sizes = sizes

        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(
                len(hashmap_size_list) * hashmap_dim, output_dim, bias=False
            ),
            # torch.nn.ReLU(),
            # torch.nn.Linear(64, output_dim),
        )

    def forward(self):
        codes = torch.split(self.codes, self.sizes, dim=0)

        outs = []
        for hashmap_size, code in zip(self.hashmap_size_list, codes):
            if self.reso <= hashmap_size:
                outs.append(code)
            else:
                map = getattr(self, "map_" + str(hashmap_size))
                out = torch.stack(
                    [code[map[:, i], i] for i in range(self.hashmap_dim)],
                    dim=-1,
                )
                outs.append(out)
        outs = torch.cat(outs, dim=1)
        outs = self.mlp_head(outs)
        return outs.to(self.codes)


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        self.means = GSHashEncoding(3, reso=self.num_points).to(self.device)
        self.scales = GSHashEncoding(3, reso=self.num_points).to(self.device)
        self.rgbs = GSHashEncoding(3, reso=self.num_points).to(self.device)
        self.quats = GSHashEncoding(4, reso=self.num_points).to(self.device)
        self.opacities = GSHashEncoding(1, reso=self.num_points).to(self.device)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )

    def train(
        self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = True
    ):
        optimizer = optim.Adam(
            [
                {"params": self.means.parameters(), "lr": lr},
                {"params": self.scales.parameters(), "lr": lr},
                {"params": self.rgbs.parameters(), "lr": lr},
                {"params": self.quats.parameters(), "lr": lr},
                {"params": self.opacities.parameters(), "lr": lr},
            ]
        )

        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=10
                ),
                torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[
                        iterations // 2,
                        iterations * 3 // 4,
                    ],
                    gamma=0.33,
                ),
            ]
        )
        mse_loss = torch.nn.MSELoss()
        frames = []

        with torch.no_grad():
            means = self.means()
            means_min = means.min(dim=0).values
            means_max = means.max(dim=0).values

        for iter in range(iterations):
            means = self.means()
            # means = (means - means_min) / (means_max - means_min) * 2 - 1
            scales = self.scales()
            rgbs = self.rgbs()
            quats = self.quats()
            opacities = self.opacities()
            (
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                cov3d,
            ) = ProjectGaussians.apply(
                means,
                scales,
                1,
                quats,
                self.viewmat,
                self.viewmat,
                self.focal,
                self.focal,
                self.H,
                self.W,
                self.tile_bounds,
            )

            out_img = RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(rgbs),
                torch.sigmoid(opacities),
                self.H,
                self.W,
            )
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 15 == 0:
                frames.append(
                    (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
                )
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            frames[0].save(
                os.getcwd() + "/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        n_params = 0
        n_params += sum(p.numel() for p in self.means.parameters())
        n_params += sum(p.numel() for p in self.scales.parameters())
        n_params += sum(p.numel() for p in self.rgbs.parameters())
        n_params += sum(p.numel() for p in self.quats.parameters())
        n_params += sum(p.numel() for p in self.opacities.parameters())
        print("Trainable parameters:", n_params)


# [10, 3]
trainable_coords = torch.randn((10, 3))
hash_mapping_x = torch.randint(low=0, high=10, size=(1000,))
hash_mapping_y = torch.randint(low=0, high=10, size=(1000,))
hash_mapping_z = torch.randint(low=0, high=10, size=(1000,))
# [1000, 3]
coords = torch.stack(
    [
        trainable_coords[hash_mapping_x, 0],
        trainable_coords[hash_mapping_y, 1],
        trainable_coords[hash_mapping_z, 2],
    ],
    dim=1,
)


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 2000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
