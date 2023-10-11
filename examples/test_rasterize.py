import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from diff_rast.project_gaussians import ProjectGaussians
from diff_rast.rasterize import RasterizeGaussians
from diff_rast.slow_rasterize import SlowRasterizeGaussians
from PIL import Image
from torch import Tensor, optim


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
        self.means = torch.empty((self.num_points, 3), device=self.device)
        self.scales = torch.empty((self.num_points, 3), device=self.device)
        self.quats = torch.empty((self.num_points, 4), device=self.device)
        self.rgbs = torch.ones((self.num_points, 3), device=self.device)
        self.opacities = torch.ones((self.num_points, 1), device=self.device)
        bd = 2
        for i in range(self.num_points):
            self.means[i] = torch.tensor(
                [
                    bd * (random.random() - 0.5),
                    bd * (random.random() - 0.5),
                    bd * (random.random() - 0.5),
                ],
                device=self.device,
            )
            self.scales[i] = torch.tensor(
                [random.random(), random.random(), random.random()], device=self.device
            )
            self.rgbs[i] = torch.tensor(
                [random.random(), random.random(), random.random()], device=self.device
            )
            u = random.random()
            v = random.random()
            w = random.random()
            self.quats[i] = torch.tensor(
                [
                    math.sqrt(1.0 - u) * math.sin(2.0 * math.pi * v),
                    math.sqrt(1.0 - u) * math.cos(2.0 * math.pi * v),
                    math.sqrt(u) * math.sin(2.0 * math.pi * w),
                    math.sqrt(u) * math.cos(2.0 * math.pi * w),
                ],
                device=self.device,
            )

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def forward_new(self):
        means2d = torch.zeros_like(self.means[:, :2])
        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            self.means,
            means2d,
            self.scales,
            1,
            self.quats,
            self.viewmat,
            self.viewmat,
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            self.tile_bounds,
        )

        return RasterizeGaussians.apply(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.H,
            self.W,
        )

    def forward_slow(self):
        means2d = torch.zeros_like(self.means[:, :2])
        xys, depths, radii, conics, num_tiles_hit, cov3d = ProjectGaussians.apply(
            self.means,
            means2d,
            self.scales,
            1,
            self.quats,
            self.viewmat,
            self.viewmat,
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            self.tile_bounds,
        )

        return SlowRasterizeGaussians.apply(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.H,
            self.W,
        )

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = True):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        for i in range(iterations):
            optimizer.zero_grad()
            slow_out = self.forward_slow()

            loss = mse_loss(slow_out, self.gt_image)
            loss.backward()
            slow_grads = [
                self.means.grad.detach(),
                self.scales.grad.detach(),
                self.quats.grad.detach(),
                self.rgbs.grad.detach(),
                self.opacities.grad.detach(),
            ]

            optimizer.zero_grad()
            new_out = self.forward_new()
            loss = mse_loss(new_out, self.gt_image)
            loss.backward()

            new_grads = [
                self.means.grad.detach(),
                self.scales.grad.detach(),
                self.quats.grad.detach(),
                self.rgbs.grad.detach(),
                self.opacities.grad.detach(),
            ]

            diff_out = slow_out.detach() - new_out.detach()  # type: ignore
            print("OUT DIFF:", diff_out.min(), diff_out.max())
            for slow_grad, new_grad in zip(slow_grads, new_grads):
                diff_grad = slow_grad - new_grad
                print("GRAD DIFF:", diff_grad.min(), diff_grad.max())
            optimizer.step()
            print(f"ITER {i}/{iterations}, LOSS: {loss.item()}")


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
    trainer.train()


if __name__ == "__main__":
    tyro.cli(main)
