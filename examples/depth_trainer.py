import math
import os
import time
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import _ProjectGaussians
from gsplat.rasterize import _RasterizeGaussians
from PIL import Image
from torch import Tensor, optim
import matplotlib


class DepthTrainer:
    """Trains random gaussians to fit a depth image."""

    def __init__(
        self,
        gt_depth: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_depth = gt_depth.to(device=self.device)
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        fov_x = math.pi / 2.0
        self.H, self.W = gt_depth.shape[0], gt_depth.shape[1]
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
        bd = 5

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device) * 0.01
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 4  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
                self.means,
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
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()

            # Depth + RGB rasterization
            # TODO: come up with a cleaner solution now that return_alpha has been merged to main
            _, _, out_depth = _RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                self.background,
                False,  # return alphas
                True,  # return depths
            )
            torch.cuda.synchronize()
            times[1] += time.time() - start

            start = time.time()
            # RGB only rasterization
            rgb_forward = _RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                self.background,
                False,  # return alphas
                False,  # return depths
            )
            # RGB only rasterization
            depth_forward = _RasterizeGaussians.apply(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths.repeat(1, 1, 3),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                self.background,
                False,  # return alphas
                False,  # return depths
            )
            times[3] += time.time() - start

            loss = mse_loss(out_depth, self.gt_depth)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()

            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                depth_colored = apply_float_colormap(out_depth)
                frames.append(
                    (depth_colored.detach().cpu().numpy() * 255).astype(np.uint8)
                )
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/depth_training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Unified RGB+Depth Rasterization: {times[1]:.3f}, Separate RGB and Depth RGB Rasterization: {times[3]:.3f}, RGB+Depth Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Unified RGB+Depth Rasterization: {times[1]/iterations:.5f}, Separate RGB and Depth Rasterization: {times[3]/iterations:.5f}, RGB+Depth Backward: {times[2]/iterations:.5f}"
        )


def apply_float_colormap(image, colormap: Literal["turbo", "grey"] = "turbo"):
    colormap = "turbo"
    image = image[..., None]
    image = image - torch.min(image)
    image = image / (torch.max(image) + 1e-5)
    image = torch.clip(image, 0, 1)
    image = torch.nan_to_num(image, 0)
    if colormap == "gray":
        return image.repeat(1, 1, 3)
    image_long = (image * 255).long()
    image_long_min = torch.min(image_long)
    image_long_max = torch.max(image_long)
    assert image_long_min >= 0, f"the min value is {image_long_min}"
    assert image_long_max <= 255, f"the max value is {image_long_max}"
    return torch.tensor(matplotlib.colormaps[colormap].colors, device=image.device)[
        image_long[..., 0]
    ]


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 10000,  # 10000
    save_imgs: bool = True,
    iterations: int = 500,
    lr: float = 0.01,
) -> None:

    # artificial depth image
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    gt_depth = np.sin(5 * np.sqrt(x**2 + y**2)) + 0.5 * np.sin(
        15 * np.sqrt(x**2 + y**2)
    )
    gt_depth = (gt_depth - np.min(gt_depth)) / (np.max(gt_depth) - np.min(gt_depth))
    gt_depth = torch.from_numpy(gt_depth).float()

    trainer = DepthTrainer(gt_depth=gt_depth, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
