import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import _ProjectGaussians
from gsplat.rasterize import _RasterizeGaussians
from PIL import Image
import pytorch3d.transforms as p3d
from torch import Tensor, optim
import torch.nn.functional as F


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
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
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
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.noisy_pose = self.viewmat.clone()
        rot_noise = p3d.euler_angles_to_matrix(
            bd
            * (torch.rand(3, device=self.device) - 0.5)
            * math.pi
            / 36.0,  # +-5 degrees
            convention="XYZ",
        )
        self.noisy_pose[:3, :3] = torch.matmul(self.noisy_pose[:3, :3], rot_noise)
        self.noisy_pose[:3, 3] += (torch.rand(3, device=self.device) - 0.5) * bd  # +-1 units
        self.pose_params = p3d.se3_log_map(
            self.noisy_pose.unsqueeze(0).permute(0, 2, 1)
        ).squeeze(0)
        self.background = torch.zeros(3, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False
        self.noisy_pose.requires_grad = False
        self.pose_params.requires_grad = True

    @staticmethod
    def save_image(img: Tensor, fname: str):
        out_img = Image.fromarray((img.detach().cpu().numpy() * 255).astype(np.uint8))
        out_dir = os.path.join(os.getcwd(), "renders")
        os.makedirs(out_dir, exist_ok=True)
        out_img.save(f"{out_dir}/{fname}.png")

    def train(self, iterations: int = 1000, lr: float = 0.01, save_imgs: bool = False):
        # optimize gaussians
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
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
            out_img = _RasterizeGaussians.apply(
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
            )
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )

        # save final image as the objective image
        self.save_image(out_img, "objective")

        # save image prior to camera pose optimazation
        xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
            self.means,
            self.scales,
            1,
            self.quats,
            p3d.se3_exp_map(self.pose_params.unsqueeze(0)).squeeze(0).permute(1, 0),
            p3d.se3_exp_map(self.pose_params.unsqueeze(0)).squeeze(0).permute(1, 0),
            self.focal,
            self.focal,
            self.W / 2,
            self.H / 2,
            self.H,
            self.W,
            self.tile_bounds,
        )
        torch.cuda.synchronize()
        out_img = _RasterizeGaussians.apply(
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
        )
        torch.cuda.synchronize()
        self.save_image(out_img, "pose_before_opt")

        # optimize camera pose
        optimizer = optim.Adam([self.pose_params], lr)
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        for iter in range(iterations):
            start = time.time()
            xys, depths, radii, conics, num_tiles_hit, cov3d = _ProjectGaussians.apply(
                self.means,
                self.scales,
                1,
                self.quats,
                p3d.se3_exp_map(self.pose_params.unsqueeze(0)).squeeze(0).permute(1, 0),
                p3d.se3_exp_map(self.pose_params.unsqueeze(0)).squeeze(0).permute(1, 0),
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
            out_img = _RasterizeGaussians.apply(
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
            )
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

        final_pose = (
            p3d.se3_exp_map(self.pose_params.unsqueeze(0)).squeeze(0).permute(1, 0)
        )
        print("target pose:\n", self.viewmat)
        print("initial pose:\n", self.noisy_pose)
        print("initial residual:\n", self.noisy_pose - self.viewmat)
        print("final pose:\n", final_pose)
        print("final residual:\n", final_pose - self.viewmat)
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/pose_training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        self.save_image(out_img, "pose_after_opt")


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
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
