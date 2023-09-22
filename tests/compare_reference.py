import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import imageio
from torch import Tensor, optim

from diff_rast.rasterize import RasterizeGaussians
from diff_rast.project_gaussians import ProjectGaussians
from ref_rast.rasterize import _RasterizeGaussians, GaussianRasterizationSettings


def random_quat_tensor(N, **kwargs):
    u = torch.rand(N, **kwargs)
    v = torch.rand(N, **kwargs)
    w = torch.rand(N, **kwargs)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
        ],
        dim=-1,
    )


def identity_quat(N, **kwargs):
    quat = torch.zeros(N, 4, **kwargs)
    quat[:, 0] = 1
    return quat


def projection_matrix(znear, zfar, fovx, fovy, **kwargs):
    t = math.tan(0.5 * fovy)
    b = -t
    r = math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, -(f + n) / (f - n), -2.0 * f * n / (f - n)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        **kwargs
    )


class CompareReference:
    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 1024,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        BLOCK_X, BLOCK_Y = 16, 16
        H, W = gt_image.shape[:2]
        fovx = math.pi / 2.0
        fx = fy = 0.5 * float(W) / math.tan(0.5 * fovx)
        fovy = 2.0 * math.atan(0.5 * float(H) / fy)
        self.fovx = fovx
        self.fovy = fovy
        self.scale_mod = 1.0
        self.tanfovx = 0.5 * W / fx
        self.tanfovy = 0.5 * H / fy
        self.W, self.H = W, H
        self.focal = fx
        self.tile_bounds = (
            (self.W + BLOCK_X - 1) // BLOCK_X,
            (self.H + BLOCK_Y - 1) // BLOCK_Y,
            1,
        )
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        self.block = torch.tensor([BLOCK_X, BLOCK_Y, 1], device=self.device)
        self.background = torch.zeros(3, device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        self.means = 2 * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        # self.quats = random_quat_tensor(self.num_points, device=self.device)
        self.quats = identity_quat(self.num_points, device=self.device)
        self.rgbs = torch.rand(self.num_points, 3, device=self.device)
        self.opacities = 0.5 * torch.ones(self.num_points, 1, device=self.device)
        self.shs = torch.Tensor([])
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        pmat = projection_matrix(0.1, 1e4, self.fovx, self.fovy, device=self.device)
        self.projmat = pmat @ self.viewmat
        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def forward_ours(self):
        xys, depths, radii, conics, num_tiles_hit = ProjectGaussians.apply(
            self.means,
            self.scales,
            self.scale_mod,
            # self.quats,
            self.quats / torch.norm(self.quats, dim=-1, keepdim=True),
            self.viewmat,
            self.projmat,
            self.focal,
            self.focal,
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
            # self.rgbs,
            # self.opacities,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.H,
            self.W,
            self.background,
        )

    def forward_ref(self):
        settings = GaussianRasterizationSettings(
            self.H,
            self.W,
            self.tanfovx,
            self.tanfovy,
            self.background,
            self.scale_mod,
            self.viewmat.T,
            self.projmat.T,
            sh_degree=0,
            campos=torch.zeros(3, device=self.device),
            prefiltered=False,
            debug=True,
        )
        means2d = torch.Tensor([])
        covs3d = torch.Tensor([])
        out_color, _ = _RasterizeGaussians.apply(
            self.means,
            means2d,
            self.shs,
            # self.rgbs,
            # self.opacities,
            torch.sigmoid(self.rgbs),
            torch.sigmoid(self.opacities),
            self.scales,
            # self.quats,
            self.quats / torch.norm(self.quats, dim=-1, keepdim=True),
            covs3d,
            settings,
        )  # (C, H, W)
        return out_color.permute(1, 2, 0)

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = True,
    ):
        names = ["rgbs", "means", "scales", "opacities", "quats"]
        params = [self.rgbs, self.means, self.scales, self.opacities, self.quats]
        optimizer = optim.Adam(params, lr)  # try training self.opacities/scales etc.
        mse_loss = torch.nn.MSELoss()
        os.makedirs("renders", exist_ok=True)
        our_writer = imageio.get_writer("renders/ours.gif")
        ref_writer = imageio.get_writer("renders/refs.gif")
        for iter in range(iterations):
            our_img = self.forward_ours()
            ref_img = self.forward_ref()
            diff = our_img - ref_img

            ref_loss = mse_loss(ref_img, self.gt_image)
            optimizer.zero_grad()
            ref_loss.backward()
            ref_grads = [x.grad for x in params]

            loss = mse_loss(our_img, self.gt_image)
            optimizer.zero_grad()
            loss.backward()
            our_grads = [x.grad for x in params]

            optimizer.step()

            print("LOSS", loss.item())
            print("our_img", our_img.min().item(), our_img.max().item())
            print("ref_img", ref_img.min().item(), ref_img.max().item())
            print("diff", diff.min().item(), diff.max().item())
            for i in range(len(params)):
                if our_grads[i] is None:
                    continue
                print(
                    names[i],
                    "our grads",
                    our_grads[i].min().item(),
                    our_grads[i].max().item(),
                )
                print(
                    names[i],
                    "ref grads",
                    ref_grads[i].min().item(),
                    ref_grads[i].max().item(),
                )

            if save_imgs and iter % 5 == 0:
                our_writer.append_data(
                    (our_img.detach().cpu().numpy() * 255).astype(np.uint8)
                )
                ref_writer.append_data(
                    (ref_img.detach().cpu().numpy() * 255).astype(np.uint8)
                )
        our_writer.close()
        ref_writer.close()


if __name__ == "__main__":
    gt_image = torch.ones((256, 256, 3)) * 1.0
    # make top left and bottom right red,blue
    gt_image[:128, :128, :] = torch.tensor([1.0, 0.0, 0.0])
    gt_image[128:, 128:, :] = torch.tensor([0.0, 0.0, 1.0])
    trainer = CompareReference(gt_image=gt_image)

    trainer.train()
