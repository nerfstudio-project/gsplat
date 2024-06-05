import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim

import matplotlib
import pdb

# os.environ['CUDA_ENABLE_COREDUMP_ON_EXCEPTION']='1'
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    import math
    return 2*math.atan(pixels/(2*focal))

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
        toy_example: bool=False,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)

        if toy_example:
            self._toy_init()
        else:
            self.num_points = num_points

            fov_x = math.pi / 2.0
            self.H, self.W = gt_image.shape[0], gt_image.shape[1]
            self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
            self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

            self._init_gaussians()

    def _toy_init(self):
        num_points = 64
        length = 0.5
        width = height = 512
        x = np.linspace(-1, 1, num_points) 
        y = np.linspace(-1, 1, num_points)
        x, y = np.meshgrid(x, y)
        means3D = torch.from_numpy(np.stack([x, y, 0], axis=-1).reshape(-1,3)).cuda().float()
        quats = torch.zeros(1,4).repeat(len(means3D), 1).cuda()
        quats[..., 0] = 1.
        scale = length / (num_points-1)
        scale = 5
        scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(scale).cuda()
        colors = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
        colors = torch.from_numpy(colors).cuda()

        opacity = torch.ones_like(means3D[:,:1])

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )        

        self.H = height
        self.W = width
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        print(self.focal)
        self.focal = 711.1111
        fov_x = focal2fov(self.focal)
        self.means = means3D.float()
        self.scales = scales.float()
        self.rgbs = colors.float()
        self.opacities = opacity.float()
        self.quats = quats.float()
        self.background = torch.zeros(3, device=self.device).float()
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
    
    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

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
        self.viewmat = self.viewmat
        # print(self.viewmat)
        self.background = torch.zeros(d, device=self.device)


        x_range = torch.linspace(-1, 1, self.gt_image.shape[0])
        y_range = torch.linspace(-1, 1, self.gt_image.shape[1])

        x, y = torch.meshgrid(x_range, y_range)

        z = torch.ones_like(x)

        # pdb.set_trace()
        # self.means = torch.stack((x, y, z), dim=-1).reshape((-1, 3)) #TODO (WZ): each pixel has one gaussian; normalized to [-1, 1]
        # self.means = self.means.to(self.device)
        # self.rgbs = self.gt_image.clone().permute((1, 0, 2)).reshape((-1, 3))
        # self.rgbs = self.rgbs.to(self.device)
        # self.scales = torch.ones_like(self.scales) * 1e-4
        # self.scales.to(self.device)



        ########### Toy Example ###############
        # num_points = 8
        # length = 0.4
        # x = np.linspace(-1, 1, num_points)
        # y = np.linspace(-1, 1, num_points)
        # x, y = np.meshgrid(x, y)
        # means3D = torch.from_numpy(np.stack([x, y, np.ones_like(x) ], axis=-1).reshape(-1,3)).cuda().float()
        # quats = torch.zeros(1,4).repeat(len(means3D), 1).cuda()
        # quats[..., 0] = 1.
        # scale = 0.6 / (num_points-1)
        # # scale = 1e-3
        # # scale = 0
        # scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(scale).cuda()
        # colors = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
        # colors = torch.from_numpy(colors).cuda()
        # opacity = torch.ones_like(means3D[:,:1])

        # self.means = means3D.float()
        # self.scales = scales.float()
        # self.rgbs = colors.float()
        # self.opacities = opacity.float()
        # self.quats = quats.float()
        # u = torch.ones_like(u)
        # v = torch.ones_like(v)
        # w = torch.ones_like(w)
        # self.quats = torch.cat(
        #     [
        #         torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        #         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        #         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        #         torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        #     ],
        #     -1,
        # )
        # self.quats.to(self.device)
        ##############

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False


    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        B_SIZE: int = 14,
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16
        # with torch.no_grad():
        for iter in range(iterations):
            start = time.time()
            # pdb.set_trace()
            (   xys,
                depths,
                radii,
                num_tiles_hit,
                cov3d,
                ray_transformations
            )  = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
            )

            # pdb.set_trace()
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()

            # pdb.set_trace()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                num_tiles_hit,
                ray_transformations,
                # self.rgbs,
                # self.opacities,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                B_SIZE,
                self.background,
            )[..., :3]
            # pdb.set_trace()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            # pdb.set_trace()
            loss.backward()
            # print("after backward")
            # print("Here!!!")
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))

            # break
        #Test
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
    # height = 512
    # width = 512
    iterations=5000
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    num_points = gt_image.shape[0] * gt_image.shape[1]
    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)
