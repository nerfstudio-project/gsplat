import math
import os
from pathlib import Path
from typing import Optional, Literal

import torch
import numpy as np
import tyro
import matplotlib
from PIL import Image

from gsplat import rasterization, rasterization_2dgs

class SimpleTrainer:
    
    def __init__(
        self,
        num_points: int = 8,
    ):
        self.device = torch.device("cuda:0")
        self.num_points = 8
        
        fov_x = math.pi / 2.0
        self.H, self.W = 256, 256
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)
        
        self._init_gaussians()
    
    def _init_gaussians(self):
        length = 0.4
        x = np.linspace(-1, 1, self.num_points)
        y = np.linspace(-1, 1, self.num_points)
        x, y = np.meshgrid(x, y)
        means3D = torch.from_numpy(np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)).cuda().float()
        quats = torch.zeros(1, 4).repeat(len(means3D), 1).cuda()
        quats[..., 0] = 1.
        scale = 0.6 / (self.num_points - 1)
        scales = torch.zeros(1, 3).repeat(len(means3D), 1).fill_(scale).cuda()
        colors = matplotlib.colormaps['Accent'](np.random.randint(1, 64, 64) / 64)[..., :3]
        colors = torch.from_numpy(colors).cuda()
        opacity = torch.ones_like(means3D[:, 0])
        
        self.viewmat = torch.tensor(
            [[-8.6086e-01,  3.7950e-01, -3.3896e-01,  6.7791e-01],
            [ 5.0884e-01,  6.4205e-01, -5.7346e-01,  1.1469e+00],
            [ 1.0934e-08, -6.6614e-01, -7.4583e-01,  1.4917e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],
            device=self.device,
        )
        
        self.means = means3D.float()
        self.scales = scales.float()
        self.rgbs = colors.float()
        self.opacities = opacity.float()
        self.quats = quats.float()
    
    def render(
        self,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        frames = []
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )
        if model_type == "3dgs":
            renders, _, _ = rasterization(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                self.opacities,
                self.rgbs,
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )

        elif model_type == "2dgs":
            renders, _, _ = rasterization_2dgs(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                self.opacities,
                self.rgbs,
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )
        else:
            raise NotImplementedError("Model not implemented")
        out_img = renders[0]
        torch.cuda.synchronize()
        
        frame = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
        frame_img = Image.fromarray(frame)
        out_dir = os.path.join(os.getcwd(), "renders")
        os.makedirs(out_dir, exist_ok=True)
        frame_img.save(f"{out_dir}/{model_type}.png")

def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 8,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:
    trainer = SimpleTrainer(num_points=num_points)
    trainer.render(
        model_type=model_type,
    )
    
if __name__ == "__main__":
    tyro.cli(main)