import math
import os
import time
from pathlib import Path
from typing import Literal, Optional, List

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
import random

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything(42)


class TileTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        tile_weights: List[float] = [0.25, 0.25, 0.25, 0.25],
        num_points: int = 2000,
    ):
        assert np.isclose(sum(tile_weights), 1.0)
        self.tile_weights = tile_weights
        self.num_tiles = len(tile_weights) if self.tile_weights is not None else 1
        self.num_tiles_x = self.num_tiles_y = int(self.num_tiles ** .5)
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2
        if self.tile_weights is None:
            self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        else:
            n_points_added = 0
            self.means = torch.zeros(self.num_points, 3, device=self.device)
            start_idx = 0
            start_x = (-1 + 1 / self.num_tiles_x) * 8
            start_y = (-1 + 1 / self.num_tiles_y) * 8
            tile_width_x = 2 / self.num_tiles_x
            tile_width_y = 2 / self.num_tiles_y
            i = 0
            for r in range(self.num_tiles_y):
                for c in range(self.num_tiles_x):
                    num_points_in_tile = int(self.num_points * self.tile_weights[i])
                    n_points_added += num_points_in_tile
                    i += 1

                    center_x = start_x + c * tile_width_x * 8
                    center_y = start_y + r * tile_width_y * 8

                    bd = 2 / self.num_tiles_x
                    means = bd * (torch.rand(num_points_in_tile, 3, device=self.device) - 0.5)
                    means[:, 0] += center_x
                    means[:, 1] += center_y
                    self.means[start_idx:start_idx + num_points_in_tile] = means
                    print(f'num points added for tile of center ({center_x}, {center_y} and weight {self.tile_weights[i-1]}): {num_points_in_tile}')
                
                    start_idx += num_points_in_tile

        assert n_points_added == self.num_points
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
        self.opacities = torch.ones((self.num_points), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

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
        save_path: str = None,
        model_type: Literal["3dgs", "2dgs"] = "3dgs",
    ):
        losses = []
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 2  # rasterization, backward
        K = torch.tensor(
            [
                [self.focal, 0, self.W / 2],
                [0, self.focal, self.H / 2],
                [0, 0, 1],
            ],
            device=self.device,
        )

        if model_type == "3dgs":
            rasterize_fnc = rasterization
        elif model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        verbose=False
        from tqdm import tqdm
        pbar = tqdm(range(iterations))
        for iter in pbar:
            start = time.time()
            

            renders = rasterize_fnc(
                self.means,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.scales,
                torch.sigmoid(self.opacities),
                torch.sigmoid(self.rgbs),
                self.viewmat[None],
                K[None],
                self.W,
                self.H,
                packed=False,
            )[0]
            out_img = renders[0]
            torch.cuda.synchronize()
            times[0] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            if iter == 0:
                init_loss = loss
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[1] += time.time() - start
            optimizer.step()
            losses.append(loss.item())
            if verbose:
                print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
            if iter == 0:
                frame = Image.fromarray(frames[0])
                frame.save("./results2/frame0.png")
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            # out_dir = os.path.join(os.getcwd(), "results")
            out_dir = './results2'
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        
        if verbose:
            print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
            print(
            f"Per step(s):\nRasterization: {times[0]/iterations:.5f}, Backward: {times[1]/iterations:.5f}"
        )

        if save_path:
            final_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
            import matplotlib.pyplot as plt
            plt.imsave(save_path, final_img)
            
        return losses


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
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:

    img_path = f'examples/images/tiled_simple.png'

    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    n = 4
    weights = [1/n] * n
    trainer = TileTrainer(
        gt_image=gt_image,
        num_points=num_points,
        tile_weights=weights
    )
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=True,
        model_type=model_type,
    )


if __name__ == "__main__":
    tyro.cli(main)
