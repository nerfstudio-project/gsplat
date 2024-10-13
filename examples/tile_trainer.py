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
        tile_weights: List[float] = [.1, .4, .4, .1],
        num_points: int = 2000,
    ):
        assert sum(tile_weights) == 1.0
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
            start_x = -1
            start_y = 1
            tile_width_x = 2 / self.num_tiles_x
            tile_width_y = 2 / self.num_tiles_y
            i = 0
            for r in range(self.num_tiles_y):
                for c in range(self.num_tiles_x):
                    num_points_in_tile = int(self.num_points * self.tile_weights[i])
                    n_points_added += num_points_in_tile
                    i += 1

                    center_x = start_x + (c + 0.5) * tile_width_x
                    center_y = start_y - (r + 0.5) * tile_width_y

                    bd = tile_width_x
                    means = bd * (torch.rand(num_points_in_tile, 3, device=self.device) - 0.5)
                    means[:, 0] += center_x
                    means[:, 1] += center_y
                    self.means[start_idx:start_idx + num_points_in_tile] = means
                    print(f'num points added for tile of center ({center_x:.2f}, {center_y:.2f}) and weight {self.tile_weights[i-1]}: {num_points_in_tile}')
                
                    start_idx += num_points_in_tile
        import matplotlib.pyplot as plt

        means_np = self.means.detach().cpu().numpy()
        gt_image_np = self.gt_image.detach().cpu().numpy()

        # Create the plot
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(gt_image_np)

        # Scale and shift the points to match image coordinates
        scaled_x = (means_np[:, 0] + 1) * self.W / 2
        scaled_y = (-means_np[:, 1] + 1) * self.H / 2

        # Determine colors for each tile based on the indices
        colors = []
        start_idx = 0
        i = 0
        for r in range(self.num_tiles_y):
            for c in range(self.num_tiles_x):
                num_points_in_tile = int(self.num_points * self.tile_weights[i])
                end_idx = start_idx + num_points_in_tile
                if r == 0 and c == 0:
                    colors.extend(['red'] * num_points_in_tile)  # Bottom-left
                elif r == 0 and c == self.num_tiles_x - 1:
                    colors.extend(['yellow'] * num_points_in_tile)  # Bottom-right
                elif r == self.num_tiles_y - 1 and c == 0:
                    colors.extend(['green'] * num_points_in_tile)  # Top-left
                elif r == self.num_tiles_y - 1 and c == self.num_tiles_x - 1:
                    colors.extend(['blue'] * num_points_in_tile)  # Top-right
                else:
                    colors.extend(['gray'] * num_points_in_tile)  # Other tiles
                start_idx = end_idx
                i += 1

        # Plot the scaled points with different colors for each tile
        ax.scatter(scaled_x, scaled_y, c=colors, s=1, alpha=0.5)

        # Set the axis limits to match the image dimensions
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)  # Reverse Y-axis to match image coordinates

        # Save the plot
        out_dir = os.path.join(os.getcwd(), "results4")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/temp.png", dpi=300, bbox_inches='tight')
        plt.close()

        
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
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            # out_dir = os.path.join(os.getcwd(), "results")
            out_dir = './results3'
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
        final_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
        return losses, final_img


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 10000,
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

    trainer = TileTrainer(gt_image=gt_image, num_points=num_points)
    losses, final_img = trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
        model_type=model_type,
    )
    print(f'final loss: {losses[-1]}, loss reduction of {losses[0] - losses[-1]}')
    weights_str = "_".join(map(str, trainer.tile_weights))
    loss_reduction = losses[0] - losses[-1]
    save_image_path = f"{weights_str}_{loss_reduction:.2f}_loss.png"
    import matplotlib.pyplot as plt
    plt.imsave(save_image_path, final_img)


if __name__ == "__main__":
    tyro.cli(main)
