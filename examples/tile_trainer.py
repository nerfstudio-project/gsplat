import math
import os
import time
from pathlib import Path
from typing import Literal, Optional, List
import matplotlib.pyplot as plt

import numpy as np
import torch
import tyro
from PIL import Image
from torch import Tensor, optim

from gsplat import rasterization, rasterization_2dgs
import random

def weights_to_str(weights):
        if weights is None:
            weights_str = 'random'
        else:
            weights_str = '_'.join([str(w) for w in weights])
        return weights_str

def seed_everything(seed: int):
    print(f"Seeding with {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

seed_everything(42)


class TileTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        tile_weights: List[float] = [0.25, 0.25, 0.25, 0.25],
        num_points: int = 2000,
        seed: int = 42,
    ):
        if tile_weights:
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

        self.seed = seed
        seed_everything(self.seed)
        self._init_gaussians()
        # print(f"Initial gaussians: {self.gaussians}")

    def _init_gaussians(self):
        """Random gaussians"""
        seed_everything(42)
        bd = 2
        # if self.tile_weights is None:
        #     self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        # else:
        #     n_points_added = 0
        #     self.means = torch.zeros(self.num_points, 3, device=self.device)
        #     start_idx = 0
        #     start_x = (-1 + 1 / self.num_tiles_x) * 8
        #     start_y = (-1 + 1 / self.num_tiles_y) * 8
        #     tile_width_x = 2 / self.num_tiles_x
        #     tile_width_y = 2 / self.num_tiles_y
        #     i = 0
        #     for r in range(self.num_tiles_y):
        #         for c in range(self.num_tiles_x):
        #             num_points_in_tile = int(self.num_points * self.tile_weights[i])
        #             n_points_added += num_points_in_tile
        #             i += 1

        #             center_x = start_x + c * tile_width_x * 8
        #             center_y = start_y + r * tile_width_y * 8

        #             bd = 2 / self.num_tiles_x
        #             means = bd * (torch.rand(num_points_in_tile, 3, device=self.device) - 0.5)
        #             means[:, 0] += center_x
        #             means[:, 1] += center_y
        #             self.means[start_idx:start_idx + num_points_in_tile] = means
        #             print(f'num points added for tile of center ({center_x}, {center_y} and weight {self.tile_weights[i-1]}): {num_points_in_tile}')
                
        #             start_idx += num_points_in_tile

        #     assert n_points_added == self.num_points
        # self.scales = torch.rand(self.num_points, 3, device=self.device)
        # d = 3
        # self.rgbs = torch.rand(self.num_points, d, device=self.device)

        # u = torch.rand(self.num_points, 1, device=self.device)
        # v = torch.rand(self.num_points, 1, device=self.device)
        # w = torch.rand(self.num_points, 1, device=self.device)
        seed_everything(self.seed)
        self.means = torch.tensor([[0.0, 0.0, 0.0]] * self.num_points, device="cuda")
        self.scales = torch.tensor([[0.1, 0.1, 0.1]] * self.num_points, device="cuda")
        self.quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * self.num_points, device="cuda")
        self.opacities = torch.tensor([0.5] * self.num_points, device="cuda")
        self.rgbs = torch.tensor([[0.5, 0.5, 0.5]] * self.num_points, device="cuda")
        # for k, v in self.gaussians.items():
        #     v.requires_grad = True

        # self.quats = torch.cat(
        #     [
        #         torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
        #         torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
        #         torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
        #         torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
        #     ],
        #     -1,
        # )
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
        d=3
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
            )

            render_colors, render_alphas, meta = renders
            print("Rasterize:")
            print("render_colors: ", render_colors)
            print("render_alphas: ", render_alphas)
            print("meta: ", meta.keys())
            print("***************************************")

            renders = renders[0]
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

            if save_imgs and iter % 25 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
            if iter == 0:
                frame = Image.fromarray(frames[0])
                frame.save("results5/frame0.png")
        if save_imgs:
            
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            # out_dir = os.path.join(os.getcwd(), "results")
            out_dir = 'results5'
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
        final_img = (out_img.detach().cpu().numpy() * 255).astype(np.uint8)
        
        if save_path:    
            
            plt.imsave(save_path, final_img)
            
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
    num_points: int = 60,
    save_imgs: bool = True,
    weights: List[float] = [0.25, 0.25, 0.25, 0.25],
    random: bool = False,

    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
    model_type: Literal["3dgs", "2dgs"] = "3dgs",
) -> None:

    seed_everything(42)

    img_path = f'examples/images/tiled_simple.png'

    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])
    if random:
        weights = None
    print(f'Weights: {weights}')
    trainer = TileTrainer(
        gt_image=gt_image,
        num_points=num_points,
        tile_weights=weights
    )
    losses, final_img = trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=True,
        model_type=model_type,
    )
    mse = losses[-1]
    psnr = 10 * np.log10(1.0 / mse)
    print(f'PSNR: {psnr}')
    OUTPUT_DIR = f'{num_points}_points_{iterations}_iter'
    # make dir if it doens't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_PLOTS_DIR = f'{OUTPUT_DIR}/plots'
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)
    OUTPUTS_IMG_DIR = f'{OUTPUT_DIR}/final_outputs'
    os.makedirs(OUTPUTS_IMG_DIR, exist_ok=True)
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss curve: {num_points} points, {iterations} iter, of {(losses[0] - losses[-1]):.3f}, PSNR of {psnr:.3f}')
    # make a weights_str that takes all elements and separates them with a comma
    plt.savefig(f'{OUTPUT_PLOTS_DIR}/train_loss_{weights_to_str(weights)}_weights.png')
    plt.show()

    
    # plot psnr
    plt.figure()
    losses = np.array(losses)
    psnrs = 10 * np.log10(1.0 / losses)
    plt.plot(psnrs)
    plt.xlabel('Iteration')
    plt.ylabel('PSNR')
    plt.title(f'PSNR curve" {num_points} points,  {iterations} iter, reduction of {(losses[0] - losses[-1]):.3f}, PSNR of {psnr:.3f}')
    plt.savefig(f'{OUTPUT_PLOTS_DIR}/psnr_{weights_to_str(weights)}_weights.png')

    plt.imsave(f'{OUTPUTS_IMG_DIR}/{psnr:.3f}_psnr_{weights_to_str(weights)}_weights.png', final_img)

if __name__ == "__main__":
    tyro.cli(main)
