import torch
import numpy as np
import imageio
import os
from utils import sh_to_rgb
import imagecodecs


def compress_splats(splats, compress_dir):
    splats_keys = ["means", "scales", "sh0", "quats", "opacities"]
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    params = torch.cat([splats[k].reshape(n_gs, -1) for k in splats_keys], dim=-1)
    grid = params.reshape((n_sidelen, n_sidelen, -1))

    # Clamp to 1st and 99th percentile
    for i in range(grid.shape[-1]):
        grid[i] = torch.clamp(
            grid[i], torch.quantile(grid[i], 0.01), torch.quantile(grid[i], 0.99)
        )

    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    grid_norm = grid_norm.detach().cpu().numpy()
    for i in range(0, grid_norm.shape[-1], 3):
        img = (grid_norm[..., i : i + 3] * 255).astype(np.uint8)
        if img.shape[-1] != 3:
            img = np.concatenate(
                [img, np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)],
                axis=-1,
            )
        # imageio.imwrite(os.path.join(compress_dir, f"img_{i}.png"), img)
        imageio.imwrite(os.path.join(compress_dir, f"img_{i}.jpg"), img)


def decompress_splats(compress_dir):
    imgs = []
    for i in range(0, 14, 3):
        img = imageio.imread(os.path.join(compress_dir, f"img_{i}.jpg"))
        imgs.append(img)
    grid_norm = np.concatenate(imgs, axis=2)[..., :14]
    print(grid_norm.shape)


def main():
    device = "cuda:0"
    ckpt_path = "examples/results/360_v2/3dgs_sh0_sort/bicycle/ckpts/ckpt_29999.pt"
    out_dir = "examples/results/compress"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=device)
    compress_splats(ckpt["splats"], out_dir)
    decompress_splats(out_dir)

    splats = ckpt["splats"]
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    params = torch.cat([splats[k].reshape(n_gs, -1) for k in splats.keys()], dim=-1)
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    grid_rgb = sh_to_rgb(grid[:, :, -3:])
    grid_rgb = torch.clamp(grid_rgb, 0.0, 1.0)
    grid_rgb = grid_rgb.detach().cpu().numpy()
    grid_rgb = (grid_rgb * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(out_dir, "rgb.png"), grid_rgb)
    imageio.imwrite(os.path.join(out_dir, "rgb.jpg"), grid_rgb)
    # imagecodecs.imwrite(os.path.join(out_dir, "rgb.jxl"), grid_rgb)


if __name__ == "__main__":
    main()
