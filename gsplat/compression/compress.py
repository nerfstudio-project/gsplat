import os
import json
import torch
import numpy as np
import imageio


def _compress_means(grid, compress_dir):
    # Clamp to 1st and 99th percentile
    for i in range(grid.shape[-1]):
        grid[i] = torch.clamp(
            grid[i], torch.quantile(grid[i], 0.01), torch.quantile(grid[i], 0.99)
        )

    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    img_norm = grid_norm.detach().cpu().numpy()
    np.savez_compressed(os.path.join(compress_dir, "means.npz"), arr=img_norm)
    return grid_mins, grid_maxs


def _decompress_means(compress_dir, grid_mins, grid_maxs):
    img_norm = np.load(os.path.join(compress_dir, "means.npz"))["arr"]
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins
    return grid


def _compress_scales(grid, compress_dir):
    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "scales.png"), img)
    return grid_mins, grid_maxs


def _decompress_scales(compress_dir, grid_mins, grid_maxs):
    img = imageio.imread(os.path.join(compress_dir, "scales.png"))
    img_norm = img / 255.0
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins
    return grid


def _compress_sh0(grid, compress_dir):
    grid = grid.squeeze()
    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "sh0.png"), img)
    return grid_mins, grid_maxs


def _decompress_sh0(compress_dir, grid_mins, grid_maxs):
    img = imageio.imread(os.path.join(compress_dir, "sh0.png"))
    img_norm = img / 255.0
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins
    return grid


def _compress_quats(grid, compress_dir):
    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "quats.png"), img)
    return grid_mins, grid_maxs


def _decompress_quats(compress_dir, grid_mins, grid_maxs):
    img = imageio.imread(os.path.join(compress_dir, "quats.png"))
    img_norm = img / 255.0
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins
    return grid


def _compress_opacities(grid, compress_dir):
    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "opacities.png"), img)
    return grid_mins, grid_maxs


def _decompress_opacities(compress_dir, grid_mins, grid_maxs):
    img = imageio.imread(os.path.join(compress_dir, "opacities.png"))
    img_norm = img / 255.0
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins
    return grid


def compress_splats(splats, compress_dir):
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    attr_names = list(splats.keys())
    attr_names.remove("shN")

    bounds = {}
    shapes = {}
    for attr_name in attr_names:
        compress_fn = eval(f"_compress_{attr_name}")
        param = splats[attr_name]
        shapes[attr_name] = list(param.shape)

        grid = param.reshape(n_sidelen, n_sidelen, *param.shape[1:])
        attr_bounds = compress_fn(grid, compress_dir)
        bounds[attr_name] = {
            "mins": attr_bounds[0].tolist(),
            "maxs": attr_bounds[1].tolist(),
        }

    meta = {
        "attr_names": attr_names,
        "bounds": bounds,
        "shapes": shapes,
    }
    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def decompress_splats(compress_dir):
    with open(os.path.join(compress_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    attr_names = meta["attr_names"]
    bounds = meta["bounds"]
    shapes = meta["shapes"]

    splats = {}
    for attr_name in attr_names:
        decompress_fn = eval(f"_decompress_{attr_name}")
        grid_mins = torch.tensor(bounds[attr_name]["mins"], dtype=torch.float32)
        grid_maxs = torch.tensor(bounds[attr_name]["maxs"], dtype=torch.float32)
        grid = decompress_fn(compress_dir, grid_mins, grid_maxs)
        splats[attr_name] = grid.reshape(shapes[attr_name])
    return splats
