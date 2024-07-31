import os
from typing import Any
import json
import torch
from torch import Tensor
import torch.nn.functional as F
from torchpq.clustering import KMeans
import numpy as np
import imageio

from plas import sort_with_plas
from gsplat.utils import sh_to_rgb, rgb_to_sh, log_transform, inverse_log_transform


def compress_splats(
    compress_dir: str,
    splats: dict[str, Tensor],
    use_sort: bool = True,
    use_kmeans: bool = True,
) -> None:
    """Compress splats with quantization, sorting, and K-means clustering of the spherical harmonic coefficents.

    Args:
        compress_dir (str): compression directory
        splats (dict[str, Tensor]): splats
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        use_kmeans (bool, optional): Whether to use K-means to compress spherical harmonics. Defaults to True.
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_diff = n_gs - n_sidelen**2
    if n_diff != 0:
        opacities = splats["opacities"]
        keep_indices = torch.argsort(opacities, descending=True)[:-n_diff]
        for k, v in splats.items():
            splats[k] = v[keep_indices]
        print(f"Number of Gaussians was not square. Removed {n_diff} Gaussians.")

    if use_sort:
        splats = sort_splats(splats)
    meta = {}
    for param_name in splats.keys():
        compress_fn = eval(f"_compress_{param_name}")
        kwargs = {}
        if param_name == "shN":
            kwargs["use_kmeans"] = use_kmeans
        meta[param_name] = compress_fn(compress_dir, splats[param_name], **kwargs)

    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def decompress_splats(compress_dir: str) -> dict[str, Tensor]:
    """Decompress splats from directory.

    Args:
        compress_dir (str): compression directory
        use_kmeans (bool, optional): Whether to use K-means to compress spherical harmonics. Defaults to True.

    Returns:
        dict[str, Tensor]: splats
    """
    with open(os.path.join(compress_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    splats = {}
    for param_name, param_meta in meta.items():
        decompress_fn = eval(f"_decompress_{param_name}")
        splats[param_name] = decompress_fn(compress_dir, param_meta)
    return splats


def sort_splats(splats: dict[str, Tensor], verbose: bool = False) -> dict[str, Tensor]:
    """Sort splats with Parallel Linear Assignment Sorting from the paper `Compact 3D Scene Representation via
    Self-Organizing Gaussian Grids <https://arxiv.org/pdf/2312.13299>`_.

    Args:
        splats (dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to False.

    Returns:
        dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    sort_keys = [k for k in splats if k != "shN"]
    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten()
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats


def _compress_means(compress_dir: str, params: Tensor) -> dict[str, Any]:
    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    params = log_transform(params)
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(os.path.join(compress_dir, "means_l.png"), img_l.astype(np.uint8))
    imageio.imwrite(os.path.join(compress_dir, "means_u.png"), img_u.astype(np.uint8))

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_means(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img_l = imageio.imread(os.path.join(compress_dir, "means_l.png")).astype(np.uint16)
    img_u = imageio.imread(os.path.join(compress_dir, "means_u.png")).astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    params = inverse_log_transform(params)
    return params


def _compress_scales(compress_dir: str, params: Tensor) -> dict[str, Any]:
    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "scales.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_scales(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "scales.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_quats(compress_dir: str, params: Tensor) -> Tensor:
    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    grid_norm = F.normalize(grid, dim=-1)
    grid_norm = grid_norm * 0.5 + 0.5
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "quats.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_quats(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "quats.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm)
    grid = grid_norm * 2 - 1

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_opacities(compress_dir: str, params: Tensor) -> Tensor:
    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    grid = params.reshape((n_sidelen, n_sidelen))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "opacities.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_opacities(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "opacities.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_sh0(compress_dir: str, params: Tensor) -> Tensor:
    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    rgb = sh_to_rgb(params)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    grid = rgb.reshape((n_sidelen, n_sidelen, -1))
    grid = grid.detach().cpu().numpy()
    img = (grid * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "sh0.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_sh0(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "sh0.png"))
    grid = img / (2**8 - 1)
    grid = torch.tensor(grid)
    params = rgb_to_sh(grid)

    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_shN(compress_dir: str, params: Tensor, use_kmeans: bool = True) -> Tensor:
    shape = params.shape
    dtype = params.dtype
    if params.numel() == 0:
        meta = {
            "shape": list(shape),
            "dtype": str(dtype).split(".")[1],
        }
        return meta

    if use_kmeans:
        kmeans = KMeans(n_clusters=2**16, distance="manhattan", verbose=True)
        x = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
        labels = kmeans.fit(x)
        labels = labels.detach().cpu().numpy()
        labels = labels.astype(np.uint16)
        params = kmeans.centroids.permute(1, 0)

    mins = torch.min(params)
    maxs = torch.max(params)
    params_norm = (params - mins) / (maxs - mins)
    params_norm = params_norm.detach().cpu().numpy()
    params_quant = (params_norm * (2**6 - 1)).round().astype(np.uint8)

    npz_dict = {"params": params_quant}
    if use_kmeans:
        npz_dict["labels"] = labels
    np.savez_compressed(os.path.join(compress_dir, "shN.npz"), **npz_dict)
    meta = {
        "shape": list(shape),
        "dtype": str(dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_shN(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
        return params

    npz_dict = np.load(os.path.join(compress_dir, "shN.npz"))
    params_quant = npz_dict["params"]

    params_norm = params_quant / (2**6 - 1)
    params_norm = torch.tensor(params_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    params = params_norm * (maxs - mins) + mins

    if "labels" in npz_dict:
        labels = npz_dict["labels"]
        params = params[labels]
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params
