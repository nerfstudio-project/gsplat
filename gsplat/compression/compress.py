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


def compress_splats(
    compress_dir: str, splats: dict[str, Tensor], sort_plas: bool = True
) -> None:
    if sort_plas:
        splats = sort_splats_with_plas(splats)

    meta = {}
    for attr_name in splats.keys():
        compress_fn = eval(f"_compress_{attr_name}")
        meta[attr_name] = compress_fn(compress_dir, splats[attr_name])

    with open(os.path.join(compress_dir, "meta.json"), "w") as f:
        json.dump(meta, f)


def decompress_splats(compress_dir: str) -> dict[str, Tensor]:
    with open(os.path.join(compress_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    splats = {}
    for attr_name, attr_meta in meta.items():
        decompress_fn = eval(f"_decompress_{attr_name}")
        splats[attr_name] = decompress_fn(compress_dir, attr_meta)
    return splats


def sort_splats_with_plas(
    splats: dict[str, Tensor], verbose: bool = True
) -> dict[str, Tensor]:
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
    def log_transform(x):
        return torch.sign(x) * torch.log1p(torch.abs(x))

    params = log_transform(params)

    n_gs = len(params)
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    grid = params.reshape((n_sidelen, n_sidelen, -1))
    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(os.path.join(compress_dir, "means_l.png"), img_l.astype(np.uint8))
    imageio.imwrite(os.path.join(compress_dir, "means_u.png"), img_u.astype(np.uint8))

    meta = {
        "shape": list(params.shape),
        "mins": grid_mins.tolist(),
        "maxs": grid_maxs.tolist(),
    }
    return meta


def _decompress_means(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    def inverse_log_transform(y):
        return torch.sign(y) * (torch.expm1(torch.abs(y)))

    img_l = imageio.imread(os.path.join(compress_dir, "means_l.png")).astype(np.uint16)
    img_u = imageio.imread(os.path.join(compress_dir, "means_u.png")).astype(np.uint16)
    img = (img_u << 8) + img_l

    img_norm = img / (2**16 - 1)
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid_mins = torch.tensor(meta["mins"], dtype=torch.float32)
    grid_maxs = torch.tensor(meta["maxs"], dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins

    params = grid.reshape(meta["shape"])
    params = inverse_log_transform(params)
    return params


def _compress_scales(compress_dir: str, params: Tensor) -> dict[str, Any]:
    n_sidelen = int(params.shape[0] ** 0.5)
    grid = params.reshape((n_sidelen, n_sidelen, -1))

    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "scales.png"), img)

    meta = {
        "shape": list(params.shape),
        "mins": grid_mins.tolist(),
        "maxs": grid_maxs.tolist(),
    }
    return meta


def _decompress_scales(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "scales.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid_mins = torch.tensor(meta["mins"], dtype=torch.float32)
    grid_maxs = torch.tensor(meta["maxs"], dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins

    params = grid.reshape(meta["shape"])
    return params


def _compress_sh0(compress_dir: str, params: Tensor) -> Tensor:
    def sh_to_rgb(sh: Tensor) -> Tensor:
        C0 = 0.28209479177387814
        return sh * C0 + 0.5

    rgb = sh_to_rgb(params)
    rgb = torch.clamp(rgb, 0.0, 1.0)
    n_sidelen = int(params.shape[0] ** 0.5)
    grid = rgb.reshape((n_sidelen, n_sidelen, -1))
    grid = grid.detach().cpu().numpy()
    img = (grid * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "sh0.png"), img)

    meta = {
        "shape": list(params.shape),
    }
    return meta


def _decompress_sh0(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    def rgb_to_sh(rgb: Tensor) -> Tensor:
        C0 = 0.28209479177387814
        return (rgb - 0.5) / C0

    img = imageio.imread(os.path.join(compress_dir, "sh0.png"))
    grid = img / (2**8 - 1)
    grid = torch.tensor(grid, dtype=torch.float32)
    rgb = grid.reshape(meta["shape"])
    params = rgb_to_sh(rgb)
    return params


def _compress_shN(
    compress_dir: str, params: Tensor, use_kmeans: bool = False
) -> Tensor:
    if params.numel() == 0:
        return {"shape": list(params.shape)}

    codebook = params
    if use_kmeans:
        kmeans = KMeans(n_clusters=2**16, distance="manhattan", verbose=True)
        x = params.reshape(params.shape[0], -1).permute(1, 0).contiguous()
        labels = kmeans.fit(x)
        labels = labels.detach().cpu().numpy()
        labels = labels.astype(np.uint16)
        codebook = kmeans.centroids.permute(1, 0)

    codebook_mins = torch.min(codebook)
    codebook_maxs = torch.max(codebook)
    codebook_norm = (codebook - codebook_mins) / (codebook_maxs - codebook_mins)
    codebook_norm = codebook_norm.detach().cpu().numpy()
    codebook = (codebook_norm * (2**6 - 1)).round().astype(np.uint8)

    npz_dict = {"codebook": codebook}
    if use_kmeans:
        npz_dict["labels"] = labels

    np.savez_compressed(os.path.join(compress_dir, "shN.npz"), **npz_dict)
    meta = {
        "shape": list(params.shape),
        "mins": codebook_mins.tolist(),
        "maxs": codebook_maxs.tolist(),
    }
    return meta


def _decompress_shN(
    compress_dir: str, meta: dict[str, Any], use_kmeans: bool = False
) -> Tensor:
    if meta["shape"][1] == 0:
        return torch.zeros(meta["shape"], dtype=torch.float32)

    npz_dict = np.load(os.path.join(compress_dir, "shN.npz"))
    codebook = npz_dict["codebook"]

    codebook_norm = codebook / (2**6 - 1)
    codebook_norm = torch.tensor(codebook_norm, dtype=torch.float32)
    codebook_mins = torch.tensor(meta["mins"], dtype=torch.float32)
    codebook_maxs = torch.tensor(meta["maxs"], dtype=torch.float32)
    codebook = codebook_norm * (codebook_maxs - codebook_mins) + codebook_mins

    if use_kmeans:
        labels = npz_dict["labels"]
        codebook = codebook[labels]
    params = codebook.reshape(meta["shape"])
    return params


def _compress_shN_codebook(
    compress_dir: str, centroids: Tensor, labels: Tensor
) -> Tensor:
    centroids_mins = torch.min(centroids)
    centroid_maxs = torch.max(centroids)
    centroids_norm = (centroids - centroids_mins) / (centroid_maxs - centroids_mins)

    labels = labels.detach().cpu().numpy()
    centroids_norm = centroids_norm.detach().cpu().numpy()
    labels = labels.astype(np.uint16)
    centroids = (centroids_norm * (2**16 - 1)).round().astype(np.uint16)
    np.savez_compressed(
        os.path.join(compress_dir, "shN.npz"), labels=labels, centroids=centroids
    )

    meta = {
        # "shape": list(params.shape),
        "mins": centroids_mins.tolist(),
        "maxs": centroid_maxs.tolist(),
    }
    return meta


def _decompress_shN_codebook(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    npz_dict = np.load(os.path.join(compress_dir, "shN.npz"))
    labels = npz_dict["labels"]
    centroids = npz_dict["centroids"]

    centroids_norm = centroids / (2**16 - 1)
    centroids_norm = torch.tensor(centroids_norm, dtype=torch.float32)
    centroids_mins = torch.tensor(meta["mins"], dtype=torch.float32)
    centroids_maxs = torch.tensor(meta["maxs"], dtype=torch.float32)
    centroids = centroids_norm * (centroids_maxs - centroids_mins) + centroids_mins

    params = centroids[labels]
    # params = params.reshape(meta["shape"])
    return params


def _compress_quats(compress_dir: str, params: Tensor) -> Tensor:
    n_sidelen = int(params.shape[0] ** 0.5)
    grid = params.reshape((n_sidelen, n_sidelen, -1))

    grid_norm = F.normalize(grid, dim=-1)
    grid_norm = grid_norm * 0.5 + 0.5
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "quats.png"), img)

    meta = {
        "shape": list(params.shape),
    }
    return meta


def _decompress_quats(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "quats.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid = grid_norm * 2 - 1

    params = grid.reshape(meta["shape"])
    return params


def _compress_opacities(compress_dir: str, params: Tensor) -> Tensor:
    n_sidelen = int(params.shape[0] ** 0.5)
    grid = params.reshape((n_sidelen, n_sidelen))

    grid_mins = torch.amin(grid, dim=(0, 1))
    grid_maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, "opacities.png"), img)

    meta = {
        "shape": list(params.shape),
        "mins": grid_mins.tolist(),
        "maxs": grid_maxs.tolist(),
    }
    return meta


def _decompress_opacities(compress_dir: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, "opacities.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm, dtype=torch.float32)
    grid_mins = torch.tensor(meta["mins"], dtype=torch.float32)
    grid_maxs = torch.tensor(meta["maxs"], dtype=torch.float32)
    grid = grid_norm * (grid_maxs - grid_mins) + grid_mins

    params = grid.reshape(meta["shape"])
    return params
