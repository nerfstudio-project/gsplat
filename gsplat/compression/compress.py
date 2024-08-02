from dataclasses import dataclass
import os
from typing import Any
import json
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import imageio

from gsplat.compression.sort import sort_splats
from gsplat.utils import log_transform, inverse_log_transform


@dataclass
class PngCompressionStrategy:
    """Compress splats with quantization, sorting, and K-means clustering of
    the spherical harmonic coefficents.

    Args:
        use_sort (bool, optional): Whether to sort splats before compression. Defaults to True.
        use_kmeans (bool, optional): Whether to use K-means to compress spherical harmonics. Defaults to True.
    """

    use_sort: bool = True
    use_kmeans: bool = True
    verbose: bool = True

    def compress(self, compress_dir: str, splats: dict[str, Tensor]) -> None:
        compress_fn_map = {
            "means": _compress_means,
            "scales": _compress_scales,
            "quats": _compress_quats,
            "opacities": _compress_opacities,
            "sh0": _compress_sh0,
            "shN": _compress_shN,
        }

        splats = _crop_square(splats)
        if self.use_sort:
            splats = sort_splats(splats)

        meta = {}
        for param_name in splats.keys():
            compress_fn = (
                compress_fn_map[param_name]
                if param_name in compress_fn_map
                else _compress_npz
            )
            kwargs = {
                "n_sidelen": int(len(splats["means"]) ** 0.5),
                "use_kmeans": self.use_kmeans,
                "verbose": self.verbose,
            }
            meta[param_name] = compress_fn(
                compress_dir, param_name, splats[param_name], **kwargs
            )

        with open(os.path.join(compress_dir, "meta.json"), "w") as f:
            json.dump(meta, f)

    def decompress(self, compress_dir: str) -> dict[str, Tensor]:
        decompress_fn_map = {
            "means": _decompress_means,
            "scales": _decompress_scales,
            "quats": _decompress_quats,
            "opacities": _decompress_opacities,
            "sh0": _decompress_sh0,
            "shN": _decompress_shN,
        }

        with open(os.path.join(compress_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        splats = {}
        for param_name, param_meta in meta.items():
            decompress_fn = (
                decompress_fn_map[param_name]
                if param_name in decompress_fn_map
                else _decompress_npz
            )
            splats[param_name] = decompress_fn(compress_dir, param_name, param_meta)
        return splats


def _crop_square(splats: dict[str, Tensor]) -> dict[str, Tensor]:
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_diff = n_gs - n_sidelen**2
    if n_diff != 0:
        opacities = splats["opacities"]
        keep_indices = torch.argsort(opacities, descending=True)[:-n_diff]
        for k, v in splats.items():
            splats[k] = v[keep_indices]
        print(
            f"Warning: Number of Gaussians was not square. Removed {n_diff} Gaussians."
        )
    return splats


def _compress_means(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> dict[str, Any]:
    params = log_transform(params)
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**16 - 1)).round().astype(np.uint16)

    img_l = img & 0xFF
    img_u = (img >> 8) & 0xFF
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_l.png"), img_l.astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(compress_dir, f"{param_name}_u.png"), img_u.astype(np.uint8)
    )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_means(
    compress_dir: str, param_name: str, meta: dict[str, Any]
) -> Tensor:
    img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
    img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
    img_u = img_u.astype(np.uint16)
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


def _compress_scales(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> dict[str, Any]:
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()

    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_scales(
    compress_dir: str, param_name: str, meta: dict[str, Any]
) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)

    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_quats(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> dict[str, Any]:
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    grid_norm = F.normalize(grid, dim=-1)
    grid_norm = grid_norm * 0.5 + 0.5
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_quats(
    compress_dir: str, param_name: str, meta: dict[str, Any]
) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm)
    grid = grid_norm * 2 - 1

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_opacities(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> dict[str, Any]:
    grid = params.reshape((n_sidelen, n_sidelen))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_opacities(
    compress_dir: str, param_name: str, meta: dict[str, Any]
) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_sh0(
    compress_dir: str, param_name: str, params: Tensor, n_sidelen: int, **kwargs
) -> dict[str, Any]:
    grid = params.reshape((n_sidelen, n_sidelen, -1))
    mins = torch.amin(grid, dim=(0, 1))
    maxs = torch.amax(grid, dim=(0, 1))
    grid_norm = (grid - mins) / (maxs - mins)
    img_norm = grid_norm.detach().cpu().numpy()
    img = (img_norm * (2**8 - 1)).round().astype(np.uint8)
    imageio.imwrite(os.path.join(compress_dir, f"{param_name}.png"), img)

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_sh0(compress_dir: str, param_name: str, meta: dict[str, Any]) -> Tensor:
    img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    img_norm = img / (2**8 - 1)
    grid_norm = torch.tensor(img_norm)
    mins = torch.tensor(meta["mins"])
    maxs = torch.tensor(meta["maxs"])
    grid = grid_norm * (maxs - mins) + mins

    params = grid.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params


def _compress_shN(
    compress_dir: str,
    param_name: str,
    params: Tensor,
    use_kmeans: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Compress spherical harmonic coefficients to a npz file.

    .. warning::
        TorchPQ must installed to use K-means clustering.

    Args:
        compress_dir (str): compression directory
        params (Tensor): parameters to compress
        use_kmeans (bool, optional): Whether to use K-means clustering during compression. Defaults to True.
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        dict[str, Any]: metadata
    """
    shape = params.shape
    dtype = params.dtype
    if params.numel() == 0:
        meta = {
            "shape": list(shape),
            "dtype": str(dtype).split(".")[1],
        }
        return meta

    if use_kmeans:
        try:
            from torchpq.clustering import KMeans
        except:
            raise ImportError(
                "Please install torchpq with 'pip install torchpq' to use K-means clustering"
            )

        kmeans = KMeans(n_clusters=2**16, distance="manhattan", verbose=verbose)
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
    np.savez_compressed(os.path.join(compress_dir, f"{param_name}.npz"), **npz_dict)
    meta = {
        "shape": list(shape),
        "dtype": str(dtype).split(".")[1],
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
    }
    return meta


def _decompress_shN(
    compress_dir: str, param_name: str, meta: dict[str, Any], **kwargs
) -> Tensor:
    if not np.all(meta["shape"]):
        params = torch.zeros(meta["shape"])
        params = params.to(dtype=getattr(torch, meta["dtype"]))
        return params

    npz_dict = np.load(os.path.join(compress_dir, f"{param_name}.npz"))
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


def _compress_npz(compress_dir: str, param_name: str, params: Tensor) -> dict[str, Any]:
    npz_dict = {"arr": params.detach().cpu().numpy()}
    np.savez_compressed(os.path.join(compress_dir, f"{param_name}.npz"), **npz_dict)
    meta = {
        "shape": params.shape,
        "dtype": str(params.dtype).split(".")[1],
    }
    return meta


def _decompress_npz(compress_dir: str, param_name: str, meta: dict[str, Any]) -> Tensor:
    arr = np.load(os.path.join(compress_dir, f"{param_name}.npz"))["arr"]
    params = torch.tensor(arr)
    params = params.reshape(meta["shape"])
    params = params.to(dtype=getattr(torch, meta["dtype"]))
    return params
