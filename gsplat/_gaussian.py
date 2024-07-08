from collections import OrderedDict
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import Tensor


def _knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def _rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


class Parameters(nn.Module):
    """A collection of Parameters.

    Arbitrary parameters can be registered, as long as the shape of the new parameter
    matches the size of this class. I.E., the registered parameter
    should have the shape of [..., <any dim>], in which ... is the size of this class
    defined at initialization.

    User can register custom parameters using `register_param` method. For example,
    ```
    # p is a Parameters object
    features = torch.randn(*gs.size(), 64) # Shape [..., 64]
    p.register_param("features", features, activation=torch.sigmoid)
    ```

    Parameters can be removed using `remove_param` method. For example,
    ```
    p.remove_param("features")
    ```

    Parameters can be accessed using the `get_param` method. For example,
    ```
    scales = p.get_param("scales", apply_act=True)
    ```

    """

    def __init__(self, size: Tuple[int, ...]):
        super().__init__()
        # TODO: when inplace modify params, _size should be updated.
        self._size = size
        self._params = nn.ParameterDict()
        self._activations = OrderedDict()

    def size(self) -> Tuple[int, ...]:
        """Returns the size of the Gaussians object."""
        return self._size

    def register_param(
        self, name: str, tensor: nn.Parameter, activation: Callable = lambda x: x
    ) -> None:
        """Register a parameter to the Gaussians object."""
        if name in self._params:
            raise Warning(f"Parameter '{name}' already exists.")
        assert isinstance(
            tensor, nn.Parameter
        ), f"Input tensor must be a nn.Parameter, but got {type(tensor)}."
        batch_size = self.size()
        batch_size_n_dim = len(batch_size)
        if batch_size_n_dim > 0:
            assert tensor.shape[:batch_size_n_dim] == batch_size, (
                f"Parameter shape {tensor.shape} does not match "
                f"the size of this Gaussian object {batch_size} "
                f"at the first {batch_size_n_dim} dimensions."
            )
        self._params[name] = tensor
        self._activations[name] = activation

    def remove_param(self, name: str) -> None:
        """Remove a parameter from the Gaussians object."""
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found in Gaussians object.")
        del self._params[name]
        del self._activations[name]

    def get_param(self, name: str, apply_act: bool = False) -> Tensor:
        """Get a parameter from the Gaussians object."""
        if name not in self._params:
            raise KeyError(f"Parameter '{name}' not found in Gaussians object.")
        tensor = self._params[name]
        if apply_act:
            activation = self._activations[name]
            return activation(tensor)
        else:
            return tensor

    # TODO: implement a public method that supports inplace modification of params.


class Gaussian3D(Parameters):
    """A collection of Gaussians."""

    @classmethod
    def from_pcl_with_knn(
        cls,
        points: Tensor,
        knn_neighbors: int = 3,
        knn_scale: float = 1.0,
        init_opacity: float = 0.1,
    ) -> "Gaussian":
        """Create a Gaussian object from a points."""

        assert (
            points.dim() == 2 and points.size(1) == 3
        ), "Points should have shape [N, 3]"

        N = points.shape[0]
        dist2_avg = (_knn(points, knn_neighbors + 1)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * knn_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        gs = cls(size=(N,))
        gs.register_param("means3d", torch.nn.Parameter(points))
        gs.register_param("scales", torch.nn.Parameter(scales), torch.exp)
        gs.register_param("quats", torch.nn.Parameter(quats), F.normalize)
        gs.register_param("opacities", torch.nn.Parameter(opacities), torch.sigmoid)
        return gs

    @classmethod
    def from_rand(cls, n: int, center: Tensor, half_edge: Tensor) -> "Gaussian":
        """Create a Gaussian object from a random distribution."""
        means3d = center + torch.rand((n, 3)) * half_edge * 2 - half_edge
        scales = torch.rand((n, 3))
        quats = torch.rand((n, 4))
        opacities = torch.rand((n,))
        gs = cls(size=(n,))
        gs.register_param("means3d", torch.nn.Parameter(means3d))
        gs.register_param("scales", torch.nn.Parameter(scales), torch.exp)
        gs.register_param("quats", torch.nn.Parameter(quats), F.normalize)
        gs.register_param("opacities", torch.nn.Parameter(opacities), torch.sigmoid)
        return gs

    def __len__(self) -> int:
        return self.size()[0]

    @property
    def means3d(self) -> Tensor:
        """Get the means of the Gaussians."""
        return self.get_param("means3d", apply_act=True)

    @property
    def scales(self) -> Tensor:
        """Get the scales of the Gaussians."""
        return self.get_param("scales", apply_act=True)

    @property
    def quats(self) -> Tensor:
        """Get the quaternions of the Gaussians."""
        return self.get_param("quats", apply_act=True)

    @property
    def opacities(self) -> Tensor:
        """Get the opacities of the Gaussians."""
        return self.get_param("opacities", apply_act=True)

    def register_sh_from_rgb(self, rgbs: Tensor, sh_degree: int = 3) -> None:
        """Register spherical harmonics coeffs from RGBs."""
        assert rgbs.dim() == 2 and rgbs.size(1) == 3, "RGBs should have shape [N, 3]"
        assert sh_degree >= 0, "sh_degree should be a non-negative integer."
        N = rgbs.shape[0]
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = _rgb_to_sh(rgbs)
        self.register_param("sh0", torch.nn.Parameter(colors[:, :1, :]))
        self.register_param("shN", torch.nn.Parameter(colors[:, 1:, :]))

    def register_color_from_rgb(self, rgbs: Tensor) -> None:
        """Register color logits from RGBs."""
        assert rgbs.dim() == 2 and rgbs.size(1) == 3, "RGBs should have shape [N, 3]"
        N = rgbs.shape[0]
        colors = torch.logit(rgbs)
        self.register_param("colors", torch.nn.Parameter(colors), torch.sigmoid)


if __name__ == "__main__":
    points = torch.randn(100, 3)
    rgbs = torch.rand(100, 3)
    features = torch.randn(100, 64)

    gs = Gaussian3D.from_pcl_with_knn(points)
    gs.register_sh_from_rgb(rgbs, sh_degree=3)
    gs.register_param("features", torch.nn.Parameter(features))

    assert gs.scales.shape == (100, 3)
    assert gs.quats.shape == (100, 4)
    assert gs.opacities.shape == (100,)
    assert gs.means3d.shape == (100, 3)
    assert gs.get_param("sh0").shape == (100, 1, 3)
    assert gs.get_param("shN").shape == (100, 15, 3)
    assert gs.get_param("features").shape == (100, 64)
