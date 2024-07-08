from typing import Dict, Optional

import torch
from torch import Tensor
from utils import knn, rgb_to_sh


class Gaussians(torch.nn.Modules):
    def __init__(self):
        self.params: torch.nn.ParameterDict = torch.nn.ParameterDict()
        self.lrs: Dict[str, float] = {}

    def __len__(self):
        if len(self.params) == 0:
            return 0
        else:
            first_param = self.params.values()[0]
            return len(first_param)

    def __getitem__(self, key: str) -> torch.nn.Parameter:
        """Allow dictionary-like access to parameters."""
        if key in self.params:
            return self.params[key]
        else:
            raise KeyError(f"Parameter '{key}' not found in Gaussians object.")

    def register_param(self, name: str, tensor: torch.Tensor):
        """Allows usage of customized parameters.
        For example,
        ```
        features = torch.randn(len(gs), 64)
        gs.register_param("features", features)
        ```
        """

        if name in self.params:
            raise Warning(f"Parameter '{name}' already exists.")
        self.params[name] = torch.nn.Parameter(tensor)

    def get_attr_names(self):
        return self.params.keys()

    def _init_attributes(
        self,
        points: Tensor,
        rgbs: Optional[Tensor] = None,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        feature_dim: Optional[int] = None,
    ):
        N = points.shape[0]
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

        params = [
            # name, value, lr
            ("means3d", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
            ("scales", torch.nn.Parameter(scales), 5e-3),
            ("quats", torch.nn.Parameter(quats), 1e-3),
            ("opacities", torch.nn.Parameter(opacities), 5e-2),
        ]

        if feature_dim is None:
            # color is SH coefficients.
            colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
            colors[:, 0, :] = rgb_to_sh(rgbs)
            params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
            params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
        else:
            # features will be used for appearance and view-dependent shading
            features = torch.rand(N, feature_dim)  # [N, feature_dim]
            params.append(("features", torch.nn.Parameter(features), 2.5e-3))
            colors = torch.logit(rgbs)  # [N, 3]
            params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

        for name, value, lr in params:
            self.register_param(name, value)
            self.lrs[name] = lr

    @classmethod
    def init_from_pcl(
        cls,
        points: Tensor,
        rgbs: Optional[Tensor] = None,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        feature_dim: Optional[int] = None,
    ) -> "Gaussians":
        """Initialize a set of gaussians from a point cloud.

        Args:
            points: the 3D points from the point cloud [N, 3]
            rgbs: the RGB values of the points, normalized to be in the range [0, 1] [N, 3]

        Returns:
            A Gaussians object with the points and rgbs registered as parameters.
        """

        instance = cls()
        instance._init_attributes(
            points, rgbs, init_scale, init_opacity, scene_scale, sh_degree, feature_dim
        )

        return instance

    @classmethod
    def init_randn(
        cls,
        center: Tensor,
        std: Tensor,
        init_num_pts: int,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        feature_dim: Optional[int] = None,
    ) -> "Gaussians":
        """
        Initialize a set of gaussians from a random normal distribution.
        """

        instance = cls()
        points = center + std * torch.randn((init_num_pts, 3))
        rgbs = torch.rand((init_num_pts, 3))

        instance._init_attributes(
            points, rgbs, init_scale, init_opacity, scene_scale, sh_degree, feature_dim
        )

        return instance

    @classmethod
    def init_rand(
        cls,
        center: Tensor,
        half_edge: Tensor,
        init_num_pts: int,
        init_extent: float = 3.0,
        init_scale: float = 1.0,
        init_opacity: float = 0.1,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        feature_dim: Optional[int] = None,
    ) -> "Gaussians":
        instance = cls()
        # points = torch.rand((init_num_pts, 3)) * 2 - 1
        # points = points * half_edge + center
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))

        instance._init_attributes(
            points, rgbs, init_scale, init_opacity, scene_scale, sh_degree, feature_dim
        )

        return instance
