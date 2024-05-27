import math
from typing import Callable, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from utils import knn, normalized_quat_to_rotmat, rgb_to_sh

from gsplat.rendering import rasterization


class Gaussians(torch.nn.Module):
    """A simple 3D Gaussian model."""

    def __init__(self):
        super().__init__()
        self.params = torch.nn.ParameterDict(
            {
                "means3d": torch.nn.Parameter(torch.empty(0, 3)),
                "scales": torch.nn.Parameter(torch.empty(0, 3)),
                "quats": torch.nn.Parameter(torch.empty(0, 4)),
                "opacities": torch.nn.Parameter(torch.empty(0)),
                "sh0": torch.nn.Parameter(torch.empty(0, 1, 3)),
                "shN": torch.nn.Parameter(torch.empty(0, 0, 3)),
            }
        )
        # Some per-GS infomation that is used for growing/pruning GSs
        self.running_stats = {}

    def __len__(self):
        return self.params["means3d"].shape[0]

    @property
    def sh_degree(self):
        return math.isqrt(self.params["sh0"].shape[1] + self.params["shN"].shape[1]) - 1

    @staticmethod
    def from_pointcloud(
        points: Tensor, rgbs: Tensor, sh_degree: int = 3, init_opac: int = 0.1
    ) -> "Gaussians":
        """Create a Gaussian model from a point cloud.

        Args:
            points: [N, 3] center of the Gaussians.
            rgbs: [N, 3] RGB colors in [0, 1].
            sh_degree: Degree of spherical harmonics.
            init_opac: Initial opacity of the Gaussians.
        """
        N = points.shape[0]
        assert points.shape == (N, 3), points.shape
        assert rgbs.shape == (N, 3), rgbs.shape

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)  # [N, 3]
        quats = torch.rand((N, 4))  # [N, 4]
        opacities = torch.logit(torch.full((N,), init_opac))  # [N,]

        # Initialize the GS color
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)

        # Set the attributes
        model = Gaussians()
        model.params = torch.nn.ParameterDict(
            {
                "means3d": torch.nn.Parameter(points),  # [N, 3]
                "scales": torch.nn.Parameter(scales),  # [N, 3]
                "quats": torch.nn.Parameter(quats),  # [N, 4]
                "opacities": torch.nn.Parameter(opacities),  # [N,]
                "sh0": torch.nn.Parameter(colors[:, :1, :]),  # [N, 1, 3]
                "shN": torch.nn.Parameter(colors[:, 1:, :]),  # [N, K-1, 3]
            }
        )
        return model

    @staticmethod
    def from_state_dict(state_dict: Dict) -> "Gaussians":
        """Load a model from a checkpoint."""
        model = Gaussians()
        model.params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v) for k, v in state_dict.items()}
        )
        return model

    def forward(self, viewmats: Tensor, Ks: Tensor, width: int, height: int, **kwargs):
        """Rasterize Gaussians to Images.

        Args:
            viewmats: [C, 4, 4] world-to-camera transformation matrices.
            Ks: [C, 3, 3] camera intrinsics.
            width: Image width.
            height: Image height.

        Returns:
            renders: [C, H, W, D], rendered images.
            alphas: [C, H, W, 1], rendered alphas.
            meta: Dict of metadata.
        """
        sh_coeffs = torch.cat([self.params["sh0"], self.params["shN"]], 1)  # [N, K, 3]
        sh_degree = kwargs.pop("sh_degree", self.sh_degree)
        renders, alphas, meta = rasterization(
            means=self.params["means3d"],  # [N, 3]
            quats=self.params["quats"],  # [N, 4] norm is fused in rasterization
            scales=torch.exp(self.params["scales"]),  # [N, 3]
            opacities=torch.sigmoid(self.params["opacities"]),  # [N,]
            colors=sh_coeffs,  # [N, K, 3]
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            sh_degree=sh_degree,
            **kwargs,
        )
        return renders, alphas, meta


@torch.no_grad()
def reset_opac(
    model: "Gaussians", optimizers: List[torch.optim.Optimizer], value: float = 0.01
):
    """Utility function to reset opacities (inplace).

    This function does:
        - Clamp the opacities to be maximum of 0.01.
        - Set the optimizer states for opacities to zeros.

    """
    param_update_fn = {
        "opacities": lambda p: torch.clamp(p, max=math.log(value / (1 - value)))
    }
    state_update_fn = {"opacities": lambda s: torch.zeros_like(s)}
    update_params(model.params, optimizers, param_update_fn, state_update_fn)
    torch.cuda.empty_cache()


@torch.no_grad()
def refine_split(
    model: "Gaussians", optimizers: List[torch.optim.Optimizer], mask: Tensor
):
    """Utility function to grow GSs (inplace).

    This function does:
        - Split the selected GS into two smaller GSs.
            - mean of the new GS is sampled based on the covariance of the GS.
            - scale of the new GS is 0.8x of the original GS.
            - other attributes are copied from the original GS.
        - Set the optimizer states for the new GSs to zeros.
    """
    device = mask.device

    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(model.params["scales"][sel])  # [N, 3]
    quats = F.normalize(model.params["quats"][sel], dim=-1)  # [N, 4]
    rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    samples = torch.einsum(
        "nij,nj,bnj->bni",
        rotmats,
        scales,
        torch.randn(2, len(scales), 3, device=device),
    )  # [2, N, 3]

    default_update_fn = lambda p: torch.cat(
        (p[rest], p[sel].repeat([2] + [1] * (p.dim() - 1)))
    )
    param_update_fn, state_update_fn = {}, {}
    for name in model.params.keys():
        if name == "means3d":
            param_update_fn[name] = lambda p: torch.cat(
                (p[rest], (p[sel] + samples).reshape(-1, 3))
            )
        elif name == "scales":
            param_update_fn[name] = lambda p: torch.cat(
                (p[rest], torch.log(scales / 1.6).repeat(2, 1))
            )
        else:
            param_update_fn[name] = default_update_fn
        state_update_fn[name] = lambda s: torch.cat(
            (s[rest], torch.zeros((2 * len(sel), *s.shape[1:]), device=s.device))
        )
    update_params(model.params, optimizers, param_update_fn, state_update_fn)

    for k, v in model.running_stats.items():
        if v is not None:
            model.running_stats[k] = default_update_fn(v)
    torch.cuda.empty_cache()


@torch.no_grad()
def refine_duplicate(
    model: "Gaussians", optimizers: List[torch.optim.Optimizer], mask: Tensor
):
    """Unility function to duplicate GSs (inplace).

    This function does:
        - Duplicate the selected GS.
        - Set the optimizer states for the new GS to zeros.
    """
    sel = torch.where(mask)[0]

    default_update_fn = lambda p: torch.cat((p, p[sel]))
    param_update_fn, state_update_fn = {}, {}
    for name in model.params.keys():
        param_update_fn[name] = default_update_fn
        state_update_fn[name] = lambda s: torch.cat(
            (s, torch.zeros((len(sel), *s.shape[1:]), device=s.device))
        )
    update_params(model.params, optimizers, param_update_fn, state_update_fn)

    for k, v in model.running_stats.items():
        if v is not None:
            model.running_stats[k] = default_update_fn(v)
    torch.cuda.empty_cache()


@torch.no_grad()
def refine_keep(
    model: "Gaussians", optimizers: List[torch.optim.Optimizer], mask: Tensor
):
    """Unility function to prune GSs (inplace).

    This function does:
        - Keep the selected GSs and remove the rest.
        - Keep the optimizer states for the selected GSs and remove the rest.
    """
    sel = torch.where(mask)[0]

    default_update_fn = lambda p: p[sel]
    state_update_fn = param_update_fn = {
        name: default_update_fn for name in model.params.keys()
    }
    update_params(model.params, optimizers, param_update_fn, state_update_fn)

    for k, v in model.running_stats.items():
        if v is not None:
            model.running_stats[k] = default_update_fn(v)
    torch.cuda.empty_cache()


def update_lr(optims: List[torch.optim.Optimizer], lrs: Dict[str, float]):
    """Update learning rates (inplace).

    Args:
        optims: List of torch optimizers. Expect to have the "name" attribute in the param_group.
        lrs: Dict of learning rates. Key is the "name" attribute in the param_group.
    """
    for optimizer in optims:
        for param_group in optimizer.param_groups:
            name = param_group["name"]
            if name in lrs:
                param_group["lr"] = lrs[name]


def update_params(
    params: torch.nn.ParameterDict,
    optimizers: List[torch.optim.Optimizer],
    param_update_fn: Dict[str, Callable[[Parameter], Parameter]],
    state_update_fn: Dict[str, Callable[[Tensor], Tensor]],
):
    """Update parameters and states (inplace).

    Args:
        params: Dict of torch parameters.
        optimizers: List of torch optimizers. Expect to have the "name" attribute in the param_group.
        param_update_fn: Dict of functions to update parameters. Each function is expected to
            the parameter as input, and return the new parameter.
        state_update_fn: Dict of functions to update states. Each function is expected to
            the state tensor as input, and return the new state tensor.

    Example:

        ```python
        # Clone the "means3d" parameter and set its states to zeros.
        param_update_fn = {"means3d": lambda p: p.repeat(2, 1)}
        state_update_fn = {"means3d": lambda s: s.zeros_().repeat(2, 1)}
        optim_update_param(params, optimizers, param_update_fn, state_update_fn)
        ```

    """
    assert set(param_update_fn.keys()) == set(state_update_fn.keys()), "Mismatched keys"
    for optimizer in optimizers:
        for i, param_group in enumerate(optimizer.param_groups):
            name = param_group["name"]
            if name not in param_update_fn:
                continue
            p = param_group["params"][0]
            state = optimizer.state[p]
            # update state first
            del optimizer.state[p]
            for key, s in state.items():
                if key != "step":
                    state[key] = state_update_fn[name](s)
            # update the param in optimizer
            p_new = torch.nn.Parameter(param_update_fn[name](p))
            optimizer.param_groups[i]["params"] = [p_new]
            optimizer.state[p_new] = state
            # update the model parameter
            params[name] = p_new
