from dataclasses import dataclass, field
from typing import Any, Dict, Union

import numpy as np
import torch
import torchvision
from torch import Tensor
import torch.nn.functional as F
from plas import sort_with_plas

from .base import Strategy
from .ops import _update_param_with_optimizer


@dataclass
class SortStrategy(Strategy):
    verbose: bool = True
    cap_max: int = 1_000_000
    sort_stop_iter: int = 25_000
    sort_every: int = 1000
    shuffle_before_sort: bool = False
    sort_attributes: list[str] = field(
        default_factory=lambda: ["sh0", "scales", "quats"]
    )

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        sorted_params = None
        sorted_mask = torch.ones(self.cap_max, dtype=bool)
        return {"sorted_params": sorted_params, "sorted_mask": sorted_mask}

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        n_gs = params["means"].shape[0]
        if n_gs != self.cap_max:
            return

        if step <= self.sort_stop_iter and step % self.sort_every == 0:
            self._sort_into_grid(params, optimizers, state)
            if self.verbose:
                print("Sorted grid.")

            torch.cuda.empty_cache()

    @torch.no_grad()
    def _sort_into_grid(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ):
        params_to_sort = torch.cat(
            [params[k].reshape(self.cap_max, -1) for k in self.sort_attributes], dim=-1
        )
        if self.shuffle_before_sort:
            shuffled_indices = torch.randperm(
                params_to_sort.shape[0], device=params_to_sort.device
            )
            params_to_sort = params_to_sort[shuffled_indices]

        n_sidelen = int(self.cap_max**0.5)
        grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
        sorted_grid, sorted_indices = sort_with_plas(
            grid.permute(2, 0, 1), improvement_break=1e-4, verbose=self.verbose
        ).permute(1, 2, 0)
        sorted_indices = sorted_indices.squeeze().flatten()
        if self.shuffle_before_sort:
            sorted_indices = shuffled_indices[sorted_indices]

        def param_fn(name: str, p: Tensor) -> Tensor:
            return torch.nn.Parameter(p[sorted_indices])

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v[sorted_indices]

        # update the parameters and the state in the optimizers
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        # update state
        state["sorted_params"] = sorted_grid.reshape(self.cap_max, -1)
        state["sorted_mask"].fill_(1)

    def nb_loss(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        sort_state: Dict[str, Any],
        strategy_state: Dict[str, Any],
    ):
        n_gs = params["means"].shape[0]
        if n_gs != self.cap_max:
            return torch.tensor(0.0, requires_grad=True)
        if sort_state["sorted_params"] is None:
            return torch.tensor(0.0, requires_grad=True)

        # update state for sorted_mask
        sort_state["sorted_mask"][strategy_state["relocated_mask"]] = 0

        params_to_sort = torch.cat(
            [params[k].reshape(self.cap_max, -1) for k in self.sort_attributes], dim=-1
        )

        # make smooth param target
        params_blurred = params_to_sort.detach().clone()
        params_blurred[~sort_state["sorted_mask"]] = sort_state["sorted_params"][
            ~sort_state["sorted_mask"]
        ]
        n_sidelen = int(n_gs**0.5)
        grid_blurred = params_blurred.reshape((n_sidelen, n_sidelen, -1))
        grid_blurred = torchvision.transforms.functional.gaussian_blur(
            grid_blurred.permute(2, 0, 1), kernel_size=5
        ).permute(1, 2, 0)
        params_blurred = grid_blurred.reshape(n_gs, -1)

        nbloss = F.huber_loss(
            params_blurred[sort_state["sorted_mask"]],
            params_to_sort[sort_state["sorted_mask"]],
        )
        return nbloss

    def debug(self, params):
        n_gs = params["means"].shape[0]
        if n_gs != self.cap_max:
            return None

        params_to_sort = torch.cat(
            [params[k].reshape(n_gs, -1) for k in self.sort_attributes], dim=-1
        )

        n_sidelen = int(n_gs**0.5)
        grid = params_to_sort.reshape(n_sidelen, n_sidelen, -1)
        grid_mins = torch.amin(grid, dim=(0, 1))
        grid_maxs = torch.amax(grid, dim=(0, 1))
        grid_norm = (grid - grid_mins) / (grid_maxs - grid_mins)

        grid_rgb = grid_norm[:, :, :3]
        grid_rgb = grid_rgb.detach().cpu().numpy()
        grid_rgb = (grid_rgb * 255).astype(np.uint8)
        return grid_rgb
