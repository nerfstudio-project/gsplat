import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor
from plas import sort_with_plas

from .base import Strategy
from .ops import (
    inject_noise_to_position,
    relocate,
    sample_add,
    _update_param_with_optimizer,
)


@dataclass
class MCMCStrategy(Strategy):
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    This strategy will:

    - Periodically teleport GSs with low opacity to a place that has high opacity.
    - Periodically introduce new GSs sampled based on the opacity distribution.
    - Periodically perturb the GSs locations.

    Args:
        cap_max (int): Maximum number of GSs. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_iter (int): Start refining GSs after this iteration. Default to 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default to 25_000.
        refine_every (int): Refine GSs every this steps. Default to 100.
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        verbose (bool): Whether to print verbose information. Default to False.

    Examples:

        >>> from gsplat import MCMCStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = MCMCStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=1e-3)

    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    refine_every: int = 1000
    min_opacity: float = 0.005
    verbose: bool = True
    sort: bool = False
    sort_every: int = 1000

    done_adding_new_gs: bool = False
    sorted_params: Tensor | None = None
    sorted_mask: Tensor | None = None

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    # def step_pre_backward(
    #     self,
    #     params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    #     optimizers: Dict[str, torch.optim.Optimizer],
    #     # state: Dict[str, Any],
    #     step: int,
    #     info: Dict[str, Any],
    # ):
    #     """Callback function to be executed before the `loss.backward()` call."""
    #     pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Callback function to be executed before the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        # move to the correct device
        state["binoms"] = state["binoms"].to(params["means"].device)

        binoms = state["binoms"]

        if (
            step <= self.refine_stop_iter
            and step >= self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            # add new GSs
            n_new_gs = self._add_new_gs(params, optimizers, binoms)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )
            if n_new_gs == 0:
                self.done_adding_new_gs = True

            torch.cuda.empty_cache()

        if (
            self.sort
            # and self.done_adding_new_gs
            and step >= self.refine_start_iter
            and step <= self.refine_stop_iter
            and step % self.sort_every == 0
        ):
            self._sort_into_grid(params, optimizers)
            if self.verbose:
                print("Sorted grid.")

            torch.cuda.empty_cache()

        # add noise to GSs
        inject_noise_to_position(
            params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"])
        dead_mask = opacities <= self.min_opacity

        if self.sort and self.sorted_mask is not None:
            self.sorted_mask[dead_mask[:len(self.sorted_mask)]] = 0

        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_target = int(n_target**0.5) ** 2
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _sort_into_grid(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        shuffle_before_sort: bool = True,
    ):
        n_gs = len(params["means"])
        n_sidelen = int(n_gs**0.5)
        reg_keys = ["means", "scales", "quats", "sh0"]
        params_to_sort = torch.cat(
            [params[k].reshape(n_gs, -1) for k in reg_keys], dim=-1
        )
        if shuffle_before_sort:
            shuffled_indices = torch.randperm(
                params_to_sort.shape[0], device=params_to_sort.device
            )
            params_to_sort = params_to_sort[shuffled_indices]

        grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
        sorted_grid, sorted_indices = sort_with_plas(
            grid.permute(2, 0, 1), improvement_break=1e-4, verbose=self.verbose
        )
        sorted_grid = sorted_grid.permute(1, 2, 0)
        sorted_indices = sorted_indices.squeeze().flatten()
        if shuffle_before_sort:
            sorted_indices = shuffled_indices[sorted_indices]

        def param_fn(name: str, p: Tensor) -> Tensor:
            return torch.nn.Parameter(p[sorted_indices])

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            return v[sorted_indices]

        # update the parameters and the state in the optimizers
        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        self.sorted_params = sorted_grid.reshape(n_gs, -1)
        self.sorted_mask = torch.ones(n_gs, dtype=bool).to(self.sorted_params.device)
