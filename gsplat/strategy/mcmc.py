# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add


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
        noise_injection_stop_iter (int): Stop injecting noise after this iteration. Default to -1 (never stop).
            Use this to stop noise injection during controller distillation while keeping refine_stop_iter separate.
        refine_every (int): Refine GSs every this steps. Default to 100.
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        verbose (bool): Whether to print verbose information. Default to False.
        preallocate (bool): Use pre-allocated buffers to eliminate memory usage in torch.cat. 
            Default=False.
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    noise_injection_stop_iter: int = -1
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False
    preallocate: bool = False

    def initialize_state(self, n_initial: int = 0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        Args:
            n_initial: umber of initial active Gaussians. 
            Only used when preallocate=True. Default to 0.
        """
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        state = {"binoms": binoms}

        if self.preallocate:
            state["n_active"] = n_initial

        return state

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
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        # move to the correct device
        state["binoms"] = state["binoms"].to(params["means"].device)

        binoms = state["binoms"]

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers, state, binoms)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            n_new_gs = self._add_new_gs(params, optimizers, state, binoms)
            if self.verbose:
                # NEW: use n_active for reporting when preallocated
                if self.preallocate:
                    n_total = state["n_active"]
                else:
                    n_total = len(params["means"])
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {n_total} GSs."
                )

            torch.cuda.empty_cache()

        # add noise to GSs
        noise_stop = (
            self.noise_injection_stop_iter
            if self.noise_injection_stop_iter >= 0
            else float("inf")
        )
        if step < noise_stop:
            # Only inject noise to Gaussians that are active
            if self.preallocate:
                n_active = state["n_active"]
                # Create a view of only active params
                active_params = {
                    k: torch.nn.Parameter(v[:n_active], requires_grad=v.requires_grad)
                    for k, v in params.items()
                }
                inject_noise_to_position(
                    params=active_params,
                    optimizers=optimizers,
                    state={},
                    scaler=lr * self.noise_lr,
                )
            else:
                inject_noise_to_position(
                    params=params,
                    optimizers=optimizers,
                    state={},
                    scaler=lr * self.noise_lr,
                )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any], 
        binoms: Tensor,
    ) -> int:
        # Only Gaussians that are active
        if self.preallocate:
            n_active = state["n_active"]
            opacities = torch.sigmoid(params["opacities"][:n_active].flatten())
        else:
            opacities = torch.sigmoid(params["opacities"].flatten())

        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            if self.preallocate:
                # Build full-buffer mask
                full_mask = torch.zeros(
                    len(params["opacities"]), dtype=torch.bool,
                    device=params["opacities"].device,
                )
                full_mask[:n_active] = dead_mask
                relocate(
                    params=params,
                    optimizers=optimizers,
                    state={},
                    mask=full_mask,
                    binoms=binoms,
                    min_opacity=self.min_opacity,
                )
            else:
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
        state: Dict[str, Any],
        binoms: Tensor,
    ) -> int:
        # Use n_active instead of len(params["means"])
        if self.preallocate:
            current_n_points = state["n_active"]
        else:
            current_n_points = len(params["means"])

        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)

        # Clamp to available buffer space
        if self.preallocate:
            buffer_size = len(params["means"])
            n_gs = min(n_gs, buffer_size - current_n_points)

        if n_gs > 0:
            result = sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
                n_active=state.get("n_active", None) if self.preallocate else None,
            )
            if self.preallocate and result is not None:
                state["n_active"] = result

        return n_gs