from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation

from .base import Strategy


@dataclass
class MCMCStrategy(Strategy):
    """Strategy that uses MCMC for GS splitting.

    "3D Gaussian Splatting as Markov Chain Monte Carlo"
    https://arxiv.org/abs/2404.09591
    """

    # Whether to print verbose information.
    verbose: bool = False
    # Maximum number of GSs.
    cap_max: int = 1_000_000
    # MCMC samping noise learning rate
    noise_lr: float = 5e5
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 25_000
    # Refine GSs every this steps
    refine_every: int = 100

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means3d", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Step before the backward pass.

        Nothing needs to be done here for MCMCStrategy.
        """
        pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Step after the backward pass.

        Relocate Gaussians, create new Gaussians, and add noise to Gaussians.
        """
        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            # add new GSs
            n_new_gs = self._add_new_gs(params, optimizers, self.cap_max)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {len(params['means3d'])} GSs."
                )

        # add noise to GSs
        self._add_noise_to_gs(params, optimizers, lr)

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        min_opacity: float = 0.005,
    ) -> int:
        dead_mask = torch.sigmoid(params["opacities"]) <= min_opacity
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        n_gs = len(dead_indices)
        if n_gs <= 0:
            return n_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(params["opacities"])[alive_indices]
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, n_gs, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(params["opacities"])[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        params["opacities"][sampled_idxs] = torch.logit(new_opacities)
        params["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in params.keys():
            params[k][dead_indices] = params[k][sampled_idxs]
        for name, optimizer in optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key][sampled_idxs] = 0
                p_new = torch.nn.Parameter(params[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                params[name] = p_new
        torch.cuda.empty_cache()
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        cap_max: int,
        min_opacity: float = 0.005,
    ) -> int:
        current_n_points = len(params["means3d"])
        n_target = min(cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs <= 0:
            return n_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(params["opacities"])
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, n_gs, replacement=True)
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(params["opacities"])[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        params["opacities"][sampled_idxs] = torch.logit(new_opacities)
        params["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in params.keys():
            params[k] = torch.cat([params[k], params[k][sampled_idxs]])
        for name, optimizer in optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sampled_idxs), *v.shape[1:]), device=v.device
                        )
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(params[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                params[name] = p_new
        torch.cuda.empty_cache()
        return n_gs

    @torch.no_grad()
    def _add_noise_to_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        lr: float,
    ):
        opacities = torch.sigmoid(params["opacities"])
        scales = torch.exp(params["scales"])
        actual_covariance, _ = quat_scale_to_covar_preci(
            params["quats"],
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(params["means3d"])
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * self.noise_lr
            * lr
        )
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        params["means3d"].add_(noise)
