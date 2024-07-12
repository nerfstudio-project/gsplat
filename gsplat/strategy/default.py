from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

from .base import Strategy
from .ops import duplicate, remove, reset_opa, split


@dataclass
class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://arxiv.org/abs/2308.04079

    If absgrad=True, it will use the absolute gradients for GS splitting, following the
    AbsGS paper:

    "AbsGS: Recovering Fine Details for 3D Gaussian Splatting"
    https://arxiv.org/abs/2404.10484
    """

    # Whether to print verbose information
    verbose: bool = False
    # The scale of the scene for calibrating the scale-related logic.
    scene_scale: float = 1.0
    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.005
    # GSs with image plane gradient above this value will be split/duplicated
    grow_grad2d: float = 0.0002
    # GSs with scale below this value will be duplicated. Above will be split
    grow_scale3d: float = 0.01
    # GSs with scale above this value will be pruned.
    prune_scale3d: float = 0.1
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Use absolute gradients for GS splitting
    absgrad: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        return {
            # running accum: the norm of the image plane gradients for each GS.
            "grad2d": None,
            # running accum: counting how many time each GS is visible.
            "count": None,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
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

        Retain the gradients of the image plane projections of the GSs.
        """
        assert (
            "means2d" in info
        ), "The 2D means of the Gaussians is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Step after the backward pass.

        Update running state and perform GS pruning, splitting, and duplicating.
        """
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in ["means2d", "width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> Tuple[int, int]:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * self.scene_scale
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)

        is_split = is_grad_high & ~is_small
        is_split = torch.cat(
            [
                is_split,
                # new GSs added by duplication will not be split
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        n_split = is_split.sum().item()
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=self.revised_opacity,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"]) < self.prune_opa
        if step > self.reset_every:
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * self.scene_scale
            )
            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
