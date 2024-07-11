from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

from .base import Strategy


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

    def __post_init__(self):
        # The following keys are required for this strategy.
        for key in ["means3d", "scales", "quats", "opacities"]:
            assert key in self.params, f"{key} is required in params but missing."

        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        self.state = {
            # running accum: the norm of the image plane gradients for each GS.
            "grad2d": None,
            # running accum: counting how many time each GS is visible.
            "count": None,
        }

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        """Step before the backward pass.

        Retain the gradients of the image plane projections of the GSs.
        """
        assert (
            "means2d" in info
        ), "The 2D means of the Gaussians is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(self, step: int, info: Dict[str, Any], packed: bool = False):
        """Step after the backward pass.

        Update running state and perform GS pruning, splitting, and duplicating.
        """
        if step >= self.refine_stop_iter:
            return

        self._update_state(info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # grow GSs
            n_dupli, n_split = self._grow_gs()
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(self.params['means3d'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(self.params['means3d'])} GSs."
                )

            # reset running stats
            self.state["grad2d"].zero_()
            self.state["count"].zero_()

        if step % self.reset_every == 0:
            self._reset_opa(value=self.prune_opa * 2.0)

    @torch.no_grad()
    def _grow_gs(self) -> Tuple[int, int]:
        count = self.state["count"]
        grads = self.state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = (
            torch.exp(self.params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * self.scene_scale
        )
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum().item()

        if n_dupli > 0:
            self._refine_duplicate(is_dupli)

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
            self._refine_split(is_split)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(self, step) -> int:
        is_prune = torch.sigmoid(self.params["opacities"]) < self.prune_opa
        if step > self.reset_every:
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            is_too_big = (
                torch.exp(self.params["scales"]).max(dim=-1).values
                > self.prune_scale3d * self.scene_scale
            )
            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            self._refine_keep(~is_prune)
        return n_prune

    def _update_state(self, info: Dict[str, Any], packed: bool = False):
        for key in ["means2d", "width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize states on the first run
        n_gaussian = len(list(self.params.values())[0])
        if self.state["grad2d"] is None:
            self.state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if self.state["count"] is None:
            self.state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
            )

    @torch.no_grad()
    def _refine_duplicate(self, mask: Tensor):
        device = mask.device
        sel = torch.where(mask)[0]
        for name, optimizer in self.optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros((len(sel), *v.shape[1:]), device=device)
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        for k, v in self.state.items():
            self.state[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _refine_split(self, mask: Tensor):
        device = mask.device
        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.params["scales"])[sel]
        quats = F.normalize(self.params["quats"], dim=-1)[sel]
        opacities = torch.sigmoid(self.params["opacities"])[sel]
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for name, optimizer in self.optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]

                # TODO: test if this is correct
                elif name == "opacities" and self.revised_opacity:
                    new_opacities = 1 - torch.sqrt(1 - opacities)
                    p_split = torch.logit(new_opacities).repeat(2, 1)
                else:
                    repeats = [2] + [1] * (p.dim() - 1)
                    p_split = p[sel].repeat(repeats)
                p_new = torch.cat([p[rest], p_split])
                p_new = torch.nn.Parameter(p_new)
                # update optimizer
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key == "step":
                        continue
                    v = p_state[key]
                    # new params are assigned with zero optimizer states
                    # (worth investigating it)
                    v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
                    p_state[key] = torch.cat([v[rest], v_split])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new

        for k, v in self.state.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.state[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for name, optimizer in self.optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        for k, v in self.state.items():
            self.state[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.params["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for name, optimizer in self.optimizers.items():
            for i, param_group in enumerate(optimizer.param_groups):
                if name != "opacities":
                    continue
                p = param_group["params"][0]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = torch.zeros_like(p_state[key])
                p_new = torch.nn.Parameter(opacities)
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        torch.cuda.empty_cache()
