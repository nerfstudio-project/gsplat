from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from torch import Tensor

import gsplat
from gsplat import quat_scale_to_covar_preci
from gsplat.relocation import compute_relocation

from examples.utils import normalized_quat_to_rotmat


class Strategy(ABC):
    @abstractmethod
    def __init__(
        self,
        params: nn.ParameterDict,
        activations: Dict[str, Callable],
        optimizers: List[torch.optim.Optimizer],
        **kwargs,
    ):
        self.params = params
        self.activations = activations
        self.optimizers = optimizers

    def get_param(self, name: str, mask: Optional[Tensor] = None, **activation_kwargs) -> Tensor:
        """Get a parameter and apply associated activation. 
        
        Notes
        - Optionally use mask to access curtain values and apply activation to them.
        - activation_kwargs can be used to pass in additional arguments to the activation function e.g. dim=-1 for F.normalize.
        - Currently, this doesn't have an elegant way to apply tensor methods like .max() on a dim before calling the activation function.
        """
        if mask is None:
            return self.activations[name](self.params[name], **activation_kwargs)
        else:
            return self.activations[name](self.params[name][mask], **activation_kwargs)

    @abstractmethod
    def step_pre_backward(self, info: Dict[str, Any]):
        pass

    @abstractmethod
    def step_post_backward(self, info: Dict[str, Any]):
        pass

class GSStrategy(Strategy):
    def __init__(
        self,
        params: nn.ParameterDict,
        activations: Dict[str, Callable],
        optimizers: List[torch.optim.Optimizer],
        schedulers: List[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device = torch.device("cuda"),

        # params specific to this strategy
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        prune_scale3d: float = 0.1,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15_000,
        reset_every: int = 3000,
        refine_every: int = 100,
        scene_scale: float = 1.0,
        absgrad: bool = False,
        batch_size: int = 1,
        packed: bool = False,
        sparse_grad: bool = False,
    ):
        self.params = params
        self.activations = activations
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device

        # Similar to `torch.optim.Optimizer`, A Strategy may also need to maintain some running states of the model
        # https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
        self.state: DefaultDict[str, Any] = defaultdict(dict)
        n_gauss = len(self.params["means3d"])
        self.state["grad2d"] = torch.zeros(n_gauss, device=self.device)
        self.state["count"] = torch.zeros(n_gauss, device=self.device, dtype=torch.int)
        self.state["step"] = 0

        self.prune_opa = prune_opa
        self.grow_grad2d = grow_grad2d
        self.grow_scale3d = grow_scale3d
        self.prune_scale3d = prune_scale3d
        self.refine_start_iter = refine_start_iter
        self.refine_stop_iter = refine_stop_iter
        self.reset_every = reset_every
        self.refine_every = refine_every
        self.scene_scale = scene_scale

        # unsure if these should be passed in. only needed for update_state()
        self.absgrad = absgrad
        self.batch_size = batch_size
        self.packed = packed
        self.sparse_grad = sparse_grad

    def step_pre_backward(self, info: Dict[str, Any]):
        """GSStrategy does nothing before loss.backward() call"""
        return

    def step_post_backward(self, info: Dict[str, Any]):
        if self.state["step"] < self.refine_stop_iter:
            self._update_state(info)

            if (self.state["step"] > self.refine_start_iter and self.state["step"] & self.refine_every == 0):
                self._grow_GSs()
                self._prune_GSs()

            self.state["grad2d"].zero_()
            self.state["count"].zero_()

        if self.state["step"] % self.reset_every == 0:
            self._reset_opa(value=self.prune_opa * 2.0)

        self.state["step"] += 1

        return

    def _grow_GSs(self):
        count = self.state["count"]
        grads = self.state["grads2d"] / count
        grads = grads.clamp_min(1)

        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(self.params["scales"].max(dim=-1).values <= self.grow_scale3d * self.scene_scale)
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum()

        if n_dupli > 0:
            self.params._duplicate(is_dupli)

        is_split = is_grad_high & ~is_small
        is_split = torch.cat(
            [
                is_split,
                # new GSs added by duplication will not be split
                torch.zeros(n_dupli),
            ]
        )

        n_split = is_split.sum().item()
        self._refine_split(is_split)
        print(
            f"Step {self.state['step']}: {n_dupli} GSs duplicated, {n_split} GSs split. "
            f"Now having {len(self.params['means3d'])} GSs."
        )

    def _prune_GSs(self):
        is_prune = torch.sigmoid(self.params["opacities"]) < self.prune_opa
        if self.state["step"] > self.reset_every:
            is_too_big = (torch.exp(self.params["scales"]).max(dim=-1).values > self.prune_scale3d * self.scene_scale)
            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        self._refine_keep(~is_prune)

        print(
            f"Step {self.state['step']}: {n_prune} GSs pruned. "
            f"Now having {len(self.params['means3d'])} GSs."
        )

        if self.state["step"] % self.reset_every == 0:
            self._reset_opa(value=self.prune_opa)

    @torch.no_grad()
    def _update_state(self, info: Dict[str, Any]):
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * self.batch_size
        grads[..., 1] *= info["height"] / 2.0 * self.batch_size

        if self.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids).int())
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids).int())

    @torch.no_grad()
    def _refine_split(self, mask: Tensor):
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = self.get_param("scales", mask=sel)
        quats = self.get_param("quats", mask=sel, dim=-1)
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
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
    def _refine_duplicate(self, mask: Tensor):
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        for k, v in self.state.items():
            self.state[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _refine_keep(self, mask: Tensor):
        """Unility function to prune GSs."""
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key] = p_state[key][sel]
                p_new = torch.nn.Parameter(p[sel])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _reset_opa(self, value: float = 0.01):
        """Utility function to reset opacities."""
        opacities = torch.clamp(
            self.params["opacities"], max=torch.logit(torch.tensor(value)).item()
        )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                if param_group["name"] != "opacities":
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
                self.params[param_group["name"]] = p_new
        torch.cuda.empty_cache()


class MCMCStrategy(Strategy):
    def __init__(
        self,
        params: nn.ParameterDict,
        activations: Dict[str, Callable],
        optimizers: List[torch.optim.Optimizer],
        schedulers: List[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device = torch.device("cuda"),
        # new to MCMC
        cap_max: int = 1_000_000,
        noise_lr: int = 5e5,
        opacity_reg: float = 0.001,
        scale_reg: float = 0.01,
        refine_start_iter: int = 500,
        # new refine_stop_iter
        refine_stop_iter: int = 25_000,
        refine_every: int = 100,
        scene_scale: float = 1.0,
        absgrad: bool = False,
        batch_size: int = 1,
        packed: bool = False,
        sparse_grad: bool = False,
    ):
        self.params = params
        self.activations = activations
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.device = device

        # Similar to `torch.optim.Optimizer`, A Strategy may also need to maintain some running states of the model
        # https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
        self.state: DefaultDict[str, Any] = defaultdict(dict)
        self.state["step"] = 0
        self.state["count"] = 0

        self.cap_max = cap_max
        self.noise_lr = noise_lr
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg

        self.refine_start_iter = refine_start_iter
        self.refine_stop_iter = refine_stop_iter
        self.refine_every = refine_every
        self.scene_scale = scene_scale

        # unsure if these should be passed in. only needed for update_state()
        self.absgrad = absgrad
        self.batch_size = batch_size
        self.packed = packed
        self.sparse_grad = sparse_grad

    def step_pre_backward(self, info: Dict[str, Any]):
        """do nothing"""
        return

    def step_post_backward(self, info: Dict[str, Any]):
        if self.state["step"] < self.refine_stop_iter:

            if self.state["step"] > self.refine_start_iter and self.state["step"] & self.refine_every == 0:
                num_relocated_gs = self._relocate_gs()
                print(f"Step {self.state['step']}: Relocated {num_relocated_gs} GSs.")

                num_new_gs = self._add_new_gs(self.cap_max)
                print(f"Step {self.state['step']}: Added {num_new_gs} GSs. Now having {len(self.params['means3d'])} GSs.")


        self.state["step"] += 1
    
    @torch.no_grad()
    def _relocate_gs(self, min_opacity: float = 0.005) -> int:
        dead_mask = self.params["opacities"] <= min_opacity
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        num_gs = len(dead_indices)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = self.params["opacities"][alive_indices]
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = compute_relocation(
            opacities=self.params["opacities"][sampled_idxs],
            scales=self.params["scales"][sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.params["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.params["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.params.keys():
            self.params[k][dead_indices] = self.params[k][sampled_idxs]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key][sampled_idxs] = 0
                p_new = torch.nn.Parameter(self.params[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        torch.cuda.empty_cache()
        return num_gs
    
    @torch.no_grad()
    def _add_new_gs(self, min_opacity: float = 0.005) -> int:
        current_num_points = len(self.params["means3d"])
        target_num = min(self.cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = self.params["opacities"]
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        new_opacities, new_scales = compute_relocation(
            opacities=self.params["opacities"][sampled_idxs],
            scales=self.params["scales"][sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.params["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.params["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.params.keys():
            self.params[k] = torch.cat(
                [self.params[k], self.params[k][sampled_idxs]]
            )
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sampled_idxs), *v.shape[1:]), device=self.device
                        )
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(self.params[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        torch.cuda.empty_cache()
        return num_gs

    @torch.no_grad()
    def _add_noise_to_gs(self, last_lr: float):
        opacities = self.params["opacities"]
        scales = self.params["scales"]
        actual_covariance, _ = quat_scale_to_covar_preci(
            self.params["quats"],
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(self.params["means3d"])
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * self.noise_lr
            * last_lr
        )
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self.params["means3d"].add_(noise)

class RevisedDensificationStrategy(GSStrategy):
    @torch.no_grad()
    def _refine_split(self, mask: Tensor):
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = self.get_param("scales", mask=sel)
        quats = self.get_param("quats", mask=sel, dim=-1)
        opacities = self.get_param("opacities", mask=sel)
        rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
        samples = torch.einsum(
            "nij,nj,bnj->bni",
            rotmats,
            scales,
            torch.randn(2, len(scales), 3, device=device),
        )  # [2, N, 3]

        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                # create new params
                if name == "means3d":
                    p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
                elif name == "scales":
                    p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
                elif name == "opacities":
                    get_new_opacity = lambda opacity: 1 - torch.sqrt(1 - opacity)
                    new_opacities = get_new_opacity(opacities)
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
    def _refine_duplicate(self, mask: Tensor):
        sel = torch.where(mask)[0]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        # new params are assigned with zero optimizer states
                        # (worth investigating it as it will lead to a lot more GS.)
                        v = p_state[key]
                        v_new = torch.zeros(
                            (len(sel), *v.shape[1:]), device=self.device
                        )
                        # v_new = v[sel]
                        p_state[key] = torch.cat([v, v_new])
                p_new = torch.nn.Parameter(torch.cat([p, p[sel]]))
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.params[name] = p_new
        for k, v in self.state.items():
            self.state[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    