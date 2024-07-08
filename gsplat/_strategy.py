from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

from gsplat._gaussian import Gaussian


class Strategy(ABC):
    """Base class for the GS optimization strategy."""

    def __init__(
        self, gs: Gaussian, optimizers: List[torch.optim.Optimizer], *args, **kwargs
    ):
        self._gs = gs
        self._optimizers = optimizers
        self._state = {}
        pass

    @abstractmethod
    def step_pre_backward(self, step: int, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def step_post_backward(self, step: int, *args, **kwargs):
        raise NotImplementedError


class NullStrategy(Strategy):
    """A null strategy that does nothing."""

    def step_pre_backward(self, step: int, *args, **kwargs):
        pass

    def step_post_backward(self, step: int, *args, **kwargs):
        pass


class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://arxiv.org/abs/2308.04079
    """

    def __init__(
        self,
        gs: Gaussian,
        optimizers: List[torch.optim.Optimizer],
        verbose: bool = False,
        # GSs with opacity below this value will be pruned
        prune_opa: float = 0.005,
        # GSs with image plane gradient above this value will be split/duplicated
        grow_grad2d: float = 0.0002,
        # GSs with scale below this value will be duplicated. Above will be split
        grow_scale3d: float = 0.01,
        # GSs with scale above this value will be pruned.
        prune_scale3d: float = 0.1,
        # Start refining GSs after this iteration
        refine_start_iter: int = 500,
        # Stop refining GSs after this iteration
        refine_stop_iter: int = 15_000,
        # Reset opacities every this steps
        reset_every: int = 3000,
        # Refine GSs every this steps
        refine_every: int = 100,
    ):
        super().__init__(gs, optimizers)
        self._verbose = verbose
        self._prune_opa = prune_opa
        self._grow_grad2d = grow_grad2d
        self._grow_scale3d = grow_scale3d
        self._prune_scale3d = prune_scale3d
        self._refine_start_iter = refine_start_iter
        self._refine_stop_iter = refine_stop_iter
        self._reset_every = reset_every
        self._refine_every = refine_every
        self._state = {
            # running accum: the norm of the image plane gradients for each GS.
            "accum_norm_grad2d": None,
            # running accum: counting how many time each GS is visible.
            "accum_visible": None,
        }

    def step_pre_backward(self, step: int, means2d: Tensor):
        """Step before the backward pass.

        Retain the gradients of the image plane projections of the GSs.

        Args:
            means2d: the 2D means of the Gaussians projected to the image plane. Shape
              of [..., 2]
        """
        means2d.retain_grad()

    def step_post_backward(
        self,
        step: int,
        means2d: Tensor,
        indices: List[Tensor],
        batch_size: int,
        width: int,
        height: int,
    ):
        """Step after the backward pass.

        TODO: WIP

        Update running state and perform GS pruning, splitting, and duplicating.

        Args:
            means2d: the 2D means of the Gaussians projected to the image plane. Shape
              of [..., 2]
            batch_size: the batch size of the rendering.
            width: the width of the image plane
            height: the height of the image plane
            indices: a list of tensors containing the indices of the active GSs, similar
              to the output of torch.where.
        """
        if step >= self._refine_stop_iter:
            return

        self._update_state(means2d, width, height, indices)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            grads = self._state["accum_norm_grad2d"] / self._state[
                "accum_visible"
            ].clamp_min(1)

            # grow GSs
            is_grad_high = grads >= self.grow_grad2d
            is_small = (
                torch.exp(self.splats["scales"]).max(dim=-1).values
                <= self.grow_scale3d * self.scene_scale
            )
            is_dupli = is_grad_high & is_small
            n_dupli = is_dupli.sum().item()
            self.refine_duplicate(is_dupli)

            is_split = is_grad_high & ~is_small
            is_split = torch.cat(
                [
                    is_split,
                    # new GSs added by duplication will not be split
                    torch.zeros(n_dupli, device=device, dtype=torch.bool),
                ]
            )
            n_split = is_split.sum().item()
            self.refine_split(is_split)
            if self._verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(self.splats['means3d'])} GSs."
                )

            # prune GSs
            is_prune = torch.sigmoid(self.splats["opacities"]) < self.prune_opa
            if step > self.reset_every:
                # The official code also implements sreen-size pruning but
                # it's actually not being used due to a bug:
                # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
                is_too_big = (
                    torch.exp(self.splats["scales"]).max(dim=-1).values
                    > self.prune_scale3d * self.scene_scale
                )
                is_prune = is_prune | is_too_big
            n_prune = is_prune.sum().item()
            self.refine_keep(~is_prune)
            if self._verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(self.splats['means3d'])} GSs."
                )

            # reset running stats
            self._state["accum_norm_grad2d"].zero_()
            self._state["accum_visible"].zero_()

    def _update_state(
        self,
        means2d: Tensor,
        indices: List[Tensor],
        batch_size: int,
        width: int,
        height: int,
    ):
        # normalize grads to [-1, 1] screen space
        grads = means2d.grad
        assert grads is not None, "Gradients of means2d are not available."
        grads[..., 0] *= width * 0.5 * batch_size
        grads[..., 1] *= height * 0.5 * batch_size
        norm_grads = torch.norm(grads, dim=-1)

        if self._state["accum_norm_grad2d"] is None:
            self._state["accum_norm_grad2d"] = torch.zeros_like(norm_grads)
        self._state["accum_norm_grad2d"] += norm_grads

        if self._state["accum_visible"] is None:
            self._state["accum_visible"] = torch.zeros_like(norm_grads)
        self._state["accum_visible"][indices] += 1


class AbaGradStrategy(DefaultStategy):
    """Strategy that uses the absolute gradients for GS splitting.

    This strategy is based on the following paper:

    "AbsGS: Recovering Fine Details for 3D Gaussian Splatting"
    https://arxiv.org/abs/2404.10484
    """

    def _update_state(
        self,
        means2d: Tensor,
        indices: List[Tensor],
        batch_size: int,
        width: int,
        height: int,
    ):
        # normalize grads to [-1, 1] screen space
        grads = means2d.absgrad
        assert grads is not None, "Gradients of means2d are not available."
        grads[..., 0] *= width * 0.5 * batch_size
        grads[..., 1] *= height * 0.5 * batch_size
        norm_grads = torch.norm(grads, dim=-1)

        if self._state["accum_norm_grad2d"] is None:
            self._state["accum_norm_grad2d"] = torch.zeros_like(norm_grads)
        self._state["accum_norm_grad2d"] += norm_grads

        if self._state["accum_visible"] is None:
            self._state["accum_visible"] = torch.zeros_like(norm_grads)
        self._state["accum_visible"][indices] += 1


class MCMCStrategy(Strategy):
    """Strategy that uses MCMC for GS splitting.

    "3D Gaussian Splatting as Markov Chain Monte Carlo"
    https://arxiv.org/abs/2404.09591
    """

    def __init__(
        self,
        gs: Gaussian,
        optimizers: List[torch.optim.Optimizer],
        verbose: bool = False,
        # Maximum number of GSs.
        cap_max: int = 1_000_000,
        # MCMC samping noise learning rate
        noise_lr: float = 5e5,
        # Start refining GSs after this iteration
        refine_start_iter: int = 500,
        # Stop refining GSs after this iteration
        refine_stop_iter: int = 25_000,
        # Refine GSs every this steps
        refine_every: int = 100,
    ):
        super().__init__(gs, optimizers)
        self._verbose = verbose
        self._cap_max = cap_max
        self._noise_lr = noise_lr
        self._refine_start_iter = refine_start_iter
        self._refine_stop_iter = refine_stop_iter
        self._refine_every = refine_every

    def step_pre_backward(self, step: int):
        # nothing needs to be done before the backward
        pass

    def step_post_backward(self, step: int, lr: float):
        if (
            step < self._refine_stop_iter
            and step > self._refine_start_iter
            and step % self._refine_every == 0
        ):
            num_relocated_gs = self._relocate_gs()
            if self._verbose:
                print(f"Step {step}: Relocated {num_relocated_gs} GSs.")

            num_new_gs = self.add_new_gs(self.cap_max)
            if self._verbose:
                print(
                    f"Step {step}: Added {num_new_gs} GSs. "
                    f"Now having {len(self.splats['means3d'])} GSs."
                )
        self.add_noise_to_gs(lr)

    @torch.no_grad()
    def _relocate_gs(self, min_opacity: float = 0.005) -> int:
        dead_mask = torch.sigmoid(self.splats["opacities"]) <= min_opacity
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        num_gs = len(dead_indices)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(self.splats["opacities"])[alive_indices]
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(self.splats["opacities"])[sampled_idxs],
            scales=torch.exp(self.splats["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.splats["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.splats["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.splats.keys():
            self.splats[k][dead_indices] = self.splats[k][sampled_idxs]
        for optimizer in self.optimizers:
            for i, param_group in enumerate(optimizer.param_groups):
                p = param_group["params"][0]
                name = param_group["name"]
                p_state = optimizer.state[p]
                del optimizer.state[p]
                for key in p_state.keys():
                    if key != "step":
                        p_state[key][sampled_idxs] = 0
                p_new = torch.nn.Parameter(self.splats[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        torch.cuda.empty_cache()
        return num_gs

    @torch.no_grad()
    def _add_new_gs(self, cap_max: int, min_opacity: float = 0.005) -> int:
        current_num_points = len(self.splats["means3d"])
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return num_gs

        # Sample for new GSs
        eps = torch.finfo(torch.float32).eps
        probs = torch.sigmoid(self.splats["opacities"])
        probs = probs / (probs.sum() + eps)
        sampled_idxs = torch.multinomial(probs, num_gs, replacement=True)
        new_opacities, new_scales = compute_relocation(
            opacities=torch.sigmoid(self.splats["opacities"])[sampled_idxs],
            scales=torch.exp(self.splats["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        )
        new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)
        self.splats["opacities"][sampled_idxs] = torch.logit(new_opacities)
        self.splats["scales"][sampled_idxs] = torch.log(new_scales)

        # Update splats and optimizers
        for k in self.splats.keys():
            self.splats[k] = torch.cat([self.splats[k], self.splats[k][sampled_idxs]])
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
                p_new = torch.nn.Parameter(self.splats[name])
                optimizer.param_groups[i]["params"] = [p_new]
                optimizer.state[p_new] = p_state
                self.splats[name] = p_new
        torch.cuda.empty_cache()
        return num_gs

    @torch.no_grad()
    def _add_noise_to_gs(self, last_lr):
        opacities = torch.sigmoid(self.splats["opacities"])
        scales = torch.exp(self.splats["scales"])
        actual_covariance, _ = quat_scale_to_covar_preci(
            self.splats["quats"],
            scales,
            compute_covar=True,
            compute_preci=False,
            triu=False,
        )

        def op_sigmoid(x, k=100, x0=0.995):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        noise = (
            torch.randn_like(self.splats["means3d"])
            * (op_sigmoid(1 - opacities)).unsqueeze(-1)
            * cfg.noise_lr
            * last_lr
        )
        noise = torch.bmm(actual_covariance, noise.unsqueeze(-1)).squeeze(-1)
        self.splats["means3d"].add_(noise)
