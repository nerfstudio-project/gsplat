from abc import ABC, abstractmethod
from typing import List, Optional, Any, DefaultDict
from config import Config
import torch
from torch import Tensor
import gsplat

from examples.utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    normalized_quat_to_rotmat,
    rgb_to_sh,
    set_random_seed,
)

class Strategy:
	# Init a Strategy with the Gaussian model, the optimizer and the schedulers.
	def __init__(
		self, 
		gaussians: gsplat.Gaussians, 
		optimizers: List[torch.optim.Optimizer], 
		schedulers: Optional[List[torch.optim.LRScheduler]] = None,
        device: torch.device = torch.device("cuda"),
        cfg: Config = Config(),
		last_epoch: int = -1,
	):
		self.gaussians = gaussians
		self.optimizers = optimizers
		self.schedulers = schedulers
		self.device = device
		self.cfg = cfg
		self.last_epoch = last_epoch
		
		# Similar to `torch.optim.Optimizer`, A Strategy may also need to maintain some running states of the model
		# https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
		self.state: DefaultDict[Tensor, Any] = defaultdict(dict)

    @abstractmethod
    def grow_GSs(self):
        raise NotImplementedError
    
    @abstractmethod
    def prune_GSs(self):
        raise NotImplementedError
    
    @abstractmethod
	def step_pre_backward(self, *args, **kwargs):
		# Inplace modify whatever needs to be modified in {gaussians, optimizers, schedulers}
		raise NotImplementedError
	
	@abstractmethod
	def step_post_backward(self, *args, **kwargs):
		# Inplace modify whatever needs to be modified in {gaussians, optimizers, schedulers}
		raise NotImplementedError


class NullStrategy(Strategy):

	def step_pre_backward(self, *args, **kwargs):
		# Do nothing
		return
		
	def step_post_backward(self, *args, **kwargs):
		# Do nothing
		return
	
class DefaultStategy(Strategy):
	
	def __init__(
		self, 
		gaussians: gsplat.Gaussians, 
		optimizers: List[torch.optim.Optimizer], 
		schedulers: Optional[List[torch.optim.LRScheduler]] = None,
        device: torch.device = torch.device("cuda"),
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        prune_scale3d: float = 0.1,
        refine_start_iter: int = 500,
        refine_stop_iter: int = 15_000,
        reset_every: int = 3000,
        refine_every: int = 100,
        scene_scale: float = 1.0,
		last_epoch: int = -1,
	):
		super().__init__(gaussians, optimizers, schedulers, device, last_epoch)
		self.prune_opa = prune_opa
		self.grow_grad2d = grow_grad2d
		self.grow_scale3d = grow_scale3d
        self.prune_scale3d = prune_scale3d
        self.refine_start_iter = refine_start_iter
        self.refine_stop_iter = refine_stop_iter
        self.reset_every = reset_every
        self.refine_every = refine_every
        self.state["grads2d"] = torch.zeros(len(self.gaussians["means3d"]), device=self.device)
        self.state["count"] = torch.zeros(len(self.gaussians["means3d"]), device=self.device)

    @torch.no_grad()
    def update_state(self, info: Dict[str, Any]):

        if self.cfg.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * self.cfg.batch_size
        grads[..., 1] *= info["height"] / 2.0 * self.cfg.batch_size

        if self.cfg.packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz] or None
            self.state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.state["count"].index_add_(
                0, gs_ids, torch.ones_like(gs_ids).int()
            )

    @torch.no_grad()
    def refine_split(self, mask: Tensor):
        device = self.device

        sel = torch.where(mask)[0]
        rest = torch.where(~mask)[0]

        scales = torch.exp(self.gaussians["scales"][sel])
        quats = F.normalize(self.gaussians["quats"][sel], dim=-1)
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
                self.splats[name] = p_new
        
        for k, v in self.state.items():
            if v is None:
                continue
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            self.state[k] = torch.cat((v[rest], v_new))
        torch.cuda.empty_cache()


    @torch.no_grad()
    def refine_duplicate(self, mask: Tensor):
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
                self.splats[name] = p_new
        for k, v in self.state.items():
            self.state[k] = torch.cat((v, v[sel]))
        torch.cuda.empty_cache()

    @torch.no_grad()
    def refine_keep(self, mask: Tensor):
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
                self.splats[name] = p_new
        for k, v in self.running_stats.items():
            self.running_stats[k] = v[sel]
        torch.cuda.empty_cache()

    def grow_GSs(self):
        count = len(self.gaussians["points"])
        grads = self.state["grads2d"] / count
        grads = grads.clamp_min(1)

        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(self.gaussians["scales"].max(dim=-1).values <= self.grow_scale3d * self.scene_scale)
        is_dupli = is_grad_high & is_small
        n_dupli = is_dupli.sum()

        self.refine_duplicate(is_dupli)

        is_split = is_grad_high & ~ is_small
        is_split = torch.cat([
                    is_split,
                    # new GSs added by duplication will not be split
                    torch.zeros(n_dupli)
                    ])
        
        n_split = is_split.sum().item()
        self.refine_split(is_split)

    def prune_GSs(self):
        is_prune = torch.sigmoid(self.gaussians["opacities"]) < self.prune_opa
        if self.last_epoch > self.reset_every:
            is_too_big = (torch.exp(self.gaussians["scales"]).max(dim=-1).values > self.prune_scale3d * self.scene_scale)
            is_prune = is_prune | is_too_big
        n_prune = is_prune.sum().item()
        
        self.refine_keep(~is_prune)

        if self.last_epoch % self.reset_every == 0:
            reset_opa(self.gaussians, value=self.prune_opa)


    
    def refine_duplicate(self):
        raise NotImplementedError
	
    def step_pre_backward(self):
		return


	def step_post_backward(self):
        if self.last_epoch > self.refine_start_iter and self.last_epoch & self.refine_every == 0:
            
            self.grow_GSs()
            self.prune_GSs()

        for optimizer in self.optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()
        return
	
class MCMCStrategy(Strategy):
	
	def __init__(
		self, 
		model: gsplat.Gaussians, e
		optimizers: List[torch.optim.Optimizer], 
		schedulers: Optional[List[torch.optim.LRScheduler]] = None,
		refine_start_iter: int = ...,
		refine_every: int = ...,
		...
		cap_max: int = 1_000_000,
        noise_lr: float = 5e5,
        opacity_reg: float = 0.01,
        scale_reg: float = 0.01,
        refine_stop_iter: int = 25_000
	):
		...
	
	def step_pre_backward(self):
		# Do nothing
		return

	def step_post_backward(self):
		self.last_epoch += 1
		
		if self.last_epoch > self.refine_start_iter and self.last_epoch & self.refine_every == 0:
			num_relocated_gs = relocate_gs(self.model, self.optimizers)
			num_new_gs = add_new_gs(self.cap_max)
			
			# add noise to GSs
			last_lr = self.schedulers[0].get_last_lr()[0]
			add_noise_to_gs(last_lr)	