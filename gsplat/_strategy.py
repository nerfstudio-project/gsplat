from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import torch
from torch import Tensor


class Strategy(ABC):
    """Base class for the GS optimization strategy."""

    def __init__(
        self,
        params: torch.nn.ParameterDict,
        activations: Dict[str, Callable],
        optimizers: List[torch.optim.Optimizer],
        *args,
        **kwargs,
    ):
        self._params = params
        self._activations = activations
        self._optimizers = optimizers
        # some running states for the strategy
        self._state: DefaultDict[str, Any] = defaultdict(dict)
        pass

    @abstractmethod
    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def step_post_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError


class NullStrategy(Strategy):
    """A null strategy that does nothing."""

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        pass

    def step_post_backward(self, step: int, info: Dict[str, Any]):
        pass


class DefaultStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://arxiv.org/abs/2308.04079

    If absgrad=True, it will use the absolute gradients for GS splitting, following the
    AbsGS paper:

    "AbsGS: Recovering Fine Details for 3D Gaussian Splatting"
    https://arxiv.org/abs/2404.10484
    """

    def __init__(
        self,
        params: torch.nn.ParameterDict,
        activations: Dict[str, Callable],
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
        # Use absolute gradients for GS splitting
        absgrad: bool = False,
    ):
        super().__init__(params, activations, optimizers)
        self._verbose = verbose
        self._prune_opa = prune_opa
        self._grow_grad2d = grow_grad2d
        self._grow_scale3d = grow_scale3d
        self._prune_scale3d = prune_scale3d
        self._refine_start_iter = refine_start_iter
        self._refine_stop_iter = refine_stop_iter
        self._reset_every = reset_every
        self._refine_every = refine_every
        self._absgrad = absgrad

        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        self._state = {
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
        if step >= self._refine_stop_iter:
            return

        self._update_state(info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # TODO: Fill things here
            ...

    def _update_state(self, info: Dict[str, Any], packed: bool = False):
        for key in ["means2d", "width", "height", "n_cameras"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self._absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize states on the first run
        n_gaussian = len(self._params.values()[0])
        if self._state["grad2d"] is None:
            self._state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if self._state["count"] is None:
            self._state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
            self.state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids))
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            self.state["grad2d"].index_add_(0, gs_ids, grads[sel].norm(dim=-1))
            self.state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids).int())


class MCMCStrategy(Strategy):
    """Strategy that uses MCMC for GS splitting.

    "3D Gaussian Splatting as Markov Chain Monte Carlo"
    https://arxiv.org/abs/2404.09591
    """

    def __init__(
        self,
        params: torch.nn.ParameterDict,
        activations: Dict[str, Callable],
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

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        # nothing needs to be done before the backward
        pass

    def step_post_backward(self, step: int, info: Dict[str, Any], lr: float):
        if (
            step < self._refine_stop_iter
            and step > self._refine_start_iter
            and step % self._refine_every == 0
        ):
            num_relocated_gs = self._relocate_gs()
            if self._verbose:
                print(f"Step {step}: Relocated {num_relocated_gs} GSs.")

            num_new_gs = self._add_new_gs(self.cap_max)
            if self._verbose:
                print(
                    f"Step {step}: Added {num_new_gs} GSs. "
                    f"Now having {len(self.splats['means3d'])} GSs."
                )
        self._add_noise_to_gs(lr)

    @torch.no_grad()
    def _relocate_gs(self, min_opacity: float = 0.005) -> int:
        # TODO: Fill things here
        ...

    @torch.no_grad()
    def _add_new_gs(self, cap_max: int, min_opacity: float = 0.005) -> int:
        # TODO: Fill things here
        ...

    @torch.no_grad()
    def _add_noise_to_gs(self, lr: float):
        # TODO: Fill things here
        ...


class RevisedDensificationStrategy(DefaultStrategy):
    # TODO: Fill things here
    ...
