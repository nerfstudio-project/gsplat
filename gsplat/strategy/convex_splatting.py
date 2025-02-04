from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch

from .base import Strategy
from .ops_convex import remove, reset_opa, split
from typing_extensions import Literal


@dataclass
class ConvexSplattingStrategy(Strategy):
    """A strategy that follows the convex splatting strategy of the paper:

    `3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes <https://arxiv.org/abs/2411.14974>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split 3D convexes with high sigma gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.
        key_for_gradient (str): Which variable uses for densification strategy.
          3DGS uses "means2d" gradient and 2DGS uses a similar gradient which stores
          in variable "gradient_2dgs".

    Examples:

        >>> from gsplat import ConvexSplattingStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = ConvexSplattingStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.02
    grow_grad_sigma: float = 0.000025
    mask_threshold: float = 0.01
    refine_scale2d_stop_iter: int = 0
    scaling_clone = 0.5
    reset_opacity_until = 9000
    refine_start_iter: int = 500
    refine_stop_iter: int = 9000
    reset_every: int = 3000
    refine_every: int = 200
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: str = "sigma"
    sigma_scaling_cloning: float = 0.88
    scaling_cloning: float = 0.63

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - grad_sigma: running accum of the norm of the 3D scale gradients for each 3D convexes.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"count": None, "scene_scale": scene_scale, "sigma": None}
        #if self.refine_scale2d_stop_iter > 0:
        state["radii"] = None
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
            * The following keys are present: {"convex_points", "opacities", "delta", "sigma", "mask"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["convex_points", "opacities", "delta", "sigma", "mask"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in info
        ), "The Sigma of the Convexes is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter and step % 1000 != 0:
            return
        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            if step < self.refine_stop_iter:
                # grow CSs
                n_split = self._grow_sigma_big(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_split} Convexes split. "
                        f"Now having {len(params['convex_points'])} Convexes."
                    )

            # prune CSs
            n_prune = self._prune_convexes(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} Convexes pruned. "
                    f"Now having {len(params['convex_points'])} Convexes."
                )

            # reset running stats
            state["count"].zero_()
            state["sigma"].zero_()
            #if self.refine_scale2d_stop_iter > 0:
            state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step <= self.reset_opacity_until:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=0.2,
            )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["sigma"] is None:
            state["sigma"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        #if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
        if state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel.permute(1, 0)]  # [nnz]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["sigma"].index_add_(0, gs_ids, grads)
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        #if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
        
        state["radii"][gs_ids] = state["radii"][gs_ids]
        # state["radii"][gs_ids] = torch.maximum(
        #     state["radii"][gs_ids],
        #     # normalize radii to [0, 1] screen space
        #     #radii / float(max(info["width"], info["height"])),
        #     radii,
        #)

    @torch.no_grad()
    def _grow_sigma_big(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        count = state["count"]
        # 3DCS
        grads_sigma = state["sigma"] / count.clamp_min(1)
        is_grad_high = grads_sigma >= self.grow_grad_sigma

        n_split = is_grad_high.sum().item()

        # is_small = (
        #     state["radii"].max(dim=-1).values
        #     <= 0.3 * state["scene_scale"]
        # )
        # print("before", is_grad_high.sum().item() )
        # is_dupli = is_grad_high & is_small
        # n_dupli = is_dupli.sum().item()
        # print("after", is_dupli.sum().item() )

        #is_large = ~is_small
        #is_split = is_grad_high & is_large
        # if step < self.refine_scale2d_stop_iter:
        #     is_split |= state["sigma"] > self.grow_scale2d
        # n_split = is_split.sum().item()
        
        if n_split > 0:
            split(params=params, optimizers=optimizers, state=state, mask=is_grad_high, sigma_scaling_cloning=self.sigma_scaling_cloning, scaling_cloning=self.scaling_cloning)
 
        return n_split

    @torch.no_grad()
    def _prune_convexes(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        #is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        is_prune = torch.logical_or((torch.sigmoid(params["mask"]) <= self.mask_threshold).squeeze(), (torch.sigmoid(params["opacities"]) < 0.03).squeeze())
  
        #if step > self.reset_every:
        is_too_big = state['radii'] >= 0.03
    #     # The official code also implements sreen-size pruning but
    #     # it's actually not being used due to a bug:
    #     # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
    #     # We implement it here for completeness but set `refine_scale2d_stop_iter`
    #     # to 0 by default to disable it.
    #     if step < self.refine_scale2d_stop_iter:
    #         is_too_big |= state["radii"] > self.prune_scale2d

        is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune
