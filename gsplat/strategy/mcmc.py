from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add


@dataclass
class MCMCStrategy(Strategy):
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_
    `Deformable Beta splatting <https://arxiv.org/abs/2501.18630>`_

    This strategy will:

    - Periodically teleport primitives with low opacity to a place that has high opacity.
    - Periodically introduce new primitives sampled based on the opacity distribution.
    - Periodically perturb the primitives locations.

    Args:
        cap_max (int): Maximum number of primitives. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_iter (int): Start refining primitives after this iteration. Default to 500.
        refine_stop_iter (int): Stop refining primitives after this iteration. Default to 25_000.
        refine_every (int): Refine primitives every this steps. Default to 100.
        min_opacity (float): primitives with opacity below this value will be pruned. Default to 0.005.
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
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        pass

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
        for key in ["means", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

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
            lr (float): Learning rate for "means" attribute of the primitive.
        """

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport primitives
            n_relocated_primitives = self._relocate_primitives(params, optimizers)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_primitives} Primitives.")

            # add new primitives
            n_new_primitive = self._add_new_primitives(params, optimizers)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_primitive} Primitives. "
                    f"Now having {len(params['means'])} Primitives."
                )

            torch.cuda.empty_cache()

        # add noise to primitives
        inject_noise_to_position(
            params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        )

    @torch.no_grad()
    def _relocate_primitives(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_primitive = dead_mask.sum().item()
        if n_primitive > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                min_opacity=self.min_opacity,
            )
        return n_primitive

    @torch.no_grad()
    def _add_new_primitives(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_primitive = max(0, n_target - current_n_points)
        if n_primitive > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_primitive,
                min_opacity=self.min_opacity,
            )
        return n_primitive
