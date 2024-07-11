import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch


@dataclass
class Strategy:
    """Base class for the GS optimization strategy."""

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize the state for the strategy.

        Returns an empty dictionary in the base class.
        """
        return {}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Check the sanity of the parameters and optimizers.

        It is not required but highly recommended for the user to call this
        function before calling `step_pre_backward()` and `step_post_backward()`.
        """
        assert set(params.keys()) == set(optimizers.keys()), (
            "params and optimizers must have the same keys, "
            f"but got {params.keys()} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Function to be executed before the backward() call.

        Does nothing in the base class.
        """
        pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Function to be executed after the backward() call.

        Does nothing in the base class.
        """
        pass
