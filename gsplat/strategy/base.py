import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch


@dataclass
class Strategy(abc.ABC):
    """Base class for the GS optimization strategy."""

    # Gaussian Parameters
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]
    # Optimizers for each parameter
    optimizers: Dict[str, torch.optim.Optimizer]

    def __post_init__(self):
        assert set(self.params.keys()) == set(self.optimizers.keys()), (
            "params and optimizers must have the same keys, "
            f"but got {self.params.keys()} and {self.optimizers.keys()}"
        )

        for optimizer in self.optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    @abc.abstractmethod
    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError

    @abc.abstractmethod
    def step_post_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError
