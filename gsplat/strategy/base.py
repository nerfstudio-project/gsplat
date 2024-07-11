from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Tuple, Union

import torch


class Strategy(ABC):
    """Base class for the GS optimization strategy."""

    def __init__(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        *args,
        **kwargs,
    ):
        self._params = params
        self._optimizers = optimizers

        # some running states for the strategy
        self._state: DefaultDict[str, Any] = defaultdict(dict)

        # sanity checks
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

    @abstractmethod
    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def step_post_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError
