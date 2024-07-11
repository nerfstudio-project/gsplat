from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Tuple

import torch


class Strategy(ABC):
    """Base class for the GS optimization strategy."""

    def __init__(
        self,
        params: torch.nn.ParameterDict,
        optimizers: List[torch.optim.Optimizer],
        *args,
        **kwargs,
    ):
        self._params = params
        self._optimizers = optimizers
        # some running states for the strategy
        self._state: DefaultDict[str, Any] = defaultdict(dict)

    @abstractmethod
    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def step_post_backward(self, step: int, info: Dict[str, Any]):
        raise NotImplementedError
