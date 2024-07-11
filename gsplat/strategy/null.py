from typing import Any, Callable, DefaultDict, Dict, List, Tuple

from .base import Strategy


class NullStrategy(Strategy):
    """A null strategy that does nothing."""

    def step_pre_backward(self, step: int, info: Dict[str, Any]):
        pass

    def step_post_backward(self, step: int, info: Dict[str, Any]):
        pass
