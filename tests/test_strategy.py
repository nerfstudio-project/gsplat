"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

import math
from typing import Optional

import pytest
import torch

device = torch.device("cuda:0")


def test_strategy():
    from gsplat._strategy import DefaultStrategy, MCMCStrategy, NullStrategy

    torch.manual_seed(42)

    # TODO: create a pesudo list of parameters, activations, and optimizers
    # to initialize the strategy

    # Then create a pesudo info dictionary as the argument for the step_pre_backward
    # and step_post_backward functions to test them out.
