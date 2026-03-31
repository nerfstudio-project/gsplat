# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Union

import torch
from torch import Tensor

from ..strategy.mcmc import MCMCStrategy


@dataclass
class NHTMCMCStrategy(MCMCStrategy):
    """MCMCStrategy variant with robust multinomial sampling.

    NHT training can produce NaN/Inf opacity logits during early iterations.
    This subclass sanitizes the opacity parameter tensor before the MCMC
    relocate and sample-add steps so that ``torch.multinomial`` does not fail.
    """

    @staticmethod
    def _sanitize_opacities(params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]):
        """Clamp opacity logits so sigmoid(opacities) is strictly in (0, 1)."""
        with torch.no_grad():
            p = params["opacities"]
            needs_fix = torch.isnan(p)
            if needs_fix.any():
                p.data[needs_fix] = 0.0

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        self._sanitize_opacities(params)
        return super()._relocate_gs(params, optimizers, binoms)

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        self._sanitize_opacities(params)
        return super()._add_new_gs(params, optimizers, binoms)
