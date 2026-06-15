# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke coverage for the relocation CUDA op (gsplat.relocation.compute_relocation).

The op is reached only through the MCMC densification strategy, which the rest
of the suite does not exercise, so this gives it direct runtime coverage: that
it runs end to end and returns finite, correctly-shaped tensors.
"""

import math

import pytest
import torch

from gsplat.relocation import compute_relocation

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _binomial_table(n_max: int, device: torch.device) -> torch.Tensor:
    # Lower-triangular Pascal's triangle, matching MCMCStrategy's state init.
    binoms = torch.zeros((n_max, n_max), device=device)
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    return binoms


def _reference_relocation(
    opacities: torch.Tensor,
    scales: torch.Tensor,
    ratios: torch.Tensor,
    binoms: torch.Tensor,
    min_opacity: float,
):
    opacities = opacities.cpu()
    scales = scales.cpu()
    ratios = ratios.cpu().to(torch.int64)
    binoms = binoms.cpu()

    new_opacities = torch.empty_like(opacities)
    new_scales = torch.empty_like(scales)
    eps = torch.finfo(opacities.dtype).eps
    for idx in range(opacities.shape[0]):
        n_idx = int(ratios[idx].item())
        new_opacity = 1.0 - (1.0 - float(opacities[idx])) ** (1.0 / n_idx)
        new_opacity = min(max(new_opacity, min_opacity), 1.0 - eps)
        new_opacities[idx] = new_opacity

        denom_sum = 0.0
        for i in range(1, n_idx + 1):
            for k in range(i):
                sign = 1.0 if k % 2 == 0 else -1.0
                denom_sum += (
                    float(binoms[i - 1, k])
                    * sign
                    * (new_opacity ** (k + 1))
                    / math.sqrt(k + 1)
                )
        new_scales[idx] = scales[idx] * (float(opacities[idx]) / denom_sum)
    return new_opacities.to(opacities.device), new_scales.to(scales.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
@pytest.mark.parametrize("n_max", [5, 51])
def test_compute_relocation_smoke(n_max: int):
    torch.manual_seed(0)
    N = 128
    binoms = _binomial_table(n_max, device)
    opacities = torch.rand(N, device=device) * 0.9 + 0.05
    scales = torch.rand(N, 3, device=device) * 0.9 + 0.1
    # ratios in [1, n_max] mirrors MCMCStrategy's clamped range; the kernel
    # indexes binoms by ratio so the table must be (n_max, n_max) or larger.
    ratios = torch.randint(1, n_max + 1, (N,), device=device).float()

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, binoms, min_opacity=0.0
    )

    assert new_opacities.shape == (N,)
    assert new_scales.shape == (N, 3)
    assert (new_opacities > 0).all() and (new_opacities <= 1).all()
    assert torch.isfinite(new_scales).all()
    assert (new_scales >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="relocation is a CUDA op")
def test_compute_relocation_min_opacity_clamps_before_scale():
    n_max = 8
    binoms = _binomial_table(n_max, device)
    opacities = torch.tensor([1e-6, 2e-3, 0.2], device=device, dtype=torch.float32)
    scales = torch.tensor(
        [[1.0, 0.5, 0.25], [0.25, 0.5, 1.0], [1.5, 0.75, 0.5]],
        device=device,
        dtype=torch.float32,
    )
    ratios = torch.tensor([8.0, 4.0, 2.0], device=device, dtype=torch.float32)
    min_opacity = 0.005

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, binoms, min_opacity=min_opacity
    )
    ref_opacities, ref_scales = _reference_relocation(
        opacities, scales, ratios, binoms, min_opacity
    )

    torch.testing.assert_close(new_opacities.cpu(), ref_opacities, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_scales.cpu(), ref_scales, rtol=1e-5, atol=1e-6)
    assert torch.all(new_opacities[:2] >= min_opacity)
    assert torch.all(new_opacities[:2] <= min_opacity + 1e-8)
