# SPDX-FileCopyrightText: Copyright 2024-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import math
from typing import Dict

import torch
from torch import Tensor


def sort_splats(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Arrange splats on a similarity-preserving 2D grid with FLAS.

    Fast Linear Assignment Sorting is described in `Improved Evaluation and
    Generation of Grid Layouts using Distance Preservation Quality and Linear
    Assignment Sorting <https://doi.org/10.1111/cgf.14718>`_.

    .. warning::
        The PNG extra must be installed to use PNG compression.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = math.isqrt(n_gs)
    if n_sidelen**2 != n_gs:
        raise ValueError("The number of splats must be a perfect square")

    import vc_flas

    sort_keys = ["means", "quats", "scales", "opacities"]
    if "sh0" in splats:
        sort_keys.append("sh0")

    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]

    # FLAS runs on the CPU. Only the sort keys cross into NumPy; the resulting
    # permutation is applied to every original field on its existing device.
    grid_features = (
        params_to_sort.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(n_sidelen, n_sidelen, -1)
        .contiguous()
        .numpy()
    )
    # Draw the native solver's seed from Torch so torch.manual_seed controls
    # the complete operation, including the initial shuffle above.
    flas_seed = torch.randint(
        torch.iinfo(torch.int32).max,
        (),
        device=params_to_sort.device,
    ).item()

    if verbose:
        print(f"Sorting {n_gs:,} splats with FLAS...")

    arrangement = vc_flas.flas(
        vc_flas.Grid.from_grid_features(grid_features),
        wrap=False,
        # These settings were selected by comparing compression size and
        # runtime against the previous PLAS implementation on real splats.
        radius_decay=0.90,
        max_swap_positions=16,
        seed=flas_seed,
    )
    sorted_indices = torch.from_numpy(arrangement.sorting.reshape(-1).copy()).to(
        device=params_to_sort.device, dtype=torch.long
    )
    sorted_indices = shuffled_indices[sorted_indices]

    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats
