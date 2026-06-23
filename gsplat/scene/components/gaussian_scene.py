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

from __future__ import annotations

import torch

from .base import Scene


class GaussianScene(Scene):
    """Wrapper around Gaussian splat parameters and extra signal data."""

    def __init__(self, id: str) -> None:
        super().__init__(id)
        self.splats = torch.nn.ParameterDict()
        self.signal: dict[str, torch.Tensor] = {}
        self.component_names: list[str] = []
        self.component_index = torch.zeros(0, dtype=torch.long)

    def put(self, name: str, component: torch.nn.ParameterDict) -> None:
        """Add a named component to the scene.

        Init-only: calling ``put()`` after optimizers have been created
        will orphan the old Parameter objects.  Pads existing signal
        rows for the new component to keep everything aligned.
        """
        if not name:
            raise ValueError("component name must not be empty")
        if name in self.component_names:
            raise ValueError(f"Component {name!r} already exists in scene")
        if not component or "means" not in component:
            raise ValueError("component splats must not be empty")

        if len(self.splats) == 0:
            self.splats = component
            self.component_names = [name]
            self.component_index = torch.zeros(
                component["means"].shape[0],
                device=self.splats["means"].device,
                dtype=torch.long,
            )
        else:
            self.splats = torch.nn.ParameterDict(
                {
                    key: torch.nn.Parameter(
                        torch.cat(
                            [self.splats[key].detach(), component[key].detach()],
                            dim=0,
                        ),
                        requires_grad=self.splats[key].requires_grad,
                    )
                    for key in self.splats
                }
            )
            self.component_names.append(name)
            self.component_index = torch.cat(
                [
                    self.component_index,
                    torch.full(
                        (component["means"].shape[0],),
                        len(self.component_names) - 1,
                        device=self.component_index.device,
                        dtype=torch.long,
                    ),
                ],
                dim=0,
            )
            for key, value in self.signal.items():
                pad = torch.zeros(
                    (component["means"].shape[0], *value.shape[1:]),
                    dtype=value.dtype,
                    device=value.device,
                )
                self.signal[key] = torch.cat([value, pad], dim=0)

        self.validate()

    @classmethod
    def from_splats(
        cls,
        splats: torch.nn.ParameterDict,
        id: str,
        signal: dict[str, torch.Tensor] | None = None,
    ) -> "GaussianScene":
        if len(splats) == 0 or "means" not in splats:
            raise ValueError(
                "from_splats requires a non-empty ParameterDict containing 'means'"
            )
        scene = cls(id)
        if signal is not None:
            device = splats["means"].device
            scene.signal = {k: v.to(device) for k, v in signal.items()}
        scene.put(id, splats)
        return scene

    def validate(self) -> None:
        # Init-only: the bounds checks call ``.item()`` on ``component_index``,
        # which forces a CUDA sync. Don't move into a per-step hot path.
        required_keys = ("means", "scales", "quats", "opacities")
        missing = [key for key in required_keys if key not in self.splats]
        if len(self.splats) > 0 and missing:
            raise ValueError(f"missing required splat keys: {missing}")

        n = self.num_gaussians()
        if not all(v.shape[0] == n for v in self.splats.values()):
            raise ValueError(
                f"every splat tensor must have leading dim == num_gaussians: {n}"
            )
        if not all(v.shape[0] == n for v in self.signal.values()):
            raise ValueError(
                f"every signal tensor must have leading dim == num_gaussians: {n}"
            )
        if self.component_index.shape != (n,):
            raise ValueError(
                f"component_index shape {tuple(self.component_index.shape)} != ({n},)"
            )
        if len(self.splats) > 0:
            if len(self.component_names) == 0:
                raise ValueError("component_names must not be empty")
            if n > 0:
                if int(self.component_index.min().item()) < 0:
                    raise ValueError("component_index must be non-negative")
                if int(self.component_index.max().item()) >= len(self.component_names):
                    raise ValueError("component_index refers to an unknown component")

    def num_gaussians(self) -> int:
        if "means" not in self.splats:
            return 0
        return self.splats["means"].shape[0]

    def _component_id(self, component: str | int) -> int:
        if isinstance(component, int):
            if component < 0 or component >= len(self.component_names):
                raise KeyError(f"Unknown component index: {component}")
            return component
        try:
            return self.component_names.index(component)
        except ValueError as exc:
            raise KeyError(f"Unknown component name: {component}") from exc

    def get(self, component: str | int) -> dict[str, object]:
        component_id = self._component_id(component)
        mask = self.component_index == component_id
        return {
            "name": self.component_names[component_id],
            "index": component_id,
            "mask": mask,
            "splats": {key: value[mask] for key, value in self.splats.items()},
            "signal": {key: value[mask] for key, value in self.signal.items()},
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "splats": self.splats.state_dict(),
            "splats_requires_grad": {
                key: bool(value.requires_grad) for key, value in self.splats.items()
            },
            "signal": {
                key: value.detach().clone() for key, value in self.signal.items()
            },
            "component_names": list(self.component_names),
            "component_index": self.component_index.detach().clone(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "GaussianScene":
        if "id" not in state:
            raise KeyError("state_dict missing required 'id' entry")

        scene = cls(state["id"])
        requires_grad_map: dict[str, bool] = state.get("splats_requires_grad", {})
        scene.splats = torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(
                    value.clone(), requires_grad=requires_grad_map.get(key, True)
                )
                for key, value in state["splats"].items()
            }
        )
        scene.signal = dict(state.get("signal", {}))  # already cloned in state_dict()
        scene.component_names = list(state.get("component_names", []))

        has_splats = "means" in scene.splats
        component_index = state.get("component_index")
        if component_index is None:
            n = scene.splats["means"].shape[0] if has_splats else 0
            device = scene.splats["means"].device if has_splats else torch.device("cpu")
            component_index = torch.zeros(n, device=device, dtype=torch.long)
        elif has_splats:
            # Align loaded `component_index` to the splats' device. Without
            # this, a checkpoint saved on CUDA + loaded with map_location="cpu"
            # (or the reverse) leaves these tensors on different devices, and
            # the next on_remove / on_duplicate / on_relocate crashes the
            # moment it indexes one with the other.
            component_index = component_index.to(scene.splats["means"].device)
        scene.component_index = component_index
        if has_splats and not scene.component_names:
            scene.component_names = [state["id"]]

        scene.validate()
        return scene

    def _cat_signal(self, indices: torch.Tensor) -> None:
        # ``indices`` must be a LongTensor; bool masks would change semantics
        # (gather True rows vs gather positions).
        for key, value in self.signal.items():
            self.signal[key] = torch.cat([value, value[indices]], dim=0)

    def on_duplicate(self, sel: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [self.component_index, self.component_index[sel]], dim=0
        )
        self._cat_signal(sel)

    def on_split(self, sel: torch.Tensor, rest: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [
                self.component_index[rest],
                self.component_index[sel],
                self.component_index[sel],
            ],
            dim=0,
        )
        for key, value in self.signal.items():
            self.signal[key] = torch.cat(
                [value[rest], value[sel], value[sel]],
                dim=0,
            )

    def on_remove(self, remove_mask: torch.Tensor) -> None:
        keep = ~remove_mask
        self.component_index = self.component_index[keep]
        for key, value in self.signal.items():
            self.signal[key] = value[keep]

    def on_relocate(
        self, dead_indices: torch.Tensor, sampled_indices: torch.Tensor
    ) -> None:
        self.component_index[dead_indices] = self.component_index[sampled_indices]
        for key, value in self.signal.items():
            value[dead_indices] = value[sampled_indices]

    def on_sample_add(self, sampled_indices: torch.Tensor) -> None:
        self.component_index = torch.cat(
            [self.component_index, self.component_index[sampled_indices]], dim=0
        )
        self._cat_signal(sampled_indices)

    def on_permute(self, order: torch.Tensor) -> None:
        self.component_index = self.component_index[order]
        for key, value in self.signal.items():
            self.signal[key] = value[order]
