# SPDX-FileCopyrightText: Copyright 2024 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""Profiling helpers and a replay harness for gsplat rasterization.

Tutorial
========

This module supports a two-step profiling workflow:

1. Capture a real rasterization call from an existing application.
2. Replay that exact input under Nsight Systems or Nsight Compute.

Capture inputs
--------------

`gsplat.rendering.rasterization()` is decorated with
`@capture_inputs(envvar="GSPLAT_INPUT_CAPTURE_RASTERIZATION")`.
Set that environment variable before running your application:

```bash
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs:1
python your_app.py
```

The value is a comma-delimited list of specs, each of the form
`<output_path>:<range_spec>`:

- `<path>:<stop>` captures calls `0..stop-1`
- `<path>:<start>:<stop>` captures calls `start..stop-1`
- `<path>:<start>:<stop>:<step>` captures strided calls

Multiple specs let you capture different call indices to different output
paths in a single run — useful when one process makes several kinds of
rasterization calls (e.g. a camera pass followed by a lidar pass) and you
want each written to its own file. The call-index ranges across specs must
not overlap.

Examples:

```bash
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs.pt:1
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/gsplat_inputs:5:8
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=raster:10:20:2
export GSPLAT_INPUT_CAPTURE_DIR=/tmp/gsplat-captures
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=raster:3
# Capture call 0 as camera and call 1 as lidar:
export GSPLAT_INPUT_CAPTURE_RASTERIZATION=/tmp/camera:1,/tmp/lidar:1:2
```

Captured files are written with a numeric suffix such as
`/tmp/gsplat_inputs_0.pt`. Under DDP (when the `RANK` environment variable is
set by a launcher), an `_r<rank>` infix is added so ranks do not race on the
same path: e.g. `/tmp/gsplat_inputs_r0_0.pt`. Outside DDP, a `_p<pid>` infix
is used to keep concurrent processes from colliding. Relative paths are
resolved under `GSPLAT_INPUT_CAPTURE_DIR` when that variable is set. Once
every requested capture has been written, the process exits intentionally
with `SystemExit`.

You can use the same decorator on other functions to build a more focused
benchmarking path, for example by capturing `isect_tiles()` inputs separately
instead of replaying the full `rasterization()` pipeline. If you do that, you
also need to adjust the replay loop in `main()` so it calls the captured
function and applies the right forward/backward logic for that workload.

Replay without a profiler
-------------------------

You can validate the captured input directly:

```bash
python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt
```

The harness loads the saved arguments, moves tensors to CUDA, restores
`requires_grad=True` on differentiable Gaussian inputs by default, then runs
warmup and profiled forward/backward iterations of `rasterization()`. Pass
`--nograd` to replay forward-only iterations under `torch.inference_mode()`;
`--grad` is the default and keeps the training-style forward/backward replay.

Replay under Nsight Systems
---------------------------

To collect rich GPU-side trace data without spending effort on secondary CPU
metrics, trace CUDA/NVTX plus the CUDA math libraries you care about, enable
CUDA memory tracking, and export a report. Use `cudaProfilerStart/Stop` to keep
the capture focused on profiled iterations:

```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --trace=cuda,nvtx,osrt,cublas,cudnn,cusolver,cusparse \
  --cuda-memory-usage=true \
  --cuda-um-cpu-page-faults=true \
  --cuda-um-gpu-page-faults=true \
  --stats=true \
  --export=sqlite \
  --output /tmp/gsplat_nsys \
  python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt --warmup 5 --iterations 20
```

The replay harness emits NVTX ranges for `iteration`, `forward`, and
`backward`, which makes it easier to inspect the trace in Nsight Systems.
Add `--sync` to the `gsplat.profile` command to synchronize the device at the
end of each iteration: each iteration then excludes the previous iteration's
in-flight kernels, and the `iteration` NVTX range spans CUDA device completion
instead of only CPU launch time.

Replay under Nsight Compute
---------------------------

To collect as many kernel metrics as practical while staying focused on the GPU
work, use the `full` set, keep NVTX filtering so the replay stays focused, and
profile a single iteration without warmup:

```bash
ncu \
  --set full \
  --nvtx \
  --nvtx-include forward \
  --target-processes all \
  --replay-mode kernel \
  --force-overwrite \
  --export /tmp/gsplat_ncu \
  python -m gsplat.profile --input /tmp/gsplat_inputs_0.pt --warmup 0 --iterations 1
```

This is still heavy-weight because `--set full` can require many replay passes.
For faster iteration, replace `--set full` with a narrower metric set and add
kernel filters as needed.

"""

import ast
import inspect
import json
import os
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

import torch
from torch.utils._pytree import tree_map

# NOTE: do NOT import anything from the ``gsplat`` package at module scope —
# importing ``gsplat`` runs its ``__init__.py``, which pulls in the CUDA
# wrapper modules and triggers JIT compilation of the C++ extension. Users
# of ``capture_inputs`` (notably ``gsplat.rendering``) should be able to
# decorate their functions without paying that cost up-front. All gsplat
# imports must be performed lazily inside the functions that need them.


def _detach_for_capture(x: Any) -> Any:
    """Break autograd-graph connectivity before ``torch.save``.

    Non-leaf tensors would otherwise drag their autograd graph into the
    pickle, bloating the file and coupling it to live model code.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x


profiler = {}

# Parameter names that should have requires_grad=True for backward pass
_GRAD_PARAMS = {"means", "quats", "scales", "opacities", "colors", "extra_signals"}


@dataclass
class ProfileWorkload:
    """Single callable selected for one `gsplat.profile` invocation."""

    name: str
    operator_name: str
    operator: Callable[..., Any]
    replay_inputs: dict[str, Any]
    losses: list[str]
    loss_contribution: Callable[[tuple[Any, ...]], torch.Tensor]
    expected_kernel_families: list[str]
    notes: list[str]


def _parse_override_value(raw: str) -> Any:
    """Parse a command-line override value into a Python scalar/container."""
    value = raw.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered == "none":
        return None
    if lowered == "nan":
        return float("nan")
    if lowered == "inf":
        return float("inf")
    if lowered == "-inf":
        return float("-inf")

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _parse_input_override(raw: str) -> tuple[str, Any]:
    """Parse a single NAME=VALUE replay-input override."""
    if "=" not in raw:
        raise ValueError(f"expected NAME=VALUE, got {raw!r}")
    name, raw_value = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"override name cannot be empty in {raw!r}")
    if "." in name or "[" in name or "]" in name:
        raise ValueError(
            f"override name must be a top-level operator argument, got {name!r}"
        )
    return name, _parse_override_value(raw_value)


def _select_replay_inputs(
    inputs: dict[str, Any],
    operator_params: dict[str, inspect.Parameter],
    workload_name: str,
) -> dict[str, Any]:
    """Project a scene/input dictionary onto one operator's keyword contract.

    Replay files are not required to come from a specific decorated function.
    They may be captures, hand-built scene dictionaries, or converted data from
    another benchmark. This helper keeps only keys accepted by the selected
    operator and checks that all required operator arguments are present.
    """
    replay_inputs: dict[str, Any] = {}
    missing_required: list[str] = []
    for name, param in operator_params.items():
        if name in inputs:
            replay_inputs[name] = inputs[name]
        elif param.default is inspect.Parameter.empty:
            missing_required.append(name)

    if missing_required:
        missing = ", ".join(missing_required)
        raise ValueError(
            f"{workload_name} input is missing required parameters: {missing}"
        )

    return replay_inputs


def _apply_3dgs_replay_preset(inputs: dict[str, Any]) -> None:
    """Force scene inputs onto the plain classic-3DGS path.

    Scene files meant for 3DGUT/eval3d often carry rays, UT flags,
    rolling-shutter/distortion parameters, normals, backgrounds, or extra
    signals. Those inputs route through the world-space 3DGUT kernels. This
    preset rewrites only top-level `rasterization()` arguments so the scene
    replays as a plain RGB classic 3DGS pass.
    """
    if inputs.get("camera_model") == "lidar" or inputs.get("lidar_coeffs") is not None:
        raise ValueError("--3dgs cannot convert lidar inputs to classic 3DGS")

    inputs.update(
        {
            "with_eval3d": False,
            "with_ut": False,
            "return_normals": False,
            "render_mode": "RGB",
            "rasterize_mode": "classic",
            "camera_model": "pinhole",
            "global_z_order": True,
            # RollingShutterType.GLOBAL is IntEnum value 4. Use the scalar here
            # so this helper stays independent of the CUDA wrapper import path.
            "rolling_shutter": 4,
            "viewmats_rs": None,
            "rays": None,
            "radial_coeffs": None,
            "tangential_coeffs": None,
            "thin_prism_coeffs": None,
            "ftheta_coeffs": None,
            "external_distortion_coeffs": None,
            # Keep the preset to plain RGB. Input extra signals can request
            # channel counts that are not part of the current build matrix.
            "extra_signals": None,
            "extra_signals_sh_degree": None,
            "backgrounds": None,
        }
    )


def _apply_3dgut_replay_preset(inputs: dict[str, Any]) -> None:
    """Force scene inputs onto the 3DGUT/eval3d path.

    3DGUT rendering in `rasterization()` is selected by the UT projection path
    plus the world-space eval3d rasterizer. Unlike `--3dgs`, this preset keeps
    camera model and signal inputs intact so camera, lidar, hit-distance, and
    normal-producing scenes remain representative; it only sets the routing
    arguments needed for the 3DGUT kernels.
    """
    inputs.update(
        {
            "with_eval3d": True,
            "with_ut": True,
            "packed": False,
            "sparse_grad": False,
            "rasterize_mode": "classic",
        }
    )


_2DGS_RENDER_MODE_MAP = {
    "d": "D",
    "Ed": "ED",
    "RGB-d": "RGB+D",
    "RGB-Ed": "RGB+ED",
}

_2DGS_DISTORTION_RENDER_MODE = "RGB+ED"
_2DGS_DISTORTION_RENDER_MODES = {"D", "ED", "RGB+D", "RGB+ED"}
_2DGS_COLOR_DEPTH_RENDER_MODES = {"RGB+D", "RGB+ED"}

_PROFILE_LOSS_ALIASES = {
    "gaussian": "gaussian_fused",
    "fused_gaussian": "gaussian_fused",
    "gaussian_fused": "gaussian_fused",
    "2dgs_distortion": "2dgs_distortion",
    "distortion": "2dgs_distortion",
}


def _adapt_2dgs_scene_inputs(
    inputs: dict[str, Any],
    rasterization_2dgs_params: dict[str, inspect.Parameter],
) -> tuple[dict[str, Any], list[str]]:
    """Adapt a scene/input dictionary to `rasterization_2dgs()`.

    This is a benchmark-oriented conversion: the Gaussian/camera tensors define
    the scene, while parameters that only exist on 3DGS/3DGUT paths are dropped.
    The resulting workload is meant to make the loaded scene meaningful for the
    2DGS operator, not to imply numerical equivalence with another operator.
    """
    if inputs.get("camera_model") == "lidar" or inputs.get("lidar_coeffs") is not None:
        raise ValueError("--2dgs cannot convert lidar inputs to 2DGS")

    adapted = _select_replay_inputs(inputs, rasterization_2dgs_params, "--2dgs")

    if adapted.get("colors") is None:
        raise ValueError("--2dgs requires colors; depth-only inputs are unsupported")

    notes: list[str] = []

    # The 2DGS entrypoint expects an integer tile size. Some scene dictionaries
    # use None to request a frontend-selected default, so resolve that to the
    # documented 2DGS default before entering warmup/profile iterations.
    if adapted.get("tile_size") is None:
        adapted["tile_size"] = 16
        notes.append("resolved tile_size=None to tile_size=16")

    # 3DGUT hit-distance render modes depend on eval3d rays. 2DGS renders
    # projection depth, so preserve the color/depth shape and expected-vs-
    # accumulated convention by mapping hit-distance modes to depth modes.
    render_mode = adapted.get("render_mode", "RGB")
    mapped_render_mode = _2DGS_RENDER_MODE_MAP.get(render_mode, render_mode)
    if mapped_render_mode != render_mode:
        adapted["render_mode"] = mapped_render_mode
        notes.append(f"mapped render_mode={render_mode!r} to {mapped_render_mode!r}")

    dropped = [
        name
        for name in (
            "rays",
            "with_ut",
            "with_eval3d",
            "rasterize_mode",
            "channel_chunk",
            "distributed",
            "camera_model",
            "segmented",
            "covars",
            "return_normals",
            "global_z_order",
            "radial_coeffs",
            "tangential_coeffs",
            "thin_prism_coeffs",
            "ftheta_coeffs",
            "lidar_coeffs",
            "external_distortion_coeffs",
            "rolling_shutter",
            "viewmats_rs",
            "ut_params",
            "extra_signals",
            "extra_signals_sh_degree",
        )
        if inputs.get(name) is not None
    ]
    if dropped:
        notes.append("dropped 3DGS/3DGUT-only inputs: " + ", ".join(dropped))

    return adapted, notes


def _default_profile_losses(replay_name: str) -> list[str]:
    """Return CUDA-backed gsplat losses that make sense for one workload."""
    losses = ["gaussian_fused"]
    if replay_name == "rasterization_2dgs":
        losses.append("2dgs_distortion")
    return losses


def _parse_profile_losses(raw_losses: str, replay_name: str) -> list[str]:
    """Parse the `--losses` option into canonical loss names.

    Bare `--losses` is represented by argparse as ``"all"``. That expands to
    the CUDA-backed loss paths that are meaningful for the selected workload.
    """
    normalized = raw_losses.strip().lower()
    if normalized in {"", "none", "false", "0"}:
        return []
    if normalized == "all":
        return _default_profile_losses(replay_name)

    losses: list[str] = []
    for raw_name in normalized.split(","):
        name = raw_name.strip()
        if not name:
            raise ValueError(f"empty loss name in --losses={raw_losses!r}")
        if name not in _PROFILE_LOSS_ALIASES:
            valid = ", ".join(sorted([*set(_PROFILE_LOSS_ALIASES), "all", "none"]))
            raise ValueError(f"unknown profile loss {name!r}; expected one of {valid}")
        canonical = _PROFILE_LOSS_ALIASES[name]
        if canonical == "2dgs_distortion" and replay_name != "rasterization_2dgs":
            raise ValueError("2dgs_distortion loss is only valid with --2dgs")
        if canonical not in losses:
            losses.append(canonical)
    return losses


def _prepare_gaussian_fused_loss_inputs(
    replay_inputs: dict[str, Any],
    scene_inputs: dict[str, Any],
) -> dict[str, Any]:
    """Prepare stable tensor inputs for the fused Gaussian loss kernels.

    The fused CUDA op expects flattened ``[N, *]`` tensors and a non-gradient
    cuboid size. Profiling does not require the scene file to carry cuboids, so
    a deterministic synthetic cuboid is created when none is provided.
    """
    required = ("means", "scales", "opacities")
    missing = [name for name in required if name not in replay_inputs]
    if missing:
        raise ValueError(
            "gaussian_fused loss requires replay inputs: " + ", ".join(missing)
        )

    positions = replay_inputs["means"].reshape(-1, 3)
    scales = replay_inputs["scales"].reshape(-1, 3)
    # rasterization() receives post-activation opacities, which are exactly the
    # density values expected by gaussian_density_reg/FusedGaussianLosses.
    densities = replay_inputs["opacities"].reshape(-1)
    if not (positions.is_cuda and scales.is_cuda and densities.is_cuda):
        raise ValueError("gaussian_fused loss requires CUDA replay tensors")
    if scales.dtype not in (torch.float32, torch.float64):
        raise ValueError(
            "gaussian_fused loss only supports float32/float64 scales, "
            f"got {scales.dtype}"
        )

    raw_cuboid_dims = scene_inputs.get("cuboid_dims")
    if raw_cuboid_dims is None:
        cuboid_dims = torch.full_like(positions, 2.0, requires_grad=False)
    elif isinstance(raw_cuboid_dims, torch.Tensor):
        cuboid_dims = raw_cuboid_dims.reshape(-1, 3).to(
            device=positions.device, dtype=positions.dtype
        )
    else:
        raise ValueError("gaussian_fused loss cuboid_dims input must be a Tensor")
    cuboid_dims = cuboid_dims.detach()
    if cuboid_dims.shape != positions.shape:
        raise ValueError(
            "gaussian_fused loss cuboid_dims must match flattened means shape, "
            f"got {tuple(cuboid_dims.shape)} vs {tuple(positions.shape)}"
        )

    raw_visibility = scene_inputs.get("visibility", scene_inputs.get("visibilities"))
    visibility = None
    if raw_visibility is not None:
        if not isinstance(raw_visibility, torch.Tensor):
            raise ValueError("gaussian_fused loss visibility input must be a Tensor")
        visibility = raw_visibility.reshape(-1).to(
            device=positions.device, dtype=positions.dtype
        )
        if visibility.numel() != positions.shape[0]:
            raise ValueError(
                "gaussian_fused loss visibility must have one value per Gaussian, "
                f"got {visibility.numel()} vs {positions.shape[0]}"
            )
        visibility = visibility.detach()

    z_scale_threshold = float(scene_inputs.get("z_scale_threshold", 0.0))
    return {
        "scales": scales,
        "densities": densities,
        "positions": positions,
        "cuboid_dims": cuboid_dims,
        "visibility": visibility,
        "z_scale_threshold": z_scale_threshold,
    }


def _apply_profile_loss_presets(
    losses: list[str],
    replay_name: str,
    replay_inputs: dict[str, Any],
) -> list[str]:
    """Apply loss-driven input mutations before user overrides.

    This is intentionally limited to semantic mode switches, such as enabling
    the 2DGS distortion path. Tensor captures for loss kernels are prepared
    after `--set` overrides so the profiled callable uses final inputs.
    """
    notes: list[str] = []

    if "2dgs_distortion" in losses:
        if replay_name != "rasterization_2dgs":
            raise ValueError("2dgs_distortion loss is only valid with --2dgs")
        if replay_inputs.get("render_mode", "RGB") not in _2DGS_DISTORTION_RENDER_MODES:
            replay_inputs["render_mode"] = _2DGS_DISTORTION_RENDER_MODE
            notes.append(
                f"set render_mode={_2DGS_DISTORTION_RENDER_MODE!r} for 2DGS distortion loss"
            )
        replay_inputs["distloss"] = True
        notes.append("enabled 2dgs_distortion loss path")

    return notes


def _build_profile_loss_contribution(
    losses: list[str],
    replay_name: str,
    replay_inputs: dict[str, Any],
    scene_inputs: dict[str, Any],
) -> tuple[Callable[[tuple[Any, ...]], torch.Tensor], list[str]]:
    """Build the profiled CUDA loss contribution callable.

    The returned callable is invoked inside the workload closure, so loss CUDA
    kernels appear in the profiled forward/backward ranges. All validation and
    stable tensor selection happens here, once, after user overrides.
    """
    notes: list[str] = []
    gaussian_state: dict[str, Any] | None = None
    fused_gaussian_losses: torch.nn.Module | None = None

    if "gaussian_fused" in losses:
        from gsplat.cuda._wrapper import has_losses
        from gsplat.losses_fused import FusedGaussianLosses

        if not has_losses():
            raise ValueError(
                "gaussian_fused loss requested, but gsplat was built without "
                "CUDA loss support"
            )
        gaussian_state = _prepare_gaussian_fused_loss_inputs(
            replay_inputs, scene_inputs
        )
        fused_gaussian_losses = FusedGaussianLosses(
            z_scale_threshold=gaussian_state["z_scale_threshold"]
        )
        notes.append("enabled gaussian_fused loss kernels")

    def _loss_contribution(outputs: tuple[Any, ...]) -> torch.Tensor:
        total: torch.Tensor | None = None

        if gaussian_state is not None and fused_gaussian_losses is not None:
            z_scales = gaussian_state["scales"][:, 2]
            loss_scale, loss_density, loss_z_scale, loss_oob = fused_gaussian_losses(
                gaussian_state["scales"],
                gaussian_state["densities"],
                z_scales,
                gaussian_state["positions"],
                gaussian_state["cuboid_dims"],
                gaussian_state["visibility"],
            )
            total = (
                loss_scale.mean()
                + loss_density.mean()
                + loss_z_scale.mean()
                + loss_oob.mean()
            )

        if "2dgs_distortion" in losses:
            render_distort = outputs[4]
            contribution = render_distort.mean()
            total = contribution if total is None else total + contribution

        if total is None:
            render_colors = outputs[0]
            return render_colors.new_zeros(())
        return total

    return _loss_contribution, notes


def _validate_profile_loss_overrides(
    losses: list[str],
    replay_name: str,
    replay_inputs: dict[str, Any],
) -> None:
    """Validate user overrides that may affect requested loss paths."""
    if "2dgs_distortion" not in losses:
        return
    if replay_name != "rasterization_2dgs":
        raise ValueError("2dgs_distortion loss is only valid with --2dgs")
    if not replay_inputs.get("distloss", False):
        raise ValueError("2dgs_distortion loss requires distloss=True")
    render_mode = replay_inputs.get("render_mode", "RGB")
    if render_mode not in _2DGS_DISTORTION_RENDER_MODES:
        valid = ", ".join(sorted(_2DGS_DISTORTION_RENDER_MODES))
        raise ValueError(
            f"2dgs_distortion loss requires a depth render_mode ({valid}), "
            f"got {render_mode!r}"
        )


def _normalize_2dgs_replay_inputs(replay_inputs: dict[str, Any]) -> list[str]:
    """Apply final 2DGS tensor-shape conversions after mode/override setup."""
    notes: list[str] = []
    render_mode = replay_inputs.get("render_mode", "RGB")
    sh_degree = replay_inputs.get("sh_degree")

    # The 2DGS RGB+depth path concatenates `colors` with projected depths.
    # With unpacked projection, depths are shaped [..., C, N, 1], so a plain
    # [..., N, D] color tensor must be expanded to [..., C, N, D] once before
    # the measured loop. Keep the converted tensor as a leaf so repeated
    # backward iterations can clear `.grad` through `_clear_replay_grads()`.
    if (
        render_mode in _2DGS_COLOR_DEPTH_RENDER_MODES
        and sh_degree is None
        and not replay_inputs.get("packed", False)
    ):
        means = replay_inputs["means"]
        viewmats = replay_inputs["viewmats"]
        colors = replay_inputs["colors"]
        batch_dims = means.shape[:-2]
        n_gaussians = means.shape[-2]
        n_cameras = viewmats.shape[-3]
        expected_scene_shape = batch_dims + (n_gaussians,)
        if colors.shape[:-1] == expected_scene_shape:
            expanded = colors[..., None, :, :].expand(
                batch_dims + (n_cameras, n_gaussians, colors.shape[-1])
            )
            replay_inputs["colors"] = (
                expanded.detach().clone().requires_grad_(colors.requires_grad)
            )
            notes.append(
                "expanded colors from scene-shaped [..., N, D] to "
                "camera-shaped [..., C, N, D] for 2DGS RGB+depth replay"
            )

    return notes


def _expected_kernel_families(
    replay_name: str,
    replay_inputs: dict[str, Any],
    losses: list[str],
) -> list[str]:
    """Describe kernel families expected from the selected single workload.

    This is intentionally a family-level list rather than a promise about exact
    template names. Exact names depend on channel count, dtype, packed mode,
    camera model, and JIT/AOT dispatch.
    """
    kernels: list[str] = []
    if replay_name == "rasterization_2dgs":
        kernels.extend(
            [
                "Projection2DGSFwd/Bwd",
                "IntersectTile",
                "IntersectOffset",
                "IntersectSort",
                "RasterizeToPixels2DGSFwd/Bwd",
            ]
        )
        if replay_inputs.get("sh_degree") is not None:
            kernels.append("SphericalHarmonicsFwd/Bwd")
    else:
        with_eval3d = bool(replay_inputs.get("with_eval3d", False))
        with_ut = bool(replay_inputs.get("with_ut", False))
        is_lidar = (
            replay_inputs.get("camera_model") == "lidar"
            or replay_inputs.get("lidar_coeffs") is not None
        )
        if with_eval3d:
            if with_ut:
                kernels.append("ProjectionUT3DGSFused")
            else:
                kernels.append("ProjectionEWA3DGSFwd/Bwd")
            kernels.append("IntersectTileLidar" if is_lidar else "IntersectTile")
            kernels.extend(
                [
                    "IntersectOffset",
                    "IntersectSort",
                    "RasterizeToPixelsFromWorld3DGSFwd/Bwd",
                    "ComputeChunksPerTile",
                ]
            )
        else:
            packed = bool(replay_inputs.get("packed", True))
            if packed:
                kernels.append("ProjectionEWA3DGSPackedFwd/Bwd")
            else:
                kernels.append("ProjectionEWA3DGSFusedFwd/Bwd")
            kernels.extend(
                [
                    "IntersectTile",
                    "IntersectOffset",
                    "IntersectSort",
                    "RasterizeToPixels3DGSFwd/Bwd",
                ]
            )
        if replay_inputs.get("sh_degree") is not None:
            kernels.append("SphericalHarmonicsFwd/Bwd")
        if replay_inputs.get("extra_signals_sh_degree") is not None:
            kernels.append("SphericalHarmonicsFwd/Bwd(extra_signals)")

    if "gaussian_fused" in losses:
        kernels.extend(["GaussianLossesFwd", "GaussianLossesBwd"])
    if "2dgs_distortion" in losses:
        kernels.append("RasterizeToPixels2DGSFwd/Bwd(distortion)")

    unique: list[str] = []
    for kernel in kernels:
        if kernel not in unique:
            unique.append(kernel)
    return unique


def _format_replay_value(value: Any) -> str:
    """Return a compact, human-readable value summary for `--describe`."""
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, requires_grad={value.requires_grad})"
        )
    return f"{type(value).__name__}({value!r})"


def _describe_workload(workload: ProfileWorkload) -> None:
    """Print the selected single workload and exit without profiling."""
    print("[gsplat.profile] Workload")
    print(f"  name: {workload.name}")
    print(f"  operator: {workload.operator_name}")
    print("  losses: " + (", ".join(workload.losses) if workload.losses else "none"))
    if workload.notes:
        print("  notes:")
        for note in workload.notes:
            print(f"    - {note}")
    print("  expected kernel families:")
    for kernel in workload.expected_kernel_families:
        print(f"    - {kernel}")
    print("  replay inputs:")
    for name in sorted(workload.replay_inputs):
        print(f"    - {name}: {_format_replay_value(workload.replay_inputs[name])}")


def _clear_replay_grads(inputs: dict[str, Any]) -> None:
    """Clear gradients on differentiable replay inputs after one iteration."""
    for k in _GRAD_PARAMS:
        if (
            k in inputs
            and isinstance(inputs[k], torch.Tensor)
            and inputs[k].grad is not None
        ):
            inputs[k].grad = None


# --- Input capture registry ---
# Tracks all active capture_inputs decorators so the program only exits when
# every one of them has finished its range.
_pending_captures: set[int] = set()
_next_capture_id: int = 0


class timeit(object):
    """Profiler that is controled by the TIMEIT environment variable.

    If TIMEIT is set to 1, the profiler will measure the time taken by the decorated function.

    Usage:

    ```python
    @timeit()
    def my_function():
        pass

    # Or

    with timeit(name="stage1"):
        my_function()

    print(profiler)
    ```
    """

    def __init__(self, name: str = "unnamed"):
        self.name = name
        self.start_time: Optional[float] = None
        self.enabled = os.environ.get("TIMEIT", "0") == "1"

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self.start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - self.start_time
            if self.name not in profiler:
                profiler[self.name] = total_time
            else:
                profiler[self.name] += total_time

    def __call__(self, f: Callable) -> Callable:
        @wraps(f)
        def decorated(*args, **kwargs):
            with self:
                self.name = f.__name__
                return f(*args, **kwargs)

        return decorated


def capture_inputs(*, envvar: str) -> Callable[[_F], _F]:
    """Decorator that captures a function's inputs to disk when envvar is set.

    When the environment variable is not set the original function is returned
    unchanged with zero overhead.

    Format: ``<envvar>=<spec>[,<spec>...]`` where each ``<spec>`` is
    ``<output_path>:<range_spec>``.

    <range_spec> follows Python range() / tensor-slice conventions:
      - ``stop``             — capture calls 0, 1, ..., stop-1
      - ``start:stop``       — capture calls start, start+1, ..., stop-1
      - ``start:stop:step``  — capture calls start, start+step, ..., < stop

    Within each spec, integers are scanned from the right; everything to the
    left of them is the output path.  If the path is relative and
    ``GSPLAT_INPUT_CAPTURE_DIR`` is set, that directory is prepended.  If no
    file extension is given, ``.pt`` is used.  Each captured call is saved as
    ``<stem>_<zero-padded index><ext>``, with enough zero-padding so that all
    indices across all specs share the same number of digits.

    Passing multiple comma-separated specs lets a single run route different
    call indices to different output paths (e.g. camera vs lidar calls).
    Ranges across specs must be disjoint.

    All gsplat C++ custom classes support pickle via ``def_pickle`` in
    ``ext.cpp``, so ``torch.save`` handles them natively.

    .. note::

       On PyTorch 2.6+, ``torch.load`` defaults to ``weights_only=True`` and
       refuses to instantiate custom ``torch::class_`` instances. Reload
       captured files with ``torch.load(path, weights_only=False)`` (the
       bundled replay harness in ``gsplat.profile.main`` already does this).

    Example::

        @capture_inputs(envvar="MY_APP_CAPTURE_FOO")
        def foo(...):
            ...
    """

    def decorator(fn: _F) -> _F:
        global _next_capture_id

        env = os.environ.get(envvar)
        if not env:
            return fn

        capture_dir = os.environ.get("GSPLAT_INPUT_CAPTURE_DIR")

        # Parse the comma-delimited list of specs. Each spec is parsed
        # independently into its own (stem, ext, capture_range).
        specs: list[tuple[str, str, range]] = []
        for spec_str in env.split(","):
            spec_str = spec_str.strip()
            if not spec_str:
                continue
            parts = spec_str.split(":")
            range_ints: list[int] = []
            path_end = len(parts)
            for i in range(len(parts) - 1, -1, -1):
                try:
                    range_ints.insert(0, int(parts[i]))
                    path_end = i
                except ValueError:
                    break
            if not range_ints:
                raise ValueError(
                    f"{envvar}: expected <path>:<stop>, <path>:<start>:<stop>, or "
                    f"<path>:<start>:<stop>:<step> in each spec, got {spec_str!r}"
                )
            if any(v < 0 for v in range_ints):
                raise ValueError(
                    f"{envvar}: negative values are not supported, got {spec_str!r}"
                )
            if len(range_ints) >= 2 and range_ints[1] < range_ints[0]:
                raise ValueError(
                    f"{envvar}: stop ({range_ints[1]}) must be >= start ({range_ints[0]}), got {spec_str!r}"
                )
            output_path = ":".join(parts[:path_end])
            capture_range = range(*range_ints)
            if not capture_range:
                raise ValueError(
                    f"{envvar}: empty range (nothing to capture), got {spec_str!r}"
                )

            # Resolve output path: apply capture dir if path is relative, add
            # default extension if none given.
            if capture_dir and not os.path.isabs(output_path):
                output_path = os.path.join(capture_dir, output_path)
            stem, ext = os.path.splitext(output_path)
            if not ext:
                ext = ".pt"

            # Create the output directory up-front so a bad path fails before
            # training starts rather than mid-run when torch.save() would
            # otherwise raise FileNotFoundError / PermissionError deep inside
            # the render loop.
            target_dir = os.path.dirname(stem)
            if target_dir:
                try:
                    os.makedirs(target_dir, exist_ok=True)
                except OSError as e:
                    raise OSError(
                        f"{envvar}: cannot create capture directory {target_dir!r} "
                        f"for spec {spec_str!r}: {e}"
                    ) from e

            specs.append((stem, ext, capture_range))

        if not specs:
            raise ValueError(f"{envvar}: no specs provided, got {env!r}")

        # Ensure call-index ranges don't overlap across specs — otherwise we'd
        # need an arbitrary tie-break when a call matches several specs.
        seen_calls: dict[int, int] = {}
        for spec_idx, (_stem, _ext, capture_range) in enumerate(specs):
            for c in capture_range:
                if c in seen_calls:
                    raise ValueError(
                        f"{envvar}: call index {c} is claimed by multiple specs "
                        f"(specs {seen_calls[c]} and {spec_idx}), got {env!r}"
                    )
                seen_calls[c] = spec_idx

        # Zero-pad index so all filenames sort lexicographically across specs.
        n_digits = len(str(max(seen_calls)))
        total_captures = sum(len(r) for _s, _e, r in specs)

        # Track all active capture decorators so multiple instrumented functions
        # can coexist in one process without exiting early.
        capture_id = _next_capture_id
        _next_capture_id += 1
        _pending_captures.add(capture_id)

        call_count = 0
        captures_done = 0
        sig = inspect.signature(fn)

        @wraps(fn)
        def _wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal call_count, captures_done

            matching_spec: Optional[tuple[str, str, range]] = None
            for spec in specs:
                if call_count in spec[2]:
                    matching_spec = spec
                    break

            if matching_spec is not None:
                stem, ext, _capture_range = matching_spec
                # Bind by signature so saved inputs are stable even when callers
                # mix positional and keyword arguments.
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                # Qualify by RANK (DDP) or PID (single-process concurrency) so
                # concurrent writers never race on the same output file.
                rank = os.environ.get("RANK")
                worker_tag = f"r{rank}" if rank is not None else f"p{os.getpid()}"
                save_path = f"{stem}_{worker_tag}_{call_count:0{n_digits}d}{ext}"
                # Detach tensors so non-leaf inputs don't drag their autograd
                # graph into the capture. main() re-applies requires_grad on
                # load for the parameters that need it.
                captured = tree_map(_detach_for_capture, bound.arguments)
                torch.save(captured, save_path)
                captures_done += 1
                print(
                    f"[gsplat.profile] Captured {fn.__name__} inputs "
                    f"({captures_done}/{total_captures}, call {call_count}) "
                    f"to {save_path}"
                )
                print(
                    "[gsplat.profile] Reload with "
                    "torch.load(path, weights_only=False) on PyTorch 2.6+."
                )
                for k, v in bound.arguments.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
                    elif v is not None:
                        print(f"  {k}: {type(v).__name__} = {v}")
                if captures_done >= total_captures:
                    _pending_captures.discard(capture_id)
                    if not _pending_captures:
                        raise SystemExit("[gsplat.profile] All captures done, exiting.")

            call_count += 1
            return fn(*args, **kwargs)

        return _wrapper  # type: ignore[return-value]

    return decorator


def main() -> None:
    """Standalone profiling harness entry point.

    Usage:
        nsys profile --capture-range=cudaProfilerApi \\
            python -m gsplat.profile --input gsplat_inputs.pt --warmup 5 --iterations 20
    """
    import argparse

    from gsplat.rendering import rasterization
    from gsplat.trace import trace_range

    parser = argparse.ArgumentParser(description="GSplat profiling harness")
    parser.add_argument("--input", required=True, help="Path to input dict (.pt file)")
    parser.add_argument(
        "--warmup", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iterations", type=int, default=20, help="Number of profiled iterations"
    )
    grad_group = parser.add_mutually_exclusive_group()
    grad_group.add_argument(
        "--grad",
        dest="grad",
        action="store_true",
        default=True,
        help=(
            "Replay training-style forward+backward iterations. This is the " "default."
        ),
    )
    grad_group.add_argument(
        "--nograd",
        dest="grad",
        action="store_false",
        help=(
            "Replay forward-only iterations under torch.inference_mode() and "
            "skip backward."
        ),
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Synchronize the CUDA device at the end of each profiled "
            "iteration so it excludes the previous iteration's in-flight "
            "kernels (models an end-of-step pipeline sync). This also makes "
            "the iteration NVTX range span GPU completion time instead of "
            "only launch-side CPU time."
        ),
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help=(
            "Set up the selected single workload, print the selected operator, "
            "losses, expected kernel families, and replay inputs, then exit "
            "before warmup/profiling."
        ),
    )
    parser.add_argument(
        "--ensure-rays",
        action="store_true",
        help=(
            "When the selected replay input has `rays=None` and the camera model "
            "is non-lidar with `with_eval3d=True`, pre-generate per-pixel "
            "rays via `_BaseCameraModel.image_point_to_world_ray_shutter_pose` "
            "and inject them into the inputs before replay. Required for "
            "BF3D measurements: the C++ dispatch in Rasterization.cpp "
            "silently falls back from BRUTE_FORCE_3D to LEGACY_CLOSEST_2D "
            "when rays is None."
        ),
    )
    parser.add_argument(
        "--no-rays",
        action="store_true",
        help=(
            "Strip the `rays` tensor from the selected replay inputs (set to None) "
            "before replay, forcing the kernel to synthesise rays from the "
            "camera model on the fly. Use this to profile the rays-synthesis "
            "code path on inputs that originally embed precomputed rays. "
            "Mutually exclusive with --ensure-rays."
        ),
    )
    parser.add_argument(
        "--3dgs",
        dest="force_3dgs",
        action="store_true",
        help=(
            "Rewrite scene inputs to use the plain RGB "
            "classic 3DGS path: disable UT/eval3d, remove rays/distortion/"
            "rolling-shutter/extra-signal inputs, force pinhole camera model, "
            "and use render_mode=RGB. Explicit --set overrides are applied "
            "after this preset."
        ),
    )
    parser.add_argument(
        "--3dgut",
        dest="force_3dgut",
        action="store_true",
        help=(
            "Rewrite scene inputs to use the 3DGUT/eval3d path: "
            "enable UT/eval3d, force unpacked dense gradients, and keep the "
            "input camera model/signals intact. If rays are missing, this "
            "also generates them before replay. Explicit --set overrides are "
            "applied after this preset."
        ),
    )
    parser.add_argument(
        "--2dgs",
        dest="force_2dgs",
        action="store_true",
        help=(
            "Adapt a camera scene to the rasterization_2dgs() "
            "operator. This drops 3DGS/3DGUT-only inputs such as rays, "
            "distortion, rolling shutter, and UT/eval3d flags, while keeping "
            "the shared Gaussian and camera tensors. Explicit --set overrides "
            "are applied after this adapter."
        ),
    )
    parser.add_argument(
        "--losses",
        nargs="?",
        const="all",
        default="none",
        metavar="LIST",
        help=(
            "Include gsplat CUDA-backed loss paths in the profiled workload. "
            "Bare --losses expands to all losses that make sense for the "
            "selected mode. Supported names: all, none, gaussian_fused, "
            "2dgs_distortion."
        ),
    )
    parser.add_argument(
        "--set",
        "--override",
        dest="overrides",
        metavar="NAME=VALUE",
        action="append",
        default=[],
        help=(
            "Override a top-level selected-operator replay argument. Values are "
            "parsed as JSON/Python literals when possible, with unquoted "
            "strings left as strings. May be passed multiple times, e.g. "
            "--set tile_size=8 --set packed=False."
        ),
    )
    args = parser.parse_args()
    if args.ensure_rays and args.no_rays:
        parser.error("--ensure-rays and --no-rays are mutually exclusive")
    if args.ensure_rays and args.force_3dgs:
        parser.error("--ensure-rays and --3dgs are mutually exclusive")
    if args.force_3dgs and args.force_3dgut:
        parser.error("--3dgs and --3dgut are mutually exclusive")
    if args.force_3dgs and args.force_2dgs:
        parser.error("--3dgs and --2dgs are mutually exclusive")
    if args.force_3dgut and args.force_2dgs:
        parser.error("--3dgut and --2dgs are mutually exclusive")
    if args.no_rays and args.force_3dgut:
        parser.error("--no-rays and --3dgut are mutually exclusive")
    if args.force_2dgs and args.ensure_rays:
        parser.error("--ensure-rays and --2dgs are mutually exclusive")
    if args.force_2dgs and args.no_rays:
        parser.error("--no-rays and --2dgs are mutually exclusive")

    print(f"[gsplat.profile] Loading inputs from {args.input}")
    inputs: dict[str, Any] = torch.load(
        args.input, map_location="cuda", weights_only=False
    )

    if args.grad:
        # Recreate the training-style inputs expected by rasterization replay.
        for k in _GRAD_PARAMS:
            if (
                k in inputs
                and isinstance(inputs[k], torch.Tensor)
                and inputs[k].is_floating_point()
            ):
                inputs[k] = inputs[k].requires_grad_(True)

    replay_operator: Callable[..., Any] = rasterization
    replay_name = "rasterization"
    replay_params = inspect.signature(replay_operator).parameters
    workload_name = "rasterization"
    workload_notes: list[str] = []

    if args.force_2dgs:
        from gsplat.rendering import rasterization_2dgs

        replay_operator = rasterization_2dgs
        replay_name = "rasterization_2dgs"
        workload_name = "2dgs"
        replay_params = inspect.signature(replay_operator).parameters
        try:
            replay_inputs, notes = _adapt_2dgs_scene_inputs(inputs, replay_params)
        except ValueError as exc:
            parser.error(str(exc))
        print("[gsplat.profile] --2dgs: adapting input to rasterization_2dgs replay")
        workload_notes.extend(notes)
        for note in notes:
            print(f"[gsplat.profile] --2dgs: {note}")
    elif args.force_3dgs:
        workload_name = "3dgs"
        try:
            replay_inputs = _select_replay_inputs(inputs, replay_params, "--3dgs")
            _apply_3dgs_replay_preset(replay_inputs)
        except ValueError as exc:
            parser.error(str(exc))
        print("[gsplat.profile] --3dgs: forcing plain RGB classic 3DGS replay")
        workload_notes.append("forced plain RGB classic 3DGS replay")
    elif args.force_3dgut:
        workload_name = "3dgut"
        try:
            replay_inputs = _select_replay_inputs(inputs, replay_params, "--3dgut")
        except ValueError as exc:
            parser.error(str(exc))
        _apply_3dgut_replay_preset(replay_inputs)
        print("[gsplat.profile] --3dgut: forcing 3DGUT/eval3d replay")
        workload_notes.append("forced 3DGUT/eval3d replay")
    else:
        try:
            replay_inputs = _select_replay_inputs(inputs, replay_params, replay_name)
        except ValueError as exc:
            parser.error(str(exc))

    try:
        profile_losses = _parse_profile_losses(args.losses, replay_name)
        loss_preset_notes = _apply_profile_loss_presets(
            profile_losses, replay_name, replay_inputs
        )
    except ValueError as exc:
        parser.error(str(exc))
    workload_notes.extend(loss_preset_notes)
    for note in loss_preset_notes:
        print(f"[gsplat.profile] --losses: {note}")

    for raw_override in args.overrides:
        try:
            name, value = _parse_input_override(raw_override)
        except ValueError as exc:
            parser.error(str(exc))
        if name not in replay_params:
            parser.error(
                f"unknown {replay_name} argument {name!r}; "
                f"expected one of {', '.join(replay_params)}"
            )
        replay_inputs[name] = value
        print(
            f"[gsplat.profile] override {name}={value!r} " f"({type(value).__name__})"
        )

    try:
        _validate_profile_loss_overrides(profile_losses, replay_name, replay_inputs)
    except ValueError as exc:
        parser.error(str(exc))

    if replay_name == "rasterization_2dgs":
        try:
            replay_normalize_notes = _normalize_2dgs_replay_inputs(replay_inputs)
        except ValueError as exc:
            parser.error(str(exc))
        workload_notes.extend(replay_normalize_notes)
        for note in replay_normalize_notes:
            print(f"[gsplat.profile] --2dgs: {note}")

    if args.no_rays:
        if replay_inputs.get("rays") is None:
            print("[gsplat.profile] --no-rays: skipping (rays already None)")
        else:
            print(
                f"[gsplat.profile] --no-rays: stripping rays "
                f"{tuple(replay_inputs['rays'].shape)} from inputs"
            )
            replay_inputs["rays"] = None

    should_ensure_rays = args.ensure_rays or args.force_3dgut
    ensure_rays_label = "--ensure-rays" if args.ensure_rays else "--3dgut"
    if should_ensure_rays and replay_inputs.get("rays") is None:
        cam_model = replay_inputs.get("camera_model", "pinhole")
        with_eval3d = replay_inputs.get("with_eval3d", False)
        if not with_eval3d:
            print(
                f"[gsplat.profile] {ensure_rays_label}: skipping "
                "(with_eval3d=False; "
                "rays unused on this path)"
            )
        else:
            from gsplat.cuda._torch_impl_eval3d import (
                _BaseCameraModel,
                _generate_rays,
            )
            from gsplat.cuda._wrapper import RollingShutterType

            original_viewmats = replay_inputs["viewmats"]
            # Preserve any leading batch dims; rasterization() requires
            # rays.shape == viewmats.shape[:-2] + (H, W, 6).
            leading_shape = original_viewmats.shape[:-2]
            viewmats = original_viewmats.reshape(-1, 4, 4)
            viewmats_rs = replay_inputs.get("viewmats_rs")
            if viewmats_rs is not None:
                viewmats_rs = viewmats_rs.reshape(-1, 4, 4)
            rs_type = replay_inputs.get("rolling_shutter")
            if rs_type is None:
                rs_type = RollingShutterType.GLOBAL

            if cam_model == "lidar":
                # Lidar dimensions come from lidar_coeffs (rendering.py:582-584
                # overrides inputs["width"]/inputs["height"] from the coeffs).
                lidar_coeffs = replay_inputs["lidar_coeffs"]
                gen_width = lidar_coeffs.n_columns
                gen_height = lidar_coeffs.n_rows
                cam = _BaseCameraModel.create(
                    camera_model="lidar",
                    rs_type=rs_type,
                    lidar_coeffs=lidar_coeffs,
                )
            else:
                Ks = replay_inputs["Ks"].reshape(-1, 3, 3)
                gen_width = replay_inputs["width"]
                gen_height = replay_inputs["height"]
                cam = _BaseCameraModel.create(
                    width=gen_width,
                    height=gen_height,
                    camera_model=cam_model,
                    focal_lengths=Ks[:, [0, 1], [0, 1]],
                    principal_points=Ks[:, [0, 1], [2, 2]],
                    rs_type=rs_type,
                    radial_coeffs=replay_inputs.get("radial_coeffs"),
                    tangential_coeffs=replay_inputs.get("tangential_coeffs"),
                    thin_prism_coeffs=replay_inputs.get("thin_prism_coeffs"),
                    ftheta_coeffs=replay_inputs.get("ftheta_coeffs"),
                )
            rays = _generate_rays(
                cam,
                gen_width,
                gen_height,
                viewmats,
                viewmats_rs,
            ).reshape(*leading_shape, gen_height, gen_width, 6)
            replay_inputs["rays"] = rays
            print(
                f"[gsplat.profile] {ensure_rays_label}: generated rays "
                f"{tuple(rays.shape)} "
                f"(camera_model={cam_model})"
            )
            workload_notes.append(
                f"{ensure_rays_label} generated rays {tuple(rays.shape)} "
                f"(camera_model={cam_model})"
            )

    try:
        profile_loss_contribution, loss_build_notes = _build_profile_loss_contribution(
            profile_losses, replay_name, replay_inputs, inputs
        )
    except ValueError as exc:
        parser.error(str(exc))
    workload_notes.extend(loss_build_notes)
    for note in loss_build_notes:
        print(f"[gsplat.profile] --losses: {note}")

    workload_descriptor = ProfileWorkload(
        name=workload_name,
        operator_name=replay_name,
        operator=replay_operator,
        replay_inputs=replay_inputs,
        losses=profile_losses,
        loss_contribution=profile_loss_contribution,
        expected_kernel_families=_expected_kernel_families(
            replay_name, replay_inputs, profile_losses
        ),
        notes=workload_notes,
    )

    n_gaussians = replay_inputs["means"].shape[-2] if "means" in replay_inputs else "?"
    width = replay_inputs.get("width", "?")
    height = replay_inputs.get("height", "?")
    print(
        f"[gsplat.profile] {replay_name}: {n_gaussians} gaussians, "
        f"{width}x{height} image"
    )
    replay_mode = "grad (forward+backward)" if args.grad else "nograd (forward-only)"
    print(f"[gsplat.profile] Replay mode: {replay_mode}")
    print(
        f"[gsplat.profile] Warmup: {args.warmup}, Profiled iterations: {args.iterations}"
    )
    if args.describe:
        _describe_workload(workload_descriptor)
        return

    if args.sync:
        print("[gsplat.profile] Synchronizing after each profiled iteration")

    def replay_forward() -> tuple[tuple[Any, ...], torch.Tensor]:
        """Run the selected operator and return outputs plus render colors."""

        outputs = workload_descriptor.operator(**workload_descriptor.replay_inputs)
        if len(outputs) == 2 and isinstance(outputs[0], tuple):
            render_colors = outputs[0][0]
        else:
            render_colors = outputs[0]
        return outputs, render_colors

    def run_iteration(iteration: int) -> None:
        # Keep the replay structure explicit so NVTX ranges map cleanly to the
        # forward and backward phases in Nsight tools.
        with trace_range("iteration", payload=iteration):
            with trace_range("forward"):
                if args.grad:
                    outputs, render_colors = replay_forward()
                    loss = render_colors.sum() + workload_descriptor.loss_contribution(
                        outputs
                    )
                else:
                    with torch.inference_mode():
                        outputs, _render_colors = replay_forward()
                        if workload_descriptor.losses:
                            workload_descriptor.loss_contribution(outputs)

            if args.grad:
                with trace_range("backward"):
                    loss.backward()
                # Clear grads in place so every iteration starts from the same state.
                _clear_replay_grads(workload_descriptor.replay_inputs)

            if args.sync:
                # Kept inside the range so the wait counts toward this iteration.
                torch.cuda.synchronize()

    print(f"[gsplat.profile] Running {args.warmup} warmup iterations...")
    for i in range(args.warmup):
        run_iteration(i)
    print("[gsplat.profile] Warmup done.")

    torch.autograd.set_multithreading_enabled(False)
    # Gate profiler collection to the measured iterations only.
    print(
        f"[gsplat.profile] Starting profiler, running {args.iterations} iterations..."
    )
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(args.iterations):
        run_iteration(i)
    # Drain in-flight kernels before stopping the profiler, even when --sync is off.
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("[gsplat.profile] Profiling done.")


if __name__ == "__main__":
    main()
