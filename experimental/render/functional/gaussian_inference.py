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


"""Stateless Gaussian Inference render API."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from libs.scene import GaussianInferenceScene

from experimental.render._common import check_inference_grad_mode
from experimental.render.kernels.gaussian_inference_ops import (
    gaussian_render_inference_only,
)
from experimental.render.types import RenderReturn


# Known unsupported features for the Inference branch (each raises TypeError).
_INFERENCE_UNSUPPORTED_FEATURES = frozenset(
    {
        "with_ut",
        "with_eval3d",
        "absgrad",
        "sparse_grad",
        "distributed",
        "packed",
        "segmented",
        "return_normals",
        "covars",
        "rays",
        "radial_coeffs",
        "tangential_coeffs",
        "thin_prism_coeffs",
        "ftheta_coeffs",
        "lidar_coeffs",
        "external_distortion_coeffs",
        "rolling_shutter",
        "viewmats_rs",
        "extra_signals",
        "extra_signals_sh_degree",
        "rasterize_mode",
        "channel_chunk",
        "global_z_order",
        "ut_params",
        "colors",
    }
)

# Accepted kwargs for the Inference branch.
_INFERENCE_ACCEPTED_KWARGS = frozenset(
    {
        "viewmat",
        "viewmats",
        "K",
        "Ks",
        "width",
        "height",
        "tile_size",
        "near_plane",
        "far_plane",
        "radius_clip",
        "eps2d",
        "background",
        "render_mode",
        "camera_model",
    }
)


def _check_grad_mode() -> None:
    check_inference_grad_mode()


def _validate_device_consistency(
    scene: GaussianInferenceScene,
    request: dict[str, Any],
    out: Optional[RenderReturn],
) -> None:
    """Ensure all request tensors live on the same CUDA device as the scene."""
    scene_device = scene.means_planar.device
    for name in ("viewmat", "viewmats", "K", "Ks", "background"):
        val = request.get(name)
        if val is not None and isinstance(val, Tensor) and val.device != scene_device:
            raise ValueError(
                f"{name} is on {val.device} but scene is on {scene_device}; "
                f"all tensors must be on the same device"
            )
    if out is not None and out.frame.device != scene_device:
        raise ValueError(
            f"out buffer is on {out.frame.device} but scene is on "
            f"{scene_device}; all tensors must be on the same device"
        )


def _validate_inference_request(request: dict[str, Any]) -> dict[str, Any]:
    """Validate and extract inference kwargs from ``**request``.

    Returns a normalised dict ready for downstream consumption.
    """
    # -- Reject sh_degree / sh_compression_mode overrides ----------------
    for key in ("sh_degree", "sh_compression_mode"):
        if key in request:
            raise TypeError(
                "sh_degree/sh_compression_mode are read from scene; "
                "cannot be overridden via request kwargs"
            )

    # -- Reject ``backgrounds`` (plural) ---------------------------------
    if "backgrounds" in request:
        raise TypeError(
            "rasterize_gaussian_inference_scene got unexpected keyword argument "
            "'backgrounds'"
        )

    # -- Reject known-unsupported features -------------------------------
    for key in _INFERENCE_UNSUPPORTED_FEATURES:
        if key in request:
            raise TypeError(f"Inference branch does not support {key}")

    # -- Reject truly unknown kwargs -------------------------------------
    for key in request:
        if key not in _INFERENCE_ACCEPTED_KWARGS:
            raise TypeError(
                f"rasterize_gaussian_inference_scene got unexpected keyword "
                f"argument '{key}'"
            )

    # -- Constrained values -----------------------------------------------
    render_mode = request.get("render_mode", "RGB")
    if render_mode != "RGB":
        raise TypeError(
            f"Inference branch supports render_mode='RGB' only; got '{render_mode}'"
        )

    camera_model = request.get("camera_model", "pinhole")
    if camera_model != "pinhole":
        raise TypeError(
            f"Inference branch supports camera_model='pinhole' only; "
            f"got '{camera_model}'"
        )

    tile_size = request.get("tile_size", 8)
    if tile_size not in (8, 16):
        raise TypeError(
            f"Inference branch supports tile_size in {{8, 16}}; got {tile_size}"
        )

    # -- Build normalised result -----------------------------------------
    return {
        "viewmat": request.get("viewmat"),
        "viewmats": request.get("viewmats"),
        "K": request.get("K"),
        "Ks": request.get("Ks"),
        "width": request.get("width"),
        "height": request.get("height"),
        "tile_size": tile_size,
        "near_plane": request.get("near_plane", 0.01),
        "far_plane": request.get("far_plane", 1e10),
        "radius_clip": request.get("radius_clip", 0.0),
        "eps2d": request.get("eps2d", 0.3),
        "background": request.get("background"),
    }


def _normalize_cameras(
    kwargs: dict[str, Any],
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Normalize viewmat/K from request kwargs.

    Returns ``(viewmat, K, cleaned_kwargs)`` where ``cleaned_kwargs`` no
    longer contains viewmat/viewmats/K/Ks.
    """
    viewmat_val = kwargs.get("viewmat")
    viewmats_val = kwargs.get("viewmats")
    K_val = kwargs.get("K")
    Ks_val = kwargs.get("Ks")

    # -- viewmat / viewmats -----------------------------------------------
    if viewmat_val is not None and viewmats_val is not None:
        raise RuntimeError("pass exactly one of viewmat or viewmats, not both")
    if viewmat_val is None and viewmats_val is None:
        raise RuntimeError("pass exactly one of viewmat or viewmats")

    if viewmats_val is not None:
        if viewmats_val.shape[0] > 1:
            raise RuntimeError(
                f"Inference branch supports single camera (leading dim == 1); "
                f"got viewmats with leading dim {viewmats_val.shape[0]}"
            )
        viewmat = viewmats_val[0]
    else:
        viewmat = viewmat_val

    # -- K / Ks -----------------------------------------------------------
    if K_val is not None and Ks_val is not None:
        raise RuntimeError("pass exactly one of K or Ks, not both")
    if K_val is None and Ks_val is None:
        raise RuntimeError("pass exactly one of K or Ks")

    if Ks_val is not None:
        if Ks_val.shape[0] > 1:
            raise RuntimeError(
                f"Inference branch supports single camera (leading dim == 1); "
                f"got Ks with leading dim {Ks_val.shape[0]}"
            )
        K = Ks_val[0]
    else:
        K = K_val

    # Build a cleaned copy without the camera keys
    cleaned = {
        k: v for k, v in kwargs.items() if k not in ("viewmat", "viewmats", "K", "Ks")
    }
    return viewmat, K, cleaned


def _validate_out_buffer(
    out: RenderReturn,
    height: int,
    width: int,
    device: torch.device,
) -> None:
    """Validate the ``out=`` buffer contract."""
    # -- frame -----------------------------------------------------------
    _validate_single_buffer(
        out.frame,
        name="out.frame",
        expected_shape=(1, height, width, 3),
        device=device,
    )
    if out.frame.requires_grad:
        raise RuntimeError("out=... buffer must not be grad-tracked")

    # -- optional alpha in metadata --------------------------------------
    alpha = out.metadata.get("alpha")
    if alpha is not None:
        _validate_single_buffer(
            alpha,
            name="out.metadata['alpha']",
            expected_shape=(1, height, width, 1),
            device=device,
        )
        if alpha.requires_grad:
            raise RuntimeError("out=... buffer must not be grad-tracked")


def _validate_single_buffer(
    t: Tensor,
    *,
    name: str,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    """Validate shape, dtype, device, contiguity for a single buffer tensor."""
    ok = (
        t.shape == expected_shape
        and t.dtype == torch.float32
        and t.device == device
        and t.is_contiguous()
    )
    if not ok:
        raise RuntimeError(
            f"{name} expected shape {list(expected_shape)}, "
            f"dtype torch.float32, device {device}, contiguous; "
            f"got shape {list(t.shape)}, dtype {t.dtype}, "
            f"device {t.device}, contiguous={t.is_contiguous()}"
        )


# ---------------------------------------------------------------------------
# rasterize_gaussian_inference_scene  (direct inference entry point)
# ---------------------------------------------------------------------------


def rasterize_gaussian_inference_scene(
    scene: GaussianInferenceScene,
    *,
    out: Optional[RenderReturn] = None,
    **request: Any,
) -> RenderReturn:
    """Render a ``GaussianInferenceScene`` via the fused Inference rasterisation op.

    This is the direct, low-level entry point.  For a polymorphic API that
    also handles ``GaussianScene``, use :func:`render_scene`.
    """
    # 1. Type check
    if not isinstance(scene, GaussianInferenceScene):
        raise TypeError(
            f"rasterize_gaussian_inference_scene requires a GaussianInferenceScene; "
            f"got {type(scene).__name__}"
        )

    # 2. Empty-scene guard
    if scene.is_empty():
        raise ValueError(
            "GaussianInferenceScene has been released and contains no packed "
            "tensors. Did you forget to rebuild the snapshot?"
        )

    # 3. Grad-mode gate
    _check_grad_mode()

    # 3b. Device consistency
    _validate_device_consistency(scene, request, out)

    # 4. Request-subset validation
    validated = _validate_inference_request(request)

    # 5. Camera normalisation
    viewmat, K, rest = _normalize_cameras(validated)

    height = rest["height"]
    width = rest["width"]
    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"height must be a positive integer, got {height!r}")
    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"width must be a positive integer, got {width!r}")

    # 6. Buffer validation
    if out is not None:
        _validate_out_buffer(out, height, width, viewmat.device)

    # 7. Build out_renders / out_alphas views
    out_renders: Optional[Tensor] = None
    out_alphas: Optional[Tensor] = None
    saved_alpha: Optional[Tensor] = None
    if out is not None:
        out_renders = out.frame[0]  # [H, W, 3]
        alpha_buf = out.metadata.get("alpha")
        if alpha_buf is not None:
            saved_alpha = alpha_buf
            out_alphas = alpha_buf[0]  # [H, W, 1]

    # 8. Call accelerated backend (stateless one-shot path)
    renders, alphas = gaussian_render_inference_only(
        scene.means_planar,
        scene.qso_packed,
        scene.colors_packed,
        viewmat,
        K,
        width,
        height,
        scene.sh_degree,
        rest["tile_size"],
        rest["near_plane"],
        rest["far_plane"],
        rest["radius_clip"],
        rest["eps2d"],
        scene.sh_compression_mode,
        rest["background"],
        out_renders=out_renders,
        out_alphas=out_alphas,
    )

    # 9. Package result
    if out is None:
        return RenderReturn(
            frame=renders[None],  # [1, H, W, 3]
            metadata={"alpha": alphas[None]},  # [1, H, W, 1]
        )
    else:
        out.metadata.clear()
        if saved_alpha is not None:
            out.metadata["alpha"] = saved_alpha
        else:
            out.metadata["alpha"] = alphas[None]
        return out


__all__ = ["rasterize_gaussian_inference_scene"]
