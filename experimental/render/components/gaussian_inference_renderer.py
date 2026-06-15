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


"""Stateful Gaussian Inference renderer component."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from libs.scene import GaussianInferenceScene

from experimental.render._common import check_inference_grad_mode
from experimental.render.kernels.gaussian_inference_ops import (
    create_native_gaussian_inference_renderer,
)
from experimental.render.types import RenderReturn


# ---------------------------------------------------------------------------
# Unsupported-feature set for the stateful renderer (same as the stateless
# path, minus the ones that are simply not parameters there like
# render_mode / camera_model, plus a few extras).
# ---------------------------------------------------------------------------

_RENDERER_UNSUPPORTED_KWARGS = frozenset(
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
        "render_mode",
        "camera_model",
        "backgrounds",
        "sh_compression_mode",
    }
)


# ---------------------------------------------------------------------------
# GaussianInferenceRenderer  (stateful, persistent buffer reuse)
# ---------------------------------------------------------------------------


class GaussianInferenceRenderer:
    """Stateful Inference renderer that caches per-scene and per-frame GPU buffers.

    Wraps the C++ ``GaussianInferenceRenderer`` for persistent buffer reuse across frames.
    Unlike the stateless :func:`rasterize_gaussian_inference_scene`, this class
        pre-allocates projection, visibility, color, and intersection buffers
        in the native renderer.  The native half4 output buffer is owned by
        Python and passed as an ``out`` tensor so rasterization writes directly
        into it.

    Usage::

        scene = GaussianInferenceScene(...)
        with GaussianInferenceRenderer(scene) as renderer:
            result = renderer.render(
                viewmat=viewmat, K=K, width=1920, height=1080,
            )
            # result.frame is [1, H, W, 4] float16 RGBT

    Parameters
    ----------
    scene : GaussianInferenceScene
        Packed scene to render.  Must not be empty.
    tile_size : int, optional
        Default tile size for rasterisation (8 or 16).  Can be overridden
        per-frame via :meth:`render`.  Default is 8.
    """

    def __init__(self, scene: Any, *, tile_size: int = 8) -> None:
        if not isinstance(scene, GaussianInferenceScene):
            raise TypeError(
                f"GaussianInferenceRenderer requires a GaussianInferenceScene; "
                f"got {type(scene).__name__}"
            )

        if scene.is_empty():
            raise ValueError(
                "GaussianInferenceScene has been released and contains no packed "
                "tensors. Did you forget to rebuild the snapshot?"
            )

        if tile_size not in (8, 16):
            raise ValueError(f"tile_size must be 8 or 16; got {tile_size}")

        # -- Create native renderer ----------------------------------------
        self._native = create_native_gaussian_inference_renderer(
            scene.means_planar,
            scene.qso_packed,
            scene.colors_packed,
            scene.sh_degree,
            scene.sh_compression_mode,
        )
        self._scene = scene
        self._tile_size = tile_size
        self._frame_buffer: Optional[Tensor] = None
        self._frame_buffer_shape: Optional[tuple[int, int, int, int]] = None

    # ------------------------------------------------------------------
    # render
    # ------------------------------------------------------------------

    def render(
        self,
        *,
        viewmat: Optional[Tensor] = None,
        viewmats: Optional[Tensor] = None,
        K: Optional[Tensor] = None,
        Ks: Optional[Tensor] = None,
        width: int,
        height: int,
        tile_size: Optional[int] = None,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        background: Optional[Tensor] = None,
        sh_degree: Optional[int] = None,
        out: Optional[RenderReturn] = None,
        **kwargs: Any,
    ) -> RenderReturn:
        """Render a single frame.

        Parameters
        ----------
        viewmat, viewmats : Tensor
            Camera extrinsics.  Pass exactly one; if *viewmats* is given it
            must have leading dim 1.
        K, Ks : Tensor
            Camera intrinsics.  Pass exactly one; if *Ks* is given it must
            have leading dim 1.
        width, height : int
            Output resolution in pixels.
        tile_size : int, optional
            Override the default tile size set at construction time.
        near_plane, far_plane : float
            Clipping planes.
        radius_clip : float
            Gaussians with projected radius below this are culled.
        eps2d : float
            Covariance regularisation epsilon.
        background : Tensor, optional
            Background color ``[3]`` float32.
        sh_degree : int, optional
            Per-frame SH degree (clamped to scene max in C++).  Defaults to
            the scene's SH degree.
        out : RenderReturn, optional
            Pre-allocated half4 output buffer to write into.

        Returns
        -------
        RenderReturn
            ``.frame`` is ``[1, H, W, 4]`` float16 with channels
            ``{R, G, B, T}``, where ``T`` is transmittance.  Callers that need
            alpha should compute ``1 - frame[..., 3:4]``.
        """
        if self._native is None or self._native.is_released():
            raise RuntimeError(
                "GaussianInferenceRenderer has been released; cannot render"
            )

        # -- Scene mutation guard ------------------------------------------
        if self._scene.num_gaussians != self._native.num_gaussians():
            raise RuntimeError(
                f"Scene was mutated after renderer creation "
                f"(scene has {self._scene.num_gaussians} Gaussians, "
                f"renderer was built for {self._native.num_gaussians()}). "
                f"Create a new GaussianInferenceRenderer after modifying the scene."
            )

        # -- Reject unsupported kwargs ------------------------------------
        for key in kwargs:
            if key in _RENDERER_UNSUPPORTED_KWARGS:
                raise TypeError(
                    f"GaussianInferenceRenderer.render() does not support {key}"
                )
            else:
                raise TypeError(
                    f"GaussianInferenceRenderer.render() got unexpected keyword "
                    f"argument '{key}'"
                )

        # -- Grad-mode gate ------------------------------------------------
        check_inference_grad_mode()

        # -- Normalize cameras ---------------------------------------------
        viewmat_t = self._normalize_viewmat(viewmat, viewmats)
        K_t = self._normalize_K(K, Ks)

        # -- tile_size -----------------------------------------------------
        effective_tile_size = tile_size if tile_size is not None else self._tile_size
        if effective_tile_size not in (8, 16):
            raise ValueError(f"tile_size must be 8 or 16; got {effective_tile_size}")

        # -- sh_degree -----------------------------------------------------
        effective_sh_degree = (
            sh_degree if sh_degree is not None else self._scene.sh_degree
        )

        # -- Validate out buffer -------------------------------------------
        if out is not None:
            self._validate_half4_out_buffer(out, height, width, viewmat_t.device)

        # -- Select half4 output framebuffer --------------------------------
        if out is not None:
            out_rgbt = out.frame  # [1, H, W, 4]
        else:
            out_rgbt = self._get_frame_buffer(height, width, viewmat_t.device)

        # -- Call C++ renderer ---------------------------------------------
        rgbt = self._native.render(
            self._scene.means_planar,
            self._scene.qso_packed,
            self._scene.colors_packed,
            viewmat_t,
            K_t,
            width,
            height,
            effective_tile_size,
            near_plane,
            far_plane,
            radius_clip,
            eps2d,
            effective_sh_degree,
            self._scene.sh_compression_mode,
            background,
            out_rgbt,
        )

        # -- Package result ------------------------------------------------
        metadata = {"format": "RGBT", "channels": "RGBT"}
        if out is None:
            return RenderReturn(
                frame=rgbt,  # [1, H, W, 4]
                metadata=metadata,
            )
        else:
            out.metadata.clear()
            out.metadata.update(metadata)
            return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release all GPU buffers held by the native renderer."""
        if self._native is not None and not self._native.is_released():
            self._native.release()
        self._native = None
        self._frame_buffer = None
        self._frame_buffer_shape = None

    @property
    def is_released(self) -> bool:
        """True if the renderer has been released."""
        return self._native is None or self._native.is_released()

    @property
    def num_gaussians(self) -> int:
        """Number of Gaussians in the scene, or 0 if released."""
        if self._native is not None and not self._native.is_released():
            return self._native.num_gaussians()
        return 0

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self) -> "GaussianInferenceRenderer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_viewmat(
        viewmat: Optional[Tensor],
        viewmats: Optional[Tensor],
    ) -> Tensor:
        """Return a single [4, 4] viewmat tensor."""
        if viewmat is not None and viewmats is not None:
            raise RuntimeError("pass exactly one of viewmat or viewmats, not both")
        if viewmat is None and viewmats is None:
            raise RuntimeError("pass exactly one of viewmat or viewmats")

        if viewmats is not None:
            if viewmats.shape[0] > 1:
                raise RuntimeError(
                    f"Inference branch supports single camera (leading dim == 1); "
                    f"got viewmats with leading dim {viewmats.shape[0]}"
                )
            return viewmats[0]
        return viewmat  # type: ignore[return-value]

    @staticmethod
    def _normalize_K(
        K: Optional[Tensor],
        Ks: Optional[Tensor],
    ) -> Tensor:
        """Return a single [3, 3] intrinsics tensor."""
        if K is not None and Ks is not None:
            raise RuntimeError("pass exactly one of K or Ks, not both")
        if K is None and Ks is None:
            raise RuntimeError("pass exactly one of K or Ks")

        if Ks is not None:
            if Ks.shape[0] > 1:
                raise RuntimeError(
                    f"Inference branch supports single camera (leading dim == 1); "
                    f"got Ks with leading dim {Ks.shape[0]}"
                )
            return Ks[0]
        return K  # type: ignore[return-value]

    @staticmethod
    def _validate_half4_out_buffer(
        out: RenderReturn,
        height: int,
        width: int,
        device: torch.device,
    ) -> None:
        """Validate the stateful renderer's native half4 ``out=`` buffer."""
        frame = out.frame
        ok = (
            frame.shape == (1, height, width, 4)
            and frame.dtype == torch.float16
            and frame.device == device
            and frame.is_contiguous()
        )
        if not ok:
            raise RuntimeError(
                "out.frame expected shape "
                f"[1, {height}, {width}, 4], dtype torch.float16, "
                f"device {device}, contiguous; got shape {list(frame.shape)}, "
                f"dtype {frame.dtype}, device {frame.device}, "
                f"contiguous={frame.is_contiguous()}"
            )
        if frame.requires_grad:
            raise RuntimeError("out=... buffer must not be grad-tracked")

    def _get_frame_buffer(
        self,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tensor:
        """Return the Python-owned default framebuffer for this resolution."""
        shape = (1, height, width, 4)
        if (
            self._frame_buffer is None
            or self._frame_buffer_shape != shape
            or self._frame_buffer.device != device
        ):
            self._frame_buffer = torch.empty(
                shape,
                device=device,
                dtype=torch.float16,
            )
            self._frame_buffer_shape = shape
        return self._frame_buffer


__all__ = ["GaussianInferenceRenderer"]
