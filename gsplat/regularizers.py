"""Regularizers ported from G-SHARP v0.2.

Public API:

- :func:`compute_tv_loss_targeted` — anisotropic total-variation loss,
  optionally restricted to a binary mask region. Ported from
  ``compute_tv_loss_targeted`` in
  ``holohub/applications/surgical_scene_recon/training/gsplat_train.py``
  (lines 1992–2028 in the source).
- :func:`dilate_mask` — pure-torch 2D mask dilation via max-pool. Replaces
  the ``cv2.dilate`` call used by G-SHARP so the core library has no OpenCV
  dependency.
- :func:`create_invisible_mask` — union of per-frame tool / dynamic-object
  masks, matching the "regions chronically occluded by instruments"
  definition from G-SHARP. Accepts either an iterable of tensors (assumed
  already in tool=1 convention) or PNG paths (loaded and inverted to match
  G-SHARP's tissue=1 on-disk convention).
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_tv_loss_targeted(image: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Anisotropic total-variation loss, optionally weighted by a binary mask.

    Without a mask, returns the unweighted TV averaged over ``image.numel()``.
    With a mask, sums each anisotropic difference weighted by the mask
    cropped to the difference shape, then divides by the number of valid
    elements (``mask.sum() * C``) plus a tiny epsilon — matching the G-SHARP
    behaviour at ``training/gsplat_train.py`` lines 1992–2028.

    Args:
        image: 4D tensor ``(B, C, H, W)``.
        mask: Optional 4D binary mask ``(B, 1, H, W)`` (or broadcastable),
            with values in ``{0, 1}``.

    Returns:
        Scalar TV loss.

    Raises:
        ValueError: if *image* isn't 4D, or if *mask* contains non-binary
            values.
    """
    if image.ndim != 4:
        raise ValueError(
            f"compute_tv_loss_targeted: expected 4D image (B, C, H, W), "
            f"got {image.ndim}D shape {tuple(image.shape)}."
        )

    tv_h = (image[:, :, 1:, :] - image[:, :, :-1, :]).abs()
    tv_w = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs()

    if mask is None:
        return (tv_h.sum() + tv_w.sum()) / image.numel()

    if not (((mask == 0) | (mask == 1)).all()):
        raise ValueError(
            "compute_tv_loss_targeted: mask must be binary (values in {0, 1}). "
            "Convert via `(mask > threshold).float()` first if needed."
        )

    mask_h = mask[:, :, 1:, :]
    mask_w = mask[:, :, :, 1:]

    tv_h_sum = (tv_h * mask_h).sum()
    tv_w_sum = (tv_w * mask_w).sum()

    channels = image.shape[1]
    num_h = mask_h.sum() * channels + 1e-8
    num_w = mask_w.sum() * channels + 1e-8

    return tv_h_sum / num_h + tv_w_sum / num_w


def dilate_mask(mask: Tensor, kernel_size: int = 3) -> Tensor:
    """Binary mask dilation via 2D max-pool.

    Pure-torch replacement for ``cv2.dilate``. With *kernel_size* = 1, this
    is the identity. With *kernel_size* = 3, each ``1`` pixel grows to a
    ``3 × 3`` neighbourhood (one-pixel safety margin). Larger odd kernel
    sizes scale accordingly.

    Args:
        mask: 2D ``(H, W)``, 3D ``(C, H, W)``, or 4D ``(B, C, H, W)`` mask.
        kernel_size: Positive odd integer (default ``3``).

    Returns:
        Dilated mask of the same shape and dtype-promotion as the input
        (cast to float internally).

    Raises:
        ValueError: if *kernel_size* is not a positive odd integer, or if
            *mask* isn't 2D/3D/4D.
    """
    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"dilate_mask: kernel_size must be a positive odd integer, "
            f"got {kernel_size!r}."
        )

    original_ndim = mask.ndim
    if original_ndim == 2:
        x = mask.unsqueeze(0).unsqueeze(0).float()
    elif original_ndim == 3:
        x = mask.unsqueeze(0).float()
    elif original_ndim == 4:
        x = mask.float()
    else:
        raise ValueError(
            f"dilate_mask: expected 2D / 3D / 4D mask, got {original_ndim}D."
        )

    pad = kernel_size // 2
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

    if original_ndim == 2:
        return dilated.squeeze(0).squeeze(0)
    if original_ndim == 3:
        return dilated.squeeze(0)
    return dilated


def create_invisible_mask(masks: Iterable[Union[Tensor, str]]) -> Tensor:
    """Union of per-frame tool / dynamic-object masks ("invisible region").

    Pixels that appear as ``1`` (tool / occluder) in *any* frame are marked
    ``1`` in the returned mask — i.e. regions that are *ever* occluded by an
    instrument across the dataset. Matches the G-SHARP semantics of
    ``create_invisible_mask_from_paths`` (lines 1925–1957 in the source).

    Two input modes are supported per element of *masks*:

    - ``Tensor`` — treated as already in tool=1 convention (no inversion).
    - ``str`` (path) — loaded via PIL, normalized to ``[0, 1]``, and
      inverted (``1 - mask``) per the G-SHARP convention that on-disk PNGs
      are tissue=1.

    Args:
        masks: Iterable of tensors or PNG paths. All must share spatial
            shape after loading.

    Returns:
        2D float tensor with values in ``[0, 1]``, the union across all
        inputs (element-wise max).

    Raises:
        ValueError: if *masks* is empty or shapes don't match.
        TypeError: if an element is neither ``Tensor`` nor ``str``.
    """
    masks_list = list(masks)
    if not masks_list:
        raise ValueError("create_invisible_mask: at least one mask is required.")

    tensors = []
    for i, m in enumerate(masks_list):
        if isinstance(m, str):
            from PIL import Image  # local import — Pillow is an optional dep

            import numpy as np

            arr = np.array(Image.open(m)) / 255.0
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            t = torch.from_numpy((1.0 - arr).astype("float32"))
        elif isinstance(m, Tensor):
            t = m.float()
        else:
            raise TypeError(
                f"create_invisible_mask: expected Tensor or str path at index {i}, "
                f"got {type(m).__name__}."
            )
        tensors.append(t)

    first_shape = tensors[0].shape
    for i, t in enumerate(tensors[1:], 1):
        if t.shape != first_shape:
            raise ValueError(
                f"create_invisible_mask: mask {i} shape {tuple(t.shape)} != "
                f"mask 0 shape {tuple(first_shape)}."
            )

    return torch.stack(tensors).amax(dim=0).clamp(0.0, 1.0)
