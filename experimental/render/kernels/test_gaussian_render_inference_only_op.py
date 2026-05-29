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

"""Tests for the ``gaussian_render_inference_only`` torch op.

Covers output correctness when called with Python-packed scene data, validates
pre-allocated output buffers, and confirms the op's dispatch table properties.
"""

import pytest
import torch


def _make_gaussians(N=100, sh_degree=None, device="cuda"):
    torch.manual_seed(12345)
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01
    opacities = torch.rand(N, device=device)
    if sh_degree is not None and sh_degree >= 0:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device)
    else:
        colors = torch.rand(N, 3, device=device)
    return means, quats, scales, opacities, colors


def _make_camera(device="cuda"):
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 3.0  # camera at z=3 looking at origin
    K = torch.tensor(
        [
            [500.0, 0.0, 128.0],
            [0.0, 500.0, 128.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return viewmat, K


def _pack_scene(
    means, quats, scales, opacities, colors, sh_degree, sh_compression="none"
):
    """Pack scene in Python matching the C++ packing logic."""
    N = means.size(0)
    means_planar = means.t().contiguous()
    inference = torch.empty(N, 8, dtype=torch.float16, device=means.device)
    inference[:, 0:4] = quats.half()
    inference[:, 4:7] = scales.half()
    inference[:, 7:8] = opacities.unsqueeze(1).half()

    sh_deg = -1 if sh_degree is None else sh_degree
    K_sh = colors.size(1) if colors.dim() == 3 else 0
    if sh_deg >= 0 and K_sh == 16:
        if sh_compression == "none":
            colors_packed = colors.half()
        elif sh_compression == "32b":
            colors_packed = colors.contiguous().view(N, 48)
        else:
            colors_packed = colors.half().view(N, 48)
    elif sh_deg >= 0 and K_sh > 0:
        colors_packed = colors.contiguous()
    else:
        c = torch.zeros(N, 4, dtype=torch.float16, device=means.device)
        c[:, 0:3] = colors.half()
        colors_packed = c
    return means_planar, inference, colors_packed


# ------------------------------------------------------------------ #
#  Forward parity: wrapper API vs native op                           #
# ------------------------------------------------------------------ #

_PARITY_CASES = [
    # (sh_degree, sh_compression)
    (None, "none"),  # pre-activated RGB
    (0, "none"),  # SH degree 0
    (1, "none"),  # SH degree 1
    (2, "none"),  # SH degree 2
    (3, "none"),  # SH degree 3, no compression
    (3, "32b"),  # SH degree 3, 32b compression
    (3, "16b"),  # SH degree 3, 16b compression
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sh_degree,sh_compression", _PARITY_CASES)
def test_forward_parity(sh_degree, sh_compression):
    """Python-packed scene and C++-packed scene render bit-exact via gaussian_render_inference_only."""
    from experimental.render.kernels._backend import _C  # noqa: F401
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    width, height = 256, 256
    means, quats, scales, opacities, colors = _make_gaussians(
        N=200, sh_degree=sh_degree
    )
    viewmat, K = _make_camera()

    sh_deg = -1 if sh_degree is None else sh_degree
    _SH_COMPRESSION_MAP = {"none": 0, "32b": 1, "16b": 2}
    sh_compression_mode = _SH_COMPRESSION_MAP[sh_compression]

    # Python-packed path
    means_planar_py, inference_py, colors_packed_py = _pack_scene(
        means, quats, scales, opacities, colors, sh_degree, sh_compression
    )

    with torch.no_grad():
        renders_py, alphas_py = torch.ops.experimental.gaussian_render_inference_only(
            means_planar_py,
            inference_py,
            colors_packed_py,
            viewmat,
            K,
            width,
            height,
            sh_deg,
            16,  # tile_size
            0.01,  # near_plane
            1e10,  # far_plane
            0.0,  # radius_clip
            0.3,  # eps2d
            sh_compression_mode,
            None,  # background
        )

    # C++-packed path (via gsplat_scene_cuda.pack_gaussian_inference_scene)
    mp_cpp, inference_cpp, cp_cpp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, sh_deg, sh_compression_mode
    )

    with torch.no_grad():
        renders_cpp, alphas_cpp = torch.ops.experimental.gaussian_render_inference_only(
            mp_cpp,
            inference_cpp,
            cp_cpp,
            viewmat,
            K,
            width,
            height,
            sh_deg,
            16,
            0.01,
            1e10,
            0.0,
            0.3,
            sh_compression_mode,
            None,
        )

    torch.testing.assert_close(
        renders_cpp, renders_py, atol=0, rtol=0, msg="renders mismatch"
    )
    torch.testing.assert_close(
        alphas_cpp, alphas_py, atol=0, rtol=0, msg="alphas mismatch"
    )


# ------------------------------------------------------------------ #
#  out_renders / out_alphas pre-allocated buffer test                  #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_out_renders_out_alphas():
    """Pre-allocated buffers must be written in-place and match the no-buffer call."""
    from experimental.render.kernels._backend import _C  # noqa: F401

    width, height = 256, 256
    means, quats, scales, opacities, colors = _make_gaussians(N=150, sh_degree=None)
    viewmat, K = _make_camera()
    means_planar, inference, colors_packed = _pack_scene(
        means, quats, scales, opacities, colors, None
    )

    sh_deg = -1
    sh_compression_mode = 0

    # Reference call (no pre-allocated buffers)
    with torch.no_grad():
        renders_ref, alphas_ref = torch.ops.experimental.gaussian_render_inference_only(
            means_planar,
            inference,
            colors_packed,
            viewmat,
            K,
            width,
            height,
            sh_deg,
            16,
            0.01,
            1e10,
            0.0,
            0.3,
            sh_compression_mode,
            None,
        )

    # Pre-allocated buffers
    buf_r = torch.empty(height, width, 3, dtype=torch.float32, device="cuda")
    buf_a = torch.empty(height, width, 1, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        ret_r, ret_a = torch.ops.experimental.gaussian_render_inference_only(
            means_planar,
            inference,
            colors_packed,
            viewmat,
            K,
            width,
            height,
            sh_deg,
            16,
            0.01,
            1e10,
            0.0,
            0.3,
            sh_compression_mode,
            None,
            out_renders=buf_r,
            out_alphas=buf_a,
        )

    # Verify returned tensors share storage with pre-allocated buffers
    assert ret_r.data_ptr() == buf_r.data_ptr(), "out_renders not written in-place"
    assert ret_a.data_ptr() == buf_a.data_ptr(), "out_alphas not written in-place"

    # Verify numerical match
    torch.testing.assert_close(
        ret_r, renders_ref, atol=0, rtol=0, msg="out_renders content mismatch"
    )
    torch.testing.assert_close(
        ret_a, alphas_ref, atol=0, rtol=0, msg="out_alphas content mismatch"
    )


# ------------------------------------------------------------------ #
#  Runtime smoke test: op existence + dispatch table properties        #
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_op_smoke():
    """The new op must exist, have a CUDA dispatch key, and NOT have AutogradCUDA."""
    from experimental.render.kernels._backend import _C  # noqa: F401

    # Confirm the op exists
    assert hasattr(
        torch.ops.experimental, "gaussian_render_inference_only"
    ), "gaussian_render_inference_only op not found"

    # Get the op's overload packet and inspect its dispatch table
    op = torch.ops.experimental.gaussian_render_inference_only
    # The default overload is callable; its _dispatch_cache or _schema
    # gives us dispatch info. We check via the operator's dispatch table.
    schema = op.default._schema
    assert schema is not None, "Op schema not found"

    # Verify CUDA dispatch key is present by running on CUDA data
    # (if it weren't registered, the call would fail with a dispatch error)
    means_planar = torch.zeros(3, 1, device="cuda")
    inference = torch.zeros(1, 8, dtype=torch.float16, device="cuda")
    colors_packed = torch.zeros(1, 4, dtype=torch.float16, device="cuda")
    viewmat = torch.eye(4, device="cuda")
    viewmat[2, 3] = 3.0
    K = torch.tensor(
        [[500.0, 0.0, 64.0], [0.0, 500.0, 64.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )

    # This call exercises the CUDA dispatch key
    renders, alphas = torch.ops.experimental.gaussian_render_inference_only(
        means_planar,
        inference,
        colors_packed,
        viewmat,
        K,
        128,
        128,
        -1,
        16,
        0.01,
        1e10,
        0.0,
        0.3,
        0,
        None,
    )
    assert renders.shape == (128, 128, 3)
    assert alphas.shape == (128, 128, 1)

    # Verify AutogradCUDA is NOT registered by confirming direct op call
    # with grad-tracked inputs runs the forward kernel and produces no grad_fn.
    # If AutogradCUDA were registered, the output would have a grad_fn.
    means_grad = torch.zeros(3, 5, device="cuda", requires_grad=True)
    inference_grad = torch.zeros(5, 8, dtype=torch.float16, device="cuda")
    colors_grad = torch.zeros(5, 4, dtype=torch.float16, device="cuda")
    viewmat_grad = torch.eye(4, device="cuda")
    viewmat_grad[2, 3] = 3.0
    K_grad = torch.tensor(
        [[500.0, 0.0, 64.0], [0.0, 500.0, 64.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )

    # This should succeed (no AutogradCUDA to intercept and fail)
    renders_g, alphas_g = torch.ops.experimental.gaussian_render_inference_only(
        means_grad,
        inference_grad,
        colors_grad,
        viewmat_grad,
        K_grad,
        128,
        128,
        -1,
        16,
        0.01,
        1e10,
        0.0,
        0.3,
        0,
        None,
    )
    # No backward kernel exists, so grad_fn should be None
    assert (
        renders_g.grad_fn is None
    ), "renders has grad_fn — AutogradCUDA should not be registered"
    assert (
        alphas_g.grad_fn is None
    ), "alphas has grad_fn — AutogradCUDA should not be registered"
