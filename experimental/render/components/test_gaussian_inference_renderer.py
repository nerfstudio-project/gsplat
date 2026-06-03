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

"""Tests for :class:`GaussianInferenceRenderer` (stateful Inference renderer).

Covers construction, native half4 RGBT rendering correctness, lifecycle
management (release, context manager), resolution changes, output-buffer
reuse, viewmat/K normalisation, and parity with the stateless
:func:`rasterize_gaussian_inference_scene`.
"""

import math

import pytest
import torch

# Skip entire module if CUDA is unavailable.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = torch.device("cuda:0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_test_scene(sh_degree=3, sh_compression="none"):
    """Create a small synthetic GaussianInferenceScene for testing."""
    N = 1000
    torch.manual_seed(42)
    means = torch.randn(N, 3, device=DEVICE)
    means[:, 2] = means[:, 2].abs() + 2.0
    quats = torch.randn(N, 4, device=DEVICE)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(torch.rand(N, 3, device=DEVICE) * 0.1 - 2.0)
    opacities = torch.sigmoid(torch.rand(N, device=DEVICE))
    K_sh = (sh_degree + 1) ** 2 if sh_degree is not None and sh_degree >= 0 else 0
    if K_sh > 0:
        colors = torch.randn(N, K_sh, 3, device=DEVICE) * 0.1
    else:
        colors = torch.sigmoid(torch.randn(N, 3, device=DEVICE))

    from experimental import GaussianInferenceScene

    return GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=sh_degree if sh_degree is not None and sh_degree >= 0 else None,
        sh_compression=sh_compression,
        id="test_scene",
    )


def make_camera(width=256, height=256):
    """Create a simple pinhole camera looking down +z."""
    focal = width / (2.0 * math.tan(math.radians(25)))
    K = torch.tensor(
        [
            [focal, 0, width / 2.0],
            [0, focal, height / 2.0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    viewmat = torch.eye(4, dtype=torch.float32, device=DEVICE)
    viewmat[2, 3] = -5.0  # camera looking at z=5
    return viewmat, K


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_construction():
    """Create a renderer and verify num_gaussians and is_released."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    renderer = GaussianInferenceRenderer(scene)
    try:
        assert renderer.num_gaussians == 1000
        assert not renderer.is_released
    finally:
        renderer.release()


# ---------------------------------------------------------------------------
# Basic rendering
# ---------------------------------------------------------------------------


def test_render_basic():
    """Render a single frame; verify output shapes and dtype."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 256, 256
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            ret = renderer.render(viewmat=viewmat, K=K, width=W, height=H)

    assert ret.frame.shape == (1, H, W, 4)
    assert ret.frame.dtype == torch.float16
    assert ret.metadata["format"] == "RGBT"
    assert ret.metadata["channels"] == "RGBT"


# ---------------------------------------------------------------------------
# Repeated rendering
# ---------------------------------------------------------------------------


def test_render_repeated():
    """Render 5 frames with slightly different viewmats; all must succeed."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            frame_ptr = None
            for i in range(5):
                vm = viewmat.clone()
                vm[0, 3] += 0.1 * i  # slight translation each frame
                ret = renderer.render(viewmat=vm, K=K, width=W, height=H)
                assert ret.frame.shape == (1, H, W, 4)
                assert ret.frame.isfinite().all()
                if frame_ptr is None:
                    frame_ptr = ret.frame.data_ptr()
                assert ret.frame.data_ptr() == frame_ptr


def test_render_repeated_stable():
    """Rendering the same viewmat twice must produce identical output."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            ret1 = renderer.render(viewmat=viewmat, K=K, width=W, height=H)
            ret2 = renderer.render(viewmat=viewmat, K=K, width=W, height=H)

    assert torch.equal(ret1.frame, ret2.frame)


# ---------------------------------------------------------------------------
# Resolution change
# ---------------------------------------------------------------------------


def test_resize():
    """Render at three different resolutions; verify shapes match each time."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    viewmat, K_256 = make_camera(256, 256)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            for w, h in [(256, 256), (512, 512), (128, 128)]:
                _, K = make_camera(w, h)
                ret = renderer.render(viewmat=viewmat, K=K, width=w, height=h)
                assert ret.frame.shape == (
                    1,
                    h,
                    w,
                    4,
                ), f"Expected (1, {h}, {w}, 4), got {ret.frame.shape}"


# ---------------------------------------------------------------------------
# Consistency with stateless path
# ---------------------------------------------------------------------------


def test_output_consistency_with_stateless():
    """Stateful and stateless renderers must produce identical output."""
    from experimental import (
        GaussianInferenceRenderer,
        rasterize_gaussian_inference_scene,
    )

    scene = make_test_scene()
    W, H = 256, 256
    viewmat, K = make_camera(W, H)

    with torch.inference_mode():
        stateless_ret = rasterize_gaussian_inference_scene(
            scene,
            viewmat=viewmat,
            K=K,
            width=W,
            height=H,
        )

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            stateful_ret = renderer.render(
                viewmat=viewmat,
                K=K,
                width=W,
                height=H,
            )

    stateful_rgb = stateful_ret.frame[..., :3].float()
    stateful_alpha = 1.0 - stateful_ret.frame[..., 3:4].float()

    torch.testing.assert_close(
        stateful_rgb,
        stateless_ret.frame,
        atol=1e-5,
        rtol=1e-4,
        msg="Stateful vs stateless frame mismatch",
    )
    torch.testing.assert_close(
        stateful_alpha,
        stateless_ret.metadata["alpha"],
        atol=1e-5,
        rtol=1e-4,
        msg="Stateful vs stateless alpha mismatch",
    )


# ---------------------------------------------------------------------------
# Output buffer reuse
# ---------------------------------------------------------------------------


def test_out_buffer_reuse():
    """Pre-allocated out buffers keep the same data_ptr across renders."""
    from experimental import GaussianInferenceRenderer, RenderReturn

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    buf = RenderReturn(
        frame=torch.empty(1, H, W, 4, device=DEVICE, dtype=torch.float16),
        metadata={},
    )
    frame_ptr = buf.frame.data_ptr()

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            for _ in range(2):
                ret = renderer.render(
                    viewmat=viewmat,
                    K=K,
                    width=W,
                    height=H,
                    out=buf,
                )
                assert ret is buf
                assert buf.frame.data_ptr() == frame_ptr
                assert buf.metadata["format"] == "RGBT"


# ---------------------------------------------------------------------------
# viewmats / Ks normalisation
# ---------------------------------------------------------------------------


def test_viewmats_ks_normalization():
    """Passing viewmats=[1,4,4] and Ks=[1,3,3] must produce same result as viewmat/K."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            ret_single = renderer.render(
                viewmat=viewmat,
                K=K,
                width=W,
                height=H,
            )
            ret_batch = renderer.render(
                viewmats=viewmat.unsqueeze(0),
                Ks=K.unsqueeze(0),
                width=W,
                height=H,
            )

    torch.testing.assert_close(
        ret_single.frame,
        ret_batch.frame,
        atol=0,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# Lifecycle: release
# ---------------------------------------------------------------------------


def test_release():
    """After release(), is_released is True and render raises RuntimeError."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    renderer = GaussianInferenceRenderer(scene)
    renderer.release()

    assert renderer.is_released

    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="has been released"):
            renderer.render(viewmat=viewmat, K=K, width=W, height=H)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_context_manager():
    """The context manager auto-releases the renderer on exit."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        assert not renderer.is_released
        with torch.inference_mode():
            ret = renderer.render(viewmat=viewmat, K=K, width=W, height=H)
        assert ret.frame.shape == (1, H, W, 4)

    assert renderer.is_released


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------


def test_construction_empty_scene_raises():
    """Constructing from a released (empty) scene raises ValueError."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    scene.release()
    assert scene.is_empty()

    with pytest.raises(ValueError, match="has been released"):
        GaussianInferenceRenderer(scene)


def test_construction_wrong_type_raises():
    """Constructing with a non-scene object raises TypeError."""
    from experimental import GaussianInferenceRenderer

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        GaussianInferenceRenderer("not_a_scene")


# ---------------------------------------------------------------------------
# SH degree override
# ---------------------------------------------------------------------------


def test_sh_degree_override():
    """Rendering with sh_degree=0 (lower than scene's 3) must not crash."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene(sh_degree=3)
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            ret = renderer.render(
                viewmat=viewmat,
                K=K,
                width=W,
                height=H,
                sh_degree=0,
            )

    assert ret.frame.shape == (1, H, W, 4)
    assert ret.frame.isfinite().all()


def test_sh_compression_mode_override_rejected():
    """SH compression is fixed by the scene used to construct the renderer."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene(sh_degree=3, sh_compression="32b")
    W, H = 128, 128
    viewmat, K = make_camera(W, H)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            with pytest.raises(TypeError, match="does not support sh_compression_mode"):
                renderer.render(
                    viewmat=viewmat,
                    K=K,
                    width=W,
                    height=H,
                    sh_compression_mode=0,
                )


# ---------------------------------------------------------------------------
# Background colour
# ---------------------------------------------------------------------------


def test_background():
    """A red background should tint transparent regions red."""
    from experimental import GaussianInferenceRenderer

    scene = make_test_scene()
    W, H = 128, 128
    viewmat, K = make_camera(W, H)
    bg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=DEVICE)

    with GaussianInferenceRenderer(scene) as renderer:
        with torch.inference_mode():
            ret = renderer.render(
                viewmat=viewmat,
                K=K,
                width=W,
                height=H,
                background=bg,
            )

    alpha = 1.0 - ret.frame[..., 3:4].float()  # [1, H, W, 1]
    # Find pixels where alpha < 0.5 (transparent regions)
    transparent_mask = alpha[0, :, :, 0] < 0.5
    if transparent_mask.any():
        # In transparent regions the red channel should dominate
        transparent_pixels = ret.frame[0, ..., :3].float()[transparent_mask]  # [K, 3]
        red_channel = transparent_pixels[:, 0]
        green_channel = transparent_pixels[:, 1]
        blue_channel = transparent_pixels[:, 2]
        assert (
            red_channel.mean() > green_channel.mean()
        ), "Red background not visible in transparent regions"
        assert (
            red_channel.mean() > blue_channel.mean()
        ), "Red background not visible in transparent regions"
