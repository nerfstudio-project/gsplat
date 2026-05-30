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

"""Integration-level verification suite for the Inference render path.

Covers:
1. Full-matrix cross-path (Inference vs direct rasterization) parity at N=1000+.
2. Vanilla-branch rejection (render_scene raises TypeError for GaussianScene).
3. Standard rasterization regression (shapes, finiteness).
4. Dispatcher/autograd verification (dispatch keys, grad-mode gate).
5. Viewer smoke test (toggle Inference on/off).
6. Benchmark timing comparison (informational, not pass/fail).
"""

import math
import statistics

import pytest
import torch
import torch.nn as nn

_CUDA = torch.cuda.is_available()
skipif_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required")

W, H = 256, 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_psnr(a, b):
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    peak = max(a.abs().max().item(), b.abs().max().item(), 1.0)
    return 10 * math.log10(peak**2 / mse)


def _make_test_gaussians(N=100, sh_degree=None, device="cuda"):
    """Create random Gaussians in *log/logit* space for GaussianScene."""
    torch.manual_seed(12345)
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01  # log-space
    opacities = torch.randn(N, device=device)  # logit-space
    if sh_degree is not None:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device) * 0.1
    else:
        colors = torch.rand(N, 3, device=device)
    return means, quats, scales, opacities, colors


def _make_camera(device="cuda"):
    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    viewmat[2, 3] = 5.0
    K = torch.tensor(
        [[500.0, 0, 128.0], [0, 500.0, 128.0], [0, 0, 1]],
        device=device,
        dtype=torch.float32,
    )
    return viewmat, K


def _make_gaussian_scene(
    N=100,
    sh_degree=None,
    device="cuda",
    *,
    requires_grad=False,
):
    """Build a GaussianScene with raw (log/logit-space) splats."""
    from libs.scene import GaussianScene

    means, quats, scales, opacities, colors = _make_test_gaussians(N, sh_degree, device)
    splats = nn.ParameterDict(
        {
            "means": nn.Parameter(means, requires_grad=requires_grad),
            "quats": nn.Parameter(quats, requires_grad=requires_grad),
            "scales": nn.Parameter(scales, requires_grad=requires_grad),
            "opacities": nn.Parameter(opacities, requires_grad=requires_grad),
            "colors": nn.Parameter(colors, requires_grad=requires_grad),
        }
    )
    return GaussianScene.from_splats(splats, id="test")


def _make_inference_scene(
    N=100, sh_degree=None, sh_compression="none", device="cuda", *, requires_grad=False
):
    """Build a GaussianInferenceScene from a GaussianScene."""
    from experimental import GaussianInferenceScene

    gs = _make_gaussian_scene(N, sh_degree, device, requires_grad=requires_grad)
    return GaussianInferenceScene.from_gaussian_scene(
        gs, id="test", sh_compression=sh_compression
    )


def _common_request(device="cuda"):
    """Return the minimal valid Inference request dict."""
    viewmat, K = _make_camera(device)
    return dict(viewmat=viewmat, K=K, width=W, height=H)


# ======================================================================
# 1. Full Integration Cross-Path Parity (Full Matrix)
# ======================================================================

# Compression modes "32b" and "16b" are only valid for SH degree 3.
_FULL_PARITY_MATRIX = [
    (None, "none"),
    (0, "none"),
    (1, "none"),
    (2, "none"),
    (3, "none"),
    (3, "32b"),
    (3, "16b"),
]


@skipif_no_cuda
@pytest.mark.parametrize("sh_degree,sh_compression", _FULL_PARITY_MATRIX)
def test_full_cross_path_parity(sh_degree, sh_compression):
    """Inference and vanilla (direct rasterization) paths produce close results at N=1000+ (PSNR >= 30 dB)."""
    from experimental import render_scene, GaussianInferenceScene
    from gsplat.rendering import rasterization

    N = 1000
    gs = _make_gaussian_scene(N=N, sh_degree=sh_degree)
    inference = GaussianInferenceScene.from_gaussian_scene(
        gs, id="parity_inference", sh_compression=sh_compression
    )

    viewmat, K = _make_camera()

    # Vanilla path via direct rasterization (render_scene no longer accepts GaussianScene)
    splats = gs.splats
    vanilla_req = dict(viewmats=viewmat[None], Ks=K[None], width=W, height=H)
    if sh_degree is not None:
        vanilla_req["sh_degree"] = sh_degree
    with torch.no_grad():
        vanilla_renders, vanilla_alphas, _info = rasterization(
            splats["means"],
            splats["quats"],
            splats["scales"].exp(),
            splats["opacities"].sigmoid(),
            splats.get("colors", splats.get("sh0")),
            **vanilla_req,
        )

    # Inference path
    inference_req = dict(viewmat=viewmat, K=K, width=W, height=H)
    with torch.inference_mode():
        inference_ret = render_scene(inference, **inference_req)

    # Verify inference render_path tag
    assert (
        inference_ret.metadata["render_path"] == "inference"
    ), "Inference path not tagged correctly"

    # PSNR check
    psnr = _compute_psnr(vanilla_renders, inference_ret.frame)
    # 16b SH compression drops CoCg for higher-order terms (k≥1), making it
    # inherently lossy for scenes with significant HO color variation; use a
    # lower floor. 32b and no-compression paths target ≥ 40 dB.
    psnr_floor = 20.0 if sh_compression == "16b" else 40.0
    assert psnr >= psnr_floor, (
        f"PSNR {psnr:.1f} dB below {psnr_floor} dB for "
        f"sh_degree={sh_degree}, sh_compression={sh_compression}"
    )

    # Alpha max-abs tolerance
    alpha_diff = (vanilla_alphas - inference_ret.metadata["alpha"]).abs()
    assert alpha_diff.max().item() < 0.1, (
        f"Alpha max diff {alpha_diff.max().item():.4f} >= 0.1 for "
        f"sh_degree={sh_degree}, sh_compression={sh_compression}"
    )


# ======================================================================
# 2. Vanilla-Branch Rejection (render_scene raises TypeError)
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize("sh_degree", [None, 0, 1, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_vanilla_passthrough_raises_typeerror(sh_degree, packed):
    """render_scene(gaussian_scene, ...) raises TypeError (vanilla path removed)."""
    from experimental import render_scene

    gs = _make_gaussian_scene(N=500, sh_degree=sh_degree)
    viewmat, K = _make_camera()
    viewmats = viewmat[None]
    Ks = K[None]
    req = dict(
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        render_mode="RGB",
        packed=packed,
    )
    if sh_degree is not None:
        req["sh_degree"] = sh_degree

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        render_scene(gs, **req)


# ======================================================================
# 3. Standard Rasterization Unchanged
# ======================================================================


@skipif_no_cuda
def test_standard_rasterization_regression():
    """Direct gsplat.rasterization(...) produces correct shapes and finite values."""
    from gsplat.rendering import rasterization

    N = 200
    torch.manual_seed(9999)
    means = torch.randn(N, 3, device="cuda")
    quats = torch.randn(N, 4, device="cuda")
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = (torch.rand(N, 3, device="cuda") * 0.1 + 0.01).exp()
    opacities = torch.rand(N, device="cuda")
    colors = torch.rand(N, 3, device="cuda")

    viewmat, K = _make_camera()
    viewmats = viewmat[None]
    Ks = K[None]

    with torch.no_grad():
        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
        )

    assert renders.shape == (1, H, W, 3), f"Unexpected shape: {renders.shape}"
    assert alphas.shape == (1, H, W, 1), f"Unexpected alpha shape: {alphas.shape}"
    assert renders.isfinite().all(), "renders contain non-finite values"
    assert alphas.isfinite().all(), "alphas contain non-finite values"
    assert isinstance(info, dict), "info should be a dict"


# ======================================================================
# 4. Dispatcher/Autograd Verification
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
class TestGradModeGateIntegration:
    """Grad-mode gate verification across both entry points."""

    def _get_fn(self, entry):
        import experimental as exp

        return getattr(exp, entry)

    def test_inference_mode_ok(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=100)
        req = _common_request()
        with torch.inference_mode():
            ret = fn(inference, **req)
        assert ret.frame.shape == (1, H, W, 3)

    def test_no_grad_ok(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=100)
        req = _common_request()
        with torch.no_grad():
            ret = fn(inference, **req)
        assert ret.frame.shape == (1, H, W, 3)

    def test_bare_grad_raises(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=100)
        req = _common_request()
        with pytest.raises(RuntimeError, match="requires torch.inference_mode"):
            fn(inference, **req)


@skipif_no_cuda
def test_dispatch_keys_rasterize_op():
    """gaussian_render_inference_only has CUDA dispatch, no AutogradCUDA."""
    from experimental.render.kernels._backend import _C  # noqa: F401

    assert hasattr(torch.ops.experimental, "gaussian_render_inference_only")

    op = torch.ops.experimental.gaussian_render_inference_only
    schema = op.default._schema
    assert schema is not None, "Op schema not found"

    # Verify CUDA dispatch works: if not registered the call below would fail.
    means_planar = torch.zeros(3, 1, device="cuda")
    inference = torch.zeros(1, 8, dtype=torch.float16, device="cuda")
    colors_packed = torch.zeros(1, 4, dtype=torch.float16, device="cuda")
    viewmat = torch.eye(4, device="cuda")
    viewmat[2, 3] = 3.0
    K = torch.tensor(
        [[500.0, 0.0, 64.0], [0.0, 500.0, 64.0], [0.0, 0.0, 1.0]],
        device="cuda",
    )

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

    # AutogradCUDA absent: grad-tracked input produces no grad_fn.
    means_grad = torch.zeros(3, 5, device="cuda", requires_grad=True)
    inference_g = torch.zeros(5, 8, dtype=torch.float16, device="cuda")
    cp_g = torch.zeros(5, 4, dtype=torch.float16, device="cuda")
    renders_g, _ = torch.ops.experimental.gaussian_render_inference_only(
        means_grad,
        inference_g,
        cp_g,
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
    assert (
        renders_g.grad_fn is None
    ), "AutogradCUDA should NOT be registered for gaussian_render_inference_only"


@skipif_no_cuda
def test_dispatch_keys_pack_op():
    """pack_gaussian_inference_scene is callable via gsplat_scene_cuda and produces no grad_fn."""
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    assert callable(_SCENE_CUDA.pack_gaussian_inference_scene)

    # Verify CUDA dispatch
    means = torch.randn(10, 3, device="cuda")
    quats = torch.randn(10, 4, device="cuda")
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(10, 3, device="cuda") * 0.1 + 0.01
    opacities = torch.rand(10, device="cuda")
    colors = torch.rand(10, 3, device="cuda")

    mp, inference, cp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, -1, 0
    )
    assert mp.shape == (3, 10)
    assert inference.dtype == torch.float16

    # No grad_fn on outputs (torch::NoGradGuard in C++)
    means_g = means.clone().requires_grad_(True)
    mp_g, inference_g, cp_g = _SCENE_CUDA.pack_gaussian_inference_scene(
        means_g, quats, scales, opacities, colors, -1, 0
    )
    assert (
        mp_g.grad_fn is None
    ), "pack_gaussian_inference_scene outputs should not have grad_fn"


# ======================================================================
# 4b. Request-subset validation re-run
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
@pytest.mark.parametrize(
    "bad_kwarg,match",
    [
        ({"render_mode": "D"}, "render_mode='RGB' only"),
        ({"with_ut": True}, "Inference branch does not support with_ut"),
        ({"absgrad": True}, "Inference branch does not support absgrad"),
        ({"tile_size": 32}, "tile_size in {8, 16}"),
        ({"sh_degree": 3}, "sh_degree/sh_compression_mode"),
        (
            {"backgrounds": torch.zeros(1, 3)},
            "unexpected keyword argument 'backgrounds'",
        ),
        ({"bogus_kwarg": 1}, "unexpected keyword argument 'bogus_kwarg'"),
    ],
)
def test_request_validation_integration(entry, bad_kwarg, match):
    """Unsupported Inference request features are rejected at integration level."""
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=100)
    req = _common_request()
    req.update(bad_kwarg)

    with torch.inference_mode():
        with pytest.raises(TypeError, match=match):
            fn(inference, **req)


# ======================================================================
# 4c. out= buffer contract re-run
# ======================================================================


@skipif_no_cuda
def test_out_buffer_numerical_match_integration():
    """out= produces same numbers as freshly-allocated path."""
    from experimental import rasterize_gaussian_inference_scene, RenderReturn

    inference = _make_inference_scene(N=500)
    req = _common_request()

    with torch.inference_mode():
        fresh = rasterize_gaussian_inference_scene(inference, **req)

    buf = RenderReturn(
        frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
        metadata={"alpha": torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32)},
    )

    with torch.inference_mode():
        reused = rasterize_gaussian_inference_scene(inference, out=buf, **req)

    assert reused is buf
    torch.testing.assert_close(reused.frame, fresh.frame, atol=0, rtol=0)
    torch.testing.assert_close(
        reused.metadata["alpha"], fresh.metadata["alpha"], atol=0, rtol=0
    )


@skipif_no_cuda
def test_out_buffer_identity_stability_integration():
    """out= data_ptr stability across multiple calls."""
    from experimental import rasterize_gaussian_inference_scene, RenderReturn

    inference = _make_inference_scene(N=100)
    req = _common_request()

    buf = RenderReturn(
        frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
        metadata={"alpha": torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32)},
    )
    frame_ptr = buf.frame.data_ptr()

    with torch.inference_mode():
        for i in range(3):
            ret = rasterize_gaussian_inference_scene(inference, out=buf, **req)
            assert ret is buf
            assert buf.frame.data_ptr() == frame_ptr


@skipif_no_cuda
def test_out_buffer_vanilla_rejection_integration():
    """render_scene(GaussianScene, out=...) raises TypeError."""
    from experimental import render_scene, RenderReturn

    gs = _make_gaussian_scene(N=50)
    viewmat, K = _make_camera()

    buf = RenderReturn(
        frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
        metadata={},
    )

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        render_scene(gs, out=buf, viewmats=viewmat[None], Ks=K[None], width=W, height=H)


# ======================================================================
# 5. Viewer Smoke Test
# ======================================================================


@skipif_no_cuda
def test_viewer_toggle_inference_on():
    """Inference on: GaussianInferenceScene -> render_scene -> valid RenderReturn."""
    from experimental import GaussianInferenceScene, render_scene, RenderReturn

    gs = _make_gaussian_scene(N=200, sh_degree=3)
    inference = GaussianInferenceScene.from_gaussian_scene(
        gs, id="viewer_inference", sh_compression="32b"
    )
    viewmat, K = _make_camera()

    with torch.inference_mode():
        ret = render_scene(inference, viewmat=viewmat, K=K, width=W, height=H)

    assert isinstance(ret, RenderReturn)
    assert ret.frame.shape == (1, H, W, 3)
    assert ret.metadata["alpha"].shape == (1, H, W, 1)
    assert ret.metadata["render_path"] == "inference"
    assert ret.frame.isfinite().all()


@skipif_no_cuda
def test_viewer_toggle_inference_off_raises():
    """Inference off: vanilla GaussianScene -> render_scene -> TypeError (vanilla path removed)."""
    from experimental import render_scene

    gs = _make_gaussian_scene(N=200)
    viewmat, K = _make_camera()

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        render_scene(gs, viewmats=viewmat[None], Ks=K[None], width=W, height=H)


@skipif_no_cuda
def test_viewer_both_paths_produce_expected_shapes():
    """Both viewer paths (direct rasterization + inference) produce matching output shapes."""
    from experimental import GaussianInferenceScene, render_scene
    from gsplat.rendering import rasterization

    gs = _make_gaussian_scene(N=300, sh_degree=1)
    viewmat, K = _make_camera()

    # Vanilla via direct rasterization (render_scene no longer accepts GaussianScene)
    splats = gs.splats
    with torch.no_grad():
        vanilla_renders, vanilla_alphas, _info = rasterization(
            splats["means"],
            splats["quats"],
            splats["scales"].exp(),
            splats["opacities"].sigmoid(),
            splats.get("colors", splats.get("sh0")),
            viewmats=viewmat[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=1,
        )

    # Inference
    inference = GaussianInferenceScene.from_gaussian_scene(
        gs, id="viewer_both", sh_compression="none"
    )
    with torch.inference_mode():
        inference_ret = render_scene(inference, viewmat=viewmat, K=K, width=W, height=H)

    # Both shapes match
    assert vanilla_renders.shape == inference_ret.frame.shape == (1, H, W, 3)
    assert vanilla_alphas.shape == inference_ret.metadata["alpha"].shape

    # Inference tag
    assert inference_ret.metadata["render_path"] == "inference"


@skipif_no_cuda
def test_unsupported_scene_type_integration():
    """render_scene rejects unknown scene types."""
    from experimental import render_scene

    class FakeScene:
        pass

    req = _common_request()
    with torch.inference_mode():
        with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
            render_scene(FakeScene(), **req)


# ======================================================================
# 6. Benchmark Timing Comparison (informational)
# ======================================================================


@skipif_no_cuda
def test_benchmark_timing_comparison():
    """Collect and print timing data for Inference vs vanilla paths.

    This is NOT a pass/fail test. It runs a small benchmark (N=10000,
    10 frames) and reports median frame times in ms.
    """
    from experimental import (
        GaussianInferenceScene,
        rasterize_gaussian_inference_scene,
        render_scene,
    )
    from gsplat.rendering import rasterization

    N = 10000
    NUM_FRAMES = 10
    device = "cuda"

    # Build activated Gaussians for both paths
    torch.manual_seed(42)
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales_raw = torch.rand(N, 3, device=device) * 0.1 + 0.01  # log-space
    opacities_raw = torch.randn(N, device=device)  # logit-space
    colors = torch.rand(N, 3, device=device)

    scales_act = scales_raw.exp()
    opacities_act = opacities_raw.sigmoid()

    viewmat, K = _make_camera()

    # Build Inference scene (from activated tensors)
    inference = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales_act,
        opacities_act,
        colors,
        sh_degree=None,
        sh_compression="none",
        id="bench",
    )

    # Helper: time a rendering closure for NUM_FRAMES
    def _bench(render_fn, label):
        # Warmup
        for _ in range(3):
            render_fn()
        torch.cuda.synchronize()

        times_ms = []
        for _ in range(NUM_FRAMES):
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            render_fn()
            end_ev.record()
            torch.cuda.synchronize()
            times_ms.append(start_ev.elapsed_time(end_ev))

        median = statistics.median(times_ms)
        return median, times_ms

    # 1. Inference via rasterize_gaussian_inference_scene
    def _inference_functional():
        with torch.inference_mode():
            rasterize_gaussian_inference_scene(
                inference, viewmat=viewmat, K=K, width=W, height=H
            )

    med_func, _ = _bench(_inference_functional, "Inference functional")

    # 2. Inference via render_scene
    def _inference_unified():
        with torch.inference_mode():
            render_scene(inference, viewmat=viewmat, K=K, width=W, height=H)

    med_unified, _ = _bench(_inference_unified, "Inference unified")

    # 3. Vanilla via gsplat.rasterization under no_grad
    viewmats = viewmat[None]
    Ks = K[None]

    def _vanilla():
        with torch.no_grad():
            rasterization(
                means=means,
                quats=quats,
                scales=scales_act,
                opacities=opacities_act,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=W,
                height=H,
            )

    med_vanilla, _ = _bench(_vanilla, "Vanilla")

    # Print results (visible in pytest -v -s output)
    print("\n" + "=" * 60)
    print("  Inference Benchmark Timing (N=10000, 10 frames)")
    print("=" * 60)
    print(
        f"  Inference functional  (rasterize_gaussian_inference_scene) : {med_func:.3f} ms"
    )
    print(
        f"  Inference unified     (render_scene)                 : {med_unified:.3f} ms"
    )
    print(f"  Vanilla         (gsplat.rasterization)         : {med_vanilla:.3f} ms")
    print("=" * 60)

    # No assertion -- this is informational only.
    # We just verify the test ran to completion without errors.
