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

"""Tests for the ``gsplat.experimental`` render surface.

Covers ``RenderReturn``, ``rasterize_gaussian_inference_scene``, and ``render_scene``:
grad-mode gating, camera normalisation, out= buffer contract, request
validation, dispatch tagging, and cross-path parity.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

_CUDA = torch.cuda.is_available()
skipif_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H = 256, 256


def _make_test_gaussians(N=100, sh_degree=None, device="cuda"):
    """Create random Gaussians in *log/logit* space for GaussianScene."""
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
# 1. Vanilla-branch passthrough parity
# ======================================================================


@skipif_no_cuda
def test_vanilla_passthrough_raises_typeerror():
    """render_scene(gaussian_scene, ...) raises TypeError (vanilla path removed)."""
    from experimental import render_scene

    gs = _make_gaussian_scene(N=200)
    viewmat, K = _make_camera()
    viewmats = viewmat[None]
    Ks = K[None]
    req = dict(viewmats=viewmats, Ks=Ks, width=W, height=H)

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        render_scene(gs, **req)


# ======================================================================
# 2. Info-key collision (alpha / render_path overwrite)
# ======================================================================


@skipif_no_cuda
def test_info_key_collision_gaussian_scene_raises():
    """render_scene rejects GaussianScene (vanilla path removed)."""
    from experimental import render_scene

    gs = _make_gaussian_scene(N=50)
    viewmat, K = _make_camera()
    req = dict(viewmats=viewmat[None], Ks=K[None], width=W, height=H)

    with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
        render_scene(gs, **req)


# ======================================================================
# 3. Inference branch dispatch + tag
# ======================================================================


@skipif_no_cuda
def test_inference_dispatch_and_tag():
    """render_scene(inference_scene, ...) returns RenderReturn with correct tag and shapes."""
    from experimental import render_scene, RenderReturn

    inference = _make_inference_scene(N=200)
    req = _common_request()

    with torch.inference_mode():
        ret = render_scene(inference, **req)

    assert isinstance(ret, RenderReturn)
    assert ret.metadata["render_path"] == "inference"
    assert ret.frame.shape == (1, H, W, 3)
    assert ret.metadata["alpha"].shape == (1, H, W, 1)


# ======================================================================
# 4. Grad-mode gate
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
class TestGradModeGate:
    """Grad-mode gate over both entry points."""

    def _get_fn(self, entry):
        import experimental as exp

        return getattr(exp, entry)

    def test_inference_mode_pass(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        req = _common_request()
        with torch.inference_mode():
            ret = fn(inference, **req)
        assert ret.frame.shape[0] == 1

    def test_no_grad_pass(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        req = _common_request()
        with torch.no_grad():
            ret = fn(inference, **req)
        assert ret.frame.shape[0] == 1

    def test_bare_grad_raises(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        req = _common_request()
        # Spy on op to verify it is NOT called
        with patch.object(
            torch.ops.experimental, "gaussian_render_inference_only", wraps=None
        ) as mock_op:
            with pytest.raises(RuntimeError, match="requires torch.inference_mode"):
                fn(inference, **req)
            mock_op.assert_not_called()


# ======================================================================
# 5. Direct op layer unguarded
# ======================================================================


@skipif_no_cuda
def test_direct_op_unguarded():
    """Direct torch.ops.experimental.gaussian_render_inference_only runs with grad enabled."""
    from experimental.render.kernels._backend import _C  # noqa: F401

    inference = _make_inference_scene(N=50)
    viewmat, K = _make_camera()

    # no_grad / inference_mode not required for the raw op
    renders, alphas = torch.ops.experimental.gaussian_render_inference_only(
        inference.means_planar,
        inference.qso_packed,
        inference.colors_packed,
        viewmat,
        K,
        W,
        H,
        inference.sh_degree,
        16,
        0.01,
        1e10,
        0.0,
        0.3,
        inference.sh_compression_mode,
        None,
    )
    assert renders.shape == (H, W, 3)
    assert alphas.shape == (H, W, 1)
    assert renders.grad_fn is None
    assert alphas.grad_fn is None


# ======================================================================
# 6. Inference request-subset validation, both entry points
# ======================================================================


_UNSUPPORTED_FEATURES = [
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
]


@skipif_no_cuda
@pytest.mark.parametrize("feature", _UNSUPPORTED_FEATURES)
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_unsupported_feature_raises(feature, entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req[feature] = True

    with torch.inference_mode():
        with pytest.raises(
            TypeError, match=f"Inference branch does not support {feature}"
        ):
            fn(inference, **req)


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_render_mode_non_rgb_raises(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["render_mode"] = "D"

    with torch.inference_mode():
        with pytest.raises(TypeError, match="render_mode='RGB' only"):
            fn(inference, **req)


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_camera_model_non_pinhole_raises(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["camera_model"] = "ortho"

    with torch.inference_mode():
        with pytest.raises(TypeError, match="camera_model='pinhole' only"):
            fn(inference, **req)


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_tile_size_invalid_raises(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["tile_size"] = 32

    with torch.inference_mode():
        with pytest.raises(TypeError, match="tile_size in {8, 16}"):
            fn(inference, **req)


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_sh_degree_override_raises(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["sh_degree"] = 3

    with torch.inference_mode():
        with pytest.raises(TypeError, match="sh_degree/sh_compression_mode"):
            fn(inference, **req)


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_unknown_kwarg_raises(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["bogus_arg"] = 42

    with torch.inference_mode():
        with pytest.raises(TypeError, match="unexpected keyword argument 'bogus_arg'"):
            fn(inference, **req)


# ======================================================================
# 7. Camera-shape normalization, both entry points
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
class TestCameraNormalization:
    def _get_fn(self, entry):
        import experimental as exp

        return getattr(exp, entry)

    def test_viewmats_1x4x4_normalised(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        viewmat, K = _make_camera()
        req = dict(viewmats=viewmat[None], Ks=K[None], width=W, height=H)
        with torch.inference_mode():
            ret = fn(inference, **req)
        assert ret.frame.shape == (1, H, W, 3)

    def test_viewmats_2x4x4_rejected(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        viewmat, K = _make_camera()
        req = dict(
            viewmats=viewmat[None].expand(2, -1, -1).contiguous(),
            Ks=K[None],
            width=W,
            height=H,
        )
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="leading dim"):
                fn(inference, **req)

    def test_both_viewmat_viewmats_rejected(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        viewmat, K = _make_camera()
        req = dict(viewmat=viewmat, viewmats=viewmat[None], K=K, width=W, height=H)
        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="not both"):
                fn(inference, **req)

    def test_neither_viewmat_viewmats_rejected(self, entry):
        fn = self._get_fn(entry)
        inference = _make_inference_scene(N=50)
        _, K = _make_camera()
        req = dict(K=K, width=W, height=H)
        with torch.inference_mode():
            with pytest.raises(
                RuntimeError, match="exactly one of viewmat or viewmats"
            ):
                fn(inference, **req)


# ======================================================================
# 8. backgrounds plural rejected
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize(
    "entry",
    ["rasterize_gaussian_inference_scene", "render_scene"],
)
def test_backgrounds_plural_rejected(entry):
    import experimental as exp

    fn = getattr(exp, entry)
    inference = _make_inference_scene(N=10)
    req = _common_request()
    req["backgrounds"] = torch.zeros(1, 3, device="cuda")

    with torch.inference_mode():
        with pytest.raises(
            TypeError, match="unexpected keyword argument 'backgrounds'"
        ):
            fn(inference, **req)


# ======================================================================
# 9. Non-GaussianInferenceScene to direct entry point
# ======================================================================


@skipif_no_cuda
def test_non_inference_scene_to_direct_raises():
    from experimental import rasterize_gaussian_inference_scene

    gs = _make_gaussian_scene(N=10)
    req = _common_request()

    with torch.inference_mode():
        with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
            rasterize_gaussian_inference_scene(gs, **req)


@skipif_no_cuda
def test_non_inference_scene_string_raises():
    from experimental import rasterize_gaussian_inference_scene

    req = _common_request()

    with torch.inference_mode():
        with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
            rasterize_gaussian_inference_scene("not_a_scene", **req)


# ======================================================================
# 10. Render on empty/released Inference scene
# ======================================================================


@skipif_no_cuda
def test_render_empty_scene_raises():
    from experimental import rasterize_gaussian_inference_scene

    inference = _make_inference_scene(N=50)
    inference.release()
    req = _common_request()

    with torch.inference_mode():
        with patch.object(
            torch.ops.experimental, "gaussian_render_inference_only"
        ) as mock_op:
            with pytest.raises(ValueError, match="has been released"):
                rasterize_gaussian_inference_scene(inference, **req)
            mock_op.assert_not_called()


@skipif_no_cuda
def test_render_empty_scene_via_render_scene():
    from experimental import render_scene

    inference = _make_inference_scene(N=50)
    inference.release()
    req = _common_request()

    with torch.inference_mode():
        with patch.object(
            torch.ops.experimental, "gaussian_render_inference_only"
        ) as mock_op:
            with pytest.raises(ValueError, match="has been released"):
                render_scene(inference, **req)
            mock_op.assert_not_called()


# ======================================================================
# 11. Unsupported Scene subclass
# ======================================================================


@skipif_no_cuda
def test_unsupported_scene_type():
    from experimental import render_scene

    class FakeScene:
        pass

    req = _common_request()
    with torch.inference_mode():
        with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
            render_scene(FakeScene(), **req)


# ======================================================================
# 12. out= buffer contract suite
# ======================================================================


@skipif_no_cuda
class TestOutBufferContract:
    """Comprehensive out= buffer tests."""

    def test_numerical_match(self):
        """out= produces same numbers as freshly-allocated path."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=200)
        req = _common_request()

        with torch.inference_mode():
            fresh = rasterize_gaussian_inference_scene(inference, **req)

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={
                "alpha": torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32)
            },
        )

        with torch.inference_mode():
            reused = rasterize_gaussian_inference_scene(inference, out=buf, **req)

        torch.testing.assert_close(reused.frame, fresh.frame, atol=0, rtol=0)
        torch.testing.assert_close(
            reused.metadata["alpha"],
            fresh.metadata["alpha"],
            atol=0,
            rtol=0,
        )

    def test_identity_guarantee(self):
        """Returned object *is* the same ``out`` instance across N=3 calls with data_ptr stability."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=50)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={
                "alpha": torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32)
            },
        )
        frame_ptr = buf.frame.data_ptr()

        with torch.inference_mode():
            alpha_ptr = None
            for i in range(3):
                ret = rasterize_gaussian_inference_scene(inference, out=buf, **req)
                assert ret is buf
                assert buf.frame.data_ptr() == frame_ptr
                if i > 0:
                    # alpha data_ptr stable after first call
                    assert buf.metadata["alpha"].data_ptr() == alpha_ptr
                alpha_ptr = buf.metadata["alpha"].data_ptr()

    def test_metadata_refresh(self):
        """Old metadata keys are cleared when out= is provided."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=50)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={
                "stale_key": "should_vanish",
                "alpha": torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32),
            },
        )

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(inference, out=buf, **req)

        assert "stale_key" not in ret.metadata
        assert "alpha" in ret.metadata

    def test_lazy_alpha_alloc(self):
        """When out= has no alpha in metadata, alpha is newly allocated."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=50)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={},
        )

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(inference, out=buf, **req)

        assert "alpha" in ret.metadata
        assert ret.metadata["alpha"].shape == (1, H, W, 1)

    def test_shape_mismatch_raises(self):
        """Wrong frame shape raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H + 1, W, 3, device="cuda", dtype=torch.float32),
            metadata={},
        )

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="out.frame expected shape"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_dtype_mismatch_raises(self):
        """Wrong frame dtype raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float16),
            metadata={},
        )

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="out.frame expected shape"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_device_mismatch_raises(self):
        """CPU frame with CUDA viewmat raises ValueError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, dtype=torch.float32),  # CPU
            metadata={},
        )

        with torch.inference_mode():
            with pytest.raises(ValueError, match="out buffer is on .* but scene is on"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_non_contiguous_raises(self):
        """Non-contiguous frame raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        # Create non-contiguous by transposing
        frame = torch.empty(1, 3, H, W, device="cuda", dtype=torch.float32).permute(
            0, 2, 3, 1
        )
        assert not frame.is_contiguous()

        buf = RenderReturn(frame=frame, metadata={})

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="out.frame expected shape"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_grad_tracked_buffer_raises(self):
        """Frame with requires_grad raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(
                1, H, W, 3, device="cuda", dtype=torch.float32
            ).requires_grad_(True),
            metadata={},
        )

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="must not be grad-tracked"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_grad_tracked_alpha_raises(self):
        """Alpha with requires_grad raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={
                "alpha": torch.empty(
                    1, H, W, 1, device="cuda", dtype=torch.float32
                ).requires_grad_(True)
            },
        )

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="must not be grad-tracked"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_vanilla_rejects_out(self):
        """render_scene with GaussianScene and out= raises TypeError."""
        from experimental import render_scene, RenderReturn

        gs = _make_gaussian_scene(N=10)
        viewmat, K = _make_camera()
        req = dict(viewmats=viewmat[None], Ks=K[None], width=W, height=H)

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={},
        )

        with pytest.raises(TypeError, match="requires a GaussianInferenceScene"):
            render_scene(gs, out=buf, **req)

    def test_alpha_shape_mismatch_raises(self):
        """Wrong alpha shape in metadata raises RuntimeError."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        buf = RenderReturn(
            frame=torch.empty(1, H, W, 3, device="cuda", dtype=torch.float32),
            metadata={
                "alpha": torch.empty(
                    1, H, W, 3, device="cuda", dtype=torch.float32
                )  # wrong last dim
            },
        )

        with torch.inference_mode():
            with pytest.raises(RuntimeError, match="out.metadata\\['alpha'\\]"):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)

    def test_metadata_unchanged_on_failure(self):
        """Metadata is not corrupted when validation fails before the op runs."""
        from experimental import rasterize_gaussian_inference_scene, RenderReturn

        inference = _make_inference_scene(N=10)
        req = _common_request()

        sentinel = object()
        original_alpha = torch.empty(1, H, W, 1, device="cuda", dtype=torch.float32)
        original_metadata = {"sentinel": sentinel, "alpha": original_alpha}

        # Shape mismatch failure (H+1 instead of H)
        buf = RenderReturn(
            frame=torch.empty(1, H + 1, W, 3, device="cuda", dtype=torch.float32),
            metadata=dict(original_metadata),
        )
        with torch.inference_mode():
            with pytest.raises(RuntimeError):
                rasterize_gaussian_inference_scene(inference, out=buf, **req)
        assert buf.metadata.get("sentinel") is sentinel
        assert "alpha" in buf.metadata


# ======================================================================
# 13. Cross-path render parity
# ======================================================================

_CROSS_PARITY_CASES = [
    # (sh_degree, sh_compression)
    (None, "none"),
    (0, "none"),
    (1, "none"),
    (2, "none"),
    (3, "none"),
    (3, "32b"),
    (3, "16b"),
]


def _compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PSNR between two images (assuming [0,1]-ish range)."""
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    # Use max of 1.0 as peak for clamp_min(0) images
    peak = max(a.abs().max().item(), b.abs().max().item(), 1.0)
    import math

    return 10 * math.log10(peak**2 / mse)


@skipif_no_cuda
@pytest.mark.parametrize("sh_degree,sh_compression", _CROSS_PARITY_CASES)
def test_cross_path_parity(sh_degree, sh_compression):
    """Inference and vanilla (direct rasterization) paths produce reasonably close results (PSNR > 30 dB)."""
    from experimental import render_scene
    from gsplat.rendering import rasterization

    torch.manual_seed(42)
    N = 500
    gs = _make_gaussian_scene(N=N, sh_degree=sh_degree)

    # Inference scene from the same GaussianScene
    from experimental import GaussianInferenceScene

    inference = GaussianInferenceScene.from_gaussian_scene(
        gs, id="test_inference", sh_compression=sh_compression
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
    assert inference_ret.metadata["render_path"] == "inference"

    # Compare: vanilla produces [1, H, W, 3], inference produces [1, H, W, 3]
    psnr = _compute_psnr(vanilla_renders, inference_ret.frame)
    assert psnr > 30.0, (
        f"PSNR {psnr:.1f} dB below 30 dB threshold for "
        f"sh_degree={sh_degree}, sh_compression={sh_compression}"
    )

    # Alpha comparison
    alpha_diff = (vanilla_alphas - inference_ret.metadata["alpha"]).abs()
    assert alpha_diff.max().item() < 0.1, (
        f"Alpha max diff {alpha_diff.max().item()} for "
        f"sh_degree={sh_degree}, sh_compression={sh_compression}"
    )
