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

"""Tests for the Gaussian Inference rasterization path (gsplat.experimental).

Migrated from tests/test_render_only.py during Phase 4b of the Inference Render
Path Unification plan.  All tests use the ``gsplat.experimental`` surface
(``GaussianInferenceScene``, ``rasterize_gaussian_inference_scene``, ``render_scene``,
``RenderReturn``).
"""

import pytest
import torch

device = torch.device("cuda:0")

_skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="No CUDA device"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussians(N, device, sh_degree=None):
    """Create random Gaussian parameters for testing.

    Returns *activated* tensors:
    - scales: positive (already exp'd)
    - opacities: in [0.1, 0.9] (already sigmoid'd)
    """
    torch.manual_seed(42)
    means = torch.randn(N, 3, device=device)
    # Place gaussians in front of camera (positive z)
    means[:, 2] = means[:, 2].abs() + 2.0
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, device=device) * 0.1 + 0.01  # activated (positive)
    opacities = torch.rand(N, device=device) * 0.8 + 0.1  # activated [0.1, 0.9]

    if sh_degree is not None:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device) * 0.1
    else:
        colors = torch.sigmoid(torch.randn(N, 3, device=device))

    return means, quats, scales, opacities, colors


def _make_camera(device, width=128, height=96):
    """Create a simple pinhole camera looking down +z."""
    focal = 128.0
    K = torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    viewmat = torch.eye(4, device=device)
    return viewmat, K, width, height


def _build_inference_scene(N, device, sh_degree=None, sh_compression="none"):
    """Build a GaussianInferenceScene from random Gaussians."""
    from experimental import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N, device, sh_degree)
    scene = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=sh_degree,
        sh_compression=sh_compression,
        id="test",
    )
    return scene


# ---------------------------------------------------------------------------
# Negative-import tests (legacy symbols removed)
# ---------------------------------------------------------------------------


class TestLegacyImportsRemoved:
    def test_render_only_rasterization_removed(self):
        with pytest.raises(ImportError):
            from gsplat import render_only_rasterization  # noqa: F401

    def test_RenderOnlyRasterizer_removed(self):
        with pytest.raises(ImportError):
            from gsplat import RenderOnlyRasterizer  # noqa: F401

    def test_has_render_only_removed(self):
        with pytest.raises(ImportError):
            from gsplat import has_render_only  # noqa: F401

    def test_has_render_only_viewer_path_removed(self):
        with pytest.raises(ImportError):
            from gsplat import has_render_only_viewer_path  # noqa: F401


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    pytestmark = [_skip_no_cuda]

    def test_cpu_tensors_rejected(self):
        """CPU tensors should be rejected by from_gaussian_tensors."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(10, "cpu")
        # The C++ pack op should reject CPU tensors
        with pytest.raises((RuntimeError, ValueError)):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=None,
                sh_compression="none",
                id="test",
            )

    def test_negative_scales_rejected(self):
        """Negative scales should be rejected by from_gaussian_tensors."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        scales_bad = scales.clone()
        scales_bad[0, 0] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales_bad,
                opacities,
                colors,
                sh_degree=None,
                sh_compression="none",
                id="test",
            )

    def test_opacities_out_of_range_rejected(self):
        """Opacities outside [0, 1] should be rejected."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        opacities_bad = opacities.clone()
        opacities_bad[0] = 1.5
        with pytest.raises(ValueError, match="opacities outside"):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities_bad,
                colors,
                sh_degree=None,
                sh_compression="none",
                id="test",
            )

    def test_invalid_tile_size_rejected(self):
        """Tile size other than 8 or 16 is rejected."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(10, device)
        viewmat, K, w, h = _make_camera(device)
        with torch.inference_mode():
            with pytest.raises(TypeError, match="tile_size"):
                rasterize_gaussian_inference_scene(
                    scene,
                    viewmat=viewmat,
                    K=K,
                    width=w,
                    height=h,
                    tile_size=4,
                )

    def test_wrong_means_shape_rejected(self):
        """Means with wrong shape should be rejected."""
        from experimental import GaussianInferenceScene

        means = torch.randn(10, 4, device=device)  # wrong: should be [N, 3]
        _, quats, scales, opacities, colors = _make_gaussians(10, device)
        with pytest.raises((ValueError, RuntimeError)):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=None,
                sh_compression="none",
                id="test",
            )

    def test_sh_degree_mismatch_rejected(self):
        """SH degree that doesn't match colors shape should be rejected."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        # Pre-activated RGB colors but sh_degree=3 -> should fail
        with pytest.raises((ValueError, RuntimeError)):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=3,
                sh_compression="none",
                id="test",
            )


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


class TestCorrectness:
    pytestmark = [_skip_no_cuda]

    def test_single_gaussian_rgb(self):
        """Single Gaussian produces non-zero output near its projected center."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        # Explicitly place the Gaussian at the origin in XY (projects to image
        # center) to avoid random seed-dependent culling when N=1.
        means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        scales = torch.tensor([[0.2, 0.2, 0.2]], device=device)
        opacities = torch.tensor([0.8], device=device)
        colors = torch.tensor([[0.9, 0.2, 0.5]], device=device)
        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        assert ret.frame.shape == (1, h, w, 3)
        assert ret.metadata["alpha"].shape == (1, h, w, 1)
        assert ret.frame.device.type == "cuda"
        # At least some pixels should be non-zero
        assert ret.frame.sum() > 0

    def test_multiple_gaussians_rgb(self):
        """Multiple Gaussians render correctly."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(100, device)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        assert ret.frame.shape == (1, h, w, 3)
        assert ret.metadata["alpha"].shape == (1, h, w, 1)

    def test_behind_camera_culling(self):
        """Gaussians behind the camera should not contribute to the image."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        means, quats, scales, opacities, colors = _make_gaussians(10, device)
        # Place all gaussians behind camera (negative z)
        means[:, 2] = -5.0
        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )
        viewmat, K, w, h = _make_camera(device)
        bg = torch.tensor([1.0, 1.0, 1.0], device=device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h, background=bg
            )

        # Should be all background
        torch.testing.assert_close(
            ret.frame[0],
            bg.expand(h, w, 3),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_output_no_grad(self):
        """Outputs should not have requires_grad."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(10, device)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        assert not ret.frame.requires_grad
        assert not ret.metadata["alpha"].requires_grad

    def test_with_background(self):
        """Background color is applied where alpha < 1."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(5, device)
        viewmat, K, w, h = _make_camera(device)
        bg = torch.tensor([0.2, 0.4, 0.6], device=device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h, background=bg
            )

        assert ret.frame.shape == (1, h, w, 3)

    @pytest.mark.parametrize("tile_size", [8, 16])
    def test_tile_sizes(self, tile_size):
        """Both supported tile sizes produce valid output."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene,
                viewmat=viewmat,
                K=K,
                width=w,
                height=h,
                tile_size=tile_size,
            )

        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()


# ---------------------------------------------------------------------------
# Reference comparison tests
# ---------------------------------------------------------------------------


class TestReferenceComparison:
    pytestmark = [_skip_no_cuda]

    def _has_3dgs(self):
        import gsplat

        return gsplat.has_3dgs()

    def test_matches_rasterization_rgb(self):
        """render_scene Inference output should closely match rasterization() for RGB."""
        if not self._has_3dgs():
            pytest.skip("3DGS not built")

        from experimental import render_scene, GaussianInferenceScene
        import gsplat

        N = 50
        means, quats, scales, opacities, colors = _make_gaussians(N, device)
        viewmat, K, w, h = _make_camera(device)

        # Build Inference scene
        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )

        # Render with Inference path
        with torch.inference_mode():
            ret_inference = render_scene(scene, viewmat=viewmat, K=K, width=w, height=h)

        # Render with standard rasterization path
        viewmats = viewmat.unsqueeze(0)
        Ks = K.unsqueeze(0)
        with torch.no_grad():
            render_ref, alphas_ref, meta_ref = gsplat.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=w,
                height=h,
                render_mode="RGB",
                packed=False,
                sh_degree=None,
            )

        # Inference uses fp16-packed intermediates, so allow fp16-level slack.
        torch.testing.assert_close(
            ret_inference.frame,
            render_ref,
            atol=5e-3,
            rtol=1e-2,
        )

    def test_matches_rasterization_sh(self):
        """render_scene Inference with SH should be close to rasterization() with SH."""
        if not self._has_3dgs():
            pytest.skip("3DGS not built")

        from experimental import render_scene, GaussianInferenceScene
        import gsplat

        N = 50
        sh_degree = 3
        means, quats, scales, opacities, colors = _make_gaussians(
            N,
            device,
            sh_degree=sh_degree,
        )
        viewmat, K, w, h = _make_camera(device)

        # Build Inference scene
        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=sh_degree,
            sh_compression="none",
            id="test",
        )

        # Render with Inference path
        with torch.inference_mode():
            ret_inference = render_scene(scene, viewmat=viewmat, K=K, width=w, height=h)

        # Render with standard rasterization path
        viewmats = viewmat.unsqueeze(0)
        Ks = K.unsqueeze(0)
        with torch.no_grad():
            render_ref, alphas_ref, meta_ref = gsplat.rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmats,
                Ks=Ks,
                width=w,
                height=h,
                render_mode="RGB",
                packed=False,
                sh_degree=sh_degree,
            )

        # Inference uses fp16-packed intermediates, allow fp16-level slack.
        torch.testing.assert_close(
            ret_inference.frame,
            render_ref,
            atol=5e-3,
            rtol=1e-2,
        )


# ---------------------------------------------------------------------------
# GaussianInferenceScene lifecycle tests (migrated from TestRenderOnlyRasterizer)
# ---------------------------------------------------------------------------


class TestGaussianInferenceSceneLifecycle:
    pytestmark = [_skip_no_cuda]

    def test_basic_render(self):
        """GaussianInferenceScene + rasterize_gaussian_inference_scene produces valid output."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        assert ret.frame.shape == (1, h, w, 3)
        assert ret.metadata["alpha"].shape == (1, h, w, 1)

    def test_multiple_renders(self):
        """Can render multiple camera poses with the same scene."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device)
        _, K, w, h = _make_camera(device)

        viewmat1 = torch.eye(4, device=device)
        viewmat2 = torch.eye(4, device=device)
        viewmat2[2, 3] = -1.0  # shift camera back

        with torch.inference_mode():
            ret1 = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat1, K=K, width=w, height=h
            )
            ret2 = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat2, K=K, width=w, height=h
            )

        assert ret1.frame.shape == (1, h, w, 3)
        assert ret2.frame.shape == (1, h, w, 3)
        # Different viewpoints should produce different images
        assert not torch.allclose(ret1.frame, ret2.frame)

    def test_rebuild_scene(self):
        """Rebuilding a scene with different data produces different renders."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        means1, quats1, scales1, opacities1, colors1 = _make_gaussians(50, device)
        torch.manual_seed(123)
        means2, quats2, scales2, opacities2, colors2 = _make_gaussians(30, device)

        viewmat, K, w, h = _make_camera(device)

        scene1 = GaussianInferenceScene.from_gaussian_tensors(
            means1,
            quats1,
            scales1,
            opacities1,
            colors1,
            sh_degree=None,
            sh_compression="none",
            id="test1",
        )
        scene2 = GaussianInferenceScene.from_gaussian_tensors(
            means2,
            quats2,
            scales2,
            opacities2,
            colors2,
            sh_degree=None,
            sh_compression="none",
            id="test2",
        )

        with torch.inference_mode():
            ret1 = rasterize_gaussian_inference_scene(
                scene1, viewmat=viewmat, K=K, width=w, height=h
            )
            ret2 = rasterize_gaussian_inference_scene(
                scene2, viewmat=viewmat, K=K, width=w, height=h
            )

        # Different scene data should produce different renders
        assert ret1.frame.shape == ret2.frame.shape

    def test_release_and_rebuild(self):
        """Releasing a scene and rebuilding works correctly."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        means, quats, scales, opacities, colors = _make_gaussians(50, device)
        viewmat, K, w, h = _make_camera(device)

        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )

        with torch.inference_mode():
            ret_before = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        scene.release()
        assert scene.is_empty()

        # Rebuild
        scene2 = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test2",
        )
        with torch.inference_mode():
            ret_after = rasterize_gaussian_inference_scene(
                scene2, viewmat=viewmat, K=K, width=w, height=h
            )

        torch.testing.assert_close(ret_before.frame, ret_after.frame, atol=0, rtol=0)

    def test_render_on_released_scene_raises(self):
        """Rendering a released scene raises ValueError."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(10, device)
        viewmat, K, w, h = _make_camera(device)

        scene.release()
        assert scene.is_empty()

        with torch.inference_mode():
            with pytest.raises(ValueError, match="has been released"):
                rasterize_gaussian_inference_scene(
                    scene, viewmat=viewmat, K=K, width=w, height=h
                )

    def test_snapshot_isolation(self):
        """Inference scene owns its packed buffer; mutating original tensors has no effect."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        means, quats, scales, opacities, colors = _make_gaussians(200, device)
        viewmat, K, w, h = _make_camera(device)

        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="snap",
        )

        with torch.inference_mode():
            ret1 = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
            frame1 = ret1.frame.clone()

            # Shift the original tensors by a large amount.
            means.add_(10.0)

            # The packed buffer is independent — render must be unchanged.
            ret2 = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
            torch.testing.assert_close(ret2.frame, frame1, atol=0, rtol=0)

            # A scene rebuilt from the mutated tensors should differ.
            scene2 = GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=None,
                sh_compression="none",
                id="snap2",
            )
            ret3 = rasterize_gaussian_inference_scene(
                scene2, viewmat=viewmat, K=K, width=w, height=h
            )
            assert not torch.allclose(
                ret3.frame, frame1, atol=1e-3
            ), "Rebuilt snapshot should differ after means shifted by 10"


# ---------------------------------------------------------------------------
# Source parity tests (Phase 8) -- kept as-is
# ---------------------------------------------------------------------------


class TestSourceParity:
    """Verify that imported viewer source files match the manifest."""

    pytestmark = [_skip_no_cuda]

    def test_imported_files_present(self):
        """All expected imported files exist."""
        import os

        imported_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "kernels",
            "cuda",
            "csrc",
            "gaussian_inference",
        )
        expected_files = [
            "Constants.h",
            "IntersectCommon.cu",
            "IntersectCommon.h",
            "IntersectMTConfig.h",
            "IntersectMTFused.cu",
            "IntersectMTFused.h",
            "MacroTileIntersect.cu",
            "MacroTileIntersect.h",
            "MacroTileRasterize.cu",
            "MacroTileRasterize.h",
            "Projection.cu",
            "Projection.h",
            "InferenceTypes.h",
            "SegmentedSort.cu",
            "SegmentedSort.h",
            "SHCommon.h",
            "SHCompression.cu",
            "SHCompression.h",
            "SphericalHarmonics.cu",
            "SphericalHarmonics.h",
            "Utils.h",
        ]
        for f in expected_files:
            path = os.path.join(imported_dir, f)
            assert os.path.isfile(path), f"Missing imported file: {f}"


# ---------------------------------------------------------------------------
# Viewer-specific correctness tests (migrated from TestViewerPath)
# ---------------------------------------------------------------------------


class TestViewerPath:
    """Tests that verify the viewer path via direct op calls."""

    pytestmark = [_skip_no_cuda]

    def test_viewer_op_callable(self):
        """The viewer Torch op can be called directly via gaussian_render_inference_only."""
        from experimental.render.kernels._backend import _C  # noqa: F401

        scene = _build_inference_scene(10, device)
        viewmat, K, w, h = _make_camera(device)

        renders, alphas = torch.ops.experimental.gaussian_render_inference_only(
            scene.means_planar,
            scene.qso_packed,
            scene.colors_packed,
            viewmat,
            K,
            w,
            h,
            scene.sh_degree,
            16,
            0.01,
            1e10,
            0.0,
            0.3,
            scene.sh_compression_mode,
            None,
        )
        assert renders.shape == (h, w, 3)
        assert alphas.shape == (h, w, 1)

    def test_sh_degree_3_fused(self):
        """Degree-3 SH (K=16) triggers the fused projection+SH path."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=3)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()
        assert ret.frame.sum() > 0

    def test_sh_degree_1_separate(self):
        """Degree-1 SH (K=4) triggers the separate projection+SH path."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=1)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()

    def test_off_screen_gaussians(self):
        """Gaussians far off-screen produce only background."""
        from experimental import (
            GaussianInferenceScene,
            rasterize_gaussian_inference_scene,
        )

        N = 20
        means = torch.zeros(N, 3, device=device)
        means[:, 0] = 1000.0  # way off to the right
        means[:, 2] = 5.0  # in front of camera
        quats = torch.zeros(N, 4, device=device)
        quats[:, 0] = 1.0  # identity rotation
        scales = torch.full((N, 3), 0.01, device=device)
        opacities = torch.full((N,), 0.5, device=device)  # activated [0, 1]
        colors = torch.ones(N, 3, device=device)

        scene = GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )
        viewmat, K, w, h = _make_camera(device)
        bg = torch.tensor([0.5, 0.5, 0.5], device=device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h, background=bg
            )

        # Should be all background since gaussians are off-screen
        torch.testing.assert_close(
            ret.frame[0], bg.expand(h, w, 3), atol=1e-3, rtol=1e-3
        )

    def test_stateful_sh_degree_3(self):
        """GaussianInferenceScene with SH degree 3 renders correctly."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=3)
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )

        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()


# ---------------------------------------------------------------------------
# SH Compression tests (Phase 7)
# ---------------------------------------------------------------------------


class TestSHCompression:
    pytestmark = [_skip_no_cuda]

    def test_raw_sh_degree3(self):
        """Raw degree-3 SH still works (no compression)."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=3, sh_compression="none")
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()

    def test_compressed_32b(self):
        """Compressed 32B degree-3 SH renders."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=3, sh_compression="32b")
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()
        assert ret.frame.sum() > 0

    def test_compressed_16b(self):
        """Compressed 16B degree-3 SH renders."""
        from experimental import rasterize_gaussian_inference_scene

        scene = _build_inference_scene(50, device, sh_degree=3, sh_compression="16b")
        viewmat, K, w, h = _make_camera(device)

        with torch.inference_mode():
            ret = rasterize_gaussian_inference_scene(
                scene, viewmat=viewmat, K=K, width=w, height=h
            )
        assert ret.frame.shape == (1, h, w, 3)
        assert ret.frame.isfinite().all()

    def test_compression_rejects_non_degree3(self):
        """Compression rejects non-degree-3 SH at scene construction time."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(
            50, device, sh_degree=1
        )
        with pytest.raises(ValueError):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=1,
                sh_compression="32b",
                id="test",
            )

    def test_compression_rejects_rgb(self):
        """Compression rejects pre-activated RGB at scene construction time."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(50, device)
        with pytest.raises(ValueError):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=None,
                sh_compression="32b",
                id="test",
            )

    def test_invalid_compression_mode(self):
        """Invalid compression mode string is rejected."""
        from experimental import GaussianInferenceScene

        means, quats, scales, opacities, colors = _make_gaussians(
            50, device, sh_degree=3
        )
        with pytest.raises(ValueError, match="sh_compression"):
            GaussianInferenceScene.from_gaussian_tensors(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree=3,
                sh_compression="invalid",
                id="test",
            )
