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

"""Tests for NHT rasterization kernels and deferred shading module.

Usage:
```bash
pytest tests/test_nht.py -s
```
"""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

import gsplat
from gsplat.nht import NHTParams
from gsplat.nht._wrapper import (
    NHT_SUPPORTED_CHANNELS as SUPPORTED_CHANNELS,
    _find_next_supported,
    get_encoding_expansion_factor,
    get_feature_divisor,
)

device = torch.device("cuda:0")

_tcnn_available = False
try:
    import tinycudann as tcnn  # noqa: F401

    _tcnn_available = True
except ImportError:
    pass


def _nht_rasterizer_available() -> bool:
    """Probe whether the compiled NHT rasterizer works."""
    if not torch.cuda.is_available() or not gsplat.has_3dgs():
        return False
    try:
        from gsplat.rendering import rasterization

        N, d = 64, "cuda"
        rasterization(
            means=torch.zeros(N, 3, device=d),
            quats=torch.tensor([1.0, 0, 0, 0], device=d).expand(N, -1).contiguous(),
            scales=torch.full((N, 3), 0.01, device=d),
            opacities=torch.full((N,), 0.5, device=d),
            colors=torch.randn(N, 16, device=d),
            viewmats=torch.linalg.inv(
                torch.eye(4, device=d).unsqueeze(0).clone().detach()
            ),
            Ks=torch.tensor([[[50.0, 0, 16], [0, 50, 12], [0, 0, 1]]], device=d),
            width=32,
            height=24,
            nht_params=NHTParams(),
            with_eval3d=True,
            with_ut=True,
            packed=False,
            sh_degree=None,
        )
        return True
    except Exception:
        return False


_nht_ok = _nht_rasterizer_available()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera(width=64, height=48, focal=100.0, device="cuda"):
    K = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    )
    c2w = torch.eye(4, device=device)
    c2w[2, 3] = -3.0
    return K.unsqueeze(0), c2w.unsqueeze(0), width, height


_lidar_cache = {}


def _make_lidar_coeffs(device="cuda"):
    """Small deterministic lidar model used for NHT lidar smoke tests."""
    key = str(device)
    if key in _lidar_cache:
        return _lidar_cache[key]

    row_elevations = torch.linspace(0.20, -0.20, 32, device=device)
    column_azimuths = torch.linspace(1.0, -1.0, 256, device=device)
    row_offsets = torch.zeros(32, device=device)
    lidar_params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
        row_elevations_rad=row_elevations,
        column_azimuths_rad=column_azimuths,
        row_azimuth_offsets_rad=row_offsets,
        spinning_frequency_hz=10.0,
        spinning_direction=gsplat.SpinningDirection.CLOCKWISE,
    )
    angles = gsplat.compute_lidar_angles_to_columns_map(
        lidar_params, resolution_factor=1
    )
    tiling = gsplat.compute_lidar_tiling(
        lidar_params,
        n_bins_elevation=4,
        max_pts_per_tile=16 * 16,
        resolution_elevation=128,
        densification_factor_azimuth=2,
    )
    lidar = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(
        lidar_params, angles, tiling
    )
    _lidar_cache[key] = lidar
    return lidar


def _make_splats(N=256, feature_dim=16, device="cuda"):
    means = torch.randn(N, 3, device=device) * 0.3
    quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
    scales = torch.rand(N, 3, device=device) * 0.5 - 2.0
    opacities = torch.logit(torch.full((N,), 0.5, device=device))
    features = torch.randn(N, feature_dim, device=device)
    return means, quats, scales, opacities, features


def _deferred_shader_raster_channels(mod, *, extra: int = 0) -> int:
    """Last dim for tensors passed to ``DeferredShaderModule.forward``."""
    if mod.enable_view_encoding:
        return mod.encoded_dim + 3 + extra
    return mod.encoded_dim + extra


def _rasterize_nht(
    feature_dim=16,
    N=256,
    width=32,
    height=24,
    requires_grad=False,
    render_mode="RGB",
    return_normals=False,
    external_distortion_coeffs=None,
    camera_model="pinhole",
):
    """Shared helper: rasterize synthetic NHT scene."""
    from gsplat.rendering import rasterization

    torch.manual_seed(42)
    means, quats, scales_raw, opacities_raw, features = _make_splats(
        N, feature_dim, device=device
    )
    scales = torch.exp(scales_raw)
    opacities = torch.sigmoid(opacities_raw)
    K, c2w, W, H = _make_camera(width=width, height=height, device=device)
    viewmats = torch.linalg.inv(c2w)

    if requires_grad:
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        features.requires_grad_(True)
    rc, ra, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=viewmats,
        Ks=K,
        width=W,
        height=H,
        camera_model=camera_model,
        nht_params=NHTParams(),
        with_eval3d=True,
        with_ut=True,
        packed=False,
        sh_degree=None,
        render_mode=render_mode,
        return_normals=return_normals,
        external_distortion_coeffs=external_distortion_coeffs,
    )
    return (
        rc,
        ra,
        info,
        means,
        scales,
        opacities,
        features,
        quats,
        K,
        c2w,
        W,
        H,
    )


def _rasterize_nht_lidar(
    feature_dim=16,
    N=256,
    render_mode="RGB",
    return_normals=False,
    requires_grad=False,
):
    """Shared helper: rasterize synthetic NHT scene through lidar camera model."""
    from gsplat.rendering import rasterization

    torch.manual_seed(7)
    means, quats, scales_raw, opacities_raw, features = _make_splats(
        N, feature_dim, device=device
    )
    scales = torch.exp(scales_raw)
    opacities = torch.sigmoid(opacities_raw)
    if requires_grad:
        means.requires_grad_(True)
        quats.requires_grad_(True)
        scales.requires_grad_(True)
        opacities.requires_grad_(True)
        features.requires_grad_(True)

    lidar = _make_lidar_coeffs(device=device)
    viewmats = torch.eye(4, device=device)[None]
    Ks = torch.eye(3, device=device)[None]

    rc, ra, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=features,
        viewmats=viewmats,
        Ks=Ks,
        width=lidar.n_columns,
        height=lidar.n_rows,
        camera_model="lidar",
        lidar_coeffs=lidar,
        nht_params=NHTParams(),
        with_eval3d=True,
        with_ut=True,
        packed=False,
        sh_degree=None,
        render_mode=render_mode,
        return_normals=return_normals,
    )
    return rc, ra, info, means, scales, opacities, features, quats, lidar


# ===================================================================
# NHT rasterization kernel tests
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not _nht_ok, reason="NHT rasterizer not functional")
class TestNHTRasterization:
    """Forward pass output shapes and basic properties."""

    @pytest.mark.parametrize("feature_dim", [16, 32, 64])
    def test_output_shapes(self, feature_dim):
        encf = get_encoding_expansion_factor()
        div = get_feature_divisor()
        rc, ra, *_ = _rasterize_nht(feature_dim=feature_dim)
        _, H, W = rc.shape[0], rc.shape[1], rc.shape[2]
        encoded_dim = (feature_dim // div) * encf
        # NHT raster appends 3 ray-direction channels after encoded features.
        expected_ch = encoded_dim + 3
        assert rc.shape == (1, H, W, expected_ch)
        assert ra.shape == (1, H, W, 1)

    def test_alpha_range(self):
        _, ra, *_ = _rasterize_nht()
        assert ra.min() >= 0.0
        assert ra.max() <= 1.0

    def test_deterministic(self):
        rc1, ra1, *_ = _rasterize_nht()
        rc2, ra2, *_ = _rasterize_nht()
        torch.testing.assert_close(rc1, rc2)
        torch.testing.assert_close(ra1, ra2)

    def test_fused_depth_layout(self):
        encf = get_encoding_expansion_factor()
        div = get_feature_divisor()
        feature_dim = 16
        rc, ra, *_ = _rasterize_nht(feature_dim=feature_dim, render_mode="RGB+D")
        encoded_dim = (feature_dim // div) * encf
        assert rc.shape[-1] == encoded_dim + 1 + 3

        encoded = rc[..., :encoded_dim]
        ray_dirs = rc[..., encoded_dim : encoded_dim + 3]
        depth = rc[..., -1:]
        assert encoded.shape[-1] == encoded_dim
        assert depth.shape[-1] == 1
        assert ray_dirs.shape[-1] == 3
        assert torch.isfinite(depth).all()
        assert torch.isfinite(ray_dirs).all()
        assert ra.shape[-1] == 1

    def test_fused_expected_depth_is_finite(self):
        rc, ra, *_ = _rasterize_nht(feature_dim=16, render_mode="RGB+ED")
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        depth = rc[..., -1:]
        valid = ra > 1e-6
        assert torch.isfinite(depth[valid.expand_as(depth)]).all()

    def test_fused_hit_distance_is_finite(self):
        rc, ra, *_ = _rasterize_nht(feature_dim=16, render_mode="RGB-d")
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        depth = rc[..., -1:]
        valid = ra > 1e-6
        assert torch.isfinite(depth[valid.expand_as(depth)]).all()

    def test_fused_normals_meta(self):
        _, _, info, *_ = _rasterize_nht(feature_dim=16, return_normals=True)
        assert "normals" in info
        normals = info["normals"]
        assert normals.shape[-1] == 3
        assert torch.isfinite(normals).all()

    def test_lidar_camera_model(self):
        rc, ra, info, *_ = _rasterize_nht_lidar(feature_dim=16)
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        assert rc.shape[-1] == encoded_dim + 3
        assert ra.shape[-1] == 1
        assert (
            info["tile_width"]
            == _make_lidar_coeffs(device=device).tiling.n_bins_azimuth
        )
        assert (
            info["tile_height"]
            == _make_lidar_coeffs(device=device).tiling.n_bins_elevation
        )
        assert torch.isfinite(rc).all()
        assert torch.isfinite(ra).all()

    def test_lidar_fused_depth_and_normals(self):
        rc, ra, info, *_ = _rasterize_nht_lidar(
            feature_dim=16, render_mode="RGB+D", return_normals=True
        )
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        depth = rc[..., -1:]
        assert depth.shape[-1] == 1
        assert "normals" in info
        assert info["normals"].shape[-1] == 3
        valid = ra > 1e-6
        assert torch.isfinite(depth[valid.expand_as(depth)]).all()
        assert torch.isfinite(info["normals"]).all()

    def test_identity_external_distortion_matches_none(self):
        from gsplat.cuda._torch_external_distortion import (
            make_identity_horizontal_poly,
            make_identity_vertical_poly,
            make_params,
        )

        rc_none, ra_none, *_ = _rasterize_nht(feature_dim=16)
        identity = make_params(
            h_poly=make_identity_horizontal_poly(),
            v_poly=make_identity_vertical_poly(),
            h_inv=make_identity_horizontal_poly(),
            v_inv=make_identity_vertical_poly(),
            device=device,
        )
        rc_identity, ra_identity, *_ = _rasterize_nht(
            feature_dim=16, external_distortion_coeffs=identity
        )
        torch.testing.assert_close(rc_identity, rc_none, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ra_identity, ra_none, atol=1e-5, rtol=1e-5)

    def test_nonzero_external_distortion_changes_output(self):
        from gsplat.cuda._torch_external_distortion import make_params

        rc_none, _, *_ = _rasterize_nht(feature_dim=16)
        perturbed = make_params(
            h_poly=[0.02, 1.01, 0.01],
            v_poly=[-0.01, 0.02, 0.98],
            h_inv=[-0.02, 0.99, -0.01],
            v_inv=[0.01, -0.02, 1.02],
            device=device,
        )
        rc_distorted, _, *_ = _rasterize_nht(
            feature_dim=16, external_distortion_coeffs=perturbed
        )
        assert (rc_distorted - rc_none).abs().max() > 1e-6

    def test_ortho_camera_model(self):
        rc, ra, *_ = _rasterize_nht(feature_dim=16, camera_model="ortho")
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        assert rc.shape[-1] == encoded_dim + 3
        assert ra.shape[-1] == 1
        assert torch.isfinite(rc).all()
        assert torch.isfinite(ra).all()

    def test_ortho_depth_and_normals(self):
        rc, ra, info, *_ = _rasterize_nht(
            feature_dim=16,
            camera_model="ortho",
            render_mode="RGB+D",
            return_normals=True,
        )
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        assert rc.shape[-1] == encoded_dim + 3 + 1
        assert "normals" in info
        assert info["normals"].shape[-1] == 3
        assert torch.isfinite(rc).all()
        assert torch.isfinite(ra).all()
        assert torch.isfinite(info["normals"]).all()

    def test_extra_signals_request_records_aov_dim(self):
        from gsplat.rendering import rasterization

        means, quats, scales_raw, opacities_raw, features = _make_splats(
            64, 16, device=device
        )
        scales = torch.exp(scales_raw)
        opacities = torch.sigmoid(opacities_raw)
        K, c2w, W, H = _make_camera(width=16, height=12, device=device)
        viewmats = torch.linalg.inv(c2w)
        extra_signals = torch.randn(64, 3, device=device)
        rc, _, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=features,
            viewmats=viewmats,
            Ks=K,
            width=W,
            height=H,
            nht_params=NHTParams(),
            with_eval3d=True,
            with_ut=True,
            packed=False,
            sh_degree=None,
            extra_signals=extra_signals,
        )
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        assert rc.shape[-1] == encoded_dim + 3
        assert meta["nht_extra_signal_dim"] == 3
        assert "render_extra_signals" not in meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not _nht_ok, reason="NHT rasterizer not functional")
class TestNHTRasterizationBackward:
    """Gradient flow through the NHT rasterizer."""

    def test_gradients_exist(self):
        rc, ra, _, means, scales, opacities, features, *_ = _rasterize_nht(
            requires_grad=True
        )
        loss = rc.sum() + ra.sum()
        loss.backward()
        for name, p in [
            ("means", means),
            ("scales", scales),
            ("opacities", opacities),
            ("features", features),
        ]:
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_gradients_nonzero(self):
        rc, ra, _, means, scales, opacities, features, *_ = _rasterize_nht(
            requires_grad=True
        )
        rc.sum().backward()
        assert features.grad.abs().sum() > 0, "Feature gradients are all zero"
        assert means.grad.abs().sum() > 0, "Means gradients are all zero"

    def test_feature_gradient_finite_diff(self):
        """Spot-check feature gradients against finite differences."""
        from gsplat.rendering import rasterization

        torch.manual_seed(123)
        N, fdim = 32, 16
        means = torch.randn(N, 3, device=device) * 0.2
        quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        scales = torch.exp(torch.full((N, 3), -1.5, device=device))
        opacities = torch.sigmoid(torch.zeros(N, device=device))
        features = torch.randn(N, fdim, device=device)
        K, c2w, W, H = _make_camera(width=16, height=12, device=device)
        viewmats = torch.linalg.inv(c2w)

        def fwd(feat):
            rc, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=feat,
                viewmats=viewmats,
                Ks=K,
                width=W,
                height=H,
                nht_params=NHTParams(),
                with_eval3d=True,
                with_ut=True,
                packed=False,
                sh_degree=None,
            )
            return rc.sum()

        features.requires_grad_(True)
        loss = fwd(features)
        loss.backward()
        analytic = features.grad.clone()

        eps = 1e-3
        for gi, fi in [(0, 0), (0, 5), (N // 2, 3), (N - 1, fdim - 1)]:
            fp = features.detach().clone()
            fm = features.detach().clone()
            fp[gi, fi] += eps
            fm[gi, fi] -= eps
            fd = (fwd(fp).item() - fwd(fm).item()) / (2 * eps)
            a = analytic[gi, fi].item()
            if abs(fd) < 1e-6 and abs(a) < 1e-6:
                continue
            rel_err = abs(a - fd) / (max(abs(a), abs(fd)) + 1e-8)
            assert (
                rel_err < 0.05
            ), f"Gradient mismatch [{gi},{fi}]: analytic={a:.6f} fd={fd:.6f}"

    @pytest.mark.parametrize("render_mode", ["RGB+D", "RGB-d"])
    def test_fused_depth_gradients_flow_to_geometry(self, render_mode):
        rc, _, _, means, scales, opacities, features, quats, *_ = _rasterize_nht(
            feature_dim=16, render_mode=render_mode, requires_grad=True
        )
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        depth = rc[..., -1:]
        depth.sum().backward()

        for name, p in [
            ("means", means),
            ("scales", scales),
            ("opacities", opacities),
        ]:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
            assert p.grad.abs().sum() > 0, f"All-zero gradient for {name}"

        if render_mode == "RGB-d":
            assert quats.grad is not None, "No gradient for quats"
            assert torch.isfinite(quats.grad).all(), "Non-finite gradient for quats"
            assert quats.grad.abs().sum() > 0, "All-zero gradient for quats"

        # Depth-only losses should not need feature gradients, but if a future
        # kernel path produces them, they must at least remain finite.
        assert features.grad is None or torch.isfinite(features.grad).all()

    def test_fused_normals_gradients_flow_to_geometry(self):
        _, _, info, means, scales, opacities, features, quats, *_ = _rasterize_nht(
            feature_dim=16, return_normals=True, requires_grad=True
        )
        info["normals"].sum().backward()

        for name, p in [
            ("means", means),
            ("scales", scales),
            ("opacities", opacities),
        ]:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"

        assert quats.grad is not None, "No gradient for quats"
        assert torch.isfinite(quats.grad).all(), "Non-finite gradient for quats"
        assert quats.grad.abs().sum() > 0, "All-zero gradient for quats"

        assert features.grad is None or torch.isfinite(features.grad).all()

    def test_lidar_fused_depth_gradients_flow_to_geometry(self):
        rc, _, _, means, scales, opacities, features, quats, _ = _rasterize_nht_lidar(
            feature_dim=16, render_mode="RGB+D", requires_grad=True
        )
        encoded_dim = (16 // get_feature_divisor()) * get_encoding_expansion_factor()
        depth = rc[..., -1:]
        depth.sum().backward()

        for name, p in [
            ("means", means),
            ("scales", scales),
            ("opacities", opacities),
        ]:
            assert p.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient for {name}"
        assert quats.grad is None or torch.isfinite(quats.grad).all()
        assert features.grad is None or torch.isfinite(features.grad).all()


# ===================================================================
# Encoding helpers
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestEncodingHelpers:
    def test_encoding_expansion_factor_positive(self):
        encf = get_encoding_expansion_factor()
        assert isinstance(encf, int) and encf >= 1

    def test_feature_divisor_positive(self):
        div = get_feature_divisor()
        assert isinstance(div, int) and div >= 1

    def test_supported_channels_sorted(self):
        for i in range(len(SUPPORTED_CHANNELS) - 1):
            assert SUPPORTED_CHANNELS[i] < SUPPORTED_CHANNELS[i + 1]

    def test_common_feature_dims_supported(self):
        # ``_find_next_supported`` is re-imported here for clarity; the
        # top-of-file import is the source of truth.

        encf = get_encoding_expansion_factor()
        for dim in [16, 32, 48, 64, 128]:
            eff = dim * encf
            if eff in SUPPORTED_CHANNELS:
                continue
            assert _find_next_supported(eff, SUPPORTED_CHANNELS) > 0


# ===================================================================
# DeferredShaderModule tests
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestDeferredShaderModule:
    def _make(self, feature_dim=16, enable_view_encoding=True, **kw):
        from gsplat.nht.deferred_shader import DeferredShaderModule

        return DeferredShaderModule(
            feature_dim=feature_dim,
            enable_view_encoding=enable_view_encoding,
            **kw,
        ).to(device)

    @pytest.mark.parametrize("feature_dim", [16, 32, 64])
    def test_construction(self, feature_dim):
        mod = self._make(feature_dim=feature_dim)
        assert mod.feature_dim == feature_dim

    @pytest.mark.parametrize("view_encoding_type", ["sh", "fourier"])
    @pytest.mark.parametrize("center_ray", [True, False])
    def test_forward_shape(self, view_encoding_type, center_ray):
        mod = self._make(
            view_encoding_type=view_encoding_type, center_ray_encoding=center_ray
        )
        _, _, W, H = _make_camera(device=device)
        inp = torch.randn(1, H, W, _deferred_shader_raster_channels(mod), device=device)
        colors, extras = mod(inp)
        assert colors.shape == (1, H, W, 3)

    def test_forward_no_view_encoding(self):
        mod = self._make(enable_view_encoding=False)
        _, _, W, H = _make_camera(device=device)
        inp = torch.randn(1, H, W, mod.encoded_dim, device=device)
        colors, _ = mod(inp)
        assert colors.shape == (1, H, W, 3)

    def test_default_shader_aux_output(self):
        mod = self._make(auxiliary_output_dim=7)
        assert mod.uses_direct_fused_output
        _, _, W, H = _make_camera(device=device)
        inp = torch.randn(1, H, W, _deferred_shader_raster_channels(mod), device=device)
        colors, extras = mod(inp)
        assert colors.shape == (1, H, W, 3)
        assert extras is not None
        assert extras.shape == (1, H, W, 7)

    def test_default_shader_large_aux_linear_readout(self):
        mod = self._make(auxiliary_output_dim=130)
        assert mod.uses_full_linear_readout
        assert mod.output_proj is not None

    def test_default_shader_split_aux_head(self):
        mod = self._make(auxiliary_output_dim=32, split_rgb_head=True)
        assert mod.uses_split_rgb_aux_linear
        assert mod.auxiliary_head is not None

    @pytest.mark.parametrize("view_encoding_type", ["sh", "fourier"])
    def test_backward(self, view_encoding_type):
        mod = self._make(view_encoding_type=view_encoding_type)
        _, _, W, H = _make_camera(width=32, height=24, device=device)
        inp = torch.randn(
            1,
            H,
            W,
            _deferred_shader_raster_channels(mod),
            device=device,
            requires_grad=True,
        )
        colors, _ = mod(inp)
        colors.sum().backward()
        assert inp.grad is not None
        assert not torch.isnan(inp.grad).any()
        has_mlp_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()
        )
        assert has_mlp_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestDeferredShaderModuleAOV:
    def _make(self, feature_dim=16, auxiliary_output_dim=8, **kw):
        from gsplat.nht.deferred_shader import DeferredShaderModuleAOV

        return DeferredShaderModuleAOV(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            auxiliary_output_dim=auxiliary_output_dim,
            **kw,
        ).to(device)

    def test_direct_fused_when_small(self):
        mod = self._make(auxiliary_output_dim=8)
        assert mod.uses_direct_fused_output
        assert mod.output_proj is None
        assert mod.auxiliary_head is None
        assert mod._total_output_dim == 11

    def test_linear_head_when_large(self):
        mod = self._make(auxiliary_output_dim=130)
        assert mod.uses_full_linear_readout
        assert mod.output_proj is not None
        assert mod.output_proj.out_features == 133

    def test_split_rgb_aux_linear(self):
        mod = self._make(auxiliary_output_dim=64, split_rgb_head=True)
        assert mod.uses_split_rgb_aux_linear
        assert mod.auxiliary_head is not None
        assert mod.auxiliary_head.out_features == 64
        assert mod.output_proj is None

    def test_fused_tcnn_sigmoid_range(self):
        mod = self._make(
            auxiliary_output_dim=4,
            fused_tcnn_sigmoid=True,
        )
        assert mod.uses_direct_fused_output
        assert mod.tcnn_emitted_sigmoid_outputs
        _, _, W, H = _make_camera(device=device)
        ch = _deferred_shader_raster_channels(mod)
        inp = torch.randn(1, H, W, ch, device=device)
        rgb_raw, aux_raw, _ = mod(inp)
        assert rgb_raw.min() >= 0.0 and rgb_raw.max() <= 1.0
        assert aux_raw is not None
        assert aux_raw.min() >= 0.0 and aux_raw.max() <= 1.0

    @pytest.mark.parametrize("aux_dim", [0, 16])
    def test_forward_shapes(self, aux_dim):
        mod = self._make(auxiliary_output_dim=aux_dim)
        _, _, W, H = _make_camera(device=device)
        ch = _deferred_shader_raster_channels(mod)
        inp = torch.randn(1, H, W, ch, device=device)
        rgb_raw, aux_raw, extras = mod(inp)
        assert rgb_raw.shape == (1, H, W, 3)
        if aux_dim == 0:
            assert aux_raw is None
        else:
            assert aux_raw.shape == (1, H, W, aux_dim)
        assert extras is None

    def test_backward(self):
        mod = self._make(auxiliary_output_dim=12)
        _, _, W, H = _make_camera(width=32, height=24, device=device)
        ch = _deferred_shader_raster_channels(mod)
        inp = torch.randn(1, H, W, ch, device=device, requires_grad=True)
        rgb_raw, aux_raw, _ = mod(inp)
        (rgb_raw.sum() + aux_raw.sum()).backward()
        assert inp.grad is not None

    def test_split_and_fused_sigmoid_mutually_exclusive(self):
        from gsplat.nht.deferred_shader import DeferredShaderModuleAOV

        with pytest.raises(ValueError, match="split_rgb_head cannot"):
            DeferredShaderModuleAOV(
                feature_dim=16,
                enable_view_encoding=True,
                auxiliary_output_dim=4,
                split_rgb_head=True,
                fused_tcnn_sigmoid=True,
            )


# ===================================================================
# Full pipeline: rasterize → deferred shader → RGB
# ===================================================================


@pytest.mark.skipif(not _nht_ok, reason="NHT rasterizer not functional")
@pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
class TestNHTPipeline:
    @pytest.mark.parametrize("feature_dim", [16, 32])
    def test_pipeline_produces_rgb(self, feature_dim):
        from gsplat.nht.deferred_shader import DeferredShaderModule

        rc, _, _, *_, W, H = _rasterize_nht(feature_dim=feature_dim)
        dm = DeferredShaderModule(
            feature_dim=feature_dim, enable_view_encoding=True
        ).to(device)
        assert rc.shape[-1] == dm.encoded_dim + 3
        colors, _ = dm(rc)
        assert colors.shape == (1, H, W, 3)
        assert not torch.isnan(colors).any()

    def test_pipeline_produces_aov_outputs(self):
        from gsplat.nht.deferred_shader import DeferredShaderModuleAOV

        feature_dim = 16
        aux_dim = 5
        rc, _, _, *_, W, H = _rasterize_nht(feature_dim=feature_dim)
        dm = DeferredShaderModuleAOV(
            feature_dim=feature_dim,
            enable_view_encoding=True,
            auxiliary_output_dim=aux_dim,
        ).to(device)
        rgb, aux, extras = dm(rc)
        assert rgb.shape == (1, H, W, 3)
        assert aux.shape == (1, H, W, aux_dim)
        assert extras is None
        assert torch.isfinite(rgb).all()
        assert torch.isfinite(aux).all()

    def test_training_loop_loss_decreases(self):
        from gsplat.nht.deferred_shader import DeferredShaderModule
        from gsplat.rendering import rasterization

        torch.manual_seed(42)
        N, fdim = 128, 16
        means = torch.nn.Parameter(torch.randn(N, 3, device=device) * 0.3)
        quats = F.normalize(torch.randn(N, 4, device=device), dim=-1)
        scales = torch.nn.Parameter(torch.full((N, 3), -1.5, device=device))
        opacities = torch.nn.Parameter(torch.zeros(N, device=device))
        features = torch.nn.Parameter(torch.randn(N, fdim, device=device))

        dm = DeferredShaderModule(feature_dim=fdim, enable_view_encoding=True).to(
            device
        )
        K, c2w, W, H = _make_camera(width=32, height=24, device=device)
        viewmats = torch.linalg.inv(c2w)
        target = torch.rand(1, H, W, 3, device=device)

        opt = torch.optim.Adam(
            [means, scales, opacities, features] + list(dm.parameters()), lr=1e-2
        )

        losses = []
        for _ in range(10):
            opt.zero_grad()
            rc, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities),
                colors=features,
                viewmats=viewmats,
                Ks=K,
                width=W,
                height=H,
                nht_params=NHTParams(),
                with_eval3d=True,
                with_ut=True,
                packed=False,
                sh_degree=None,
            )
            colors, _ = dm(rc)
            loss = F.l1_loss(colors, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert (
            losses[-1] < losses[0]
        ), f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ===================================================================
# NHT exporter
# ===================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestNHTExporter:
    def test_export_ply(self):
        from gsplat.nht.exporter import export_splats_nht

        N = 50
        data = export_splats_nht(
            means=torch.randn(N, 3),
            scales=torch.randn(N, 3),
            quats=F.normalize(torch.randn(N, 4), dim=-1),
            opacities=torch.rand(N),
            features=torch.randn(N, 16),
        )
        assert isinstance(data, bytes) and len(data) > 0

    @pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
    def test_export_with_deferred_module(self):
        from gsplat.nht.deferred_shader import DeferredShaderModule
        from gsplat.nht.exporter import export_splats_nht

        N, fdim = 50, 16
        dm = DeferredShaderModule(feature_dim=fdim, enable_view_encoding=True).to(
            device
        )
        state = {"state_dict": dm.state_dict(), "config": {"feature_dim": fdim}}

        with tempfile.TemporaryDirectory() as tmp:
            ply_path = os.path.join(tmp, "test.ply")
            export_splats_nht(
                means=torch.randn(N, 3),
                scales=torch.randn(N, 3),
                quats=F.normalize(torch.randn(N, 4), dim=-1),
                opacities=torch.rand(N),
                features=torch.randn(N, fdim),
                deferred_module=state,
                save_to=ply_path,
            )
            assert os.path.exists(ply_path)
            assert os.path.exists(ply_path.replace(".ply", ".deferred.pt"))

    def test_export_features_fp16_roundtrip(self):
        """``f_nht_*`` properties round-trip through fp16 in the PLY binary.

        Verifies (a) the header advertises ``ushort`` features and carries the
        NHT_F16 marker comment, (b) the per-feature byte width is 2 (half),
        and (c) reading the raw bits back as fp16 matches a manual cast of the
        input features. Means/scales/quats stay fp32.
        """
        import numpy as np

        from gsplat.nht.exporter import (
            NHT_PLY_FP16_COMMENT,
            export_splats_nht,
        )

        N, fdim = 32, 12
        means = torch.randn(N, 3)
        scales = torch.randn(N, 3)
        quats = F.normalize(torch.randn(N, 4), dim=-1)
        opacities = torch.rand(N)
        features = torch.randn(N, fdim) * 3.0  # exercise a real fp16 range

        data = export_splats_nht(
            means=means,
            scales=scales,
            quats=quats,
            opacities=opacities,
            features=features,
        )

        header_end = data.find(b"end_header\n") + len(b"end_header\n")
        header = data[:header_end].decode("ascii")
        assert NHT_PLY_FP16_COMMENT in header
        assert f"property ushort f_nht_0" in header
        assert f"property ushort f_nht_{fdim - 1}" in header
        # opacities / means / quats / scales must still be float32
        assert "property float x" in header
        assert "property float opacity" in header

        # ``f4`` for the 7 fp32 fields (x,y,z,opacity,scale,rot) + ``u2`` for
        # each fp16 feature.
        dtype_fields = [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
        ]
        dtype_fields += [(f"f_nht_{j}", "<u2") for j in range(fdim)]
        dtype_fields.append(("opacity", "<f4"))
        dtype_fields += [(f"scale_{i}", "<f4") for i in range(3)]
        dtype_fields += [(f"rot_{i}", "<f4") for i in range(4)]
        dt = np.dtype(dtype_fields)
        assert dt.itemsize == 3 * 4 + fdim * 2 + 1 * 4 + 3 * 4 + 4 * 4

        body = np.frombuffer(data[header_end:], dtype=dt)
        assert body.shape == (N,)

        feature_bits = np.stack([body[f"f_nht_{j}"] for j in range(fdim)], axis=-1)
        decoded = feature_bits.view(np.float16).astype(np.float32)
        expected = features.to(torch.float16).to(torch.float32).numpy()
        np.testing.assert_array_equal(decoded, expected)

    @pytest.mark.skipif(not _tcnn_available, reason="tinycudann not available")
    def test_export_deferred_module_backbone_fp16(self):
        """Companion ``.deferred.pt`` stores tcnn backbone weights as fp16."""
        from gsplat.nht.deferred_shader import DeferredShaderModule
        from gsplat.nht.exporter import export_splats_nht

        N, fdim = 8, 16
        dm = DeferredShaderModule(
            feature_dim=fdim,
            enable_view_encoding=False,
        ).to(device)
        # auxiliary_output_dim == 0 ⇒ no Linear heads; the backbone is the
        # only learnable submodule and is the one we expect to be downcast.
        original = {k: v.detach().clone() for k, v in dm.state_dict().items()}
        state = {"state_dict": dm.state_dict(), "config": {"feature_dim": fdim}}

        with tempfile.TemporaryDirectory() as tmp:
            ply_path = os.path.join(tmp, "test.ply")
            export_splats_nht(
                means=torch.randn(N, 3),
                scales=torch.randn(N, 3),
                quats=F.normalize(torch.randn(N, 4), dim=-1),
                opacities=torch.rand(N),
                features=torch.randn(N, fdim),
                deferred_module=state,
                save_to=ply_path,
            )
            companion = ply_path.replace(".ply", ".deferred.pt")
            assert os.path.exists(companion)
            loaded = torch.load(companion, map_location="cpu", weights_only=True)
            saved_sd = loaded["state_dict"]

            assert "backbone.params" in saved_sd
            assert saved_sd["backbone.params"].dtype == torch.float16
            # Roundtrip should be bit-exact when the source is already fp32-
            # quantized-as-fp16 — and at minimum within fp16 representational
            # accuracy of the original training weights.
            upcast = saved_sd["backbone.params"].to(torch.float32)
            orig = original["backbone.params"].cpu().to(torch.float32)
            assert torch.allclose(upcast, orig, atol=0, rtol=0) or torch.allclose(
                upcast, orig.to(torch.float16).to(torch.float32), atol=0
            )

    def test_cast_state_dict_helpers(self):
        from gsplat.nht.exporter import (
            cast_state_dict_to_fp16,
            cast_state_dict_to_fp32,
        )

        sd = {
            "backbone.params": torch.randn(64, dtype=torch.float32),
            "output_proj.weight": torch.randn(8, 16, dtype=torch.float32),
            "output_proj.bias": torch.randn(8, dtype=torch.float32),
            "step": torch.tensor(42, dtype=torch.int64),
        }

        # Prefix-restricted downcast: backbone-only.
        down = cast_state_dict_to_fp16(sd, prefix="backbone.")
        assert down["backbone.params"].dtype == torch.float16
        assert down["output_proj.weight"].dtype == torch.float32
        assert down["output_proj.bias"].dtype == torch.float32
        assert down["step"].dtype == torch.int64

        # Unrestricted downcast: all fp32 → fp16, integer kept.
        all_down = cast_state_dict_to_fp16(sd)
        assert all(v.dtype == torch.float16 for k, v in all_down.items() if k != "step")
        assert all_down["step"].dtype == torch.int64

        # Round trip preserves shapes and is the inverse on storage dtype.
        up = cast_state_dict_to_fp32(down)
        for k in sd:
            assert up[k].shape == sd[k].shape
            assert up[k].dtype == sd[k].dtype
