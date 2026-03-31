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
from gsplat.cuda._wrapper import (
    SUPPORTED_CHANNELS,
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
            Ks=torch.tensor(
                [[[50.0, 0, 16], [0, 50, 12], [0, 0, 1]]], device=d
            ),
            width=32,
            height=24,
            nht=True,
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


def _rasterize_nht(feature_dim=16, N=256, width=32, height=24, requires_grad=False):
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
        nht=True,
        with_eval3d=True,
        with_ut=True,
        packed=False,
        sh_degree=None,
    )
    return rc, ra, info, means, scales, opacities, features, K, c2w, W, H


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
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=feat, viewmats=viewmats, Ks=K, width=W, height=H,
                nht=True, with_eval3d=True, with_ut=True, packed=False,
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
            assert rel_err < 0.05, (
                f"Gradient mismatch [{gi},{fi}]: analytic={a:.6f} fd={fd:.6f}"
            )


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
        from gsplat.cuda._wrapper import _find_next_supported

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

    @pytest.mark.parametrize("view_encoding_type", ["sh", "fourier"])
    def test_backward(self, view_encoding_type):
        mod = self._make(view_encoding_type=view_encoding_type)
        _, _, W, H = _make_camera(width=32, height=24, device=device)
        inp = torch.randn(
            1, H, W, _deferred_shader_raster_channels(mod), device=device, requires_grad=True
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

        rc, _, _, _, _, _, _, _, _, W, H = _rasterize_nht(feature_dim=feature_dim)
        dm = DeferredShaderModule(
            feature_dim=feature_dim, enable_view_encoding=True
        ).to(device)
        assert rc.shape[-1] == dm.encoded_dim + 3
        colors, _ = dm(rc)
        assert colors.shape == (1, H, W, 3)
        assert not torch.isnan(colors).any()

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

        dm = DeferredShaderModule(feature_dim=fdim, enable_view_encoding=True).to(device)
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
                means=means, quats=quats, scales=torch.exp(scales),
                opacities=torch.sigmoid(opacities), colors=features,
                viewmats=viewmats, Ks=K, width=W, height=H,
                nht=True, with_eval3d=True, with_ut=True, packed=False,
                sh_degree=None,
            )
            colors, _ = dm(rc)
            loss = F.l1_loss(colors, target)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )


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
        dm = DeferredShaderModule(feature_dim=fdim, enable_view_encoding=True).to(device)
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
