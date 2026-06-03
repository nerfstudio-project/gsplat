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

"""Inference Scene Packing tests.

Tests for GaussianInferenceScene class, pack_gaussian_inference_scene C++ op,
classmethod constructors, activation-contract validation, fp16 clamp
policy, detach-and-warn behavior, distributed rejection, and parity.
"""

import pytest
import warnings
import torch

gsplat_scene = pytest.importorskip("gsplat_scene")
SHCompressionMode = gsplat_scene.SHCompressionMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(device, seed):
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def _sh_compression_mode(sh_compression):
    from gsplat_scene.sh_compression import SH_COMPRESSION_MAP

    return SH_COMPRESSION_MAP[sh_compression]


def _make_gaussians(N=100, sh_degree=None, device="cuda", seed=11):
    """Create activated Gaussian parameters for testing."""
    generator = _make_generator(device, seed)
    means = torch.randn(N, 3, device=device, generator=generator)
    quats = torch.randn(N, 4, device=device, generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = (
        torch.rand(N, 3, device=device, generator=generator) * 0.1 + 0.01
    )  # positive (activated)
    opacities = torch.rand(N, device=device, generator=generator)  # [0, 1] (activated)
    if sh_degree is not None and sh_degree >= 0:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device, generator=generator)
    else:
        colors = torch.rand(N, 3, device=device, generator=generator)
    return means, quats, scales, opacities, colors


def _make_gaussian_scene(N=100, sh_degree=None, device="cuda", seed=23):
    """Build a GaussianScene with raw (log-space) splats."""
    from gsplat_scene import GaussianScene

    generator = _make_generator(device, seed)
    means = torch.randn(N, 3, device=device, generator=generator)
    quats = torch.randn(N, 4, device=device, generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales_raw = torch.randn(N, 3, device=device, generator=generator)  # log-space
    opacities_raw = torch.randn(N, device=device, generator=generator)  # logit-space
    if sh_degree is not None and sh_degree >= 0:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(N, K, 3, device=device, generator=generator)
    else:
        colors = torch.rand(N, 3, device=device, generator=generator)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales_raw),
            "opacities": torch.nn.Parameter(opacities_raw),
            "colors": torch.nn.Parameter(colors),
        }
    )
    return GaussianScene.from_splats(splats, id="test_scene")


def _make_packed_component(N=10, device="cuda"):
    return {
        "means_planar": torch.empty(3, N, dtype=torch.float32, device=device),
        "qso_packed": torch.empty(N, 8, dtype=torch.float16, device=device),
        "colors_packed": torch.empty(N, 4, dtype=torch.float16, device=device),
        "sh_degree": -1,
        "sh_compression_mode": SHCompressionMode.NONE,
    }


def _make_camera(device="cuda"):
    viewmat = torch.eye(4, device=device)
    viewmat[2, 3] = 3.0
    K = torch.tensor(
        [
            [500.0, 0.0, 128.0],
            [0.0, 500.0, 128.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )
    return viewmat, K


def _pack_scene_python(
    means, quats, scales, opacities, colors, sh_degree, sh_compression="none"
):
    """Pack scene in Python matching the C++ scene packing contract."""
    N = means.size(0)
    means_planar = means.t().contiguous()
    qso_packed = torch.empty(N, 8, dtype=torch.float16, device=means.device)
    qso_packed[:, 0:4] = quats.half()
    qso_packed[:, 4:7] = scales.half()
    qso_packed[:, 7:8] = opacities.unsqueeze(1).half()

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
    return means_planar, qso_packed, colors_packed


# ======================================================================
# A. Smoke test for pack op
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_op_smoke():
    """pack_gaussian_inference_scene op is callable via gsplat_scene_cuda and produces no grad_fn."""
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    assert callable(_SCENE_CUDA.pack_gaussian_inference_scene)

    # Verify CUDA dispatch by calling it
    means, quats, scales, opacities, colors = _make_gaussians(N=10)
    mp, qso_packed, cp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, -1, 0
    )
    assert mp.shape == (3, 10)
    assert qso_packed.shape == (10, 8)
    assert qso_packed.dtype == torch.float16

    # Verify no grad_fn on outputs (torch::NoGradGuard in C++)
    means_g = means.clone().requires_grad_(True)
    mp_g, qso_packed_g, cp_g = _SCENE_CUDA.pack_gaussian_inference_scene(
        means_g, quats, scales, opacities, colors, -1, 0
    )
    assert mp_g.grad_fn is None


# ======================================================================
# B. Parity between Python packing and C++ pack op
# ======================================================================

_PARITY_PACK_CASES = [
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
@pytest.mark.parametrize("sh_degree,sh_compression", _PARITY_PACK_CASES)
def test_pack_op_parity_with_python(sh_degree, sh_compression):
    """C++ pack op must be bit-exact with Python packing (when values in fp16 range)."""
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    means, quats, scales, opacities, colors = _make_gaussians(
        N=200, sh_degree=sh_degree
    )

    # Python packing with values in fp16 range by construction.
    mp_py, qso_packed_py, cp_py = _pack_scene_python(
        means, quats, scales, opacities, colors, sh_degree, sh_compression
    )

    # C++ packing
    sh_deg = -1 if sh_degree is None else sh_degree
    sh_compression_mode = _sh_compression_mode(sh_compression)
    mp_cpp, qso_packed_cpp, cp_cpp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, sh_deg, int(sh_compression_mode)
    )

    torch.testing.assert_close(
        mp_cpp, mp_py, atol=0, rtol=0, msg="means_planar mismatch"
    )
    torch.testing.assert_close(
        qso_packed_cpp, qso_packed_py, atol=0, rtol=0, msg="qso_packed mismatch"
    )
    torch.testing.assert_close(
        cp_cpp, cp_py, atol=0, rtol=0, msg="colors_packed mismatch"
    )


# ======================================================================
# C. Constructor detach-and-warn tests
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_detach_no_grad_no_warning():
    """All inputs grad-free: no RuntimeWarning, packed tensors grad-free."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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
    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warns) == 0, f"unexpected warnings: {runtime_warns}"
    assert not scene.means_planar.requires_grad
    assert not scene.qso_packed.requires_grad
    assert not scene.colors_packed.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_detach_with_grad_warns():
    """Some inputs grad=True: single RuntimeWarning naming affected tensors."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    means_g = means.clone().requires_grad_(True)
    scales_g = scales.clone().requires_grad_(True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scene = GaussianInferenceScene.from_gaussian_tensors(
            means_g,
            quats,
            scales_g,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test",
        )
    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warns) >= 1
    msg = str(runtime_warns[0].message)
    assert "means" in msg
    assert "scales" in msg
    assert "detached grad-tracked inputs" in msg

    # Packed tensors must be grad-free
    assert not scene.means_planar.requires_grad
    assert not scene.qso_packed.requires_grad
    assert not scene.colors_packed.requires_grad

    # Bit-exact vs explicit detach
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        scene2 = GaussianInferenceScene.from_gaussian_tensors(
            means_g.detach(),
            quats,
            scales_g.detach(),
            opacities,
            colors,
            sh_degree=None,
            sh_compression="none",
            id="test2",
        )
    torch.testing.assert_close(scene.means_planar, scene2.means_planar, atol=0, rtol=0)
    torch.testing.assert_close(scene.qso_packed, scene2.qso_packed, atol=0, rtol=0)
    torch.testing.assert_close(
        scene.colors_packed, scene2.colors_packed, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pack_op_with_grad_no_grad_fn():
    """Direct pack op with grad-tracked input: no grad_fn on outputs."""
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    means, quats, scales, opacities, colors = _make_gaussians(N=20)
    means_g = means.clone().requires_grad_(True)
    mp, qso_packed, cp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means_g, quats, scales, opacities, colors, -1, 0
    )
    assert mp.grad_fn is None
    assert qso_packed.grad_fn is None
    assert cp.grad_fn is None


# ======================================================================
# D. Distributed rejection test
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_distributed_rejection_from_gaussian_scene():
    """from_gaussian_scene raises RuntimeError when world_size > 1."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=50)

    # Simulate distributed init with world_size=1 via file store
    import os
    import tempfile

    tmpfile = tempfile.NamedTemporaryFile(delete=False)
    tmpfile.close()
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
                init_method=f"file://{tmpfile.name}",
            )

        # world_size=1 should NOT raise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            inference_scene = GaussianInferenceScene.from_gaussian_scene(
                scene, id="test", sh_compression="none"
            )
            assert not inference_scene.is_empty()

        # Simulate a multi-rank process group.
        orig_ws = torch.distributed.get_world_size
        try:
            torch.distributed.get_world_size = lambda: 2
            with pytest.raises(RuntimeError, match="not supported under distributed"):
                GaussianInferenceScene.from_gaussian_scene(
                    scene, id="test2", sh_compression="none"
                )
        finally:
            torch.distributed.get_world_size = orig_ws
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        try:
            os.unlink(tmpfile.name)
        except FileNotFoundError:
            pass
        for key in ("MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_tensors_no_distributed_check():
    """from_gaussian_tensors has no distributed check."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)

    # This should succeed regardless of distributed state
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
    assert not scene.is_empty()


# ======================================================================
# E. fp16 clamp/warn tests
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_all_safe_no_warning():
    """All-safe: no warning, packed values bit-exact to simple fp16 cast."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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

    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warns) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_one_lane_exceeds():
    """One lane > fp16 max: single RuntimeWarning, packed value = fp16 max."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    fp16_max = torch.finfo(torch.float16).max

    # Set one scale value beyond fp16 range
    scales[0, 0] = fp16_max + 1000.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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

    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warns) == 1
    msg = str(runtime_warns[0].message)
    assert "scales" in msg
    assert "clamped 1 elements" in msg

    # Verify the packed value is clamped to fp16 max
    # Inference lanes 4-7 are scales; check first Gaussian, first scale dim
    packed_scale_0 = scene.qso_packed[0, 4].float().item()
    assert packed_scale_0 == pytest.approx(fp16_max, rel=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp16_many_lanes_exceed():
    """Many lanes > fp16 max: warning still single-shot, count accurate."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    fp16_max = torch.finfo(torch.float16).max

    # Set 10 scale values beyond fp16 range
    scales[:5, :2] = fp16_max + 5000.0  # 10 elements total

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
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

    runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
    assert len(runtime_warns) == 1
    msg = str(runtime_warns[0].message)
    assert "scales" in msg
    assert "clamped 10 elements" in msg


# ======================================================================
# F. Activation-contract validation tests
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_non_positive_scales():
    """Non-positive scales: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    scales[0, 0] = -0.1

    with pytest.raises(ValueError, match="scales contain non-positive values"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_opacities_out_of_range():
    """Opacities outside [0,1]: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    opacities[0] = 1.5

    with pytest.raises(ValueError, match="opacities outside \\[0, 1\\]"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_nan_inf():
    """NaN/Inf in tensor: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    means[0, 0] = float("nan")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
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

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    colors[0, 0] = float("inf")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_nan_inf_non_contiguous_tensor():
    """Activation validation should report non-finite values in non-contiguous tensors."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, _ = _make_gaussians(N=50)
    colors = torch.rand(3, 50, device="cuda", generator=_make_generator("cuda", 37)).t()
    assert not colors.is_contiguous()
    colors[0, 0] = float("inf")

    with pytest.raises(ValueError, match="contains NaN or Inf"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_invalid_sh_compression():
    """Invalid sh_compression: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)

    with pytest.raises(ValueError, match="sh_compression must be one of"):
        GaussianInferenceScene.from_gaussian_tensors(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree=None,
            sh_compression="invalid",
            id="test",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_skips_activation_checks():
    """from_gaussian_scene not affected by activation checks (raw negative scales ok)."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=50)
    # Raw scales may be negative (log-space). from_gaussian_scene applies exp()
    # internally, so activation checks should be skipped.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_scene = GaussianInferenceScene.from_gaussian_scene(
            scene, id="test", sh_compression="none"
        )
        assert not inference_scene.is_empty()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_rgb_uses_safe_default_compression():
    """RGB scenes should convert without explicitly overriding sh_compression."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=50)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_scene = GaussianInferenceScene.from_gaussian_scene(scene, id="test")

    assert inference_scene.sh_degree == -1
    assert inference_scene.sh_compression_mode is SHCompressionMode.NONE
    assert inference_scene.colors_packed.shape == (50, 4)
    assert inference_scene.colors_packed.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_concatenates_sh0_shn():
    """GaussianScene with sh0/shN but no colors should preserve all SH coefficients."""
    from gsplat_scene import GaussianInferenceScene, GaussianScene

    N = 20
    generator = _make_generator("cuda", 41)
    means = torch.randn(N, 3, device="cuda", generator=generator)
    quats = torch.randn(N, 4, device="cuda", generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales_raw = torch.randn(N, 3, device="cuda", generator=generator)
    opacities_raw = torch.randn(N, device="cuda", generator=generator)
    colors = torch.randn(N, 9, 3, device="cuda", generator=generator)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales_raw),
            "opacities": torch.nn.Parameter(opacities_raw),
            "sh0": torch.nn.Parameter(colors[:, :1]),
            "shN": torch.nn.Parameter(colors[:, 1:]),
        }
    )
    scene = GaussianScene.from_splats(splats, id="sh_scene")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_scene = GaussianInferenceScene.from_gaussian_scene(scene, id="test")

    assert inference_scene.sh_degree == 2
    assert inference_scene.colors_packed.shape == (N, 9, 3)
    assert inference_scene.colors_packed.dtype == torch.float32
    torch.testing.assert_close(inference_scene.colors_packed, colors, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_normalizes_non_unit_quats():
    """Raw GaussianScene quaternions need not already be unit length."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=30)
    raw_quats = scene.splats["quats"].detach().clone() * 3.0
    scene.splats["quats"].data.copy_(raw_quats)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_scene = GaussianInferenceScene.from_gaussian_scene(scene, id="test")

    expected = raw_quats / raw_quats.norm(dim=1, keepdim=True)
    torch.testing.assert_close(
        inference_scene.qso_packed[:, :4], expected.half(), atol=0, rtol=0
    )


# ======================================================================
# G. Scene state error tests
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_put_duplicate_name():
    """Duplicate put() name: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=10)
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    mp, qso_packed, cp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, -1, 0
    )

    scene = GaussianInferenceScene("test")
    component = {
        "means_planar": mp,
        "qso_packed": qso_packed,
        "colors_packed": cp,
        "sh_degree": -1,
        "sh_compression_mode": SHCompressionMode.NONE,
    }
    scene.put("comp1", component)

    with pytest.raises(ValueError, match="already present in GaussianInferenceScene"):
        scene.put("comp1", component)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_put_empty_name():
    """Empty put() name: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    scene = GaussianInferenceScene("test")
    with pytest.raises(ValueError, match="component name must not be empty"):
        scene.put("", {"means_planar": None})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_put_rejects_non_contiguous_packed_tensors():
    """Packed scene tensors must satisfy the render-time layout contract."""
    from gsplat_scene import GaussianInferenceScene

    n = 10
    component = {
        "means_planar": torch.empty(n, 3, device="cuda").t(),
        "qso_packed": torch.empty(n, 8, dtype=torch.float16, device="cuda"),
        "colors_packed": torch.empty(n, 4, dtype=torch.float16, device="cuda"),
        "sh_degree": -1,
        "sh_compression_mode": SHCompressionMode.NONE,
    }

    assert not component["means_planar"].is_contiguous()
    scene = GaussianInferenceScene("test")
    with pytest.raises(ValueError, match="means_planar must be contiguous"):
        scene.put("comp", component)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case,match",
    [
        ("means_shape", "means_planar must have shape"),
        ("qso_packed_shape", "qso_packed must have shape"),
        ("colors_rows", "colors_packed.shape\\[0\\] must match"),
        ("means_dtype", "means_planar must have dtype"),
        ("colors_device", "colors_packed device"),
    ],
)
def test_put_rejects_bad_packed_tensor_invariants(case, match):
    """put() validates packed shapes, devices, and dtypes before storing state."""
    from gsplat_scene import GaussianInferenceScene

    component = _make_packed_component(N=10)
    override = {
        "means_shape": {
            "means_planar": torch.empty(10, 3, dtype=torch.float32, device="cuda")
        },
        "qso_packed_shape": {
            "qso_packed": torch.empty(10, 7, dtype=torch.float16, device="cuda")
        },
        "colors_rows": {
            "colors_packed": torch.empty(11, 4, dtype=torch.float16, device="cuda")
        },
        "means_dtype": {
            "means_planar": torch.empty(3, 10, dtype=torch.float16, device="cuda")
        },
        "colors_device": {"colors_packed": torch.empty(10, 4, dtype=torch.float16)},
    }[case]
    component.update(override)
    scene = GaussianInferenceScene("test")

    with pytest.raises((TypeError, ValueError), match=match):
        scene.put("comp", component)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_unknown_component():
    """Unknown get() component: KeyError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=10)

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

    with pytest.raises(KeyError):
        scene.get("nonexistent")

    with pytest.raises(KeyError):
        scene.get(99)


# ======================================================================
# H. Parity between two classmethods
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_classmethod_parity():
    """from_gaussian_scene and from_gaussian_tensors with manual activation must be bit-exact."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=100)
    splats = scene.splats

    # from_gaussian_scene (auto-activates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_a = GaussianInferenceScene.from_gaussian_scene(
            scene, id="test_a", sh_compression="none"
        )

    # from_gaussian_tensors (manual activation)
    means = splats["means"].detach()
    quats = splats["quats"].detach()
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = splats["scales"].detach().exp()
    opacities = splats["opacities"].detach().sigmoid()
    colors = splats["colors"].detach()

    inference_b = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=None,
        sh_compression="none",
        id="test_b",
    )

    torch.testing.assert_close(
        inference_a.means_planar,
        inference_b.means_planar,
        atol=0,
        rtol=0,
        msg="means_planar mismatch between classmethods",
    )
    torch.testing.assert_close(
        inference_a.qso_packed,
        inference_b.qso_packed,
        atol=0,
        rtol=0,
        msg="qso_packed mismatch between classmethods",
    )
    torch.testing.assert_close(
        inference_a.colors_packed,
        inference_b.colors_packed,
        atol=0,
        rtol=0,
        msg="colors_packed mismatch between classmethods",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_classmethod_parity_sh3():
    """from_gaussian_scene and from_gaussian_tensors parity with SH degree 3."""
    from gsplat_scene import GaussianInferenceScene

    scene = _make_gaussian_scene(N=100, sh_degree=3)
    splats = scene.splats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_a = GaussianInferenceScene.from_gaussian_scene(
            scene, id="test_a", sh_compression="32b"
        )

    means = splats["means"].detach()
    quats = splats["quats"].detach()
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales = splats["scales"].detach().exp()
    opacities = splats["opacities"].detach().sigmoid()
    colors = splats["colors"].detach()

    inference_b = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=3,
        sh_compression="32b",
        id="test_b",
    )

    torch.testing.assert_close(
        inference_a.means_planar, inference_b.means_planar, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inference_a.qso_packed, inference_b.qso_packed, atol=0, rtol=0
    )
    torch.testing.assert_close(
        inference_a.colors_packed, inference_b.colors_packed, atol=0, rtol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "sh_compression,expected_mode,expected_dtype,expected_shape",
    [
        ("none", SHCompressionMode.NONE, torch.float16, (40, 16, 3)),
        ("32b", SHCompressionMode.PACKED_32B, torch.float32, (40, 48)),
        ("16b", SHCompressionMode.PACKED_16B, torch.float16, (40, 48)),
    ],
)
def test_sh3_compression_metadata_layout_and_values(
    sh_compression, expected_mode, expected_dtype, expected_shape
):
    """GaussianInferenceScene records SH3 mode metadata and mode-specific layouts."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=40, sh_degree=3)

    scene = GaussianInferenceScene.from_gaussian_tensors(
        means,
        quats,
        scales,
        opacities,
        colors,
        sh_degree=3,
        sh_compression=sh_compression,
        id="test",
    )

    assert scene.sh_degree == 3
    assert scene.sh_compression_mode is expected_mode
    assert scene.colors_packed.dtype == expected_dtype
    assert scene.colors_packed.shape == expected_shape

    component = scene.get("test")
    assert component["sh_degree"] == 3
    assert component["sh_compression_mode"] is expected_mode
    assert component["colors_packed"].dtype == expected_dtype
    assert component["colors_packed"].shape == expected_shape

    if sh_compression == "none":
        expected = colors.half()
    elif sh_compression == "32b":
        expected = colors.contiguous().view(40, 48)
    else:
        expected = colors.half().view(40, 48)
    torch.testing.assert_close(scene.colors_packed, expected, atol=0, rtol=0)


# ======================================================================
# I. is_empty / release tests
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_is_empty_and_release():
    """After construction: not empty. After release: empty. Release is idempotent."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)

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

    # After construction
    assert not scene.is_empty()
    assert scene.num_gaussians == 50

    # After release
    scene.release()
    assert scene.is_empty()
    assert scene.num_gaussians == 0
    assert scene.means_planar is None
    assert scene.qso_packed is None
    assert scene.colors_packed is None
    assert scene.component_names == []
    assert scene.component_index.shape == (0,)

    # Idempotent
    scene.release()
    assert scene.is_empty()


def test_zero_length_component_is_empty_and_replaceable():
    """Zero-length packed components should behave like an empty scene."""
    from gsplat_scene import GaussianInferenceScene

    scene = GaussianInferenceScene("test")
    scene.put("empty", _make_packed_component(N=0, device="cpu"))

    assert scene.is_empty()
    assert scene.num_gaussians == 0

    scene.put("nonempty", _make_packed_component(N=2, device="cpu"))

    assert not scene.is_empty()
    assert scene.num_gaussians == 2
    assert scene.component_names == ["nonempty"]
    torch.testing.assert_close(scene.component_index, torch.zeros(2, dtype=torch.long))


# ======================================================================
# J. Parity between one-shot and packed-scene APIs
# ======================================================================

_RENDER_PARITY_CASES = [
    # (sh_degree, sh_compression)
    (None, "none"),  # RGB
    (1, "none"),  # SH1
    (3, "none"),  # SH3, no compression
    (3, "32b"),  # SH3, 32b compression
    (3, "16b"),  # SH3, 16b compression
]


@pytest.mark.skip(reason="experimental.render not available in this repo")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sh_degree,sh_compression", _RENDER_PARITY_CASES)
def test_render_parity_cpp_packed_vs_python_packed(sh_degree, sh_compression):
    """Pack with C++ op, render via gaussian_render_inference_only, compare with Python packing."""
    from experimental.render.kernels._backend import _C  # noqa: F401
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    width, height = 256, 256
    means, quats, scales, opacities, colors = _make_gaussians(
        N=200, sh_degree=sh_degree
    )
    viewmat, K = _make_camera()

    sh_deg = -1 if sh_degree is None else sh_degree
    sh_compression_mode = _sh_compression_mode(sh_compression)

    # Python-packed reference path
    means_planar_py, qso_packed_py, colors_packed_py = _pack_scene_python(
        means, quats, scales, opacities, colors, sh_degree, sh_compression
    )

    with torch.no_grad():
        renders_py, alphas_py = torch.ops.experimental.gaussian_render_inference_only(
            means_planar_py,
            qso_packed_py,
            colors_packed_py,
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
            int(sh_compression_mode),
            None,
        )

    # Pack with C++ op
    mp, qso_packed, cp = _SCENE_CUDA.pack_gaussian_inference_scene(
        means, quats, scales, opacities, colors, sh_deg, int(sh_compression_mode)
    )

    # Render with C++-packed scene
    with torch.no_grad():
        renders_new, alphas_new = torch.ops.experimental.gaussian_render_inference_only(
            mp,
            qso_packed,
            cp,
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
            int(sh_compression_mode),
            None,
        )

    torch.testing.assert_close(
        renders_new, renders_py, atol=0, rtol=0, msg="renders mismatch"
    )
    torch.testing.assert_close(
        alphas_new, alphas_py, atol=0, rtol=0, msg="alphas mismatch"
    )


# ======================================================================
# Additional: Multi-component put/get test
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multi_component_put_get():
    """Multiple components can be put and retrieved correctly."""
    from gsplat_scene import GaussianInferenceScene
    from gsplat_scene.kernels._backend import _SCENE_CUDA  # noqa: F401

    N1, N2 = 30, 50
    means1, quats1, scales1, opacities1, colors1 = _make_gaussians(N=N1)
    means2, quats2, scales2, opacities2, colors2 = _make_gaussians(N=N2)

    mp1, qso_packed1, cp1 = _SCENE_CUDA.pack_gaussian_inference_scene(
        means1, quats1, scales1, opacities1, colors1, -1, 0
    )
    mp2, qso_packed2, cp2 = _SCENE_CUDA.pack_gaussian_inference_scene(
        means2, quats2, scales2, opacities2, colors2, -1, 0
    )

    scene = GaussianInferenceScene("multi")
    scene.put(
        "comp_a",
        {
            "means_planar": mp1,
            "qso_packed": qso_packed1,
            "colors_packed": cp1,
            "sh_degree": -1,
            "sh_compression_mode": SHCompressionMode.NONE,
        },
    )
    scene.put(
        "comp_b",
        {
            "means_planar": mp2,
            "qso_packed": qso_packed2,
            "colors_packed": cp2,
            "sh_degree": -1,
            "sh_compression_mode": SHCompressionMode.NONE,
        },
    )

    assert scene.num_gaussians == N1 + N2
    assert scene.means_planar.shape == (3, N1 + N2)
    assert scene.qso_packed.shape == (N1 + N2, 8)

    # Get by name
    comp_a = scene.get("comp_a")
    assert comp_a["name"] == "comp_a"
    assert comp_a["index"] == 0
    assert comp_a["qso_packed"].shape == (N1, 8)
    torch.testing.assert_close(comp_a["qso_packed"], qso_packed1, atol=0, rtol=0)

    # Get by index
    comp_b = scene.get(1)
    assert comp_b["name"] == "comp_b"
    assert comp_b["qso_packed"].shape == (N2, 8)
    torch.testing.assert_close(comp_b["qso_packed"], qso_packed2, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_rejects_app_opt():
    """from_gaussian_scene raises ValueError for scenes with 'features'."""
    from gsplat_scene import GaussianInferenceScene, GaussianScene

    N = 20
    generator = _make_generator("cuda", 51)
    means = torch.randn(N, 3, device="cuda", generator=generator)
    quats = torch.randn(N, 4, device="cuda", generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales_raw = torch.randn(N, 3, device="cuda", generator=generator)
    opacities_raw = torch.randn(N, device="cuda", generator=generator)
    colors = torch.rand(N, 3, device="cuda", generator=generator)
    features = torch.randn(N, 32, device="cuda", generator=generator)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales_raw),
            "opacities": torch.nn.Parameter(opacities_raw),
            "colors": torch.nn.Parameter(colors),
            "features": torch.nn.Parameter(features),
        }
    )
    scene = GaussianScene.from_splats(splats, id="app_opt_scene")

    with pytest.raises(ValueError, match="appearance-optimized"):
        GaussianInferenceScene.from_gaussian_scene(scene, id="test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_activation_non_unit_quats():
    """Non-unit-norm quaternions: ValueError."""
    from gsplat_scene import GaussianInferenceScene

    means, quats, scales, opacities, colors = _make_gaussians(N=50)
    quats = quats * 5.0  # break unit-norm

    with pytest.raises(ValueError, match="quats are not unit-norm"):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_nan_input_raises():
    """from_gaussian_scene raises ValueError when activation produces Inf."""
    from gsplat_scene import GaussianInferenceScene, GaussianScene

    N = 20
    generator = _make_generator("cuda", 53)
    means = torch.randn(N, 3, device="cuda", generator=generator)
    quats = torch.randn(N, 4, device="cuda", generator=generator)
    scales_raw = torch.randn(N, 3, device="cuda", generator=generator)
    scales_raw[0, 0] = 1e6  # exp(1e6) = Inf
    opacities_raw = torch.randn(N, device="cuda", generator=generator)
    colors = torch.rand(N, 3, device="cuda", generator=generator)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales_raw),
            "opacities": torch.nn.Parameter(opacities_raw),
            "colors": torch.nn.Parameter(colors),
        }
    )
    scene = GaussianScene.from_splats(splats, id="nan_scene")

    with pytest.raises(ValueError, match="contains NaN or Inf after activation"):
        GaussianInferenceScene.from_gaussian_scene(scene, id="test")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quat_column_order_wxyz():
    """Packed qso quaternion columns preserve wxyz input order."""
    from gsplat_scene import GaussianInferenceScene

    N = 5
    quats = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * N, device="cuda")
    quats = quats / quats.norm(dim=1, keepdim=True)
    means = torch.randn(N, 3, device="cuda", generator=_make_generator("cuda", 55))
    scales = (
        torch.rand(N, 3, device="cuda", generator=_make_generator("cuda", 56)) + 0.01
    )
    opacities = torch.rand(N, device="cuda", generator=_make_generator("cuda", 57))
    colors = torch.rand(N, 3, device="cuda", generator=_make_generator("cuda", 58))

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

    torch.testing.assert_close(
        scene.qso_packed[:, :4],
        quats.half(),
        atol=0,
        rtol=0,
        msg="qso_packed quaternion columns must preserve wxyz order",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_gaussian_scene_rejects_multi_component():
    """from_gaussian_scene raises ValueError for multi-component scenes."""
    from gsplat_scene import GaussianInferenceScene, GaussianScene

    N = 20
    generator = _make_generator("cuda", 61)
    means = torch.randn(N, 3, device="cuda", generator=generator)
    quats = torch.randn(N, 4, device="cuda", generator=generator)
    quats = quats / quats.norm(dim=1, keepdim=True)
    scales_raw = torch.randn(N, 3, device="cuda", generator=generator)
    opacities_raw = torch.randn(N, device="cuda", generator=generator)
    colors = torch.rand(N, 3, device="cuda", generator=generator)
    splats = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "quats": torch.nn.Parameter(quats),
            "scales": torch.nn.Parameter(scales_raw),
            "opacities": torch.nn.Parameter(opacities_raw),
            "colors": torch.nn.Parameter(colors),
        }
    )
    scene = GaussianScene.from_splats(splats, id="first")

    means2 = torch.randn(10, 3, device="cuda", generator=generator)
    quats2 = torch.randn(10, 4, device="cuda", generator=generator)
    quats2 = quats2 / quats2.norm(dim=1, keepdim=True)
    scales2 = torch.randn(10, 3, device="cuda", generator=generator)
    opacities2 = torch.randn(10, device="cuda", generator=generator)
    colors2 = torch.rand(10, 3, device="cuda", generator=generator)
    splats2 = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means2),
            "quats": torch.nn.Parameter(quats2),
            "scales": torch.nn.Parameter(scales2),
            "opacities": torch.nn.Parameter(opacities2),
            "colors": torch.nn.Parameter(colors2),
        }
    )
    scene.put("second", splats2)

    with pytest.raises(ValueError, match="multi-component"):
        GaussianInferenceScene.from_gaussian_scene(scene, id="test")
