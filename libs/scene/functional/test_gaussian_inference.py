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

"""Functional tests for pack_gaussian_inference_scene.

Run with:
    pytest libs/scene/functional/test_gaussian_inference.py -v
"""

from __future__ import annotations

import pytest
import torch

gsplat_scene = pytest.importorskip("gsplat_scene")
SHCompressionMode = gsplat_scene.SHCompressionMode

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda"
N = 128  # default Gaussian count for tests


def _make_generator(device, seed):
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(seed)
    return generator


def _make_inputs(
    n: int = N,
    sh_degree: int = -1,
    dtype: torch.dtype = torch.float32,
    device: str = DEVICE,
    seed: int = 11,
):
    """Return a dict of valid activated Gaussian tensors."""
    generator = _make_generator(device, seed)
    means = torch.randn(n, 3, dtype=dtype, device=device, generator=generator)
    quats = torch.randn(n, 4, dtype=dtype, device=device, generator=generator)
    quats = quats / quats.norm(dim=-1, keepdim=True)  # unit quaternions
    scales = (
        torch.rand(n, 3, dtype=dtype, device=device, generator=generator) + 0.01
    )  # positive
    opacities = torch.sigmoid(
        torch.randn(n, dtype=dtype, device=device, generator=generator)
    )

    if sh_degree == -1:
        colors = torch.randn(n, 3, dtype=dtype, device=device, generator=generator)
    else:
        K = (sh_degree + 1) ** 2
        colors = torch.randn(n, K, 3, dtype=dtype, device=device, generator=generator)

    return dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )


def _call(
    n: int = N,
    sh_degree: int = -1,
    sh_compression_mode: SHCompressionMode = SHCompressionMode.NONE,
    seed: int = 11,
):
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(n=n, sh_degree=sh_degree, seed=seed)
    return pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=sh_degree,
        sh_compression_mode=sh_compression_mode,
    )


# ---------------------------------------------------------------------------
# Import test
# ---------------------------------------------------------------------------


def test_import():
    from gsplat_scene.functional import pack_gaussian_inference_scene  # noqa: F401

    assert callable(pack_gaussian_inference_scene)


def test_import_from_kernels():
    from gsplat_scene.kernels.gaussian_inference_ops import (
        pack_gaussian_inference_scene,
    )  # noqa: F401

    assert callable(pack_gaussian_inference_scene)


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_means_planar_shape():
    means_planar, qso_packed, _ = _call(N, sh_degree=-1)
    assert means_planar.shape == (3, N), f"expected (3, {N}), got {means_planar.shape}"


def test_qso_packed_shape():
    _, qso_packed, _ = _call(N, sh_degree=-1)
    assert qso_packed.shape == (N, 8), f"expected ({N}, 8), got {qso_packed.shape}"


@pytest.mark.parametrize(
    "sh_degree,sh_compression_mode",
    [
        (-1, SHCompressionMode.NONE),
        (0, SHCompressionMode.NONE),
        (1, SHCompressionMode.NONE),
        (2, SHCompressionMode.NONE),
        (3, SHCompressionMode.NONE),
        (3, SHCompressionMode.PACKED_32B),
        (3, SHCompressionMode.PACKED_16B),
    ],
)
def test_shapes_all_valid_modes(sh_degree, sh_compression_mode):
    means_planar, qso_packed, colors_packed = _call(
        N, sh_degree=sh_degree, sh_compression_mode=sh_compression_mode
    )
    assert means_planar.shape == (3, N)
    assert qso_packed.shape == (N, 8)
    # colors_packed must have N rows on leading dim
    assert colors_packed.shape[0] == N


@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize(
    "sh_degree,sh_compression_mode",
    [
        (-1, SHCompressionMode.NONE),
        (0, SHCompressionMode.NONE),
        (3, SHCompressionMode.PACKED_32B),
    ],
)
def test_shapes_small_n_supported(n, sh_degree, sh_compression_mode):
    means_planar, qso_packed, colors_packed = _call(
        n, sh_degree=sh_degree, sh_compression_mode=sh_compression_mode
    )
    assert means_planar.shape == (3, n)
    assert qso_packed.shape == (n, 8)
    assert colors_packed.shape[0] == n


# ---------------------------------------------------------------------------
# Output dtypes
# ---------------------------------------------------------------------------


def test_means_planar_dtype():
    means_planar, _, _ = _call(N, sh_degree=-1)
    assert means_planar.dtype == torch.float32


def test_qso_packed_dtype():
    _, qso_packed, _ = _call(N, sh_degree=-1)
    assert qso_packed.dtype == torch.float16


def test_colors_rgb_dtype_and_shape():
    """RGB mode: colors_packed should be [N, 4] float16."""
    _, _, colors_packed = _call(
        N, sh_degree=-1, sh_compression_mode=SHCompressionMode.NONE
    )
    assert colors_packed.dtype == torch.float16
    assert colors_packed.shape == (N, 4)


@pytest.mark.parametrize(
    "sh_compression_mode,expected_dtype,expected_shape",
    [
        (SHCompressionMode.NONE, torch.float16, (N, 16, 3)),
        (SHCompressionMode.PACKED_32B, torch.float32, (N, 48)),
        (SHCompressionMode.PACKED_16B, torch.float16, (N, 48)),
    ],
)
def test_colors_sh3_mode_dtype_and_shape(
    sh_compression_mode, expected_dtype, expected_shape
):
    """SH3 compression modes must produce distinct color layouts."""
    _, _, colors_packed = _call(N, sh_degree=3, sh_compression_mode=sh_compression_mode)
    assert colors_packed.dtype == expected_dtype
    assert colors_packed.shape == expected_shape


@pytest.mark.parametrize("sh_compression_mode", list(SHCompressionMode))
def test_colors_sh3_mode_values(sh_compression_mode):
    """SH3 packing preserves values exactly except for fp16 modes' casts."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=3)
    _, _, colors_packed = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=3,
        sh_compression_mode=sh_compression_mode,
    )

    if sh_compression_mode is SHCompressionMode.NONE:
        expected = inp["colors"].to(torch.float16)
    elif sh_compression_mode is SHCompressionMode.PACKED_32B:
        expected = inp["colors"].contiguous().view(N, 48)
    else:
        expected = inp["colors"].to(torch.float16).view(N, 48)
    torch.testing.assert_close(colors_packed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("sh_degree", [0, 1, 2])
def test_colors_low_sh_dtype(sh_degree):
    """SH 0-2: colors_packed should be float32 (pass-through)."""
    _, _, colors_packed = _call(
        N, sh_degree=sh_degree, sh_compression_mode=SHCompressionMode.NONE
    )
    assert colors_packed.dtype == torch.float32


# ---------------------------------------------------------------------------
# Value correctness
# ---------------------------------------------------------------------------


def test_means_planar_is_transpose():
    """means_planar must be the exact transpose of the input means."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    means_planar, _, _ = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=-1,
        sh_compression_mode=SHCompressionMode.NONE,
    )
    expected = inp["means"].t().contiguous()
    assert torch.allclose(
        means_planar, expected
    ), "means_planar is not the transpose of means"


def test_qso_packed_quats_columns():
    """qso_packed columns 0-3 must round-trip quaternions through fp16 accurately."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    _, qso_packed, _ = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=-1,
        sh_compression_mode=SHCompressionMode.NONE,
    )
    qso_packed_quats = qso_packed[:, :4].float()
    expected = inp["quats"].to(torch.float16).float()
    assert torch.allclose(
        qso_packed_quats, expected
    ), "qso_packed quaternion columns mismatch"


def test_qso_packed_scales_columns():
    """qso_packed columns 4-6 must match scales clamped to fp16 range."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    _, qso_packed, _ = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=-1,
        sh_compression_mode=SHCompressionMode.NONE,
    )
    FP16_MAX = 65504.0
    qso_packed_scales = qso_packed[:, 4:7].float()
    expected = inp["scales"].clamp(-FP16_MAX, FP16_MAX).to(torch.float16).float()
    assert torch.allclose(
        qso_packed_scales, expected
    ), "qso_packed scale columns mismatch"


def test_qso_packed_opacity_column():
    """qso_packed column 7 must match opacities cast to fp16."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    _, qso_packed, _ = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=-1,
        sh_compression_mode=SHCompressionMode.NONE,
    )
    qso_packed_opacity = qso_packed[:, 7].float()
    expected = inp["opacities"].to(torch.float16).float()
    assert torch.allclose(
        qso_packed_opacity, expected
    ), "qso_packed opacity column mismatch"


def test_colors_rgb_padding_zero():
    """RGB mode: the 4th channel (padding) in colors_packed must be 0."""
    _, _, colors_packed = _call(
        N, sh_degree=-1, sh_compression_mode=SHCompressionMode.NONE
    )
    padding = colors_packed[:, 3].float()
    assert (padding == 0).all(), "RGB padding channel is not zero"


# ---------------------------------------------------------------------------
# fp16 clamping
# ---------------------------------------------------------------------------


def test_fp16_clamping_no_error():
    """Scales above fp16 max are silently clamped; no exception raised."""
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    # Set one scale way above fp16 max
    inp["scales"][0, 0] = 1e9
    means_planar, qso_packed, colors_packed = pack_gaussian_inference_scene(
        inp["means"],
        inp["quats"],
        inp["scales"],
        inp["opacities"],
        inp["colors"],
        sh_degree=-1,
        sh_compression_mode=SHCompressionMode.NONE,
    )
    # Should not raise; clamped value should equal fp16 max
    FP16_MAX = torch.finfo(torch.float16).max
    assert float(qso_packed[0, 4].float()) == pytest.approx(FP16_MAX, rel=1e-3)


# ---------------------------------------------------------------------------
# Input validation errors (Python wrapper)
# ---------------------------------------------------------------------------


def test_error_cpu_tensor():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(ValueError, match="CUDA"):
        pack_gaussian_inference_scene(
            inp["means"].cpu(),
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_wrong_dtype():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(TypeError, match="float32"):
        pack_gaussian_inference_scene(
            inp["means"].to(torch.float16),
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_wrong_means_shape():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(ValueError, match="means"):
        pack_gaussian_inference_scene(
            inp["means"][:, :2],  # wrong last dim
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_invalid_sh_degree():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(ValueError, match="sh_degree"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=5,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_invalid_sh_compression_mode():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(TypeError, match="SHCompressionMode"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=3,
        )


def test_error_non_tensor():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(TypeError, match="torch.Tensor"):
        pack_gaussian_inference_scene(
            "not_a_tensor",
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_sh_degree_not_int():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(TypeError, match="sh_degree"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=1.0,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_rgb_colors_with_sh_degree():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(ValueError, match="requires colors to be 3D"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_wrong_sh_k():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=2)
    inp["colors"] = inp["colors"][:, :4, :]
    with pytest.raises(ValueError, match=r"requires colors shape .*9"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=2,
            sh_compression_mode=SHCompressionMode.NONE,
        )


@pytest.mark.parametrize("sh_degree", [-1, 0, 1, 2])
@pytest.mark.parametrize(
    "sh_compression_mode",
    [SHCompressionMode.PACKED_32B, SHCompressionMode.PACKED_16B],
)
def test_error_compression_requires_sh3(sh_degree, sh_compression_mode):
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=sh_degree)
    with pytest.raises(ValueError, match="requires sh_degree=3"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=sh_degree,
            sh_compression_mode=sh_compression_mode,
        )


def test_error_rgb_mode_rejects_3d_colors():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=0)
    with pytest.raises(ValueError, match="colors"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_sh_last_dim_must_be_three():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=0)
    inp["colors"] = torch.randn(
        N, 1, 4, device=DEVICE, generator=_make_generator(DEVICE, 29)
    )
    with pytest.raises(ValueError, match=r"requires colors shape .*1, 3"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"],
            sh_degree=0,
            sh_compression_mode=SHCompressionMode.NONE,
        )


def test_error_colors_leading_dim_must_match_means():
    from gsplat_scene.functional import pack_gaussian_inference_scene

    inp = _make_inputs(N, sh_degree=-1)
    with pytest.raises(ValueError, match="colors dim 0"):
        pack_gaussian_inference_scene(
            inp["means"],
            inp["quats"],
            inp["scales"],
            inp["opacities"],
            inp["colors"][:-1],
            sh_degree=-1,
            sh_compression_mode=SHCompressionMode.NONE,
        )


# ---------------------------------------------------------------------------
# Device consistency
# ---------------------------------------------------------------------------


def test_output_on_cuda():
    means_planar, qso_packed, colors_packed = _call(N, sh_degree=-1)
    assert means_planar.device.type == "cuda"
    assert qso_packed.device.type == "cuda"
    assert colors_packed.device.type == "cuda"
